"""MagicMakeup (lip mode) over the multi-sample set on a RunPod GPU (8회차 "샘플별 지표·가속화", 2026-11-23).

Run from the Magic-Makeup repo root (commit b4d16f5e), after face masks exist for every input (preprocess/faceparsing.py):
    python run_mm_multi.py <inputs_dir> <masks_dir> <out_dir> <flux_kontext_path> <lora_path> [variants=base,steps14,steps8,res768,full]

Uses the repo's own test_single.load_models / run_single (as the 11-20 run, logs/run_mm.py). Reference setting = the 11-20
A6000 run: label lip, 1024, guidance 2.5, 28 steps (the pipeline default), memory_mode model_offload, seed 0.
  base     all pairs in samples.json
  others   only the pairs in TIMED: steps14 / steps8 = fewer denoising steps (num_inference_steps passed into the pipeline
           call), res768 = 768x768, full = memory_mode full (whole pipeline on the GPU, no CPU offload), 28 steps, 1024
Writes <out_dir>/magicmakeup/<variant>/<id>.png, time.json ({id: sec, "_all": median}) and run_log.json.
Timing = run_single() only, CUDA-synchronised; the first pair of each variant runs once more at the end and that time is kept.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time

import torch

sys.path.insert(0, os.getcwd())
import test_single as T  # noqa: E402

TIMED = ['p01', 'p05', 'u01']
CFG = {'base': (28, 1024, 'model_offload'), 'steps14': (14, 1024, 'model_offload'), 'steps8': (8, 1024, 'model_offload'),
       'res768': (28, 768, 'model_offload'), 'full': (28, 1024, 'full')}


class Steps:
    """Pass num_inference_steps into every pipeline call; everything else goes to the real pipeline."""

    def __init__(self, pipe, n):
        self.pipe, self.n = pipe, n

    def __call__(self, **kw):
        return self.pipe(num_inference_steps=self.n, **kw)

    def __getattr__(self, k):
        return getattr(self.pipe, k)


def main():
    inp, masks, out, flux, lora = sys.argv[1:6]
    variants = next((a.split('=', 1)[1].split(',') for a in sys.argv[6:] if a.startswith('variants=')), list(CFG))
    pairs = [p['id'] for p in json.load(open(os.path.join(inp, 'samples.json'), encoding='utf-8'))['pairs']]
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'], text=True).strip()
    log = {'gpu': gpu, 'torch': torch.__version__, 'repo_commit': subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip(),
           'reference_setting': 'lip, 1024, guidance 2.5, 28 steps, model_offload, seed 0 (11-20 A6000 run)', 'variants': {}}
    loaded, models = None, None
    for v in variants:
        steps, res, mode = CFG[v]
        if loaded != mode:
            models = pipe = g = c = None   # drop every reference to the previous pipeline before loading the next one
            torch.cuda.empty_cache()
            t0 = time.time()
            a = argparse.Namespace(model_path=flux, lora_path=lora, memory_mode=mode)
            models = T.load_models(a)
            loaded = mode
            log.setdefault('load_sec', {})[mode] = round(time.time() - t0, 1)
        pipe, g, c = models
        vdir = os.path.join(out, 'magicmakeup', v)
        os.makedirs(vdir, exist_ok=True)
        ids = pairs if v == 'base' else [i for i in TIMED if i in pairs]
        tm = {}
        torch.cuda.reset_peak_memory_stats()
        for k, pid in enumerate(ids + ids[:1]):
            dst = os.path.join(vdir, f'{pid}.png') if k < len(ids) else os.path.join(vdir, '_warm_repeat.png')
            a = argparse.Namespace(source_image=os.path.join(inp, f'{pid}_src_1024.png'), source_mask=os.path.join(masks, f'{pid}_src_1024.png'),
                                   reference_image=os.path.join(inp, f'{pid}_ref_1024.png'), reference_mask=os.path.join(masks, f'{pid}_ref_1024.png'),
                                   output_path=dst, save_panel=False, panel_path='', label='lip', guidance_scale=2.5,
                                   height=res, width=res, seed=0, memory_mode=mode)
            torch.cuda.synchronize()
            t = time.time()
            T.run_single(a, Steps(pipe, steps), g, c)
            torch.cuda.synchronize()
            tm[pid] = round(time.time() - t, 1)
            print(v, pid, tm[pid], flush=True)
        if os.path.exists(os.path.join(vdir, '_warm_repeat.png')):
            os.remove(os.path.join(vdir, '_warm_repeat.png'))
        tm['_all'] = round(statistics.median([tm[i] for i in ids]), 1)
        json.dump(tm, open(os.path.join(vdir, 'time.json'), 'w'), indent=1)
        log['variants'][v] = {'steps': steps, 'res': res, 'memory_mode': mode, 'pairs': ids, 'sec_median': tm['_all'],
                              'peak_vram_gb': round(torch.cuda.max_memory_allocated() / 1e9, 2)}
        json.dump(log, open(os.path.join(out, 'magicmakeup', 'run_log.json'), 'w'), indent=1)
    print('ALL_DONE', json.dumps({k: v['sec_median'] for k, v in log['variants'].items()}), flush=True)


if __name__ == '__main__':
    main()

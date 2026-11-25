"""Stable-Makeup over the multi-sample set on a RunPod GPU (8회차 "샘플별 지표·가속화", 2026-11-23).

Run from the Stable-Makeup repo root (commit 88c8ca51), with the repo's weights in ./models as for infer_kps.py:
    python run_sm_multi.py <inputs_dir> <out_dir> <sd15_path> [variants=base,steps15,steps8,dpm15,fp16]

Same model build as the repo's infer_kps.py (its module-level code is not imported: the file has a syntax error on the
model_id line at this commit). Reference setting = the 11-20 Colab run: 512, DDIM 30 steps, guidance 1.6, fp32, seed 0.
  base     all pairs in samples.json
  others   only the pairs in TIMED (speed-ups): steps15 / steps8 = DDIM with fewer steps, dpm15 = DPM-Solver++ 15 steps,
           fp16 = the whole model in float16, DDIM 30 steps
Writes <out_dir>/stablemakeup/<variant>/<id>.png, time.json ({id: sec, "_all": median}) and run_log.json.
Timing = generate() only, CUDA-synchronised; the first image of each variant is a warm-up and timed again afterwards.
11-23 A40 run (runpod_handoff/RUNPOD_RESULT.md): run with PYTHONPATH=<repo root> so the repo's bundled diffusers is used; the
Pod copy also loaded weights with weights_only=True and had a face-detector fallback (the run log shows no detector error).
"""
import json
import os
import statistics
import subprocess
import sys
import time

import torch
from diffusers import ControlNetModel, DDIMScheduler, DPMSolverMultistepScheduler
from diffusers import UNet2DConditionModel as OriginalUNet2DConditionModel
from diffusers.utils import load_image
from PIL import Image

sys.path.insert(0, os.getcwd())
from detail_encoder.encoder_plus import detail_encoder  # noqa: E402
from facelib import FaceDetector  # noqa: E402
from spiga.inference.config import ModelConfig  # noqa: E402
from spiga.inference.framework import SPIGAFramework  # noqa: E402
from spiga_draw import spiga_process, spiga_segmentation  # noqa: E402
from pipeline_sd15 import StableDiffusionControlNetPipeline  # noqa: E402  (repo root at this commit, not utils/)

TIMED = ['p01', 'p05', 'u01']
STEPS = {'base': 30, 'steps15': 15, 'steps8': 8, 'dpm15': 15, 'fp16': 30}


def build(sd15, dtype):
    unet = OriginalUNet2DConditionModel.from_pretrained(sd15, subfolder='unet', torch_dtype=dtype).to('cuda')
    id_enc, pose_enc = ControlNetModel.from_unet(unet), ControlNetModel.from_unet(unet)
    mk = detail_encoder(unet, './models/image_encoder_l', 'cuda', dtype=dtype)
    mk.load_state_dict(torch.load('./models/stablemakeup/pytorch_model.bin'), strict=False)
    id_enc.load_state_dict(torch.load('./models/stablemakeup/pytorch_model_1.bin'), strict=False)
    pose_enc.load_state_dict(torch.load('./models/stablemakeup/pytorch_model_2.bin'), strict=False)
    for m in (id_enc, pose_enc, mk):
        m.to('cuda', dtype=dtype)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(sd15, safety_checker=None, unet=unet,
                                                             controlnet=[id_enc, pose_enc], torch_dtype=dtype).to('cuda')
    return pipe, mk


def main():
    inp, out, sd15 = sys.argv[1:4]
    variants = next((a.split('=', 1)[1].split(',') for a in sys.argv[4:] if a.startswith('variants=')), list(STEPS))
    pairs = [p['id'] for p in json.load(open(os.path.join(inp, 'samples.json'), encoding='utf-8'))['pairs']]
    detector = FaceDetector(weight_path='./models/mobilenet0.25_Final.pth')
    SPIGAFramework(ModelConfig('300wpublic'))   # loads the SPIGA weights once (spiga_draw keeps its own processor)
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'], text=True).strip()
    log = {'gpu': gpu, 'torch': torch.__version__, 'repo_commit': subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip(),
           'reference_setting': '512, DDIM 30, guidance 1.6, fp32, seed 0 (11-20 Colab T4 run)', 'variants': {}}
    loaded = None
    for v in variants:
        dtype = torch.float16 if v == 'fp16' else torch.float32
        if loaded != dtype:
            t0 = time.time()
            pipe, mk = build(sd15, dtype)
            loaded = dtype
            log.setdefault('load_sec', {})[str(dtype)] = round(time.time() - t0, 1)
        base_cfg = pipe.scheduler.config
        pipe.scheduler = (DPMSolverMultistepScheduler.from_config(base_cfg, algorithm_type='dpmsolver++') if v == 'dpm15'
                          else DDIMScheduler.from_config(base_cfg))
        vdir = os.path.join(out, 'stablemakeup', v)
        os.makedirs(vdir, exist_ok=True)
        ids = pairs if v == 'base' else [i for i in TIMED if i in pairs]
        tm = {}
        torch.cuda.reset_peak_memory_stats()
        for k, pid in enumerate(ids + ids[:1]):          # the first id once more at the end: its first run was the warm-up
            src = load_image(os.path.join(inp, f'{pid}_src_512.png')).resize((512, 512))
            ref = load_image(os.path.join(inp, f'{pid}_ref_512.png')).resize((512, 512))
            spigas = spiga_process(src, detector)
            pose = Image.new('RGB', src.size) if spigas is False else spiga_segmentation(spigas, size=512)
            torch.manual_seed(0)
            torch.cuda.synchronize()
            t = time.time()
            img = mk.generate(id_image=[src, pose], makeup_image=ref, pipe=pipe, guidance_scale=1.6,
                              num_inference_steps=STEPS[v])
            torch.cuda.synchronize()
            sec = round(time.time() - t, 2)
            if k < len(ids):
                img.save(os.path.join(vdir, f'{pid}.png'))
            tm[pid] = sec                                  # warm-up value is overwritten by the repeat at the end
            print(v, pid, sec, flush=True)
        tm['_all'] = round(statistics.median([tm[i] for i in ids]), 2)
        json.dump(tm, open(os.path.join(vdir, 'time.json'), 'w'), indent=1)
        log['variants'][v] = {'steps': STEPS[v], 'scheduler': 'DPM-Solver++' if v == 'dpm15' else 'DDIM', 'dtype': str(dtype),
                              'pairs': ids, 'sec_median': tm['_all'], 'peak_vram_gb': round(torch.cuda.max_memory_allocated() / 1e9, 2)}
        json.dump(log, open(os.path.join(out, 'stablemakeup', 'run_log.json'), 'w'), indent=1)
    print('ALL_DONE', json.dumps({k: v['sec_median'] for k, v in log['variants'].items()}), flush=True)


if __name__ == '__main__':
    main()

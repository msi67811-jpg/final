"""FPMT on the CPU: reference run over the multi-sample set plus speed-up variants (8회차 "가속화 가능성", 2026-11-23).

    python research/makeup_gen/fpmt_accel.py <fpmt/quick_start> <root=research/data/makeup_gen> [variants=base,threads1,...]

Same model, weights and CPU patches as run_fpmt.py (nothing in the repo is edited). For every pair in multi/samples.json and
every variant: one transfer, mapped back onto the source photo (exact inverse of the repo's resize + centre crop), saved as
multi/out/fpmt/<variant>/<id>.png. Timing: forward pass only (face parsing excluded, as run_fpmt.py), on the first TIMED pairs,
1 warm-up + 9 runs, median; time.json holds the median over those pairs as "_all" plus each pair's median.

Variants
  base       torch default threads, input 512 (resize 573, crop 512)  = the reference setting
  threadsN   torch.set_num_threads(N)
  in256      input 256 (resize 287, crop 256), same 512 weights; output upscaled back by the frame mapping
  onnx       the same graph exported with torch.onnx (opset 17) and run in ONNX Runtime CPU (default session options)
Not run: dynamic int8 (torch.ao quantize_dynamic only covers nn.Linear / RNN; FPMT has none, so the model would be unchanged);
static int8 needs per-layer calibration data and was out of scope.
"""
import json
import os
import platform
import statistics
import subprocess
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image

TIMED = ['p01', 'p05', 'u01']
REPEATS = 9


def load_model(qs):
    sys.path.insert(0, qs)
    sys.argv = [sys.argv[0], '--gpu', '-1']
    import networks
    import utils
    _gk = networks.Lap_Pyramid_Conv.gauss_kernel
    networks.Lap_Pyramid_Conv.gauss_kernel = lambda self, device=torch.device('cpu'), channels=3: _gk(self, device, channels)

    def init_net_cpu(net, gpu, init_type='normal', gain=0.02):
        net.to(gpu)
        utils.init_weights(net, init_type, gain)
        return net
    utils.init_net = init_net_cpu
    from dataset import MakeupDataset
    from model import FPMT
    from options import Options
    opts = Options().parse()
    opts.phase = 'test_pair'
    model = FPMT(opts)
    ck = torch.load(os.path.join(qs, 'weights', 'FPMT_L3_512.pth'), map_location='cpu', weights_only=False)
    model.gen.load_state_dict(ck['gen'], strict=False)
    model.eval()
    return model, opts, MakeupDataset


class Gen(torch.nn.Module):
    """Only the transfer image, so the graph has one output for ONNX."""

    def __init__(self, gen):
        super().__init__()
        self.gen = gen

    def forward(self, s, sp, r, rp):
        return self.gen(source_img=s, source_parse=sp, ref_img=r, ref_parse=rp, is_train=True)[2][:, 0:3]


def to_img(t):
    return Image.fromarray(((t[0].clamp(-1, 1).permute(1, 2, 0).numpy() + 1) * 127.5).round().astype(np.uint8))


def back_to_frame(out_img, src_path, S, C):
    src = np.asarray(Image.open(src_path).convert('RGB'))
    H, W = src.shape[:2]
    out = np.asarray(out_img.resize((C, C), Image.LANCZOS)) if out_img.size[0] != C else np.asarray(out_img)
    o = (S - C) // 2
    M = np.float32([[W / S, 0, o * W / S], [0, H / S, o * H / S]])
    warped = cv2.warpAffine(out, M, (W, H), flags=cv2.INTER_LANCZOS4)
    cover = cv2.warpAffine(np.ones((C, C), np.float32), M, (W, H), flags=cv2.INTER_LINEAR)[..., None]
    return Image.fromarray(np.clip(warped * cover + src * (1 - cover), 0, 255).round().astype(np.uint8))


def main():
    qs, root = os.path.abspath(sys.argv[1]), sys.argv[2]
    variants = next((a.split('=', 1)[1].split(',') for a in sys.argv[3:] if a.startswith('variants=')),
                    ['base', 'threads1', 'threads2', 'threads8', 'in256', 'onnx'])
    multi = os.path.join(root, 'multi')
    pairs = json.load(open(os.path.join(multi, 'samples.json'), encoding='utf-8'))['pairs']
    model, opts, MakeupDataset = load_model(qs)
    default_threads = torch.get_num_threads()
    gen = Gen(model.gen).eval()
    try:
        cpu = subprocess.check_output(['powershell', '-NoProfile', '-Command', '(Get-CimInstance Win32_Processor).Name'], text=True).strip()
    except Exception:
        cpu = platform.processor()
    log = {'cpu': cpu, 'logical_cores': os.cpu_count(), 'torch': torch.__version__, 'default_threads': default_threads,
           'timed_pairs': TIMED, 'repeats': f'1 warm-up + {REPEATS}', 'variants': {}}
    for v in variants:
        torch.set_num_threads(int(v[7:]) if v.startswith('threads') else default_threads)
        size = 256 if v == 'in256' else 512
        opts.crop_size, opts.resize_size = size, round(size * 573 / 512)
        out_dir = os.path.join(multi, 'out', 'fpmt', v)
        os.makedirs(out_dir, exist_ok=True)
        sess, note = None, ''
        tm = {}
        for p in pairs:
            opts.data_root = os.path.join(root, p['fpmt_root'], 'images')
            ds = MakeupDataset(opts)
            data = {k: v_[None] for k, v_ in ds[0].items()}
            args = (data['non_makeup_img'], data['non_makeup_split_parse'], data['makeup_img'], data['makeup_split_parse'])
            if v == 'onnx' and sess is None:
                import onnxruntime as ort
                path = os.path.join(out_dir, 'fpmt_512.onnx')
                t0 = time.perf_counter()
                try:
                    torch.onnx.export(gen, args, path, opset_version=17, dynamo=False,
                                      input_names=['s', 'sp', 'r', 'rp'], output_names=['out'])
                    sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
                except Exception as e:   # record the reason, keep the other variants going
                    log['variants'][v] = {'error': f'{type(e).__name__}: {str(e)[:400]}'}
                    print(v, 'export failed', log['variants'][v]['error'], flush=True)
                    break
                note = f'export {time.perf_counter() - t0:.1f} s, file {os.path.getsize(path) / 1e6:.1f} MB'

            def run():
                if v == 'onnx':
                    feed = {n: a.numpy() for n, a in zip(['s', 'sp', 'r', 'rp'], args)}
                    return torch.from_numpy(sess.run(None, feed)[0])
                with torch.no_grad():
                    return gen(*args)
            n = 1 + REPEATS if p['id'] in TIMED else 1
            ts = []
            for _ in range(n):
                t = time.perf_counter()
                out = run()
                ts.append(time.perf_counter() - t)
            if p['id'] in TIMED:
                tm[p['id']] = round(statistics.median(ts[1:]), 4)
            src_path = os.path.join(ds.non_makeup_path[0])
            back_to_frame(to_img(out), src_path, opts.resize_size, opts.crop_size).save(os.path.join(out_dir, f"{p['id']}.png"))
            print(v, p['id'], f'{ts[-1]:.2f}s', flush=True)
        if v in log['variants']:
            continue
        tm['_all'] = round(statistics.median([tm[k] for k in TIMED if k in tm]), 4) if tm else None
        json.dump(tm, open(os.path.join(out_dir, 'time.json'), 'w'), indent=1)
        log['variants'][v] = {'threads': torch.get_num_threads(), 'input': size, 'sec_median': tm['_all'], 'per_pair': tm, 'note': note}
        print(v, 'median', tm['_all'], note, flush=True)
    log['not_run'] = {'int8_dynamic': 'quantize_dynamic covers nn.Linear / RNN only; FPMT has none (model unchanged)',
                      'int8_static': 'needs per-layer calibration; out of scope'}
    json.dump(log, open(os.path.join(multi, 'out', 'fpmt', 'accel_log.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(json.dumps({k: v.get('sec_median', v.get('error')) for k, v in log['variants'].items()}))


if __name__ == '__main__':
    main()

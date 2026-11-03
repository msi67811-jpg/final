"""CIEDE2000 of the mean lip colour, AR render vs the real applied photo, as hw01_compare computes it (no SSIM alignment: fast)."""
import re, sys
from pathlib import Path
import cv2, numpy as np
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'research/lip_reference'))
from hw01_compare import Faces, ciede2000, lip_mask, stats, to_lab
from render_ar_offline import fit
RAW = ROOT / 'research/data/ar_eval_public/hw01_raw'; HERE = Path(__file__).resolve().parent
faces = Faces(); cache = {}
for real_p in sorted(RAW.glob('p*_3.jpg')):
    real = fit(cv2.imread(str(real_p))); bare = fit(cv2.imread(str(real_p.with_name(real_p.name.replace('_3.jpg', '_0.jpg')))))
    rp, bp = faces.points(real), faces.points(bare); mw = np.linalg.norm(rp[61] - rp[291]); inset = max(1, round(0.03 * mw))
    cache[real_p.stem] = (real, bare, lip_mask(rp, real.shape[:2], inset), lip_mask(bp, bare.shape[:2], inset))
for d in sys.argv[1:]:
    vals = []
    for stem, (real, bare, rm, bm) in cache.items():
        ar = cv2.imread(str(HERE / d / (stem + '.png')))
        if ar.shape[:2] != bare.shape[:2]: ar = cv2.resize(ar, bare.shape[1::-1])
        vals.append(float(ciede2000(stats(to_lab(ar), bm)['mean'], stats(to_lab(real), rm)['mean'])))
    print(f'{d:14s} dE median {np.median(vals):.2f}  mean {np.mean(vals):.2f}  (<=4: {sum(v <= 4 for v in vals)}/{len(vals)})')

"""The gloss layer of an AR lip render, isolated: render with gloss minus the same render with the gloss terms switched off.
Local only.

  area       share of the lip pixels where the gloss raises L* by more than DL = 4 (the highlight area)
  (the other figures use the pixels above 30% of the peak, so a dim highlight is judged on its own shape)
  peak       99th percentile of the L* rise (how bright the brightest highlight is)
  plateau    among highlight pixels, the share within 80% of the peak: high = a flat-topped, uniformly bright patch
  edge       mean L* rise gradient at the highlight boundary / peak: high = a hard-edged patch (reads as a drawn line)
  chroma     median C*ab of highlight pixels with gloss / without gloss: below 1 = the colour is washed out
  dh         median hue shift (degrees) of highlight pixels
  db         mean b* shift: positive = yellower
  centre     share of the highlight area inside the middle 40% of the mouth width (focus on the centre of the lips)

    python gloss_layer.py CUR_AR_DIR NOGLOSS_AR_DIR [--glossy]
"""
import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "research/lip_reference"))
from eval_lapa import LIPS_INNER, LIPS_OUTER  # noqa: E402
from hw01_compare import Faces, to_lab  # noqa: E402
from render_ar_offline import fit  # noqa: E402

RAW = ROOT / "research/data/ar_eval_public/hw01_raw"
DL = 4.0
GLOSSY = {"m15", "m18", "m21", "m24", "m17", "m05", "m16"}      # texture "glossy" in hw01_raw/meta.csv


def measure(cur_dir, ref_dir, only_glossy):
    faces = Faces()
    rows = []
    for real_p in sorted(RAW.glob("p*_3.jpg")):
        if only_glossy and re.search(r"_(m\d+)_3", real_p.name).group(1) not in GLOSSY:
            continue
        bare = fit(cv2.imread(str(real_p.with_name(real_p.name.replace("_3.jpg", "_0.jpg")))))
        a = cv2.imread(str(cur_dir / (real_p.stem + ".png"))); b = cv2.imread(str(ref_dir / (real_p.stem + ".png")))
        if a.shape != bare.shape: a = cv2.resize(a, bare.shape[1::-1])
        if b.shape != bare.shape: b = cv2.resize(b, bare.shape[1::-1])
        pts = faces.points(bare); mw = np.linalg.norm(pts[61] - pts[291])
        m = np.zeros(bare.shape[:2], np.uint8)
        cv2.fillPoly(m, [np.round(pts[LIPS_OUTER]).astype(np.int32)], 1); cv2.fillPoly(m, [np.round(pts[LIPS_INNER]).astype(np.int32)], 0)
        m = m.astype(bool)
        La, Lb = to_lab(a), to_lab(b)
        D = (La[..., 0] - Lb[..., 0]); Dm = D[m]
        peak0 = float(np.percentile(Dm, 99))
        hi_abs = m & (D > DL)                                   # absolute: the area figure
        hi = m & (D > max(2.5, 0.3 * peak0))                    # relative to the peak: shape, edge and colour of the highlight itself
        if hi.sum() < 20:
            rows.append({"name": real_p.stem[:3], "area": float(hi_abs.sum() / m.sum()), "peak": float(np.percentile(Dm, 99)), "plateau": np.nan, "edge": np.nan, "chroma": np.nan, "dh": np.nan, "db": np.nan, "centre": np.nan})
            continue
        peak = float(np.percentile(Dm, 99))
        gx = cv2.Sobel(cv2.GaussianBlur(D, (0, 0), 1.0), cv2.CV_32F, 1, 0); gy = cv2.Sobel(cv2.GaussianBlur(D, (0, 0), 1.0), cv2.CV_32F, 0, 1)
        grad = np.hypot(gx, gy) / 8.0
        ring = m & cv2.dilate(hi.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ~cv2.erode(hi.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        Ca = np.hypot(La[..., 1], La[..., 2]); Cb = np.hypot(Lb[..., 1], Lb[..., 2])
        ha = np.degrees(np.arctan2(La[..., 2], La[..., 1])); hb = np.degrees(np.arctan2(Lb[..., 2], Lb[..., 1]))
        dh = (ha - hb + 180) % 360 - 180
        ys, xs = np.nonzero(hi)
        cx = (pts[0][0] + pts[17][0]) / 2
        rows.append({"name": real_p.stem[:3], "area": float(hi_abs.sum() / m.sum()), "peak": peak, "plateau": float((D[hi] > 0.8 * peak).mean()),
                     "edge": float(grad[ring].mean() / max(peak, 1e-6)) if ring.any() else np.nan,
                     "chroma": float(np.median(Ca[hi] / np.maximum(Cb[hi], 1e-3))), "dh": float(np.median(np.abs(dh[hi]))),
                     "db": float((La[..., 2] - Lb[..., 2])[hi].mean()), "centre": float((np.abs(xs - cx) < 0.2 * mw).mean())})
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cur"); ap.add_argument("ref"); ap.add_argument("--glossy", action="store_true")
    a = ap.parse_args()
    here = Path(__file__).resolve().parent
    for cur in a.cur.split(","):
        rows = measure(here / cur, here / a.ref, a.glossy)
        med = lambda k: float(np.nanmedian([r[k] for r in rows]))
        print(f"{cur:12s} n={len(rows):2d}  area {100 * med('area'):4.1f}%  peak {med('peak'):4.1f}  plateau {100 * med('plateau'):3.0f}%  edge {med('edge'):4.2f}  "
              f"chroma x{med('chroma'):4.2f}  hue shift {med('dh'):4.1f}deg  db {med('db'):+4.1f}  centre {100 * med('centre'):3.0f}%")

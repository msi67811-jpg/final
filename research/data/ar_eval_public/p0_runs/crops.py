"""Lip crops side by side: bare | AR run 1 | AR run 2 ... | real (AFTER). Local file only.

    python crops.py OUT.jpg p01 p05 p08 --ar base_ar [--ar p01_ar ...]
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "research/lip_reference"))
from eval_lapa import LIPS_OUTER, make_landmarker  # noqa: E402
from render_ar_offline import fit  # noqa: E402
import mediapipe as mp  # noqa: E402

RAW = ROOT / "research/data/ar_eval_public/hw01_raw"
HERE = Path(__file__).resolve().parent

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("people", nargs="+")
ap.add_argument("--ar", action="append", default=[])
a = ap.parse_args()
lm = make_landmarker()
rows = []
for person in a.people:
    real_p = next(RAW.glob(f"{person}_*_3.jpg"))
    bare = fit(cv2.imread(str(real_p.with_name(real_p.name.replace("_3.jpg", "_0.jpg")))))
    real = fit(cv2.imread(str(real_p)))
    res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bare, cv2.COLOR_BGR2RGB)))
    h, w = bare.shape[:2]
    pts = np.array([[p.x * w, p.y * h] for p in res.face_landmarks[0]])[LIPS_OUTER]
    x0, y0 = pts.min(0); x1, y1 = pts.max(0)
    mw = x1 - x0
    box = lambda im: cv2.resize(im[max(0, int(y0 - 0.35 * mw)):int(y1 + 0.35 * mw), max(0, int(x0 - 0.2 * mw)):int(x1 + 0.2 * mw)], (360, 220))
    tiles = [box(bare)]
    for d in a.ar:
        ar = cv2.imread(str(HERE / d / (real_p.stem + ".png")))
        tiles.append(box(cv2.resize(ar, (w, h)) if ar.shape[:2] != (h, w) else ar))
    # Real AFTER: same framing via its own landmarks.
    res2 = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(real, cv2.COLOR_BGR2RGB)))
    h2, w2 = real.shape[:2]
    p2 = np.array([[p.x * w2, p.y * h2] for p in res2.face_landmarks[0]])[LIPS_OUTER]
    x0, y0 = p2.min(0); x1, y1 = p2.max(0); mw = x1 - x0
    tiles.append(box(real))
    rows.append(np.hstack(tiles))
cv2.imwrite(str(HERE / a.out), np.vstack(rows))
print("labels: bare |", " | ".join(a.ar), "| real(AFTER)")

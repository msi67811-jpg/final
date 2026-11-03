"""Lip edge metrics on offline AR renders (P0-2). Local only; nothing leaves the PC.

For each pair the paint amount D = |Lab(AR) - Lab(bare)| (CIE76) is sampled along the normals of the outer-lip
landmark ring, and the photo's own lip/skin edge is read from the bare photo (redness).

  edge_ar     10-90% rise width of the paint profile, in px (mouth-width normalised too)
  edge_photo  10-90% rise width of the bare photo's own lip edge (redness profile)
  ratio       edge_ar / edge_photo   (< 1: the paint edge is sharper than the photo's own lip edge)
  corner      paint at the mouth corners (within 0.06 mouth widths of 61 / 291) / paint at the lip middle
  spill       share of changed pixels (D > 6) outside the landmark ring grown by 3% of the mouth width

    python edge_metrics.py base_ar [s3_ar ...]
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "research/lip_reference"))
from eval_lapa import LIPS_OUTER  # noqa: E402
from hw01_compare import Faces, to_lab  # noqa: E402
from render_ar_offline import fit  # noqa: E402

RAW = ROOT / "research/data/ar_eval_public/hw01_raw"
HERE = Path(__file__).resolve().parent
D = np.arange(-8.0, 8.01, 0.5)


def width(profile, d=D):
    """10-90% rise width of a profile that goes from ~0 (outside, d > 0) to a plateau (inside, d < 0)."""
    inside = profile[(d <= -3) & (d >= -7)].mean()
    outside = profile[d >= 5].mean()
    if abs(inside - outside) < 1e-6:
        return np.nan
    n = np.clip((profile - outside) / (inside - outside), 0, 1)
    # n runs 1 (inside) → 0 (outside) with growing d; find where it crosses 0.9 and 0.1.
    def cross(level):
        for i in range(len(d) - 1):
            if n[i] >= level > n[i + 1]:
                return d[i] + (n[i] - level) / max(n[i] - n[i + 1], 1e-6) * (d[i + 1] - d[i])
        return np.nan
    return cross(0.1) - cross(0.9)


def sample(img, pts):
    return ndimage.map_coordinates(img, [pts[:, 1], pts[:, 0]], order=1, mode="nearest")


def measure(ar_dir):
    faces = Faces()
    rows = []
    for real_p in sorted(RAW.glob("p*_3.jpg")):
        bare = fit(cv2.imread(str(real_p.with_name(real_p.name.replace("_3.jpg", "_0.jpg")))))
        ar = cv2.imread(str(ar_dir / (real_p.stem + ".png")))
        if ar.shape[:2] != bare.shape[:2]:
            ar = cv2.resize(ar, bare.shape[1::-1])
        pts = faces.points(bare)
        ring = pts[LIPS_OUTER].astype(np.float64)
        mw = np.linalg.norm(pts[61] - pts[291])
        # Densify the ring, skipping the corners (the two points 61 and 291).
        n = len(ring)
        dense = np.vstack([np.linspace(ring[i], ring[(i + 1) % n], 6, endpoint=False) for i in range(n)])
        c = dense.mean(0)
        tang = np.roll(dense, -1, 0) - np.roll(dense, 1, 0)
        nor = np.stack([tang[:, 1], -tang[:, 0]], 1)
        nor /= np.linalg.norm(nor, axis=1, keepdims=True) + 1e-9
        nor *= np.sign(np.sum(nor * (dense - c), 1, keepdims=True) + 1e-9)  # outward
        away = np.minimum(np.linalg.norm(dense - pts[61], axis=1), np.linalg.norm(dense - pts[291], axis=1)) > 0.15 * mw
        lab_a, lab_b = to_lab(ar), to_lab(bare)
        dE = np.linalg.norm(lab_a - lab_b, axis=2).astype(np.float32)
        b32 = bare.astype(np.float32)
        red = ((b32[..., 2] - b32[..., 1]) / (b32.sum(2) + 1e-3)).astype(np.float32)
        profs_d, profs_r = [], []
        for p, nv in zip(dense[away], nor[away]):
            xy = p[None, :] + D[:, None] * nv[None, :]
            profs_d.append(sample(dE, xy)); profs_r.append(sample(red, xy))
        pd, pr = np.mean(profs_d, 0), np.mean(profs_r, 0)
        real = fit(cv2.imread(str(real_p)))
        rpts = faces.points(real)
        rring = rpts[LIPS_OUTER].astype(np.float64)
        rd = np.vstack([np.linspace(rring[i], rring[(i + 1) % n], 6, endpoint=False) for i in range(n)])
        rc = rd.mean(0)
        rt = np.roll(rd, -1, 0) - np.roll(rd, 1, 0)
        rn = np.stack([rt[:, 1], -rt[:, 0]], 1)
        rn /= np.linalg.norm(rn, axis=1, keepdims=True) + 1e-9
        rn *= np.sign(np.sum(rn * (rd - rc), 1, keepdims=True) + 1e-9)
        rmw = np.linalg.norm(rpts[61] - rpts[291])
        raway = np.minimum(np.linalg.norm(rd - rpts[61], axis=1), np.linalg.norm(rd - rpts[291], axis=1)) > 0.15 * rmw
        r32 = real.astype(np.float32)
        rred = ((r32[..., 2] - r32[..., 1]) / (r32.sum(2) + 1e-3)).astype(np.float32)
        rprof = np.mean([sample(rred, q[None, :] + D[:, None] * v[None, :]) for q, v in zip(rd[raway], rn[raway])], 0)
        wr = width(rprof)
        # Mouth width scale: the 10-90 widths are in px; also report relative to the mouth.
        wa, wp = width(pd), width(pr)
        # Corners vs middle paint.
        yy, xx = np.mgrid[0:dE.shape[0], 0:dE.shape[1]]
        poly = np.zeros(dE.shape, np.uint8)
        cv2.fillPoly(poly, [ring.astype(np.int32)], 1)
        inner = cv2.erode(poly, np.ones((3, 3), np.uint8)).astype(bool)
        corner = np.zeros(dE.shape, bool)
        for k in (61, 291):
            corner |= (np.hypot(xx - pts[k][0], yy - pts[k][1]) < 0.06 * mw) & inner
        middle = inner & (np.hypot(xx - (pts[0][0] + pts[17][0]) / 2, yy - (pts[0][1] + pts[17][1]) / 2) < 0.2 * mw)
        corner_ratio = float(dE[corner].mean() / max(dE[middle].mean(), 1e-3)) if corner.any() and middle.any() else np.nan
        k = max(3, int(0.03 * mw)) * 2 + 1
        grown = cv2.dilate(poly, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(bool)
        changed = dE > 6
        cover = float(changed[inner].mean())
        spill = float((changed & ~grown).sum() / max(changed.sum(), 1))
        rows.append((real_p.stem[:3], wa, wp, wa / wr if wr and not np.isnan(wr) else np.nan, corner_ratio, spill, mw, wr, cover))
    return rows


if __name__ == "__main__":
    for name in sys.argv[1:]:
        rows = measure(HERE / name)
        a = np.array([r[1:] for r in rows], dtype=float)
        med = lambda i: np.nanmedian(a[:, i])
        print(f"{name:10s} edge_ar {med(0):5.2f}px  bare {med(1):5.2f}px  real(AFTER) {med(6):5.2f}px  ar/real {med(2):4.2f}  "
              f"cover {med(7) * 100:4.1f}%  corner/middle {med(3):4.2f}  spill {med(4) * 100:4.1f}%  (n={len(rows)}, mouth {med(5):.0f}px)")

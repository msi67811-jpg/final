"""How far behind the face does the foundation skin mask fall while the head moves? (P-found-lag, 11-02)

The page paints foundation over the landmark face oval, gated by a skin mask from the selfie-multiclass segmenter. The
mesh is drawn from the current frame's landmarks, but the mask is a screen-space texture that is only refreshed every
TUNING.foundSegEveryMs (120 ms; 400 ms when the frame rate is under 20). Between refreshes the face moves and the mask
stays, so the paint is cut by an old skin / hair boundary.

For a clip this measures, over the face oval of frame t, the mask error when the mask is k frames old:
  stale  : mask(t-k) used as it is
  moved  : mask(t-k) moved with the face (similarity transform from the landmarks of t to those of t-k)
error = mean |skin(t) - mask| over the oval (0 = identical, 1 = completely different), plus the share of the oval that is
skin now but unpainted with the old mask ("uncovered") and not skin now but painted ("spilled").

Local use only: clip and outputs stay under research/data/ (git-ignored); nothing is sent anywhere.

    python research/lip_reference/eval_foundation_lag.py CLIP.mp4 [--out DIR]
"""
import argparse
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

ROOT = Path(__file__).resolve().parents[2]
FACE_MODEL = ROOT / "research/data/models/face_landmarker.task"
SEG_MODEL = ROOT / "research/data/models/selfie_multiclass_256x256.tflite"
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152,
             148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
# Points that move with the head far more than with the expression: eye corners, nose, mouth corners, chin, forehead, cheeks.
RIGID = [33, 133, 263, 362, 1, 4, 61, 291, 152, 10, 234, 454, 168, 6]
SKIN_PROB = (0.35, 0.75)   # TUNING.foundSkinProb
KS = [1, 2, 3, 4, 6, 8, 12]


def smoothstep(lo, hi, x):
    t = np.clip((x - lo) / (hi - lo), 0, 1)
    return t * t * (3 - 2 * t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    for m in (FACE_MODEL, SEG_MODEL):
        if not m.exists():
            sys.exit(f"missing {m}")
    out = a.out or a.clip.parent
    cap = cv2.VideoCapture(str(a.clip))
    if not cap.isOpened():
        sys.exit(f"cannot open {a.clip} (use an ASCII path)")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(FACE_MODEL)), running_mode=vision.RunningMode.VIDEO, num_faces=1))
    seg = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(SEG_MODEL)), running_mode=vision.RunningMode.VIDEO,
        output_category_mask=True, output_confidence_masks=True))

    pts, masks, frames = [], [], []
    i = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        H, W = bgr.shape[:2]
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        ts = int(round(i * 1000 / fps))
        r = face.detect_for_video(img, ts)
        s = seg.segment_for_video(img, ts)
        if r.face_landmarks:
            pts.append(np.array([[p.x * W, p.y * H] for p in r.face_landmarks[0]]))
            prob = s.confidence_masks[3].numpy_view().astype(np.float32)
            cat = s.category_mask.numpy_view()
            skin = smoothstep(*SKIN_PROB, prob) * (cat != 1)          # as applyFoundSkinMask: hair gets 0
            masks.append(cv2.resize(skin.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR))
        else:
            pts.append(None)
            masks.append(None)
        frames.append(bgr if i in (60, 90, 100, 104) else None)
        i += 1
    n = len(pts)
    W, H = masks[next(k for k in range(n) if masks[k] is not None)].shape[::-1]
    print(f"{a.clip.name}: {n} frames, {fps:.0f} fps, {W}x{H}")

    def oval(p):
        m = np.zeros((H, W), np.uint8)
        cv2.fillPoly(m, [np.round(p[FACE_OVAL]).astype(np.int32)], 1)
        return m.astype(bool)

    speed = np.full(n, np.nan)
    for t in range(1, n):
        if pts[t] is not None and pts[t - 1] is not None:
            speed[t] = np.linalg.norm(pts[t].mean(0) - pts[t - 1].mean(0))
    face_w = float(np.median([np.linalg.norm(p[234] - p[454]) for p in pts if p is not None]))
    fast_cut = np.nanpercentile(speed, 75)
    very_cut = np.nanpercentile(speed, 90)
    print(f"face width {face_w:.0f}px; head speed median {np.nanmedian(speed):.1f} px/frame, p90 {np.nanpercentile(speed, 90):.1f}, p75 {fast_cut:.1f}")

    rows = {k: {"stale": [], "moved": [], "unc_s": [], "unc_m": [], "spl_s": [], "spl_m": [], "fast": [], "very": []} for k in KS}
    for t in range(max(KS), n):
        if pts[t] is None or masks[t] is None:
            continue
        R = oval(pts[t])
        area = R.sum()
        now_skin = masks[t][R]
        for k in KS:
            if pts[t - k] is None or masks[t - k] is None:
                continue
            M, _ = cv2.estimateAffinePartial2D(pts[t][RIGID].astype(np.float32), pts[t - k][RIGID].astype(np.float32))
            if M is None:
                continue
            moved = cv2.warpAffine(masks[t - k], M, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderValue=0)
            # warpAffine with WARP_INVERSE_MAP: dst(x) = src(M x), i.e. the old mask sampled where the face was.
            # (M maps current coordinates to old ones, so pass it as the inverse map.)
            old = masks[t - k][R]
            mv = moved[R]
            e = rows[k]
            e["stale"].append(np.abs(old - now_skin).mean()); e["moved"].append(np.abs(mv - now_skin).mean())
            e["unc_s"].append(((now_skin > 0.5) & (old < 0.5)).sum() / area); e["unc_m"].append(((now_skin > 0.5) & (mv < 0.5)).sum() / area)
            e["spl_s"].append(((now_skin < 0.5) & (old > 0.5)).sum() / area); e["spl_m"].append(((now_skin < 0.5) & (mv > 0.5)).sum() / area)
            e["fast"].append(bool(speed[t] >= fast_cut))
            e["very"].append(bool(speed[t] >= very_cut))

    p = lambda v, q: float(np.percentile(v, q)) if len(v) else float("nan")
    print("\nmask error over the face oval (mean |skin(now) - mask|; 0 = same). 'moved' = the old mask moved with the face")
    print(f"{'age':>12s} | {'all frames':^27s} | {'head fast (top 25%)':^27s} | {'head very fast (top 10%)':^27s}")
    print(f"{'':>12s} | {'stale':>8s} {'moved':>8s} {'gain':>7s} | {'stale':>8s} {'moved':>8s} {'gain':>7s} | {'stale':>8s} {'moved':>8s} {'gain':>7s}")
    for k in KS:
        e = rows[k]
        fast = np.array(e["fast"]); very = np.array(e["very"])
        cells = []
        for sel in (np.ones(len(fast), bool), fast, very):
            s = np.array(e["stale"])[sel]; m = np.array(e["moved"])[sel]
            cells.append(f"{np.median(s):8.3f} {np.median(m):8.3f} {100 * (1 - np.median(m) / max(np.median(s), 1e-9)):6.0f}%")
        print(f"{k:3d} fr {1000 * k / fps:4.0f}ms | {cells[0]} | {cells[1]} | {cells[2]}")
    print("\nshare of the face oval: uncovered = skin now but old mask says no; spilled = not skin now but old mask says yes (median, fast frames)")
    for k in (4, 12):
        e = rows[k]
        fast = np.array(e["fast"])
        g = lambda key: 100 * np.median(np.array(e[key])[fast])
        print(f"  {k:2d} frames old ({1000 * k / fps:.0f} ms): uncovered {g('unc_s'):4.1f}% -> {g('unc_m'):4.1f}% moved;  spilled {g('spl_s'):4.1f}% -> {g('spl_m'):4.1f}% moved")

    # Picture: fast frame, mask 4 frames old, moved, and fresh.
    tt = int(np.nanargmax(np.where(np.arange(n) >= 12, speed, np.nan)))
    k = 4
    R = oval(pts[tt]); M, _ = cv2.estimateAffinePartial2D(pts[tt][RIGID].astype(np.float32), pts[tt - k][RIGID].astype(np.float32))
    moved = cv2.warpAffine(masks[tt - k], M, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderValue=0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, tt); _, bgr = cap.read()
    def paint(m, name):
        v = bgr.copy(); v[(m > 0.5) & R] = (0.45 * v[(m > 0.5) & R] + 0.55 * np.array([0, 160, 255])).astype(np.uint8)
        cv2.putText(v, name, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        return cv2.resize(v, (W // 2, H // 2))
    sheet = np.hstack([paint(masks[tt - k], f"mask {k} frames old"), paint(moved, "old mask moved with face"), paint(masks[tt], "fresh mask")])
    cv2.imwrite(str(out / f"{a.clip.stem}_lag_masks.jpg"), sheet)
    print(f"\npicture (frame {tt}, head speed {speed[tt]:.0f} px/frame): {out / (a.clip.stem + '_lag_masks.jpg')}")


if __name__ == "__main__":
    main()

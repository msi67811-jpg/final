"""Lip tracking stability on one recorded video (AR_SPEC 8, P1-2). Same video, any device or build.

Runs the MediaPipe Python Face Landmarker in VIDEO mode on every frame of a clip and re-runs the app's own smoothing
on the lip points, so the numbers show what the AR page's tracking does with that clip:

  * the app's filter (faceTrackerHtml.ts filterLandmarks): anchors follow the raw landmarks with no filter (the face
    centre, and the mouth centre near the mouth, blended by canonical UV distance), and One Euro smooths only each
    point's offset from its anchor. The constants are read from the page (TUNING.smoothMinCutoff / smoothBeta /
    smoothDCutoff), not copied.
  * a frame counts as "still" when the face centre moves less than STILL_REL mouth widths since the last frame.

Metrics, raw landmarks and smoothed:
  jitter_rel   frame-to-frame movement of the lip points / mouth width, on still frames (median, p95)
  jitter_shape the same after taking out the movement of the lip centre (the filter anchors that part on
               purpose, unfiltered, so a shake moves the makeup in the same frame; it smooths only the shape)
  lip_iou_t    IoU of the lip mask (outer minus inner polygon) in consecutive frames: flicker (all frames, still frames)
  lost_rate    share of frames with no face; longest run of lost frames
  lag_frames   how far the smoothed lip centre trails the raw one while the head moves (best shift, frames)
  trail_rel    largest distance between smoothed and raw lip points / mouth width

A recording of the AR screen works as input too (the lip is then painted), but the paint is not read back: a colour
threshold cannot tell a pale gloss reflection or shadow on the paint from skin, so it would give wrong numbers. For
paint position against the lip, compare with a raw camera clip of the same take.

Local use only: the clip and the outputs stay under research/data/ (git-ignored); nothing is sent anywhere.

    python research/lip_reference/eval_stability.py CLIP.mp4 [--out DIR]
"""
import argparse
import csv
import math
import re
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "research/data/models/face_landmarker.task"
PAGE = ROOT / "components/ar/faceTrackerHtml.ts"
UV = ROOT / "components/ar/faceUV.ts"

LIPS_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
LIPS_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
LIPS = LIPS_OUTER + LIPS_INNER
N_MESH, N_LANDMARKS = 468, 478
STILL_REL = 0.01      # face centre movement per frame, in mouth widths, below which a frame is "still"
MIN_STILL = 10        # fewer still frames than this and the jitter figure is not representative


def tuning():
    """Smoothing constants as the page has them."""
    src = PAGE.read_text(encoding="utf-8")
    get = lambda k: float(re.search(k + r":\s*([0-9.]+)", src).group(1))
    return get("smoothMinCutoff"), get("smoothBeta"), get("smoothDCutoff")


def face_uv():
    src = UV.read_text(encoding="utf-8")
    return np.array([float(v) for v in re.search(r"FACE_UV: number\[\] = \[(.*?)\];", src, re.S).group(1).split(",")]).reshape(-1, 2)


def mouth_weights(uv):
    """Per landmark: 1 near the mouth, 0 far from it (faceTrackerHtml.ts mouthWeights)."""
    w = np.zeros(N_LANDMARKS)
    m = uv[LIPS_OUTER].mean(0)
    mw = np.linalg.norm(uv[61] - uv[291])
    d = np.linalg.norm(uv[:N_MESH] - m, axis=1) / mw
    s = np.clip((d - 0.6) / 0.9, 0, 1)
    w[:N_MESH] = 1 - s * s * (3 - 2 * s)
    return w


class AppFilter:
    """The page's landmark filter, for the lip points only (each point is filtered on its own)."""

    def __init__(self, weights, min_cutoff, beta, d_cutoff):
        self.w, self.mc, self.beta, self.dc = weights, min_cutoff, beta, d_cutoff
        self.ready, self.t = False, 0.0
        self.o = self.dv = None

    @staticmethod
    def alpha(cutoff, dt):
        tau = 1 / (2 * math.pi * cutoff)
        return 1 / (1 + tau / dt)

    def __call__(self, face, t):
        """face: (478, 2) normalised landmarks, t seconds. Returns smoothed (40, 2) for LIPS."""
        c = face.mean(0)
        m = face[LIPS].mean(0)
        anchor = c[None, :] + self.w[LIPS][:, None] * (m - c)[None, :]
        off = face[LIPS] - anchor
        if not self.ready or t - self.t > 0.5:
            self.o, self.dv = off.copy(), np.zeros_like(off)
            self.ready = True
        else:
            dt = max(0.001, t - self.t)
            ad = self.alpha(self.dc, dt)
            self.dv = self.dv + ad * ((off - self.o) / dt - self.dv)
            speed = np.linalg.norm(self.dv, axis=1)
            a = self.alpha(self.mc + self.beta * speed, dt)[:, None]
            self.o = self.o + a * (off - self.o)
        self.t = t
        return anchor + self.o


def make_video_landmarker():
    opts = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL)),
        running_mode=vision.RunningMode.VIDEO, num_faces=1, min_face_detection_confidence=0.5, min_tracking_confidence=0.5)
    return vision.FaceLandmarker.create_from_options(opts)


def lip_mask(pts, shape):
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.round(pts[:20]).astype(np.int32)], 1)
    cv2.fillPoly(m, [np.round(pts[20:]).astype(np.int32)], 0)
    return m.astype(bool)


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else float("nan")


def pct(v, q):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    return float(np.percentile(v, q)) if len(v) else float("nan")


def best_lag(raw, sm, idx, max_lag=6.0, step=0.25):
    """Shift L (frames) minimising |smoothed(t) - raw(t - L)| over the frames in idx (linear interpolation)."""
    t = np.arange(len(raw))
    best, best_err = 0.0, float("inf")
    for lag in np.arange(0, max_lag + 1e-9, step):
        ti = t[idx] - lag
        ok = ti >= 0
        if ok.sum() < 5:
            continue
        rx = np.interp(ti[ok], t, raw[:, 0]); ry = np.interp(ti[ok], t, raw[:, 1])
        err = float(np.mean((sm[idx][ok, 0] - rx) ** 2 + (sm[idx][ok, 1] - ry) ** 2))
        if err < best_err:
            best, best_err = float(lag), err
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip", type=Path)
    ap.add_argument("--out", type=Path, default=None, help="output folder (default: next to the clip)")
    a = ap.parse_args()
    if not MODEL.exists():
        sys.exit(f"missing {MODEL}")
    out = a.out or a.clip.parent
    out.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(a.clip))
    if not cap.isOpened():
        sys.exit(f"cannot open {a.clip} (use an ASCII path: OpenCV cannot read some non-ASCII Windows paths)")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    mc, beta, dc = tuning()
    filt = AppFilter(mouth_weights(face_uv()), mc, beta, dc)
    lm = make_video_landmarker()

    frames = []          # per frame dict (None when the face is lost)
    i = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), int(round(i * 1000 / fps)))
        if res.face_landmarks:
            face = np.array([[p.x, p.y] for p in res.face_landmarks[0]])[:N_LANDMARKS]
            sm = filt(face, i / fps)
            scale = np.array([W, H])
            frames.append({"i": i, "raw": face[LIPS] * scale, "sm": sm * scale, "center": face.mean(0) * scale})
        else:
            frames.append(None)
            filt.ready = False
        i += 1
    cap.release()
    n = len(frames)
    det = [f for f in frames if f]
    if len(det) < 5:
        sys.exit(f"face found in only {len(det)} of {n} frames")

    mouth_w = float(np.median([np.linalg.norm(f["raw"][0] - f["raw"][10]) for f in det]))  # LIPS[0]=61, LIPS[10]=291
    # Per frame metrics against the previous frame (only when both have a face).
    rows = []
    prev = None
    for f in frames:
        row = {"frame": len(rows), "t": round(len(rows) / fps, 3), "detected": int(f is not None)}
        if f is not None and prev is not None:
            row["face_move_rel"] = float(np.linalg.norm(f["center"] - prev["center"]) / mouth_w)
            for k in ("raw", "sm"):
                row[f"jitter_{k}"] = float(np.linalg.norm(f[k] - prev[k], axis=1).mean() / mouth_w)
                shape = (f[k] - f[k].mean(0)) - (prev[k] - prev[k].mean(0))
                row[f"shape_{k}"] = float(np.linalg.norm(shape, axis=1).mean() / mouth_w)
                row[f"iou_{k}"] = iou(lip_mask(f[k], (H, W)), lip_mask(prev[k], (H, W)))
            row["trail"] = float(np.linalg.norm(f["sm"] - f["raw"], axis=1).max() / mouth_w)
        rows.append(row)
        prev = f
    got = [r for r in rows if "face_move_rel" in r]
    still = [r for r in got if r["face_move_rel"] < STILL_REL]

    summary = {
        "frames": n, "fps": round(fps, 2), "size": f"{W}x{H}", "mouth_px": round(mouth_w, 1),
        "filter": f"minCutoff {mc} beta {beta} dCutoff {dc} (read from the page)",
        "lost_rate": 1 - len(det) / n,
    }
    run = best = 0
    for f in frames:
        run = run + 1 if f is None else 0
        best = max(best, run)
    summary["lost_longest_run"] = best
    summary["still_frames"] = len(still)
    for k, label in (("raw", "raw"), ("sm", "smoothed")):
        summary[f"jitter_rel_{label}_median"] = pct([r[f"jitter_{k}"] for r in still], 50)
        summary[f"jitter_rel_{label}_p95"] = pct([r[f"jitter_{k}"] for r in still], 95)
        summary[f"jitter_shape_{label}_median"] = pct([r[f"shape_{k}"] for r in still], 50)
        summary[f"jitter_shape_{label}_p95"] = pct([r[f"shape_{k}"] for r in still], 95)
        summary[f"lip_iou_t_{label}_median_all"] = pct([r[f"iou_{k}"] for r in got], 50)
        summary[f"lip_iou_t_{label}_p05_all"] = pct([r[f"iou_{k}"] for r in got], 5)
        summary[f"lip_iou_t_{label}_median_still"] = pct([r[f"iou_{k}"] for r in still], 50)
    summary["trail_rel_max"] = pct([r["trail"] for r in got], 100)
    summary["trail_rel_p95"] = pct([r["trail"] for r in got], 95)
    # Lag while the head moves (the faster half of the frames).
    di = [k for k, f in enumerate(frames) if f]
    raw_c = np.array([frames[k]["raw"].mean(0) for k in di]); sm_c = np.array([frames[k]["sm"].mean(0) for k in di])
    speed = np.r_[0, np.linalg.norm(np.diff(raw_c, axis=0), axis=1)]
    fast = np.nonzero(speed >= np.median(speed))[0]
    summary["lag_frames"] = best_lag(raw_c, sm_c, fast) if len(fast) > 10 else float("nan")

    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("frame", "t", "detected"), k))
    with open(out / f"{a.clip.stem}_stability_frames.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)
    with open(out / f"{a.clip.stem}_stability_summary.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerows(summary.items())

    print(f"{a.clip.name}: {n} frames, {fps:.0f} fps, {W}x{H}, mouth {mouth_w:.0f}px, filter {summary['filter']}")
    print(f"lost {summary['lost_rate'] * 100:.1f}% (longest run {best} frames);  still frames {len(still)} of {len(got)}"
          + ("" if len(still) >= MIN_STILL else f"   WARNING: fewer than {MIN_STILL} still frames, jitter is not representative"))
    print(f"{'':12s} {'raw':>10s} {'smoothed':>10s}")
    f3 = lambda x: f"{x:10.4f}"
    print(f"{'jitter med':12s}" + f3(summary["jitter_rel_raw_median"]) + f3(summary["jitter_rel_smoothed_median"]) + "   (mouth widths / frame, still frames)")
    print(f"{'jitter p95':12s}" + f3(summary["jitter_rel_raw_p95"]) + f3(summary["jitter_rel_smoothed_p95"]))
    print(f"{'shape med':12s}" + f3(summary["jitter_shape_raw_median"]) + f3(summary["jitter_shape_smoothed_median"]) + "   (centre movement taken out)")
    print(f"{'shape p95':12s}" + f3(summary["jitter_shape_raw_p95"]) + f3(summary["jitter_shape_smoothed_p95"]))
    print(f"{'IoU_t med':12s}" + f3(summary["lip_iou_t_raw_median_all"]) + f3(summary["lip_iou_t_smoothed_median_all"]) + "   (consecutive frames, all)")
    print(f"{'IoU_t p05':12s}" + f3(summary["lip_iou_t_raw_p05_all"]) + f3(summary["lip_iou_t_smoothed_p05_all"]))
    print(f"{'IoU_t still':12s}" + f3(summary["lip_iou_t_raw_median_still"]) + f3(summary["lip_iou_t_smoothed_median_still"]))
    print(f"smoothed vs raw: lag {summary['lag_frames']:.2f} frames while moving;  trail p95 {summary['trail_rel_p95']:.3f}  max {summary['trail_rel_max']:.3f} mouth widths")
    print(f"-> {out / (a.clip.stem + '_stability_summary.csv')}")


if __name__ == "__main__":
    main()

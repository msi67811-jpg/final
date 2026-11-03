"""Offset between the foundation edge recorded in the clip (live pipeline) and the same foundation repainted with no lag
(app shader + raw landmarks of that frame): per scanline on the viewer's left side of the face. Local only."""
import sys, cv2, numpy as np
sys.path.insert(0, 'research/lip_reference')
import eval_foundation_lag as E
import mediapipe as mp
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision
base = 'research/data/ar_eval_public/p0_runs'
vid = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(base_options=mpp.BaseOptions(model_asset_path=str(E.FACE_MODEL)), running_mode=vision.RunningMode.VIDEO, num_faces=1))
cap = cv2.VideoCapture('research/data/ar_eval_public/stability/foundation_clip.mp4'); fr = []
while True:
    ok, f = cap.read()
    if not ok: break
    fr.append(f)
P = {}
for t, f in enumerate(fr):
    H, W = f.shape[:2]; r = vid.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(f, cv2.COLOR_BGR2RGB)), int(t * 1000 / 30))
    P[t] = np.array([[q.x * W, q.y * H] for q in r.face_landmarks[0]]) if r.face_landmarks else None
print(f"{'frame':>6s} {'head vx':>8s} {'vy':>6s} | live paint edge minus zero-lag repaint edge (px; + = the live paint is smaller / further inside)")
for k in (60, 98, 100, 102, 104, 16):
    src = cv2.imread(f'{base}/fl_raw/f{k:03d}.jpg'); out = cv2.imread(f'{base}/fl_out/f{k:03d}.png')
    if out.shape != src.shape: out = cv2.resize(out, src.shape[1::-1])
    green = (src[..., 2].astype(int) - out[..., 2].astype(int)) > 40         # the repaint pulls the red channel down a lot
    p = P[k]; lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float32)
    offs = []
    for frac in (0.25, 0.45, 0.65, 0.85):
        y = int(p[1][1] + frac * (p[152][1] - p[1][1])); ys = slice(y - 4, y + 5)
        xs = np.nonzero(green[ys, :].mean(0) > 0.5)[0]
        if len(xs) < 20: continue
        xg = int(xs.min())
        seg = lab[ys, :, 1:3].mean(0); seg = np.stack([np.convolve(seg[:, c], np.ones(5) / 5, 'same') for c in (0, 1)], 1)
        g = np.linalg.norm(np.gradient(seg, axis=0), axis=1)
        lo, hi = max(xg - 45, 0), xg + 45
        xl = lo + int(np.argmax(g[lo:hi])); offs.append((frac, xl - xg, float(g[xl])))
    v = ((P[k + 2] - P[k - 2]).mean(0) / 4.0) if P[k + 2] is not None and P[k - 2] is not None else (np.nan, np.nan)
    print(f"{k:6d} {v[0]:8.1f} {v[1]:6.1f} | " + '  '.join(f"y{int(f*100)}%: {o:+4d}px (edge {g:.2f})" for f, o, g in offs))

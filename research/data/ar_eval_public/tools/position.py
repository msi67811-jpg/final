"""위치·경계: '화장 전후로 색이 바뀐 영역'을 정답으로 삼아 AR이 바꾼 영역과 비교 (IoU·넘침·빠짐).
실제 화장 사진은 맨얼굴에 랜드마크로 정렬 + 주변 피부 기준 노출 보정 후, 픽셀 ΔE2000 > T 인 곳을 '바뀐 곳'으로 본다.
AR 렌더는 맨얼굴과 픽셀 위치가 같으므로 그대로 비교."""
import sys, csv, glob, os, json
import cv2, numpy as np
from pathlib import Path as _P
REPO = _P(__file__).resolve().parents[4]  # research/data/ar_eval_public/tools → 저장소 루트
sys.path.insert(0, str(REPO / 'research/lip_reference'))
sys.path.insert(0, str(_P(__file__).resolve().parent))
from hw01_compare import ciede2000
from eval_lapa import LIPS_OUTER, LIPS_INNER
from lm import landmarks, fit
from warp import piecewise
T = 6.0  # ΔE2000: 맨얼굴 재촬영 수준의 차이(<6)는 '안 바뀜'으로 봄

def lab(img): return cv2.cvtColor(img.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
def poly(pts, idx, shape):
    m = np.zeros(shape, np.uint8); cv2.fillPoly(m, [np.round(pts[idx]).astype(np.int32)], 255); return m

def zone_lip(pts, shape):
    mw = np.linalg.norm(pts[61] - pts[291]); k = lambda n: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))
    lips = poly(pts, LIPS_OUTER, shape) & ~poly(pts, LIPS_INNER, shape)
    search = cv2.dilate(poly(pts, LIPS_OUTER, shape), k(max(5, round(0.25 * mw)))) & ~poly(pts, LIPS_INNER, shape)
    ring = cv2.dilate(search, k(max(5, round(0.25 * mw)))) & ~search
    return lips, search, ring

def changed(a, b, search, ring, core):
    """b가 a보다 '화장 색 쪽으로' 바뀐 픽셀. 크기: ΔE2000 > T, 방향: 입술 안쪽 평균 변화와 같은 방향(cos > 0.6).
    방향 조건은 조명·그림자처럼 화장과 무관한 피부 변화를 걸러낸다."""
    A, B = lab(a), lab(b)
    r = ring > 0
    B = B + (A[r].mean(0) - B[r].mean(0))
    d = cv2.GaussianBlur(ciede2000(A, B).astype(np.float32), (0, 0), 1.5)
    D = cv2.GaussianBlur(B - A, (0, 0), 1.5)
    v = D[core > 0].mean(0); v = v / (np.linalg.norm(v) + 1e-6)
    cos = (D @ v) / (np.linalg.norm(D, axis=-1) + 1e-6)
    return (d > T) & (cos > 0.6) & (search > 0)

def run(pairs, ardir, out):
    rows = []
    for bare_f, real_f, ar_f, pid in pairs:
        bare = fit(cv2.imread(bare_f)); real = fit(cv2.imread(real_f))
        ar = cv2.imread(ar_f); ar = cv2.resize(ar, bare.shape[1::-1]) if ar.shape[:2] != bare.shape[:2] else ar
        bp, rp = landmarks(bare), landmarks(real)
        real_al, cov = piecewise(real, rp, bp, bare.shape)
        lips, search, ring = zone_lip(bp, bare.shape[:2])
        core = cv2.erode(lips, np.ones((5,5),np.uint8))
        cr = changed(bare, real_al, search, ring, core); ca = changed(bare, ar, search, ring, core)
        inter = (cr & ca).sum(); uni = (cr | ca).sum()
        rows.append({'pair': pid, 'iou': inter / max(uni, 1), 'spill': (ca & ~cr).sum() / max(ca.sum(), 1),
                     'miss': (cr & ~ca).sum() / max(cr.sum(), 1), 'real_px': int(cr.sum()), 'lip_px': int((lips>0).sum()), 'real_cover': float((cr & (lips>0)).sum() / max((lips>0).sum(),1)), 'ar_px': int(ca.sum())})
        vis = bare.copy(); vis[cr & ca] = (0, 200, 0); vis[cr & ~ca] = (0, 0, 255); vis[ca & ~cr] = (255, 0, 0)
        x, y, w, h = cv2.boundingRect(search); cv2.imwrite(f'{out}/pos_{pid}.jpg', np.hstack([real_al[y:y+h, x:x+w], ar[y:y+h, x:x+w], vis[y:y+h, x:x+w]]))
    return rows

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('--raw', required=True); ap.add_argument('--ar', required=True)
    ap.add_argument('--out', default='lip_position.csv'); ap.add_argument('--vis', default='pos_vis')
    a = ap.parse_args(); os.makedirs(a.vis, exist_ok=True); pairs = []
    for f in sorted(glob.glob(f'{a.raw}/p*_0.jpg')):
        stem = os.path.basename(f)[:-6]
        ap_ = [p for p in glob.glob(f'{a.raw}/{stem}_[13].jp*g')]
        if ap_: n = ap_[0].rsplit('_', 1)[1].split('.')[0]; pairs.append((f, ap_[0], f'{a.ar}/{stem}_{n}.png', stem.split('_')[0]))
    rows = run(pairs, a.ar, a.vis)
    w = csv.DictWriter(open(a.out, 'w', newline=''), fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    c = [r for r in rows if r['real_cover'] >= 0.5]
    print(f'전체 {len(rows)}쌍 / 색 변화가 선명한(입술의 50% 이상) {len(c)}쌍 중앙값:',
          {k: round(float(np.median([r[k] for r in c])), 3) for k in ['iou', 'spill', 'miss']})

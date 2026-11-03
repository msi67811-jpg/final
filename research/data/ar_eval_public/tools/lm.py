import sys
from pathlib import Path as _P
REPO = _P(__file__).resolve().parents[4]  # research/data/ar_eval_public/tools → 저장소 루트
sys.path.insert(0, str(REPO / 'research/lip_reference'))
sys.path.insert(0, str(_P(__file__).resolve().parent))
from eval_lapa import make_landmarker
import mediapipe as mp, cv2, numpy as np
_lm = None
def landmarks(img):
    global _lm
    if _lm is None: _lm = make_landmarker()
    r = _lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    if not r.face_landmarks: return None
    h, w = img.shape[:2]
    return np.array([[l.x*w, l.y*h] for l in r.face_landmarks[0]], np.float32)
def fit(img, M=1280):
    h, w = img.shape[:2]; s = min(1.0, M/max(h, w))
    return cv2.resize(img, (round(w*s), round(h*s)), interpolation=cv2.INTER_AREA) if s < 1 else img
def crop(img, pts, pad=0.25, size=360):
    x0,y0 = pts.min(0); x1,y1 = pts.max(0); cx,cy=(x0+x1)/2,(y0+y1)/2; s=max(x1-x0,y1-y0)*(1+pad)/2
    M = np.float32([[size/(2*s),0,size/2-cx*size/(2*s)],[0,size/(2*s),size/2-cy*size/(2*s)]])
    return cv2.warpAffine(img, M, (size,size), borderValue=(255,255,255))

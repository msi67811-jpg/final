import json, cv2, numpy as np
import re
from pathlib import Path
_uv = (Path(__file__).resolve().parents[4] / 'components/ar/faceUV.ts').read_text(encoding='utf-8')
TRIS = np.array(json.loads(re.search(r'FACE_TRIANGLES: number\[\] = (\[.*?\]);', _uv, re.S).group(1))).reshape(-1, 3)
def piecewise(src, sp, dp, shape):
    """src 이미지를 랜드마크 삼각형별로 dp 위치에 맞춰 붙인다 (입 모양 차이까지 정렬)."""
    H, W = shape[:2]; out = np.zeros((H, W, 3), np.uint8); cover = np.zeros((H, W), np.uint8)
    for t in TRIS:
        s = sp[t].astype(np.float32); d = dp[t].astype(np.float32)
        x, y, w, h = cv2.boundingRect(d)
        if w < 1 or h < 1 or x < 0 or y < 0 or x + w > W or y + h > H: continue
        M = cv2.getAffineTransform(s, d - np.float32([x, y]))
        patch = cv2.warpAffine(src, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        m = np.zeros((h, w), np.uint8); cv2.fillConvexPoly(m, np.round(d - [x, y]).astype(np.int32), 1)
        roi = out[y:y+h, x:x+w]; roi[m > 0] = patch[m > 0]; cover[y:y+h, x:x+w] |= m
    return out, cover > 0

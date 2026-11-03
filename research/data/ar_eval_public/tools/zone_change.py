"""4부위 AR 렌더가 맨얼굴에서 부위별로 얼마나 바뀌었는지 (ΔE2000 평균).
    python zone_change.py --raw RAW_DIR --ar AR_DIR --meta meta_levels.csv --out zone_change.csv
열: lip(립), eyelid(눈두덩: 아이섀도+파운데이션), cheek(볼: 블러셔+파운데이션), forehead(이마: 파운데이션만).
eye_add·blush_add: 눈두덩·볼 평균색이 이마(파운데이션만 바름) 평균색과 다른 정도가 맨얼굴보다 얼마나 커졌는지 = 아이섀도·블러셔가 눈에 띄는 정도."""
import argparse, csv
import cv2, numpy as np
from lm import landmarks, fit
from hw01_compare import ciede2000
from eval_lapa import LIPS_OUTER, LIPS_INNER

LID_L = [33, 246, 161, 160, 159, 158, 157, 173, 133, 190, 56, 28, 27, 29, 30, 247]
LID_R = [263, 466, 388, 387, 386, 385, 384, 398, 362, 414, 286, 258, 257, 259, 260, 467]
CHEEK_L = [117, 118, 101, 36, 205, 187, 123]
CHEEK_R = [346, 347, 330, 266, 425, 411, 352]
FOREHEAD = [67, 109, 10, 338, 297, 299, 9, 69]

def lab(i): return cv2.cvtColor(i.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
def poly(pts, idx, shape):
    m = np.zeros(shape, np.uint8); cv2.fillPoly(m, [np.round(pts[idx]).astype(np.int32)], 1); return m > 0

def measure(bare, ar):
    pts = landmarks(bare); s = bare.shape[:2]
    if ar.shape[:2] != s: ar = cv2.resize(ar, s[::-1])
    d = ciede2000(lab(bare), lab(ar))
    lip = poly(pts, LIPS_OUTER, s) & ~poly(pts, LIPS_INNER, s)
    out = {'lip': d[lip].mean(),
           'eyelid': np.mean([d[poly(pts, LID_L, s)].mean(), d[poly(pts, LID_R, s)].mean()]),
           'cheek': np.mean([d[poly(pts, CHEEK_L, s)].mean(), d[poly(pts, CHEEK_R, s)].mean()]),
           'forehead': d[poly(pts, FOREHEAD, s)].mean()}
    # 아이섀도·블러셔가 보이는 정도: 그 부위 평균색이 이마(파운데이션만) 평균색과 얼마나 다른지, AR에서 맨얼굴보다 얼마나 커졌는지
    A, B = lab(ar), lab(bare)
    fh = poly(pts, FOREHEAD, s)
    lid = poly(pts, LID_L, s) | poly(pts, LID_R, s); chk = poly(pts, CHEEK_L, s) | poly(pts, CHEEK_R, s)
    con = lambda X, m: float(ciede2000(X[m].mean(0), X[fh].mean(0)))
    out['eye_add'] = con(A, lid) - con(B, lid); out['blush_add'] = con(A, chk) - con(B, chk)
    return {k: round(float(v), 2) for k, v in out.items()}

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--raw', required=True); ap.add_argument('--ar', required=True)
    ap.add_argument('--meta', default='meta_levels.csv'); ap.add_argument('--out', default='zone_change.csv')
    a = ap.parse_args(); rows = []
    for r in csv.DictReader(open(f'{a.raw}/{a.meta}', encoding='utf-8-sig')):
        ar = cv2.imread(f'{a.ar}/{r["name"]}.png')
        if ar is None: continue
        rows.append({'name': r['name'], 'person': r['name'].split('_')[0], 'level': r['name'].split('_L')[-1] if '_L' in r['name'] else '',
                     **measure(fit(cv2.imread(f'{a.raw}/{r["bare"]}')), ar)})
    w = csv.DictWriter(open(a.out, 'w', newline='', encoding='utf-8-sig'), fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    for lv in sorted({r['level'] for r in rows}):
        rr = [r for r in rows if r['level'] == lv]
        print(f'단계 {lv or "-"} ({len(rr)}장) 중앙값:', {k: round(float(np.median([r[k] for r in rr])), 2) for k in ['lip', 'forehead', 'eye_add', 'blush_add']})

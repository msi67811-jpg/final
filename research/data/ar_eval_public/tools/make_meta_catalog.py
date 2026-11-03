"""4부위 AR 약·중·강 meta를 '우리 카탈로그 제품 색'으로 만든다.
    python make_meta_catalog.py BARE_DIR CATALOG_JSON [--seed 2026]   → BARE_DIR/meta_catalog.csv
BARE_DIR 의 *_0.jpg(맨얼굴)마다:
  - 파운데이션: 카탈로그 파운데이션 셰이드 중 본인 볼 피부색과 ΔE2000이 가장 가까운 것(셰이드 매칭), 단계별 강도만 다름
  - 립·블러셔·아이섀도: 진하기(채도 + 본인 피부보다 어두운 정도)로 셰이드를 3등분 → 약=연한 셰이드, 강=진한 셰이드, 그 안에서 무작위
  - 강도(슬라이더): 약 0.35~0.45 / 중 0.65~0.7 / 강 1.0, 질감은 제품의 texture 값
모든 행에 제품 브랜드·이름·셰이드명을 남긴다."""
import argparse, csv, glob, json, os, random
import cv2, numpy as np
from lm import landmarks, fit
from hw01_compare import ciede2000

CHEEK = [50, 101, 118, 117, 123, 147, 187, 205, 280, 330, 347, 346, 352, 376, 411, 425]
INT = {1: {'lip_i': 0.45, 'found_i': 0.35, 'blush_i': 0.35, 'eye_i': 0.4},
       2: {'lip_i': 0.7, 'found_i': 0.65, 'blush_i': 0.65, 'eye_i': 0.7},
       3: {'lip_i': 1.0, 'found_i': 1.0, 'blush_i': 1.0, 'eye_i': 1.0}}
TEX = {'matte', 'velvet', 'cream', 'satin', 'sheer', 'glossy', 'shimmer'}

def hex2lab(h):
    rgb = np.array([[[int(h[i:i + 2], 16) for i in (5, 3, 1)]]], np.uint8)  # BGR
    return cv2.cvtColor(rgb.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)[0, 0]

def shades(cat, zone):
    out = []
    for p in cat['products']:
        if p.get('makeupZone') != zone: continue
        for s in p.get('shades', []):
            h = s.get('hex') or ''
            if len(h) != 7: continue
            lab = hex2lab(h); chroma = float(np.hypot(lab[1], lab[2]))
            if zone == 'lip' and (chroma < 15 or lab[0] > 85): continue          # 립케어·투명 제외
            name = p.get('name', '')
            if zone == 'blush' and (chroma < 18 or lab[0] < 35 or ('하이라이트' in name and '블러' not in name) or name.endswith('브러쉬') or name.endswith('브러시')): continue
            if zone == 'eyeshadow' and lab[0] > 92: continue                     # 흰 하이라이트 제외
            out.append({'hex': h.upper(), 'lab': lab, 'tex': p.get('texture') if p.get('texture') in TEX else 'matte',
                        'brand': p.get('brand', ''), 'product': p.get('name', '')[:60], 'shade': s.get('name', '')})
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('bare'); ap.add_argument('catalog'); ap.add_argument('--seed', type=int, default=2026)
    a = ap.parse_args(); rnd = random.Random(a.seed)
    cat = json.load(open(a.catalog, encoding='utf-8'))
    pool = {z: shades(cat, z) for z in ('foundation', 'lip', 'blush', 'eyeshadow')}
    print({z: len(v) for z, v in pool.items()})
    rows = []
    for f in sorted(glob.glob(os.path.join(a.bare, '*_0.jpg'))):
        b = os.path.basename(f); person = b.split('_')[0]
        img = fit(cv2.imread(f)); pts = landmarks(img)
        if pts is None: print('얼굴 없음', b); continue
        skin = np.median(cv2.cvtColor(img.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)[pts[CHEEK][:, 1].astype(int), pts[CHEEK][:, 0].astype(int)], 0)
        dist = lambda s: float(ciede2000(s['lab'], skin))
        # 진하기 = 채도 + 피부보다 어두운 정도 (밝은 펄·하이라이트 색은 '약' 쪽으로)
        depth = lambda s: float(np.hypot(s['lab'][1], s['lab'][2]) + max(0.0, skin[0] - s['lab'][0]))
        found = min(pool['foundation'], key=dist)
        pick = {}
        for z in ('lip', 'blush', 'eyeshadow'):
            srt = sorted(pool[z], key=depth); n = len(srt)
            pick[z] = {lv: rnd.choice(srt[(lv - 1) * n // 3: lv * n // 3]) for lv in (1, 2, 3)}
        for lv in (1, 2, 3):
            L, B, E = pick['lip'][lv], pick['blush'][lv], pick['eyeshadow'][lv]
            rows.append({'name': f'{person}_L{lv}', 'bare': b, 'lip': L['hex'], 'lip_tex': L['tex'], 'found': found['hex'],
                         'blush': B['hex'], 'eye': E['hex'], 'eye_tex': 'shimmer' if E['tex'] in ('shimmer', 'glossy') else 'matte', **INT[lv],
                         'found_match_dE': round(dist(found), 1),
                         'lip_product': f"{L['brand']} {L['product']} / {L['shade']}", 'found_product': f"{found['brand']} {found['product']} / {found['shade']}",
                         'blush_product': f"{B['brand']} {B['product']} / {B['shade']}", 'eye_product': f"{E['brand']} {E['product']} / {E['shade']}"})
    out = os.path.join(a.bare, 'meta_catalog.csv')
    w = csv.DictWriter(open(out, 'w', newline='', encoding='utf-8-sig'), fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print(len(rows), '행 →', out)

if __name__ == '__main__': main()

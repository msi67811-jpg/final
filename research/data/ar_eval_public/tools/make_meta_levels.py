"""4부위 AR을 약·중·강 3단계로 만들 meta_levels.csv 생성.
    python make_meta_levels.py RAW_DIR   (RAW_DIR/meta_full.csv 가 먼저 있어야 함: make_meta_full.py)
단계마다 강도(슬라이더 값)와 색 진하기를 같이 올린다. 립 색=실제 사진 평균색, 파운데이션=본인 피부톤(톤 맞춤),
블러셔·아이섀도=임의 팔레트(제품 hex 아님)."""
import csv, sys
RAW = sys.argv[1]
LEVELS = {
    1: {'i': {'lip_i': 0.45, 'found_i': 0.35, 'blush_i': 0.35, 'eye_i': 0.4},
        'blush': ['#E9A39A', '#E6A5B0', '#E2A48F'], 'eye': [('#B3917B', 'matte'), ('#C4A09C', 'shimmer'), ('#A88878', 'matte')]},
    2: {'i': {'lip_i': 0.7, 'found_i': 0.65, 'blush_i': 0.65, 'eye_i': 0.7},
        'blush': ['#E8857A', '#E38E9E', '#DB8B74'], 'eye': [('#9C7458', 'matte'), ('#B98682', 'shimmer'), ('#8C6A5A', 'matte')]},
    3: {'i': {'lip_i': 1.0, 'found_i': 1.0, 'blush_i': 1.0, 'eye_i': 1.0},
        'blush': ['#E0606A', '#D9667F', '#D9704F'], 'eye': [('#7A4E3A', 'matte'), ('#9E5A6A', 'shimmer'), ('#6B4A6E', 'matte')]},
}
base = list(csv.DictReader(open(f'{RAW}/meta_full.csv', encoding='utf-8-sig')))
rows = []
for k, b in enumerate(base):
    p = b['name'].split('_')[0]
    for lv, L in LEVELS.items():
        eye, tex = L['eye'][k % 3]
        rows.append({'name': f'{p}_L{lv}', 'bare': b['bare'], 'lip': b['lip'], 'lip_tex': b['lip_tex'], 'found': b['found'],
                     'blush': L['blush'][k % 3], 'eye': eye, 'eye_tex': tex, **L['i']})
w = csv.DictWriter(open(f'{RAW}/meta_levels.csv', 'w', newline='', encoding='utf-8-sig'), fieldnames=rows[0].keys())
w.writeheader(); w.writerows(rows); print(len(rows), 'rows →', f'{RAW}/meta_levels.csv')

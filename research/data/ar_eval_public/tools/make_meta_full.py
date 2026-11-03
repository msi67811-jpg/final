# python make_meta_full.py RAW_DIR → RAW_DIR/meta_full.csv
# 맨얼굴 20명 각각에 4부위 색을 정한다. 립=실제 화장 사진 평균색(meta.csv), 파운데이션=본인 볼 피부색(톤 맞춤),
# 블러셔·아이섀도=고정 팔레트에서 사람별로 번갈아(제품 hex 아님, 임의 팔레트).
import csv, cv2, numpy as np, glob, os, sys
RAW = sys.argv[1]  # hw01_raw 형식 폴더 (pNN_..._0.jpg + meta.csv)
from lm import landmarks, fit
CHEEK = [50, 101, 118, 117, 123, 147, 187, 205, 280, 330, 347, 346, 352, 376, 411, 425]
BLUSH = ['#E8857A', '#E38E9E', '#DB8B74']; EYE = [('#9C7458', 'matte'), ('#B98682', 'shimmer'), ('#8C6A5A', 'matte')]
meta = {r['product']: r for r in csv.DictReader(open(f'{RAW}/meta.csv', encoding='utf-8-sig'))}
rows = []
for i, f in enumerate(sorted(glob.glob(f'{RAW}/p*_0.jpg'))):
    b = os.path.basename(f); p, _, _, m, _ = b.split('_')
    img = fit(cv2.imread(f)); pts = landmarks(img)
    vals = np.array([img[int(y), int(x)] for x, y in pts[CHEEK]], np.float32)
    med = np.median(vals, 0)  # BGR
    rgb = np.clip(med[::-1] * 1.03 + 3, 0, 255).astype(int)
    rows.append({'name': f'{p}_full', 'bare': b, 'lip': meta[m]['hex'], 'lip_tex': meta[m]['texture'],
                 'found': '#%02X%02X%02X' % tuple(rgb), 'blush': BLUSH[i % 3], 'eye': EYE[i % 3][0], 'eye_tex': EYE[i % 3][1]})
w = csv.DictWriter(open(f'{RAW}/meta_full.csv', 'w', newline='', encoding='utf-8-sig'), fieldnames=rows[0].keys())
w.writeheader(); w.writerows(rows); print(len(rows)); print(rows[0])

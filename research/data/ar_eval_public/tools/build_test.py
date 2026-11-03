"""자연스러움 블라인드 테스트 HTML 만들기 (오프라인, 파일 하나).

pool.csv: person,real,ar,full1,full2,full3  (real·ar 이 빈 사람은 4부위 AR 전용, full 이 빈 사람은 쌍 전용)
  real = 실제 화장 사진, ar = 같은 사람 맨얼굴에 입힌 AR(실제 사진과 같은 부위만),
  full1~3 = 같은 맨얼굴에 4부위 AR을 약·중·강으로 입힌 렌더(선택)
평가자마다 서로 다른 사람 N명을 무작위로 뽑는다(같은 사람은 한 번만).
  - 그중 --full 명은 4부위 AR(단계 무작위, 되도록 약·중·강이 고루)
  - 나머지는 실제/AR 반반 → 실제/AR 맞히기는 이 사진들로만 계산
    python build_test.py pool.csv OUT_DIR --per-rater 10 --full 4
OUT_DIR 에 naturalness_test.html, key.csv 를 만든다. key.csv(정답표)는 평가자에게 주지 않는다.
"""
import argparse, base64, csv, json, random, sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lm import landmarks, fit

W, H = 480, 600

def face_crop(path):
    img = cv2.imread(str(path)); img = fit(img, 1600); p = landmarks(img)
    if p is None: raise SystemExit(f'얼굴을 못 찾음: {path}')
    x0, y0 = p.min(0); x1, y1 = p.max(0); cx, cy = (x0 + x1) / 2, (y0 + y1) / 2 - 0.04 * (y1 - y0)
    s = max((x1 - x0) / W, (y1 - y0) / H) * 1.45
    M = np.float32([[1 / s, 0, W / 2 - cx / s], [0, 1 / s, H / 2 - cy / s]])
    out = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)
    ok, buf = cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, 85])  # 실제·AR 모두 같은 크기·압축으로
    return 'data:image/jpeg;base64,' + base64.b64encode(buf).decode()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pool', type=Path); ap.add_argument('out', type=Path)
    ap.add_argument('--per-rater', type=int, default=10); ap.add_argument('--full', type=int, default=4)
    ap.add_argument('--parts', default='입술,피부,눈,볼')
    ap.add_argument('--title', default='메이크업 사진 평가')
    ap.add_argument('--seed', type=int, default=2026)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(a.pool, encoding='utf-8-sig')))
    rnd = random.Random(a.seed)
    ids = [f'{k:04d}' for k in rnd.sample(range(1000, 10000), 5 * len(rows))]  # 사람당 최대 5장  # 이미지 번호는 정답과 무관한 무작위 숫자
    items, key = [], []
    has_full = any(r.get('full1') for r in rows)
    for g, r in enumerate(rows):
        for truth, col in ((('REAL', 'real'), ('AR', 'ar')) if r.get('real') and r.get('ar') else ()):
            iid = 'i' + ids.pop()
            v = rnd.randrange(100, 1000)
            if (v + g) % 2 != (0 if truth == 'REAL' else 1): v += 1  # 실제/AR 5:5 균형용 표시 (화면엔 안 나옴)
            items.append({'id': iid, 'g': g, 'k': 'L', 'v': v, 'src': face_crop(r[col])})
            key.append({'image_id': iid, 'truth': truth, 'person': r['person'], 'kind': 'pair', 'level': ''})
        if r.get('full1'):
            for lv in (1, 2, 3):
                iid = 'i' + ids.pop()
                v = rnd.randrange(100, 1000); v += (lv - 1 - (v + g) % 3) % 3  # 단계 고루 뽑기용 표시
                items.append({'id': iid, 'g': g, 'k': 'F', 'v': v, 'src': face_crop(r[f'full{lv}'])})
                key.append({'image_id': iid, 'truth': 'AR', 'person': r['person'], 'kind': 'full', 'level': lv})
    rnd.shuffle(items)
    tpl = (Path(__file__).resolve().parent / 'test_template.html').read_text(encoding='utf-8')
    cfg = {'title': a.title, 'per': a.per_rater, 'full': a.full if has_full else 0, 'parts': a.parts.split(','), 'groups': len(rows)}
    html = tpl.replace('/*CFG*/null', json.dumps(cfg, ensure_ascii=False)).replace('/*ITEMS*/[]', json.dumps(items))
    (a.out / 'naturalness_test.html').write_text(html, encoding='utf-8')
    with open(a.out / 'key.csv', 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=['image_id', 'truth', 'person', 'kind', 'level']); w.writeheader(); w.writerows(sorted(key, key=lambda k: k['image_id']))
    print(f'{len(rows)}명, 사진 {len(items)}장, 평가자당 {a.per_rater}장(4부위 {cfg["full"]}장) → {a.out}')

if __name__ == '__main__': main()

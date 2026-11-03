"""4부위(립·파운데이션·아이섀도·블러셔) 동시 AR 렌더 — 맨얼굴 사진 위에 앱 셰이더 그대로.
python render_full.py --raw DIR --out DIR [--meta meta_full.csv]
(meta: name,bare,lip,lip_tex,found,blush,eye,eye_tex[,lip_i,found_i,blush_i,eye_i] — 강도 열은 선택)
"""
import sys, csv, json, re, shutil, subprocess, argparse
from pathlib import Path
from pathlib import Path as _P
REPO = _P(__file__).resolve().parents[4]  # research/data/ar_eval_public/tools → 저장소 루트
sys.path.insert(0, str(REPO / 'research/lip_reference'))
sys.path.insert(0, str(_P(__file__).resolve().parent))
import render_ar_offline as r
import cv2, numpy as np
from lm import landmarks, fit

OLD = "makeup = { lip: { hex: job.hex, intensity: job.intensity, texture: job.texture }, blush: null, eyeshadow: null };"
NEW = "makeup = Object.assign({ lip: null, blush: null, eyeshadow: null, eyeliner: null, mascara: null, highlighter: null, contour: null, brow: null, foundation: null }, job.makeup);"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--raw', type=Path); ap.add_argument('--out', type=Path); ap.add_argument('--chrome', default=str(r.CHROME)); ap.add_argument('--meta', default='meta_full.csv')
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    work = a.out / '_work'; shutil.rmtree(work, ignore_errors=True); work.mkdir()
    jobs = []
    for row in csv.DictReader(open(a.raw / a.meta, encoding='utf-8-sig')):
        img = fit(cv2.imread(str(a.raw / row['bare']))); p = landmarks(img)
        if p is None: print('skip', row['name']); continue
        h, w = img.shape[:2]; p3 = r_lm(img)
        mk = {}
        it = lambda k, d: float(row.get(k) or d)  # 강도 열(lip_i 등)이 있으면 쓰고, 없으면 기본값
        if row['lip']: mk['lip'] = {'hex': row['lip'], 'intensity': it('lip_i', 1.0), 'texture': row['lip_tex'] or 'matte'}
        if row['found']: mk['foundation'] = {'hex': row['found'], 'intensity': it('found_i', 0.6), 'texture': 'matte'}
        if row['blush']: mk['blush'] = {'hex': row['blush'], 'intensity': it('blush_i', 0.5), 'texture': 'matte'}
        if row['eye']: mk['eyeshadow'] = {'hex': row['eye'], 'intensity': it('eye_i', 0.6), 'texture': row['eye_tex'] or 'matte'}
        frame = f'{len(jobs)}.png'; cv2.imwrite(str(work / frame), img)
        jobs.append({'name': row['name'], 'frame': frame, 'w': w, 'h': h, 'x': p3[0], 'y': p3[1], 'z': p3[2], 'makeup': mk,
                     'hex': '#000000', 'texture': 'matte', 'intensity': 1})
    html = r.page_html(REPO / 'components/ar/faceTrackerHtml.ts')
    assert OLD in html; html = html.replace(OLD, NEW)
    (work / 'jobs.json').write_text(json.dumps(jobs)); (work / 'index.html').write_text(html, encoding='utf-8')
    httpd = r.serve(work); url = f'http://127.0.0.1:{httpd.server_address[1]}/index.html'
    extra = ['--no-sandbox'] if sys.platform.startswith('linux') else []
    dom = subprocess.run([a.chrome, *extra, '--headless=new', '--use-angle=swiftshader', '--enable-unsafe-swiftshader',
                          f'--virtual-time-budget={60000 + 15000 * len(jobs)}', '--dump-dom', url],
                         capture_output=True, text=True, encoding='utf-8', timeout=1500).stdout
    httpd.shutdown()
    t = re.search(r'<title>([^<]*)', dom); print(t.group(1)[:400] if t else 'no title')
    import base64; n = 0
    for name, data in re.findall(r'<img class="result" data-name="([^"]+)" src="data:image/png;base64,([^"]+)"', dom):
        (a.out / f'{name}.png').write_bytes(base64.b64decode(data)); n += 1
    shutil.rmtree(work, ignore_errors=True); print('saved', n)

def r_lm(img):
    import mediapipe as mp
    from lm import _lm  # noqa
    import lm
    res = lm._lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    f = res.face_landmarks[0]
    return [l.x for l in f], [l.y for l in f], [l.z for l in f]

if __name__ == '__main__': main()

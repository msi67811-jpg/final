"""Fix the multi-sample set for the 8회차 checklist items "샘플별 지표·평균" and "가속화" (2026-11-23).

    python research/makeup_gen/prep_samples.py <fpmt_repo=research/data/fpmt> <root=research/data/makeup_gen>

Public pairs: FPMT's own example faces (MT-dataset style, 361x361, 19-class parsing included in the repo).
  - 10 bare faces: the 13 in examples/images/non_makeup minus three that are open-mouthed or turned (gap or yaw, below).
  - 5 references with different lip colours, all with closed mouths: coral, red, rose, nude, burgundy.
  - pair i = source i + reference i % 5, so every reference is used twice.
  - target = the reference's lip CIELAB median (MediaPipe lip mask, same as eval_makeup.py): "how well is this person's lip
    colour carried over", not a worn-colour prediction.
User pair (u01, added at the user's request): bare crop A + worn photo E. Target = E's lip (the worn colour; E is also the
reference, so it is circular, as in COMPARE_LOCAL.md).

Writes <root>/multi/:
  samples.json                 pair list, targets, file paths (relative to <root>)
  faces/<id>_{src,ref}_{512,1024}.png   model inputs (public faces upscaled from 361 with LANCZOS)
  faces/<id>_frame_720x960.png the AR still frame (public: 720x720 face, 120 px reflected rows above and below)
  fpmt/<id>/images|seg1/{non_makeup,makeup}/  one-pair FPMT data root (run_fpmt.py runs every src x ref of a root)
"""
import json
import os
import shutil
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_makeup import landmarks, lip_lab, lip_mask, load  # noqa: E402

SOURCES = ['vSYYZ241', 'xfsy_0129', 'xfsy_0202', 'xfsy_0242', 'xfsy_0387',
           'xfsy_0405', 'xfsy_0429', 'xfsy_0435', 'xfsy_0515', 'xfsy_0520']
EXCLUDED = {'xfsy_0491': 'mouth open (gap 0.345 IOD), yaw -0.23', 'xfsy_0058': 'yaw -0.25', 'xfsy_0409': 'mouth open (gap 0.106 IOD)'}
REFS = [('vFG112', 'coral'), ('vFG511', 'red'), ('vHX129', 'rose'),
        ('8ed6feb8f72b89632d02fbafcc2511a9', 'nude'), ('XYH-109', 'burgundy')]


def lab_hex(img):
    lm = landmarks(img)
    m = lip_mask(img, lm)
    med = lip_lab(img, m)[0]
    return [round(float(v), 2) for v in med], '#%02X%02X%02X' % tuple(int(v) for v in np.median(img[m], 0))


def up(img, size):
    return Image.fromarray(img).resize((size, size), Image.LANCZOS)


def ar_frame(img):
    sq = np.asarray(up(img, 720))
    return Image.fromarray(np.pad(sq, ((120, 120), (0, 0), (0, 0)), mode='reflect'))


def main():
    repo, root = sys.argv[1:3]
    ex = os.path.join(repo, 'examples')
    out = os.path.join(root, 'multi')
    faces = os.path.join(out, 'faces')
    os.makedirs(faces, exist_ok=True)
    pairs = []
    for i, src in enumerate(SOURCES):
        ref, colour = REFS[i % len(REFS)]
        pid = f'p{i + 1:02d}'
        s_img = load(os.path.join(ex, 'images/non_makeup', src + '.png'))
        r_img = load(os.path.join(ex, 'images/makeup', ref + '.png'))
        for size in (512, 1024):
            up(s_img, size).save(os.path.join(faces, f'{pid}_src_{size}.png'))
            up(r_img, size).save(os.path.join(faces, f'{pid}_ref_{size}.png'))
        ar_frame(s_img).save(os.path.join(faces, f'{pid}_frame_720x960.png'))
        d = os.path.join(out, 'fpmt', pid)
        for kind, stem in (('non_makeup', src), ('makeup', ref)):
            for sub in ('images', 'seg1'):
                os.makedirs(os.path.join(d, sub, kind), exist_ok=True)
                shutil.copy(os.path.join(ex, sub, kind, stem + '.png'), os.path.join(d, sub, kind, stem + '.png'))
        t_lab, t_hex = lab_hex(r_img)
        b_lab, b_hex = lab_hex(s_img)
        pairs.append({'id': pid, 'kind': 'public', 'src': src, 'ref': ref, 'ref_colour': colour,
                      'target': 'reference lip (361 original)', 'target_lab': t_lab, 'target_hex': t_hex,
                      'bare_lab': b_lab, 'bare_hex': b_hex, 'fpmt_root': f'multi/fpmt/{pid}',
                      'fpmt_out': f'fpmt_{src}__{ref}_srcframe.png'})
    # user pair: existing crops and FPMT root (fpmt_in), nothing re-cropped
    inp = os.path.join(root, 'input')
    for size in (512, 1024):
        shutil.copy(os.path.join(inp, f'bare_crop_{size}.png'), os.path.join(faces, f'u01_src_{size}.png'))
        shutil.copy(os.path.join(inp, f'ref_crop_{size}.png'), os.path.join(faces, f'u01_ref_{size}.png'))
    shutil.copy(os.path.join(inp, 'bare_frame_720x960.png'), os.path.join(faces, 'u01_frame_720x960.png'))
    t_lab, t_hex = lab_hex(load(os.path.join(inp, 'ref_crop_1024.png')))
    b_lab, b_hex = lab_hex(load(os.path.join(inp, 'bare_crop_1024.png')))
    pairs.append({'id': 'u01', 'kind': 'user', 'src': 'A_bare', 'ref': 'E_worn', 'ref_colour': 'worn (롬앤 03 베어 그레이프)',
                  'target': 'worn photo E lip (1024 crop); E is also the reference -> circular', 'target_lab': t_lab,
                  'target_hex': t_hex, 'bare_lab': b_lab, 'bare_hex': b_hex, 'fpmt_root': 'fpmt_in',
                  'fpmt_out': 'fpmt_A_bare__E_worn_srcframe.png'})
    meta = {'created': '2026-11-23', 'excluded_sources': EXCLUDED,
            'public_source': 'FPMT repo examples (MT-dataset style faces, 361x361; research use)',
            'upscale': 'LANCZOS 361 -> 512 / 1024; AR frame = 720x720 + 120 px reflect padding top and bottom',
            'pairs': pairs}
    json.dump(meta, open(os.path.join(out, 'samples.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    for p in pairs:
        print(p['id'], p['src'].ljust(12), p['ref'][:12].ljust(12), p['ref_colour'].ljust(9), p['target_hex'], p['target_lab'])


if __name__ == '__main__':
    main()

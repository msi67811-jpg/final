"""Per-sample metrics and method averages over the multi-sample set (prep_samples.py), 2026-11-23.

    python research/makeup_gen/eval_multi.py <root=research/data/makeup_gen> [only=<comma ids>]

Same metric functions and definitions as COMPARE_LOCAL.md (no new metric):
  colour / region   eval_makeup.compare: lip dE00 vs the pair's target, outside-lip dE00 and SSIM, lip shape IoU
  texture           texture_metric.measure: wrinkle corr / strength without gloss spots, gloss-spot share
  post-processing   region_composite.composite: paste (A) and colour only (B), on the bare input, for every model output
Differences from COMPARE_LOCAL.md, applied to every sample alike:
  - face region for "outside lip" = convex hull of the bare face's 478 landmarks (the AR rows used this already; the 11-20
    MagicMakeup face-parsing mask exists only for the user photo). Outside-lip numbers for u01 can differ slightly from the table.
  - each model is measured at its own output size against the bare input of that size, as in COMPARE_LOCAL.md
    (FPMT and Stable-Makeup 512, MagicMakeup 1024; resizing MagicMakeup to 512 moved its lip dE00 3.61 -> 3.12 on u01).
  - AR: the frame with the lip against the AR-off frame of the same session (same pixels), at 720x960.

Inputs looked up under <root>/multi/ (a missing file only drops that row):
  ar/<id>/{off,glossy065,matte050}.png                    ar_tune_render.mjs output per face
  out/<model>/<variant>/<id>.png                           model outputs; variant 'base' = the reference setting, others = speed-ups
  out/<model>/<variant>/time.json                          {"<id>": seconds, ...} or {"_all": seconds} (optional)
FPMT outputs are taken as the _srcframe image (already on the bare photo's pixels -> no landmark alignment).
Writes multi/eval/multi_metrics.json and multi/eval/multi_table.md.
"""
import glob
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image
from skimage.color import deltaE_ciede2000, rgb2lab

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_makeup import compare, landmarks, load  # noqa: E402
from region_composite import composite  # noqa: E402
from texture_metric import measure, prep  # noqa: E402

MODELS = {'fpmt': ('FPMT', False, 512), 'stablemakeup': ('Stable-Makeup', True, 512), 'magicmakeup': ('MagicMakeup', True, 1024)}
AR = {'glossy065': 'AR glossy 0.65', 'matte050': 'AR matte 0.50'}
KEYS = {'lip_dE': 'dE00_lip_vs_E_real', 'out_dE': 'change_dE00_mean_outside_lip(face)', 'out_ssim': 'SSIM_outside_lip(face)',
        'iou': 'lip_shape_IoU_vs_bare', 'wc': 'wrinkle_corr_nohl', 'we': 'wrinkle_energy_nohl', 'hl': 'highlight_share'}
LABEL = {'lip_dE': 'lip dE00 vs target', 'out_dE': 'outside-lip dE00', 'out_ssim': 'outside SSIM', 'iou': 'lip IoU',
         'wc': 'wrinkle corr', 'we': 'wrinkle strength', 'hl': 'gloss spots', 'sec': 'sec / image'}


def rs(img, size):
    return img if img.shape[:2] == (size, size) else np.asarray(Image.fromarray(img).resize((size, size), Image.LANCZOS))


def hull_mask(lm, shape):
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(m, [cv2.convexHull(lm.astype(np.int32))], 1)
    return m.astype(bool)


class Base:
    """A bare image with its landmarks, face hull and texture reference, built once per sample and size."""

    def __init__(self, img):
        self.img = img
        self.lm = landmarks(img)
        self.face = hull_mask(self.lm, img.shape)
        self.tex = prep(img)


def row(img, base, target_lab, align):
    r, _ = compare('x', img, base.img, base.lm, base.face, target_lab)
    if 'error' in r:
        return {'error': r['error']}
    t, _ = measure(img, base.tex, align)
    out = {k: r[v] for k, v in KEYS.items() if v in r} | {k: t[v] for k, v in KEYS.items() if v in t}
    out['lip_hex'] = r['lip_hex_median']
    return out


def times(folder):
    p = os.path.join(folder, 'time.json')
    return json.load(open(p, encoding='utf-8')) if os.path.exists(p) else {}


def stats(rows):
    out = {}
    for k in list(KEYS) + ['sec']:
        vals = [(r[k], r['id']) for r in rows if isinstance(r.get(k), (int, float))]
        if not vals:
            continue
        v = np.array([x for x, _ in vals], float)
        worst = max(vals, key=lambda x: x[0]) if k in ('lip_dE', 'out_dE') else min(vals, key=lambda x: x[0])
        out[k] = {'mean': round(float(v.mean()), 3), 'std': round(float(v.std(ddof=1)) if len(v) > 1 else 0.0, 3),
                  'min': round(float(v.min()), 3), 'max': round(float(v.max()), 3), 'worst_id': worst[1], 'n': len(v)}
    return out


def main():
    root = sys.argv[1]
    only = next((a.split('=', 1)[1].split(',') for a in sys.argv[2:] if a.startswith('only=')), None)
    multi = os.path.join(root, 'multi')
    meta = json.load(open(os.path.join(multi, 'samples.json'), encoding='utf-8'))
    pairs = [p for p in meta['pairs'] if not only or p['id'] in only]
    rows = []           # one dict per (group, sample)
    for p in pairs:
        pid, tgt = p['id'], np.array(p['target_lab'])
        bases = {s: Base(load(os.path.join(multi, 'faces', f'{pid}_src_{s}.png'))) for s in (512, 1024)}
        b512 = bases[512]
        rows.append({'group': 'bare (no makeup)', 'id': pid} | row(b512.img, b512, tgt, False))
        # AR: against its own off frame
        off = os.path.join(multi, 'ar', pid, 'off.png')
        if os.path.exists(off):
            boff = Base(load(off))
            for f, label in AR.items():
                q = os.path.join(multi, 'ar', pid, f + '.png')
                if os.path.exists(q):
                    rows.append({'group': label, 'id': pid, 'sec': 'real time'} | row(load(q), boff, tgt, False))
        # models: raw, paste, colour only
        for model, (name, align, size) in MODELS.items():
            bb = bases[size]
            for vdir in sorted(glob.glob(os.path.join(multi, 'out', model, '*'))):
                variant = os.path.basename(vdir)
                q = os.path.join(vdir, f'{pid}.png')
                if not os.path.exists(q):
                    continue
                tm = times(vdir)
                sec = tm.get(pid, tm.get('_all'))
                raw = rs(load(q), size)
                tag = name if variant == 'base' else f'{name} [{variant}]'
                rows.append({'group': tag, 'id': pid, 'sec': sec} | row(raw, bb, tgt, align))
                if variant != 'base':
                    ref = os.path.join(multi, 'out', model, 'base', f'{pid}.png')
                    if os.path.exists(ref):   # how far the speed-up moved the picture from the reference setting
                        rows[-1]['dE_vs_base_mean'] = round(float(deltaE_ciede2000(rgb2lab(rs(load(ref), size)), rgb2lab(raw)).mean()), 3)
                    continue
                paste, delta, _ = composite(bb.img, raw, bb.lm, align)
                for kind, img in (('paste', paste), ('colour only', delta)):
                    rows.append({'group': f'{tag} + {kind}', 'id': pid, 'sec': sec} | row(img, bb, tgt, False))
        print(pid, 'done', sum(1 for r in rows if r['id'] == pid), 'rows', flush=True)
    groups = list(dict.fromkeys(r['group'] for r in rows))
    summary = {}
    for g in groups:
        gr = [r for r in rows if r['group'] == g and 'error' not in r]
        summary[g] = {'public': stats([r for r in gr if r['id'].startswith('p')]), 'all': stats(gr),
                      'errors': [r['id'] for r in rows if r['group'] == g and 'error' in r]}
        dv = [r['dE_vs_base_mean'] for r in gr if 'dE_vs_base_mean' in r]
        if dv:
            summary[g]['dE_vs_base_mean'] = round(float(np.mean(dv)), 3)
    ev = os.path.join(multi, 'eval')
    os.makedirs(ev, exist_ok=True)
    json.dump({'method': __doc__.strip().split('\n\n')[1], 'pairs': [p['id'] for p in pairs], 'rows': rows, 'summary': summary},
              open(os.path.join(ev, 'multi_metrics.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

    def cell(s, k):
        if k not in s:
            return '-'
        return f"{s[k]['mean']:.2f} ± {s[k]['std']:.2f}" if k != 'hl' else f"{s[k]['mean'] * 100:.0f}% ± {s[k]['std'] * 100:.0f}"
    cols = list(KEYS) + ['sec']
    md = []
    for scope, title in (('public', 'public pairs p01-p10'), ('all', 'all pairs incl. user u01')):
        md += [f'### mean ± std ({title})', '', '| method | n | ' + ' | '.join(LABEL[c] for c in cols) + ' | worst lip dE |',
               '|---|---|' + '---|' * (len(cols) + 1)]
        for g in groups:
            s = summary[g][scope]
            if not s:
                continue
            n = max(v['n'] for v in s.values())
            worst = f"{s['lip_dE']['max']:.2f} ({s['lip_dE']['worst_id']})" if 'lip_dE' in s else '-'
            md.append(f'| {g} | {n} | ' + ' | '.join(cell(s, c) for c in cols) + f' | {worst} |')
        md.append('')
    md += ['### per sample: lip dE00 vs target', '', '| method | ' + ' | '.join(p['id'] for p in pairs) + ' |',
           '|---|' + '---|' * len(pairs)]
    for g in groups:
        by = {r['id']: r for r in rows if r['group'] == g}
        md.append(f'| {g} | ' + ' | '.join(f"{by[p['id']]['lip_dE']:.2f}" if 'lip_dE' in by.get(p['id'], {}) else '-' for p in pairs) + ' |')
    open(os.path.join(ev, 'multi_table.md'), 'w', encoding='utf-8').write('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()

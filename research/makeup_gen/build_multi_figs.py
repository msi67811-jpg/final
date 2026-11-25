"""Figures for MULTI_SAMPLE.md from multi/eval/multi_metrics.json (eval_multi.py output), 2026-11-23.

    python research/makeup_gen/build_multi_figs.py <root=research/data/makeup_gen>

Writes multi/eval/fig_multi_lip_dE.png (public 10 pairs: mean bar + one dot per sample, per method) and
multi/eval/fig_multi_speedup.png (one panel per model: sec / image vs mean dE00 against the reference-setting output).
"""
import json
import sys

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

SURF, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
FAM = {'AR': '#2a78d6', 'FPMT': '#eb6834', 'Stable-Makeup': '#1baf7a', 'MagicMakeup': '#eda100', 'bare': '#8a8984'}
GROUPS = [('bare (no makeup)', '맨얼굴', 'bare'), ('AR glossy 0.65', 'AR glossy 0.65', 'AR'), ('AR matte 0.50', 'AR matte 0.50', 'AR'),
          ('FPMT', 'FPMT', 'FPMT'), ('FPMT + colour only', 'FPMT + 색 차이만', 'FPMT'),
          ('Stable-Makeup', 'Stable-Makeup', 'Stable-Makeup'), ('Stable-Makeup + colour only', 'Stable-Makeup + 색 차이만', 'Stable-Makeup'),
          ('MagicMakeup', 'MagicMakeup', 'MagicMakeup'), ('MagicMakeup + colour only', 'MagicMakeup + 색 차이만', 'MagicMakeup')]
PANELS = [('FPMT (노트북 CPU)', 'FPMT', ['FPMT', 'FPMT [threads8]', 'FPMT [threads2]', 'FPMT [threads1]', 'FPMT [onnx]', 'FPMT [in256]']),
          ('Stable-Makeup (A40)', 'Stable-Makeup', ['Stable-Makeup', 'Stable-Makeup [fp16]', 'Stable-Makeup [steps15]',
                                                    'Stable-Makeup [dpm15]', 'Stable-Makeup [steps8]']),
          ('MagicMakeup (A40)', 'MagicMakeup', ['MagicMakeup', 'MagicMakeup [full]', 'MagicMakeup [steps14]', 'MagicMakeup [steps8]'])]
LABEL_OFFSET = {('FPMT', 'threads8'): (-30, 12), ('FPMT', 'base'): (-8, -16), ('FPMT', 'onnx'): (4, 24)}   # points that sit together


def tidy(ax):
    ax.set_facecolor(SURF)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


def lip_fig(rows, summ, path):
    fig, ax = plt.subplots(figsize=(9, 5.2), facecolor=SURF)
    tidy(ax)
    ax.grid(axis='y', visible=False)
    for i, (g, _, fam) in enumerate(GROUPS):
        y = len(GROUPS) - 1 - i
        m = summ[g]['public']['lip_dE']['mean']
        ax.barh(y, m, height=0.5, color=FAM[fam], alpha=0.35, edgecolor='none')
        vals = [r['lip_dE'] for r in rows if r['group'] == g and r['id'].startswith('p') and 'lip_dE' in r]
        ax.scatter(vals, [y] * len(vals), s=28, color=FAM[fam], edgecolor=SURF, linewidth=1.5, zorder=3)
        ax.text(max(vals) + 0.8, y, f'{m:.1f}', va='center', color=INK, fontsize=9)
    ax.axvline(2.3, color=INK2, lw=1, ls=(0, (3, 3)))
    ax.text(2.4, len(GROUPS) - 0.45, '구분 한계 2.3', color=INK2, fontsize=8)
    ax.set_yticks(range(len(GROUPS)))
    ax.set_yticklabels([lab for _, lab, _ in GROUPS][::-1])
    ax.set_xlabel('입술 ΔE00 (레퍼런스 입술색 대비, 낮을수록 정확)')
    ax.set_title('공개 10쌍 입술 색 오차: 막대 = 평균, 점 = 샘플', loc='left', color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURF)


def speed_fig(summ, path):
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.3), facecolor=SURF)
    for ax, (title, fam, gs) in zip(axs, PANELS):
        tidy(ax)
        for g in gs:
            sec = summ[g]['all']['sec']['mean']
            d = summ[g].get('dE_vs_base_mean', 0.0)
            name = 'base' if '[' not in g else g.split('[')[1].rstrip(']')
            ax.scatter(sec, d, s=60 if name == 'base' else 40, color=FAM[fam], edgecolor=SURF, linewidth=2, zorder=3,
                       marker='D' if name == 'base' else 'o')
            ax.annotate(name, (sec, d), textcoords='offset points', xytext=LABEL_OFFSET.get((fam, name), (6, 5)), fontsize=8.5, color=INK)
        ax.set_title(title, loc='left', color=INK, fontsize=11)
        ax.set_xlabel('장당 처리 시간 (초)')
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=-0.1)
    axs[0].set_ylabel('기준 출력 대비 변화 (이미지 평균 ΔE00)')
    fig.suptitle('가속 설정: 왼쪽 아래일수록 빠르고 기준과 같다 (◆ = 기준 설정)', x=0.01, ha='left', color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURF)


def main():
    ev = f'{sys.argv[1]}/multi/eval'
    j = json.load(open(f'{ev}/multi_metrics.json', encoding='utf-8'))
    plt.rcParams.update({'font.family': 'Malgun Gothic', 'axes.unicode_minus': False, 'font.size': 10, 'axes.edgecolor': INK2,
                         'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2})
    lip_fig(j['rows'], j['summary'], f'{ev}/fig_multi_lip_dE.png')
    speed_fig(j['summary'], f'{ev}/fig_multi_speedup.png')
    print('wrote', ev)


if __name__ == '__main__':
    main()

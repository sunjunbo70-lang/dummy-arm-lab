"""把 python -m dummy_loop.wall study 的结果画成一张图（需要 matplotlib）。

  python tools/simulation/plot_wall_study.py experiments/<日期>_wall_sim_chain/raw/sensitivity.json
左：随机误差下每条 episode 的覆盖率（每个点一条），按补偿方式分行。
右：单因素误差下，无补偿 vs 探触+压缩闭环的覆盖率。
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

INK, MUTED, GRID = '#1f1f1e', '#6b6a64', '#e6e5e0'
SERIES = {'none': '#2a78d6', 'probe+servo': '#eb6834'}     # 分类色 1、2，已过色觉校验
LABEL = {'none': 'no compensation', 'probe': 'probe only', 'servo': 'compression servo only',
         'probe+servo': 'probe + servo'}


def main(path):
    r = json.loads(Path(path).read_text(encoding='utf-8'))
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 6.2), gridspec_kw={'width_ratios': [1, 1.25]})
    for ax in (a, b):
        for s in ('top', 'right'): ax.spines[s].set_visible(False)
        for s in ('left', 'bottom'): ax.spines[s].set_color(GRID)
        ax.tick_params(colors=MUTED); ax.grid(axis='x', color=GRID, lw=0.8); ax.set_axisbelow(True)

    modes = ['none', 'servo', 'probe', 'probe+servo']
    rng = np.random.default_rng(0)
    for i, m in enumerate(modes):
        cov = [e['coverage'] for e in r['randomized'][m]['per_episode'] if 'coverage' in e]
        y = i + rng.uniform(-0.12, 0.12, len(cov))
        a.scatter(cov, y, s=36, color='#2a78d6', alpha=0.75, edgecolor='white', linewidth=0.8, zorder=3)
        mean = float(np.mean(cov))
        a.plot([mean, mean], [i - 0.28, i + 0.28], color=INK, lw=2, zorder=4)
        a.text(mean, i + 0.34, f'mean {mean:.3f}  ·  worst {min(cov):.3f}', ha='center', va='bottom', fontsize=9, color=INK)
    a.set_yticks(range(len(modes))); a.set_yticklabels([LABEL[m] for m in modes], color=INK)
    a.set_xlim(0.2, 1.02); a.set_ylim(-0.6, len(modes) - 0.2)
    a.set_xlabel('coverage of target region', color=MUTED)
    n = r['randomized']['none']['n']
    a.set_title(f'{n} random pre-calibration error sets (one dot per episode)', loc='left', fontsize=11, color=INK)

    rows = []
    for fac, levels in r['single_factor'].items():
        for val, modes_ in levels.items():
            rows.append((f'{fac} = {val}', modes_['none']['coverage'], modes_['probe+servo']['coverage']))
    rows.append(('no error', r['baseline_no_error']['none']['coverage'], r['baseline_no_error']['probe+servo']['coverage']))
    ys = np.arange(len(rows))[::-1]
    for y, (name, c0, c1) in zip(ys, rows):
        b.plot([c0, c1], [y, y], color=GRID, lw=2.5, zorder=2, solid_capstyle='round')
        b.scatter([c0], [y], s=48, color=SERIES['none'], zorder=3, edgecolor='white', linewidth=1)
        b.scatter([c1], [y], s=48, color=SERIES['probe+servo'], zorder=3, edgecolor='white', linewidth=1)
    b.set_yticks(ys); b.set_yticklabels([r_[0] for r_ in rows], fontsize=8.5, color=INK)
    b.set_xlim(0.68, 1.02); b.set_xlabel('coverage of target region', color=MUTED)
    b.set_title('One error at a time', loc='left', fontsize=11, color=INK)
    b.scatter([], [], s=48, color=SERIES['none'], label=LABEL['none'])
    b.scatter([], [], s=48, color=SERIES['probe+servo'], label=LABEL['probe+servo'])
    b.legend(loc='lower left', frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle('Wall-trowel simulation: coverage under sim-to-real errors  (L1, reference model, assumed parameters)',
                 x=0.01, ha='left', fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = Path(path).with_name('sensitivity.png'); fig.savefig(out, dpi=120, facecolor='white')
    print(out)


if __name__ == '__main__':
    main(sys.argv[1])

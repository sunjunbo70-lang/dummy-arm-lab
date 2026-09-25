"""把 python -m dummy_loop.wall rl-train 的结果画成一张图（需要 matplotlib）。

  python tools/simulation/plot_rl_training.py experiments/v0.1/r0/runs/rl/training.json

左：学习曲线（训练回报、评估回报，与脚本基线 / 示教预热 / 随机策略对比）。
右：一刀抹完之后墙上的料层厚度沿高度的分布——脚本基线 vs 学到的策略 vs 目标厚度。
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

INK, MUTED, GRID = '#1f1f1e', '#6b6a64', '#e6e5e0'
LEARNED, SCRIPTED = '#2a78d6', '#eb6834'      # 分类色 1、2，已过色觉校验


def thickness_profile(policy_path=None):
    from dummy_loop.wall.rl import make_env, load_policy, Scaled
    from dummy_loop.wall.stroke_env import rollout
    env = make_env(1000)
    policy = None
    if policy_path is not None:
        policy = Scaled(env, load_policy(env, policy_path), deterministic=True)
    rollout(env, policy)
    f = env.field
    v = f.cv[:, 0]
    inside = f.inside.any(axis=1)
    prof = np.where(f.inside, f.h, np.nan)
    return v[inside] * 1000, np.nanmean(prof[inside], axis=1) * 1000, env.metrics()


def main(path):
    r = json.loads(Path(path).read_text(encoding='utf-8'))
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={'width_ratios': [1.25, 1]})
    for ax in (a, b):
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        for s in ('left', 'bottom'):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=MUTED); ax.grid(axis='y', color=GRID, lw=0.8); ax.set_axisbelow(True)

    upd = [row['update'] for row in r['curve']]
    a.plot(upd, [row['train_return_mean'] for row in r['curve']], color=LEARNED, lw=1.2, alpha=0.5,
           label='training return (with exploration noise)')
    ev = [(row['update'], row['eval']['return_mean']) for row in r['curve'] if 'eval' in row]
    a.plot([e[0] for e in ev], [e[1] for e in ev], color=LEARNED, lw=2.2, marker='o', ms=4,
           label='learned policy (evaluation)')
    for value, label, color, style in [
            (r['scripted_baseline']['return_mean'], 'scripted stroke', SCRIPTED, '--'),
            (r['after_behaviour_clone']['return_mean'] if r.get('after_behaviour_clone') else None,
             'after behaviour cloning', MUTED, ':')]:
        if value is not None:
            a.axhline(value, color=color, lw=1.6, ls=style)
            a.text(upd[-1], value, f' {label}: {value:.2f}', va='center', fontsize=9, color=color)
    a.set_xlabel('PPO update', color=MUTED); a.set_ylabel('episode return', color=MUTED)
    a.set_title(f"Learning to spread one stroke  (random policy: {r['random_policy']['return_mean']:.1f})",
                loc='left', fontsize=11, color=INK)
    a.legend(loc='lower right', frameon=False, fontsize=9, labelcolor=INK)

    policy = Path(path).with_name('policy.npz')
    target = r['env']['material']['target_thickness'] * 1000
    v_s, h_s, m_s = thickness_profile(None)
    b.plot(h_s, v_s, color=SCRIPTED, lw=2, label=f"scripted (rms {m_s['rms_error_mm']:.2f} mm)")
    if policy.is_file():
        v_l, h_l, m_l = thickness_profile(policy)
        b.plot(h_l, v_l, color=LEARNED, lw=2, label=f"learned (rms {m_l['rms_error_mm']:.2f} mm)")
    b.axvline(target, color=INK, lw=1.4, ls='--')
    b.text(target, v_s[-1], f' target {target:.1f} mm', fontsize=9, color=INK, va='top')
    b.set_xlabel('layer thickness (mm)', color=MUTED); b.set_ylabel('height on the wall (mm)', color=MUTED)
    b.set_title('Layer left by one stroke', loc='left', fontsize=11, color=INK)
    b.legend(loc='lower right', frameon=False, fontsize=9, labelcolor=INK)

    fig.suptitle('Trowel stroke RL  (L1 simulation; material is a reduced-order proxy model, parameters assumed)',
                 x=0.01, ha='left', fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = Path(path).with_name('training.png'); fig.savefig(out, dpi=120, facecolor='white')
    print(out)


if __name__ == '__main__':
    main(sys.argv[1])

"""对比两次抹涂训练（工人手法预热 vs 平刀预热），并跑整片作业。需要 matplotlib。

  python tools/simulation/rl_compare.py --technique outputs/wall/rl_tech --flat outputs/wall/rl_flat \
      --out outputs/wall/rl_compare

产出（全部是 L1 仿真，不连硬件）：
  training.png   学习曲线 + 抹出来的料层 RMS 误差
  technique.png  手法轨迹：刀面俯仰角、下缘间隙、刀上剩料 随刀高度的变化（四条策略对比）
  session.png    整片作业的料层厚度图（脚本 vs 两个学到的策略）
  report.json    上面三张图背后的全部数字
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dummy_loop.wall.ppo import PPOConfig                      # noqa: E402
from dummy_loop.wall.rl import Scaled, load_policy, make_env   # noqa: E402
from dummy_loop.wall.session import SessionConfig, run_session  # noqa: E402
from dummy_loop.wall.stroke_env import StrokeConfig            # noqa: E402

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'WenQuanYi Zen Hei', 'Source Han Sans SC',
                                   'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
INK, GRID = '#1f1f1e', '#e6e5e0'


def policy_for(env, path):
    if path is None:
        return None
    return Scaled(env, load_policy(env, path, PPOConfig()), deterministic=True)


def trace(env, policy=None):
    """跑一刀，逐步记录刀面俯仰、下缘间隙、刀上剩料、力。"""
    rows, obs, done = [], env.reset(), False
    while not done:
        a = env.scripted_action() if policy is None else policy(obs)
        low, _ = env.blade_edges()
        rows.append({'step': len(rows), 'pitch_deg': float(np.rad2deg(env.blade_pitch())),
                     'lower_gap_mm': float(max(-low[2], 0.0)) * 1000, 'v_mm': float(low[1]) * 1000,
                     'load_frac': float(env.field.load / max(env.field.load0, 1e-12)),
                     'force_N': float(env.last_force)})
        obs, _, done, _ = env.step(a)
    return rows, env.metrics()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--technique', type=Path, required=True, help='工人手法预热那次训练的输出目录')
    ap.add_argument('--flat', type=Path, required=True, help='平刀（不用手法）预热那次训练的输出目录')
    ap.add_argument('--out', type=Path, default=Path('outputs/wall/rl_compare'))
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    runs = {'technique': a.technique, 'flat': a.flat}
    report = {'evidence_level': 'L1', 'hardware_motion': False, 'runs': {}}

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for k, d in runs.items():
        t = json.loads((d / 'training.json').read_text(encoding='utf-8'))
        ev = [(r['update'], r['eval']) for r in t['curve'] if 'eval' in r]
        ax[0].plot([r['update'] for r in t['curve']], [r['train_return_mean'] for r in t['curve']], alpha=.5)
        ax[0].plot([e[0] for e in ev], [e[1]['return_mean'] for e in ev], 'o-', label=f'{k} 评估回报')
        ax[1].plot([e[0] for e in ev], [e[1]['rms_error_mm'] for e in ev], 'o-', label=k)
        report['runs'][k] = {x: t[x] for x in ('scripted_baseline', 'after_behaviour_clone', 'learned_final',
                                               'random_policy', 'updates', 'steps', 'elapsed_s')}
    ax[0].set(xlabel='PPO 更新轮', ylabel='单刀回报', title='学习曲线')
    ax[1].set(xlabel='PPO 更新轮', ylabel='厚度 RMS 误差 mm', title='抹出来的料层误差')
    for x in ax:
        x.grid(color=GRID); x.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(a.out / 'training.png', dpi=130); plt.close(fig)

    cases = [('脚本-工人手法', 'technique', None), ('脚本-不用手法', 'flat', None),
             ('学到的(手法预热)', 'technique', a.technique), ('学到的(平刀预热)', 'flat', a.flat)]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    report['traces'] = {}
    for name, style, pol in cases:
        env = make_env(1000, 0.0, StrokeConfig(script_style=style))
        rows, m = trace(env, policy_for(env, pol and pol / 'policy.npz'))
        report['traces'][name] = {'metrics': m, 'rows': rows}
        v = [r['v_mm'] for r in rows]
        ax[0].plot(v, [r['pitch_deg'] for r in rows], label=name)
        ax[1].plot(v, [r['lower_gap_mm'] for r in rows], label=name)
        ax[2].plot(v, [r['load_frac'] for r in rows], label=name)
    for x, t in zip(ax, ('刀面俯仰角 (deg)', '下缘间隙 (mm)', '刀上剩料 / 一份')):
        x.set(xlabel='刀下缘高度 v (mm)', title=t); x.grid(color=GRID); x.legend(fontsize=7)
    ax[1].axhline(2.0, color=INK, ls='--', lw=.8)       # 目标厚度
    fig.tight_layout(); fig.savefig(a.out / 'technique.png', dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    report['sessions'] = {}
    for i, (name, style, pol) in enumerate([c for c in cases if c[0] != '脚本-不用手法']):
        env = make_env(2000, 0.0, StrokeConfig(script_style=style))
        m, strokes, field = run_session(env, policy_for(env, pol and pol / 'policy.npz'), SessionConfig())
        report['sessions'][name] = {'area': m, 'strokes': strokes}
        im = ax[i].pcolormesh(field.cu * 100, field.cv * 100, field.h * 1000, cmap='viridis', vmin=0, vmax=4)
        ax[i].set(title=f'{name}\n覆盖 {m["coverage"]:.2f} / RMS {m["rms_error_mm"]:.2f} mm',
                  xlabel='u (cm)', ylabel='v (cm)'); ax[i].set_aspect('equal')
        fig.colorbar(im, ax=ax[i], label='厚度 mm')
    fig.tight_layout(); fig.savefig(a.out / 'session.png', dpi=130); plt.close(fig)

    (a.out / 'report.json').write_text(json.dumps(report, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({k: v['area'] for k, v in report['sessions'].items()}, ensure_ascii=False, indent=1))
    print('written', a.out)


if __name__ == '__main__':
    main()

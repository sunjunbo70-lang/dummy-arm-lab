"""Figures and summary for the v0.3 whole-wall experiment (two runs: flat / technique warm start).

    python tools/simulation/wall_cycle_report.py --flat <run_dir> --technique <run_dir> --out <dir>

Writes (L1 simulation evidence):
  curves.png       validation return / coverage / waste / mean deposit pitch per PPO update
  strokes.png      distribution of the pitch and force the policies actually use on deposit
                   strokes: teacher vs DAgger-BC vs selected PPO checkpoint, for both runs
  walls.png        final wall thickness on one test seed (co-simulation), teacher vs selected
  summary.json     the numbers behind every figure + the like-for-like test tables
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
from dummy_loop.wall.ppo import PPO, PPOConfig                 # noqa: E402
from dummy_loop.wall_cycle.area import load_work_area            # noqa: E402
from dummy_loop.wall_cycle.config import CycleConfig             # noqa: E402
from dummy_loop.wall_cycle import train as T                     # noqa: E402

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'WenQuanYi Zen Hei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
C_TEACH, C_BC, C_PPO = '#8a8a86', '#eb6834', '#2a78d6'


def load(run):
    return json.loads((run / 'training.json').read_text(encoding='utf-8'))


def agent(run, name, env):
    return PPO(env.obs_dim, env.act_dim, PPOConfig(hidden=(128, 128))).load(run / name)


def stroke_stats(cfg, style, pol, episodes=20, seed=50000):
    """Pitch/force of every deposit/reuse stroke actually EXECUTED (after the safety layer)."""
    ps, fs = [], []
    for i in range(episodes):
        env = T.make_env(cfg, seed + i, style, 'table')
        o = env.reset(); done = False
        while not done:
            a = env.teacher_action() if pol is None else pol.act(o, deterministic=True)[0]
            d = env.decode(a)
            n_proj = env.projected
            o, r, done, info = env.step(a)
            if d.mode in ('DEPOSIT', 'REUSE') and env.projected == n_proj:
                ps.append(np.rad2deg(d.pitch_start)); fs.append(d.force_N)
    return np.asarray(ps), np.asarray(fs)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--flat', type=Path, required=True)
    ap.add_argument('--technique', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--stroke-episodes', type=int, default=20)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    cfg = load_work_area(CycleConfig())
    runs = {'flat': a.flat, 'technique': a.technique}
    tr = {k: load(v) for k, v in runs.items()}
    summary = {'evidence_level': 'L1', 'hardware_motion': False, 'runs': {}}

    # ---- curves
    fig, ax = plt.subplots(1, 4, figsize=(17, 3.8))
    for k, t in tr.items():
        ev = [(r['update'], r['val']) for r in t['curve'] if 'val' in r]
        u = [e[0] for e in ev]
        ls = '-' if k == 'technique' else '--'
        ax[0].plot(u, [e[1]['return_mean'] for e in ev], ls, marker='o', label=f'{k} 预热')
        ax[1].plot(u, [e[1]['coverage_mean'] for e in ev], ls, marker='o', label=k)
        ax[2].plot(u, [e[1]['waste_frac_mean'] for e in ev], ls, marker='o', label=k)
        ax[3].plot(u, [e[1].get('deposit_pitch_start_deg_mean', 0) for e in ev], ls, marker='o', label=k)
    for x, ttl in zip(ax, ('验证集回报', '覆盖率', '浪费率', '抹涂刀次平均起始俯仰 (°)')):
        x.set_title(ttl); x.set_xlabel('PPO 更新轮'); x.grid(alpha=.3); x.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(a.out / 'curves.png', dpi=130); plt.close(fig)

    # ---- executed pitch / force distributions
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    for col, (k, run) in enumerate(runs.items()):
        env = T.make_env(cfg, 0, k, 'table')
        pols = {'教师': None, 'DAgger-BC': agent(run, 'policy_bc.npz', env), '选中模型': agent(run, 'policy.npz', env)}
        colours = {'教师': C_TEACH, 'DAgger-BC': C_BC, '选中模型': C_PPO}
        st = {}
        for name, pol in pols.items():
            p, f = stroke_stats(cfg, k, pol, a.stroke_episodes)
            st[name] = {'pitch_deg_mean': float(p.mean()) if len(p) else None,
                        'pitch_deg_quartiles': [float(x) for x in np.percentile(p, [25, 50, 75])] if len(p) else None,
                        'force_N_mean': float(f.mean()) if len(f) else None, 'strokes': int(len(p))}
            if len(p):
                ax[0, col].hist(p, bins=np.arange(0, 37, 2.5), alpha=.55, color=colours[name], label=name)
                ax[1, col].hist(f, bins=np.linspace(0.5, 15, 30), alpha=.55, color=colours[name], label=name)
        ax[0, col].set_title(f'{k} 预热：实际执行的起始俯仰 (°)'); ax[1, col].set_title(f'{k} 预热：接触力 (N)')
        for r_ in (0, 1):
            ax[r_, col].legend(fontsize=8); ax[r_, col].grid(alpha=.3)
        summary['runs'][k] = {'stroke_distributions': st, 'test': tr[k]['test'], 'test_cosim': tr[k].get('test_cosim'),
                              'selected_checkpoint': tr[k]['selected_checkpoint'], 'ppo_steps': tr[k]['ppo_steps'],
                              'elapsed_s': tr[k]['elapsed_s'], 'exploration': tr[k].get('exploration')}
    fig.tight_layout(); fig.savefig(a.out / 'strokes.png', dpi=130); plt.close(fig)

    # ---- final walls in co-simulation, one test seed
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    seed = T.TEST_SEED
    i = 0
    for k, run in runs.items():
        env0 = T.make_env(cfg, 0, k, 'cosim')
        for name, pol in (('教师', None), ('选中模型', agent(run, 'policy.npz', env0))):
            env = T.make_env(cfg, seed, k, 'cosim')
            o = env.reset(); done = False
            while not done:
                act = env.teacher_action() if pol is None else pol.act(o, deterministic=True)[0]
                o, r, done, info = env.step(act)
            m = info['metrics']
            im = ax[i].imshow(env.material.wall * 1000, origin='lower', cmap='viridis', vmin=0, vmax=4,
                              extent=[-cfg.width_m / 2 * 100, cfg.width_m / 2 * 100, 0, cfg.height_m * 100])
            ax[i].set_title(f'{k}·{name}\n覆盖 {m["coverage"]:.2f}  RMSE {m["rmse_mm"]:.2f} mm  浪费 {m["waste_frac"]:.2f}', fontsize=9)
            ax[i].set_xlabel('u (cm)'); ax[i].set_ylabel('v (cm)')
            summary['runs'][k].setdefault('cosim_wall_seed', {})[name] = {'seed': seed, **{kk: float(vv) for kk, vv in m.items()}}
            i += 1
    fig.colorbar(im, ax=ax, label='厚度 mm', shrink=.8)
    fig.savefig(a.out / 'walls.png', dpi=130, bbox_inches='tight'); plt.close(fig)
    (a.out / 'summary.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({k: v['stroke_distributions'] for k, v in summary['runs'].items()}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()

"""墙面抹涂仿真命令行。全部是仿真，不连接硬件。

  python -m dummy_loop.wall scene   --out experiments/v0.1/r0/runs/scene.xml     导出场景 MJCF（可用 MuJoCo 查看器打开）
  python -m dummy_loop.wall layout  --out experiments/v0.1/r0/runs/layout.json   工位布局搜索
  python -m dummy_loop.wall demo    [--viewer] [--probe] [--servo] [--random-scale 1]
  python -m dummy_loop.wall collect --episodes 50 --out experiments/v0.1/r0/runs/episodes [--random-scale 1] [--probe] [--servo]
  python -m dummy_loop.wall replay  experiments/v0.1/r0/runs/episodes/ep_0000.npz
  python -m dummy_loop.wall study   --out experiments/v0.1/r0/runs/study
  python -m dummy_loop.wall render  --out experiments/v0.1/r0/runs/camera        固定相机 RGB 与深度图
  python -m dummy_loop.wall rl-train --updates 60 --out experiments/v0.1/r0/runs/rl   抹涂手法强化学习（示教预热 + PPO）
  python -m dummy_loop.wall rl-eval  --policy experiments/v0.1/r0/runs/rl/policy.npz  评估学到的策略并与脚本基线对比
  python -m dummy_loop.wall rl-session [--policy ...] --out experiments/v0.1/r0/runs/session   一刀一刀把整片区域抹完
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np


def emit(x):
    print(json.dumps(x, ensure_ascii=False, indent=2))


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m dummy_loop.wall', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('scene'); s.add_argument('--out', type=Path, default=Path('experiments/v0.1/r0/runs/scene.xml'))
    s = sub.add_parser('layout'); s.add_argument('--out', type=Path, default=Path('experiments/v0.1/r0/runs/layout.json'))
    s.add_argument('--joint-limit-cap-deg', type=float, default=None,
                   help='在 V2 固件限位之外再加的对称保守包络（度）；默认不加')
    for name in ('demo', 'collect'):
        s = sub.add_parser(name)
        s.add_argument('--probe', action='store_true', help='开工前探触墙面并修正墙面坐标系')
        s.add_argument('--servo', action='store_true', help='按压缩量闭环修正压入深度')
        s.add_argument('--random-scale', type=float, default=0.0, help='域随机化强度；0 = 无误差')
        s.add_argument('--seed', type=int, default=0)
        if name == 'demo':
            s.add_argument('--viewer', action='store_true')
        else:
            s.add_argument('--episodes', type=int, default=20)
            s.add_argument('--out', type=Path, default=Path('experiments/v0.1/r0/runs/episodes'))
    s = sub.add_parser('replay'); s.add_argument('episode', type=Path)
    s = sub.add_parser('study'); s.add_argument('--out', type=Path, default=Path('experiments/v0.1/r0/runs/study'))
    s.add_argument('--random', type=int, default=20); s.add_argument('--random-scale', type=float, default=1.0)
    s = sub.add_parser('render'); s.add_argument('--out', type=Path, default=Path('experiments/v0.1/r0/runs/camera'))
    s = sub.add_parser('rl-train'); s.add_argument('--out', type=Path, default=Path('experiments/v0.1/r0/runs/rl'))
    s.add_argument('--updates', type=int, default=60); s.add_argument('--steps-per-update', type=int, default=2048)
    s.add_argument('--seed', type=int, default=0); s.add_argument('--random-scale', type=float, default=0.0)
    s.add_argument('--lr', type=float, default=5e-5); s.add_argument('--init-log-std', type=float, default=-3.0)
    s.add_argument('--no-warm-start', action='store_true', help='不做示教预热（从零探索，基本学不动，用于对照）')
    s.add_argument('--eval-every', type=int, default=10); s.add_argument('--eval-episodes', type=int, default=8)
    s.add_argument('--script-style', choices=('technique', 'flat'), default='technique',
                   help='示教预热用的手写手法：technique = 工人手法（斜着贴墙再放平）；flat = 不用手法（对照组）')
    s = sub.add_parser('rl-eval'); s.add_argument('--policy', type=Path, required=True)
    s.add_argument('--episodes', type=int, default=20); s.add_argument('--seed', type=int, default=1000)
    s.add_argument('--random-scale', type=float, default=0.0)
    s = sub.add_parser('rl-session'); s.add_argument('--out', type=Path, default=Path('experiments/v0.1/r0/runs/session'))
    s.add_argument('--policy', type=Path, default=None, help='不给就用手写脚本跑')
    s.add_argument('--seed', type=int, default=2000)
    s.add_argument('--columns', type=float, nargs='+', default=[-0.08, 0.0, 0.08], help='每条带的中心 u（m）')
    s.add_argument('--band-v', type=float, nargs=2, default=[0.0, 0.07], help='每条带的竖直范围（m）')
    s.add_argument('--simulate-transit', action='store_true',
                   help='把带与带之间的横移也仿真出来（已知会卡在腕部支解切换上，见 session.py 说明）')
    for name, sp in sub.choices.items():
        if name in ('scene', 'demo', 'collect', 'study'):
            sp.add_argument('--tool', choices=('rigid', 'spring'), default='rigid',
                            help='末端工具：rigid = 实际刚性抹刀（默认）；spring = 早期弹簧滑轨方案，仅对比用')
    for name in ('scene', 'demo', 'collect'):
        sub.choices[name].add_argument('--tool-profile', choices=('legacy', 'lab_20260922'), default='legacy')
    a = ap.parse_args(argv)

    def selected_scene():
        from .scene import SceneConfig
        if getattr(a, 'tool_profile', 'legacy') == 'lab_20260922':
            from .lab_tool import scene_config
            return scene_config(tool_mount=a.tool)
        return SceneConfig(tool_mount=a.tool)

    from .errors import Perturbation
    if a.cmd == 'scene':
        from .scene import SceneConfig, export_xml
        emit({'written': str(export_xml(selected_scene(), a.out))})
    elif a.cmd == 'layout':
        from .layout import search
        r = search(joint_limit_cap_deg=a.joint_limit_cap_deg)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(r, indent=1, ensure_ascii=False), encoding='utf-8')
        emit({'n_feasible': r['n_feasible'], 'best': r['ranked'][:3], 'written': str(a.out)})
    elif a.cmd == 'demo':
        from .task import WallTask
        from .teacher import RasterTeacher
        from .scene import SceneConfig
        pert = Perturbation.sample(np.random.default_rng(a.seed), a.random_scale) if a.random_scale > 0 else Perturbation()
        env = WallTask(selected_scene(), perturbation=pert, seed=a.seed)
        teacher = RasterTeacher(env, probe=a.probe, servo=a.servo)
        viewer = None
        if a.viewer:
            import mujoco.viewer
            viewer = mujoco.viewer.launch_passive(env.model, env.data)
            viewer.cam.lookat[:] = [0.22, 0, 0.2]; viewer.cam.distance = 0.9
            viewer.cam.azimuth = -130; viewer.cam.elevation = -15

            def on_step(obs, action, info):
                t = time.monotonic(); viewer.sync()
                if not viewer.is_running():
                    raise KeyboardInterrupt
                time.sleep(max(0.0, env.spec.dt - (time.monotonic() - t)))
        else:
            on_step = None
        try:
            m, log = teacher.run(on_step=on_step)
            emit({'metrics': m, 'teacher_log': log, 'perturbation': pert.to_dict(),
                  'scope': 'simulation only, L1; scene dimensions are placeholders'})
            if viewer:
                print('episode finished; close the viewer window to exit')
                while viewer.is_running():
                    time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            if viewer:
                viewer.close()
    elif a.cmd == 'collect':
        from .pipeline import record_episode
        from .scene import SceneConfig
        rng = np.random.default_rng(a.seed); a.out.mkdir(parents=True, exist_ok=True); index = []
        for i in range(a.episodes):
            pert = Perturbation.sample(rng, a.random_scale) if a.random_scale > 0 else Perturbation()
            path = a.out / f'ep_{i:04d}.npz'
            try:
                m = record_episode(path, pert, seed=a.seed + i, probe=a.probe, servo=a.servo,
                                   scene=selected_scene())
                index.append({'file': path.name, **m})
            except RuntimeError as e:
                index.append({'file': None, 'failed': str(e)})
        (a.out / 'index.json').write_text(json.dumps(index, indent=1, ensure_ascii=False), encoding='utf-8')
        ok = [r for r in index if 'coverage' in r]
        emit({'episodes': len(index), 'failed': len(index) - len(ok),
              'coverage_mean': round(float(np.mean([r['coverage'] for r in ok])), 4) if ok else None, 'out': str(a.out)})
    elif a.cmd == 'replay':
        from .pipeline import replay_episode
        rec, rep = replay_episode(a.episode)
        emit({'recorded': rec, 'replayed': rep, 'identical': rec == rep})
        if rec != rep:
            raise SystemExit(1)
    elif a.cmd == 'study':
        from .pipeline import run_study
        from .scene import SceneConfig
        r = run_study(a.out, a.random, a.random_scale, scene=selected_scene())
        emit({'randomized': r['randomized'], 'elapsed_s': r['elapsed_s'], 'written': str(a.out / 'sensitivity.json')})
    elif a.cmd == 'rl-train':
        from .rl import train
        from .ppo import PPOConfig
        from .stroke_env import StrokeConfig
        cfg = PPOConfig(steps_per_update=a.steps_per_update, seed=a.seed, lr=a.lr,
                        init_log_std=a.init_log_std, entropy_coef=0.0, epochs=5)
        r = train(a.out, updates=a.updates, cfg=cfg, stroke=StrokeConfig(script_style=a.script_style),
                  random_scale=a.random_scale, seed=a.seed,
                  eval_every=a.eval_every, eval_episodes=a.eval_episodes, warm_start=not a.no_warm_start,
                  log=lambda s: print(s, flush=True))
        emit({'scripted_baseline': r['scripted_baseline'], 'after_behaviour_clone': r['after_behaviour_clone'],
              'learned_final': r['learned_final'], 'random_policy': r['random_policy'],
              'elapsed_s': r['elapsed_s'], 'written': str(a.out / 'training.json')})
    elif a.cmd == 'rl-eval':
        from .rl import make_env, evaluate, load_policy
        env = make_env(a.seed, a.random_scale)
        agent = load_policy(env, a.policy)
        emit({'learned': evaluate(env, agent, a.episodes), 'scripted': evaluate(env, None, a.episodes),
              'scope': 'simulation only, L1; material is a reduced-order proxy model'})
    elif a.cmd == 'rl-session':
        from .rl import make_env, load_policy, Scaled
        from .session import SessionConfig, run_session
        env = make_env(a.seed)
        policy = None
        if a.policy is not None:
            policy = Scaled(env, load_policy(env, a.policy), deterministic=True)
        cfg = SessionConfig(column_centres=tuple(a.columns), band_v=tuple(a.band_v),
                            simulate_transit=a.simulate_transit)
        m, strokes, field = run_session(env, policy, cfg)
        a.out.mkdir(parents=True, exist_ok=True)
        np.savez(a.out / 'thickness.npz', h=field.h, u=field.cu, v=field.cv, inside=field.inside)
        r = {'evidence_level': 'L1', 'hardware_motion': False, 'policy': str(a.policy) if a.policy else 'scripted',
             'session': cfg.to_dict(), 'area': m, 'strokes': strokes,
             'note': 'strokes[].return 不能与单刀训练回报比较：终局奖励是在整片共用高度场上算的'}
        (a.out / 'session.json').write_text(json.dumps(r, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
        emit({'area': m, 'strokes': strokes, 'written': str(a.out / 'session.json')})
    elif a.cmd == 'render':
        from .render import render_camera
        emit(render_camera(a.out))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, RuntimeError, OSError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr); raise SystemExit(2)

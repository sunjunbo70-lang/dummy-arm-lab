"""墙面抹涂仿真命令行。全部是仿真，不连接硬件。

  python -m dummy_loop.wall scene   --out outputs/wall/scene.xml     导出场景 MJCF（可用 MuJoCo 查看器打开）
  python -m dummy_loop.wall layout  --out outputs/wall/layout.json   工位布局搜索
  python -m dummy_loop.wall demo    [--viewer] [--probe] [--servo] [--random-scale 1]
  python -m dummy_loop.wall collect --episodes 50 --out outputs/wall/episodes [--random-scale 1] [--probe] [--servo]
  python -m dummy_loop.wall replay  outputs/wall/episodes/ep_0000.npz
  python -m dummy_loop.wall study   --out outputs/wall/study
  python -m dummy_loop.wall render  --out outputs/wall/camera        固定相机 RGB 与深度图
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
    s = sub.add_parser('scene'); s.add_argument('--out', type=Path, default=Path('outputs/wall/scene.xml'))
    s = sub.add_parser('layout'); s.add_argument('--out', type=Path, default=Path('outputs/wall/layout.json'))
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
            s.add_argument('--out', type=Path, default=Path('outputs/wall/episodes'))
    s = sub.add_parser('replay'); s.add_argument('episode', type=Path)
    s = sub.add_parser('study'); s.add_argument('--out', type=Path, default=Path('outputs/wall/study'))
    s.add_argument('--random', type=int, default=20); s.add_argument('--random-scale', type=float, default=1.0)
    s = sub.add_parser('render'); s.add_argument('--out', type=Path, default=Path('outputs/wall/camera'))
    for name, sp in sub.choices.items():
        if name in ('scene', 'demo', 'collect', 'study'):
            sp.add_argument('--tool', choices=('rigid', 'spring'), default='rigid',
                            help='末端工具：rigid = 实际刚性抹刀（默认）；spring = 早期弹簧滑轨方案，仅对比用')
    a = ap.parse_args(argv)

    from .errors import Perturbation
    if a.cmd == 'scene':
        from .scene import SceneConfig, export_xml
        emit({'written': str(export_xml(SceneConfig(tool_mount=a.tool), a.out))})
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
        env = WallTask(SceneConfig(tool_mount=a.tool), perturbation=pert, seed=a.seed)
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
                                   scene=SceneConfig(tool_mount=a.tool))
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
        r = run_study(a.out, a.random, a.random_scale, scene=SceneConfig(tool_mount=a.tool))
        emit({'randomized': r['randomized'], 'elapsed_s': r['elapsed_s'], 'written': str(a.out / 'sensitivity.json')})
    elif a.cmd == 'render':
        from .render import render_camera
        emit(render_camera(a.out))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, RuntimeError, OSError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr); raise SystemExit(2)

"""墙面可达区域分析：抹刀正对墙面时，末端能压到墙上的哪些位置。

软件分析（L1），不连接硬件，不发送任何指令。

方法：墙是位于机械臂正前方（模型 -Y 方向）、垂直于 Y 轴的平面，距 J1 轴 D 米。
在墙面上按网格取点，对每个点求逆运动学：要求刀面中心（法兰沿 J6 轴外伸
tool_length 米）落在该点，且 J6 轴与墙面法线夹角 < 3°。J6 绕自身轴的转角不影响
这两个约束，所以这是 5 个约束对 6 个关节。能解出来的点即为可达。

关节范围未经标定（configs/ 下 profile 的限位全为 null），因此按三类假设分别计算：
  S1  上位机当前调试范围（dummy_loop/live_control.py 的 LOWER/UPPER，固件角度）。
      假设固件 HOME (0,0,90,0,0,0) 对应模型零位；J2、J3 的正方向未标定，四种组合都算。
  S2  仿真执行器 ctrlrange，全部 ±40°。
  S3  全部 ±90°，只反映连杆几何，不代表实机限位。

用法：
  python tools/simulation/wall_workspace_ik.py --out experiments/<日期>_wall_workspace/raw
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / 'models' / 'dummy_reference.xml'
D2R = np.pi / 180
WALL_NORMAL = np.array([0., -1., 0.])   # 刀具须指向前方墙面
W_ORI = 0.2                              # 方向误差权重：1 rad ≈ 20 cm 位置误差
POS_TOL_M, ANG_TOL_DEG = 2e-3, 3.0


def scenarios():
    s = {'S2_sim_ctrlrange_pm40': ([-40] * 6, [40] * 6),
         'S3_geometry_only_pm90': ([-90] * 6, [90] * 6)}
    for s2 in (1, -1):
        for s3 in (1, -1):
            j2 = sorted([s2 * -75, s2 * 20])
            j3 = sorted([0, s3 * 90])
            key = f'S1_gui_envelope_J2{"+" if s2 > 0 else "-"}_J3{"+" if s3 > 0 else "-"}'
            s[key] = ([-20, j2[0], j3[0], -15, -15, -15], [20, j2[1], j3[1], 15, 15, 15])
    return s


class Arm:
    def __init__(self, tool_length):
        self.m = mujoco.MjModel.from_xml_path(str(MODEL))
        self.d = mujoco.MjData(self.m)
        self.fl = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, 'link6_1_1')
        self.j5 = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, 'Joint5')
        self.j6 = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, 'Joint6')
        self.L = tool_length

    def fk(self, q):
        m, d = self.m, self.d
        d.qpos[:6] = q
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        p = d.xpos[self.fl].copy()
        a = d.xaxis[self.j6].copy()
        if a @ (p - d.xanchor[self.j5]) < 0:   # 让 a 指向远离手腕的方向
            a = -a
        return p + self.L * a, a

    def ik(self, target, q0, lo, hi, iters=80):
        q = np.clip(q0, lo, hi)
        for _ in range(iters):
            t, a = self.fk(q)
            ang = np.degrees(np.arccos(np.clip(a @ WALL_NORMAL, -1, 1)))
            if np.linalg.norm(t - target) < 1e-3 and ang < 2:
                return q, True
            jp = np.zeros((3, self.m.nv)); jr = np.zeros((3, self.m.nv))
            mujoco.mj_jac(self.m, self.d, jp, jr, t, self.fl)
            ja = np.cross(jr[:, :6].T, a).T
            J = np.r_[jp[:, :6], W_ORI * ja]
            r = np.r_[t - target, W_ORI * (a - WALL_NORMAL)]
            q = np.clip(q - np.linalg.solve(J.T @ J + 1e-3 * np.eye(6), J.T @ r), lo, hi)
        t, a = self.fk(q)
        ang = np.degrees(np.arccos(np.clip(a @ WALL_NORMAL, -1, 1)))
        return q, bool(np.linalg.norm(t - target) < POS_TOL_M and ang < ANG_TOL_DEG)


def scan(arm, lo_deg, hi_deg, D, xs, zs, rng):
    lo, hi = np.array(lo_deg) * D2R, np.array(hi_deg) * D2R
    reach = np.zeros((len(zs), len(xs)), bool)
    last = np.clip(np.zeros(6), lo, hi)
    for i, z in enumerate(zs):
        for j, x in enumerate(xs):
            tgt = np.array([x, -D, z])
            seeds = [last, np.clip(np.zeros(6), lo, hi)] + [lo + (hi - lo) * rng.random(6) for _ in range(3)]
            for s in seeds:
                q, ok = arm.ik(tgt, s, lo, hi)
                if ok:
                    reach[i, j] = True
                    last = q
                    break
    return reach


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tool-length', type=float, default=0.10, help='法兰到刀面中心的距离 (m)')
    ap.add_argument('--walls', type=float, nargs='+', default=[0.20, 0.25, 0.30, 0.35, 0.40, 0.45])
    ap.add_argument('--grid', type=float, default=0.02)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--plot', action='store_true', help='另存 workspace.png（需要 matplotlib）')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    arm = Arm(args.tool_length)
    rng = np.random.default_rng(args.seed)
    xs = np.round(np.arange(-0.30, 0.30 + 1e-9, args.grid), 4)
    zs = np.round(np.arange(args.grid, 0.50 + 1e-9, args.grid), 4)
    cell_cm2 = (args.grid * 100) ** 2
    results, grids = {}, {}
    for name, (lo, hi) in scenarios().items():
        t0 = time.time(); rows = {}
        for D in args.walls:
            R = scan(arm, lo, hi, D, xs, zs, rng)
            grids[(name, D)] = R
            ext = None
            if R.any():
                ii, jj = np.nonzero(R)
                ext = {'x_m': [float(xs[jj.min()]), float(xs[jj.max()])],
                       'z_m': [float(zs[ii.min()]), float(zs[ii.max()])]}
            rows[f'{D:.2f}'] = {'area_cm2': round(float(R.sum() * cell_cm2), 1), 'extent': ext,
                                'touches_grid_edge': bool(R[:, 0].any() or R[:, -1].any() or R[-1].any())}
        best = max(rows, key=lambda k: rows[k]['area_cm2'])
        results[name] = {'lower_deg': lo, 'upper_deg': hi, 'best_wall_distance_m': best, 'by_wall_distance_m': rows}
        print(f'{name:30s} best D={best} m  area={rows[best]["area_cm2"]:7.1f} cm2  [{time.time() - t0:.0f}s]', flush=True)

    record = {
        'evidence_level': 'L1', 'hardware_motion': False,
        'model': 'models/dummy_reference.xml (reference geometry, not calibrated)',
        'tool_length_m': args.tool_length, 'grid_m': args.grid, 'seed': args.seed,
        'orientation_constraint': f'J6 axis within {ANG_TOL_DEG} deg of wall normal (trowel square to wall)',
        'position_tolerance_m': POS_TOL_M,
        'wall_distance_reference': 'distance from J1 axis along model -Y',
        'grid_x_m': [float(xs[0]), float(xs[-1])], 'grid_z_m': [float(zs[0]), float(zs[-1])],
        'results': results,
    }
    (args.out / f'wall_workspace_L{int(round(args.tool_length * 100))}cm.json').write_text(
        json.dumps(record, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')

    if args.plot:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        picks = [max(((k, D) for (k, D) in grids if k.startswith('S1')), key=lambda kd: grids[kd].sum()),
                 ('S2_sim_ctrlrange_pm40', float(results['S2_sim_ctrlrange_pm40']['best_wall_distance_m'])),
                 ('S3_geometry_only_pm90', float(results['S3_geometry_only_pm90']['best_wall_distance_m']))]
        titles = ['Current GUI debug envelope (best sign case)', 'Sim control range, all joints ±40°',
                  'Geometry only, all joints ±90° (hypothetical)']
        fig, ax = plt.subplots(1, 3, figsize=(13, 4.6), sharey=True)
        for a, (k, D), t in zip(ax, picks, titles):
            R = grids[(k, D)]
            h = args.grid * 50
            a.imshow(R, origin='lower', cmap='Oranges', vmin=0, vmax=1.4,
                     extent=[xs[0] * 100 - h, xs[-1] * 100 + h, zs[0] * 100 - h, zs[-1] * 100 + h])
            a.set_title(f'{t}\nwall {D * 100:.0f} cm from J1 axis: {R.sum() * cell_cm2:.0f} cm²', fontsize=10)
            a.set_xlabel('across the wall (cm)'); a.grid(alpha=.3)
            a.plot([0], [0], marker='^', color='k')
        ax[0].set_ylabel('height above base origin (cm)')
        fig.suptitle(f'Where a {args.tool_length * 100:.0f} cm trowel can press square onto a wall '
                     '(IK on grid, reference model, uncalibrated, L1)', fontsize=10)
        fig.tight_layout(); fig.savefig(args.out / 'workspace.png', dpi=110)


if __name__ == '__main__':
    main()

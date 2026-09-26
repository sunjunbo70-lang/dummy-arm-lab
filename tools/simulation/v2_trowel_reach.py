"""Dummy V2 + 刚性抹刀：墙面上真正够得到的区域，以及按用户规则选出的作业正方形。

用户规则（2026-09-23 修正）：
    在机械臂操作范围内找「最大的圆」，圆面积先留 80%（同心缩圆），再取缩小后圆的内接正方形
    作为评分区；未缩减那个圆的内接正方形作为仿真/物理区（越界不扣分，仍逐刀受可行性检查）。
    2026-09-22 的最初版本把 80% 直接乘在内接正方形边长上，不是这条规则，边长少留了约 11.8%
    （面积少留了约 25%），2026-09-23 修正。

做法（L1 软件分析，不连接硬件）：
  1. 墙是机械臂正前方的竖直平面（墙面系 u=+Y 水平、v=+Z 向上、n=+X 指向墙内），
     对若干墙距 D 各算一次。
  2. 墙面按网格取点。一个点算「可作业」，要同时满足：
       - 抹刀在 --pitches 给出的每个俯仰角都有逆解（刀长方向保持水平 psi=0，roll 弱约束，与 StrokeEnv 一致）。
         严格版 `--pitches 0 30`：刀面正对墙与「立起 30°」都要够得到；
         宽松版 `--pitches 0`：只要求刀面能正对墙（2026-09-22 起的默认作业区用这个，
         因为倾斜角改为由强化学习自己学，够不到的倾斜由执行器逐刀拒绝）；
       - 刀面正对墙、离墙 2.5 cm 的待命位也要有逆解（能贴上墙却退不出来的点不能作业）；
       - 关节在 V2 固件限位内（J6 收到 ±180°）；
       - 解出来的姿态下，除刀片外的连杆、电机、减速器离墙 ≥ 5 mm，所有部件（含刀片）离桌面 ≥ 5 mm
         （用 mj_geomDistance 几何距离判定：场景里臂的几何体 contype=0，普通碰撞检测不会报它们）。
  3. 在可作业区域里找最大内切圆（距离变换），圆心可以不在正中。
     r_scored = r·√0.8；评分区边长 = √2·r_scored；仿真/物理区边长 = √2·r（同一圆心）。
  4. 挑出作业正方形最大的墙距。

限位全部是 V2 固件候选值（M3 实测前未标定）；只检查运动学与连杆碰墙，不检查力矩。

  python tools/simulation/v2_trowel_reach.py --out experiments/v0.1/r0/runs/reach
"""
import argparse, json, time
from pathlib import Path
import sys

import numpy as np
import mujoco
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dummy_loop.wall.scene import SceneConfig, build_scene, wall_frame, tool_frame_matrix, joint_limits, blade_outline  # noqa: E402
from dummy_loop.wall.controller import ToolIK, WallFrame  # noqa: E402

PITCHES_DEG = (0.0, 30.0)
SAFETY = 0.8


class Reach:
    def __init__(self, distance, z_centre=0.25, pitches=PITCHES_DEG):
        self.pitches = tuple(pitches)
        self.cfg = SceneConfig(wall_distance=distance, wall_center_z=z_centre, wall_size=(1.2, 1.0))
        self.m, _ = build_scene(self.cfg)
        self.d = mujoco.MjData(self.m)
        lo, hi = joint_limits(self.cfg)
        lo, hi = np.array(lo, float), np.array(hi, float)
        lo[5], hi[5] = max(lo[5], -np.pi), min(hi[5], np.pi)
        self.ik = ToolIK(self.m, lo, hi, tool_frame_matrix(self.cfg), w_roll=0.1)
        self.frame = WallFrame(*wall_frame(self.cfg))
        wall = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, 'wall_geom')
        self.wall = wall
        # 刀颈焊在刀片背面、与刀片几何体重叠，属于抹刀本身，和刀片一起豁免
        names = ('blade_geom', 'blade_tip_geom', 'neck_geom')
        self.blade = {mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in names} - {-1}
        # 臂上的几何体在场景里只做显示（contype=0），碰撞检测不会报它们，
        # 所以这里直接算它们到墙面的几何距离（mj_geomDistance 不看 contype）。
        floor = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
        self.floor = floor
        self.arm = [g for g in range(self.m.ngeom) if g not in self.blade | {wall, floor}]
        # 底座本身就放在桌面上，不参与「碰桌面」检查
        base = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, 'base_link_visual')
        self.above_floor = [g for g in range(self.m.ngeom) if g not in {wall, floor, base}]
        o = blade_outline(self.cfg); self.hw = o['half_width']
        self.rng = np.random.default_rng(0)

    def collides(self, q, clearance=0.005):
        """碰撞判定（臂上几何体在场景里 contype=0，只能用几何距离判）：
        除刀片外的任何部件离墙面 < clearance，或任何部件（含刀片）离桌面 < clearance。
        第一版漏了桌面：墙距 33 cm 时「可达区域」一直延伸到桌面高度 z=0。"""
        self.d.qpos[:] = 0; self.d.qpos[:6] = q
        mujoco.mj_forward(self.m, self.d)
        for g in self.arm:
            if mujoco.mj_geomDistance(self.m, self.d, g, self.wall, 0.05, None) < clearance:
                return True
        for g in self.above_floor:
            if mujoco.mj_geomDistance(self.m, self.d, g, self.floor, 0.05, None) < clearance:
                return True
        return False

    def solve(self, u, v, seeds):
        """两个工作姿态都要解得出且不碰墙；返回俯仰 0° 的解（给下一个点作初值）。"""
        z_world_v = v - self.cfg.wall_center_z          # 墙面系 v 以 wall_center_z 为原点
        first = None
        # 工作姿态集合：每个俯仰角贴墙（刀面中心要离墙 半宽·sin(俯仰)，否则倾斜时后缘插进墙里——
        # 第一版的严格扫描犯了这个错），外加刀面正对墙、离墙 2.5 cm 的待命位（够得着却退不出来的点不能干活）
        poses = [(np.deg2rad(pt), -self.hw * np.sin(np.deg2rad(pt))) for pt in self.pitches] + [(0.0, -0.025)]
        for pitch, n in poses:
            p = self.frame.to_world([u, z_world_v, n])
            R = self.frame.tool_rotation(0.0, pitch)
            good = None
            for s in seeds + [self.ik.lo + (self.ik.hi - self.ik.lo) * self.rng.random(6) for _ in range(4)]:
                q, ok, _, _ = self.ik.solve(p, R, s, iters=120)
                if ok and not self.collides(q):
                    good = q; break
            if good is None:
                return None
            if first is None:
                first = good
            seeds = [good] + seeds
        return first


def scan(distance, step=0.01, u_half=0.45, v_range=(0.0, 0.60), passes=4, pitches=PITCHES_DEG):
    """逐行扫描（用相邻点的解作初值），再做几轮补洞：数值逆解偶尔会漏掉可达点，
    对每个未解出、但邻格已解出的点，用邻格的解重新求一次，直到不再变化。"""
    r = Reach(distance, pitches=pitches)
    us = np.arange(-u_half, u_half + 1e-9, step)
    vs = np.arange(v_range[0], v_range[1] + 1e-9, step)
    ok = np.zeros((len(vs), len(us)), bool)
    Q = {}
    prev_row = [None] * len(us)
    for i, v in enumerate(vs):
        left = None
        for j, u in enumerate(us):
            seeds = [s for s in (left, prev_row[j], np.zeros(6)) if s is not None]
            q = r.solve(u, v, seeds)
            ok[i, j] = q is not None
            if q is not None:
                Q[i, j] = q
            left = q; prev_row[j] = q if q is not None else prev_row[j]
    for _ in range(passes):
        changed = 0
        for i in range(len(vs)):
            for j in range(len(us)):
                if ok[i, j]:
                    continue
                seeds = [Q[k] for k in ((i-1, j), (i+1, j), (i, j-1), (i, j+1),
                                        (i-1, j-1), (i+1, j+1), (i-1, j+1), (i+1, j-1)) if k in Q]
                if not seeds:
                    continue
                q = r.solve(us[j], vs[i], seeds)
                if q is not None:
                    ok[i, j] = True; Q[i, j] = q; changed += 1
        if not changed:
            break
    return us, vs, ok


def work_square(us, vs, ok, area_safety=SAFETY):
    """最大内切圆 → 面积先按 area_safety 同心缩圆 → 内接正方形 = 评分区（score_square）；
    未缩减的内接正方形 = 仿真/物理区（sim_square，越界不扣分，仍逐刀受 IK/碰撞检查约束）。

    2026-09-23 修正：此前把 safety 直接乘在内接正方形的边长上（side = safety * side_inscribed），
    不是用户设计意图的"圆面积先留 area_safety，同心缩圆，再在缩小后的圆里取最大内接正方形"
    （r_scored = r * sqrt(area_safety)），边长系数应是 sqrt(area_safety) 而不是 area_safety 本身，
    面积相差约 1/area_safety 倍（area_safety=0.8 时评分区面积比旧算法多约 25%）。"""
    step = float(us[1] - us[0])
    # 可达区域外一圈当作不可达，距离变换给出每个可达格到最近不可达格的距离
    padded = np.pad(ok, 1, constant_values=False)
    dist = ndimage.distance_transform_edt(padded)[1:-1, 1:-1] * step
    i, j = np.unravel_index(int(np.argmax(dist)), dist.shape)
    r = float(dist[i, j]) - step / 2          # 格心到边界：保守减半格
    r_scored = r * np.sqrt(area_safety)
    side_scored = np.sqrt(2) * r_scored        # 缩圆后的内接正方形 = 评分区边长
    side_sim = np.sqrt(2) * r                    # 原始（未缩减）内接正方形 = 仿真/物理区边长
    cu, cv = float(us[j]), float(vs[i])
    return {'circle_centre_uv_m': [round(cu, 4), round(cv, 4)], 'circle_radius_m': round(r, 4),
            'area_safety_factor': area_safety,
            'score_square_side_m': round(side_scored, 4),
            'sim_square_side_m': round(side_sim, 4),
            'score_square_u_m': [round(cu - side_scored / 2, 4), round(cu + side_scored / 2, 4)],
            'score_square_v_m': [round(cv - side_scored / 2, 4), round(cv + side_scored / 2, 4)],
            'sim_square_u_m': [round(cu - side_sim / 2, 4), round(cu + side_sim / 2, 4)],
            'sim_square_v_m': [round(cv - side_sim / 2, 4), round(cv + side_sim / 2, 4)],
            'score_area_cm2': round(side_scored * side_scored * 1e4, 1),
            'sim_area_cm2': round(side_sim * side_sim * 1e4, 1),
            'reachable_area_cm2': round(float(ok.sum()) * step * step * 1e4, 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, default=Path('experiments/v0.1/r0/runs/reach'))
    ap.add_argument('--distances', type=float, nargs='+', default=[0.25, 0.30, 0.35, 0.40, 0.45])
    ap.add_argument('--step', type=float, default=0.01)
    ap.add_argument('--pitches', type=float, nargs='+', default=list(PITCHES_DEG),
                    help='working pitches every point must reach (deg). 0 30 = strict (default); 0 = loose')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for D in a.distances:
        t = time.time()
        us, vs, ok = scan(D, a.step, pitches=a.pitches)
        sq = work_square(us, vs, ok)
        rows.append({'wall_distance_m': D, **sq, 'elapsed_s': round(time.time() - t, 1)})
        np.savez(a.out / f'reach_D{int(round(D * 100))}.npz', u=us, v=vs, ok=ok)
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    best = max(rows, key=lambda r: r['score_square_side_m'])
    result = {'evidence_level': 'L1', 'hardware_motion': False,
              'rule': ('largest circle inside the reachable set -> reduce AREA by area_safety, '
                       'same centre -> inscribed square = scored work area; the un-reduced '
                       'circle\'s own inscribed square = simulation/physical work area'),
              'poses': {'pitch_deg': list(a.pitches), 'psi_deg': 0, 'roll': 'weak (w_roll=0.1)'},
              'joint_limits': 'V2 firmware candidates, J6 clipped to +/-180 deg; NOT calibrated (M3)',
              'collision': 'arm geoms other than the blade >= 5 mm from the wall; every geom incl. blade >= 5 mm above the table',
              'standoff_pose': 'face-on, 2.5 cm off the wall, must also be solvable',
              'v_origin': 'v is height above the floor plane of the base (world z)',
              'rows': rows, 'best': best}
    (a.out / 'reach.json').write_text(json.dumps(result, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    print('best', json.dumps(best, ensure_ascii=False))


if __name__ == '__main__':
    main()

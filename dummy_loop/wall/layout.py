"""工位布局搜索：墙放多远、工作区多高、刀面倾角多少，才能在整片工作区内「好干活」。

「能够到」不等于「好干活」：靠近墙时可达面积大，但很多点只能靠腕部大幅扭转、
贴着关节限位或在奇异附近才够得到。这里对每个候选布局，在工作区网格上沿路径连续求逆解，
取最差点的三项指标：
  sigma_min        位置+法线 5 维任务雅可比的最小奇异值（越大越远离奇异；单位 m/rad）
  limit_margin_deg 离最近关节限位的角度
  max_twist_deg    J1/J4/J6 的最大绝对值（腕部扭转，实机上意味着线缆缠绕）
L1 软件分析。关节范围用 V2 固件限位（候选值），M3 实测限位后应重跑。
"""
import itertools
import numpy as np
import mujoco
from .scene import SceneConfig, build_scene, wall_frame, tool_frame_matrix, joint_limits
from .controller import EEController, WallFrame

NATURAL_PLANE_U = 0.0     # V2 固件 DH 各连杆共面（无横向偏置），工作区以 u=0 为中心


def conditioning(ctrl, q):
    p, R = ctrl.ik.fk(q)
    mujoco.mj_jacSite(ctrl.ik.m, ctrl.ik.d, ctrl.ik.jp, ctrl.ik.jr, ctrl.ik.site)
    n = R[:, 1]
    B = np.linalg.svd(np.eye(3) - np.outer(n, n))[0][:, :2]
    J = np.r_[ctrl.ik.jp[:, :6], 0.1 * B.T @ ctrl.ik.jr[:, :6]]
    return float(np.linalg.svd(J, compute_uv=False)[-1])


def evaluate(cfg: SceneConfig, half=(0.06, 0.05), grid=5):
    lo, hi = joint_limits(cfg)
    m, _ = build_scene(cfg)
    c = EEController(m, WallFrame(*wall_frame(cfg)), lo, hi, tool_R=tool_frame_matrix(cfg))
    uc = NATURAL_PLANE_U
    try:
        c.reset([uc, 0, 0], 0, np.zeros(6))
    except ValueError:
        return {'feasible': False, 'reason': 'centre unreachable'}
    smin, margin, twist = np.inf, np.inf, 0.0
    us = np.linspace(uc - half[0], uc + half[0], grid); vs = np.linspace(-half[1], half[1], grid)
    for i, v in enumerate(vs):
        for u in (us if i % 2 == 0 else us[::-1]):
            tgt = np.array([u, v, 0.])
            for _ in range(80):
                d = tgt - c.target
                if np.linalg.norm(d) < 1e-6:
                    break
                _, st, _ = c.step(np.r_[np.clip(d, -0.005, 0.005), 0])
                if st == 'unreachable':
                    return {'feasible': False, 'reason': f'unreachable at u={u:.3f} v={v:.3f}'}
            for _ in range(40):
                _, st, _ = c.step([0, 0, 0, 0])
                if st == 'ok':
                    break
            q = c.q_cmd
            smin = min(smin, conditioning(c, q))
            margin = min(margin, float(np.degrees(np.minimum(q - lo, hi - q)).min()))
            twist = max(twist, float(np.degrees(np.abs(q[[0, 3, 5]])).max()))
    return {'feasible': True, 'sigma_min': round(smin, 4), 'limit_margin_deg': round(margin, 1),
            'max_twist_deg': round(twist, 1)}


def search(tilts=(-45, -30, 0, 30, 45, 60), distances=(0.25, 0.30, 0.35, 0.40),
           heights=(0.15, 0.20, 0.25, 0.30), joint_limit_cap_deg=None, half=(0.06, 0.05)):
    rows = []
    for tilt, D, z in itertools.product(tilts, distances, heights):
        cfg = SceneConfig(joint_limit_cap_deg=joint_limit_cap_deg, blade_tilt_deg=tilt, wall_distance=D, wall_center_z=z)
        r = evaluate(cfg, half)
        rows.append({'blade_tilt_deg': tilt, 'wall_distance_m': D, 'work_centre_z_m': z, **r})
    feas = [r for r in rows if r['feasible']]
    # 综合分：最小奇异值，离限位不足 15° 时按比例打折
    for r in feas:
        r['score'] = round(r['sigma_min'] * min(1.0, r['limit_margin_deg'] / 15.0), 5)
    feas.sort(key=lambda r: r['score'], reverse=True)
    return {'joint_limits': 'V2 firmware' + ('' if joint_limit_cap_deg is None else f' capped at +/-{joint_limit_cap_deg} deg'), 'patch_half_size_m': list(half),
            'n_candidates': len(rows), 'n_feasible': len(feas), 'ranked': feas,
            'infeasible': [r for r in rows if not r['feasible']]}

"""静力矩对比：夹持 1 kg 砝码 vs 抹刀压墙，各关节需要多大力矩。

软件分析（L1），不连接硬件，不发送任何指令。

方法：关闭模型里的理想重力补偿（gravcomp），在给定姿态下用雅可比转置
tau = -J^T f 计算抵抗外力所需的关节力矩，再加上手臂自重力矩。只算静态，
不含加减速带来的惯性力矩。

默认用 V2 模型 models/dummy_v2.xml（质量 = CAD 体积×估计密度 + 资料质量，未称重），
并与 V2 各关节的额定 / 启停峰值转矩（dummy_loop/v2.py）对比，给出裕量比。
--model reference 可复现 2026-09-20 用参考模型得到的旧结果。结果仍是估计值。

J6 部分：抹刀装在 J6 轴上时，J6 要承受的扭矩 = 沿墙拖曳力 × 刀刃接触中心偏离
J6 轴线的距离。工具固定在 J6 外壳上时这部分为 0（J6 不在受力路径上），
J5 及以前各轴的负担不变。

用法：
  python tools/simulation/static_joint_torque.py --out experiments/<日期>_static_torque/raw
  （工具长度从 V2 裸轴端面量起；参考模型从法兰量起）
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
MODELS = {'v2': (ROOT / 'models' / 'dummy_v2.xml', 'link6'),
          'reference': (ROOT / 'models' / 'dummy_reference.xml', 'link6_1_1')}
G = 9.81
POSES = {
    'zero_L_pose_forearm_horizontal': [0, 0, 0, 0, 0, 0],
    'elbow_bent': [0, 0.5, -0.9, 0, 0.4, 0],
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', choices=sorted(MODELS), default='v2')
    ap.add_argument('--tool-length', type=float, default=0.20, help='安装面（V2 裸轴端面 / 参考模型法兰）到刀刃的距离 (m)')
    ap.add_argument('--tool-mass', type=float, default=0.15, help='抹刀+转接件质量 (kg)，质心取刀长一半')
    ap.add_argument('--payload-offset', type=float, default=0.05, help='夹持物重心离法兰的距离 (m)')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    path, link = MODELS[args.model]
    m = mujoco.MjModel.from_xml_path(str(path)); d = mujoco.MjData(m)
    m.body_gravcomp[:] = 0
    fl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, link)
    tip = m.site('shaft_tip').pos.copy() if args.model == 'v2' else np.zeros(3)
    j5 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'Joint5')
    j6 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'Joint6')
    up = np.array([0, 0, 1.])

    def need(point, force):
        jp = np.zeros((3, m.nv)); mujoco.mj_jac(m, d, jp, None, point, fl)
        return -(jp.T @ force)[:6]

    out = {'evidence_level': 'L1', 'hardware_motion': False,
           'model': f'{path.relative_to(ROOT).as_posix()}, gravcomp disabled, masses are estimates (not weighed)',
           'units': 'N*m, absolute value per joint J1..J6', 'static_only': True,
           'tool_length_m': args.tool_length, 'tool_mass_kg': args.tool_mass,
           'payload_offset_m': args.payload_offset, 'poses': {}}
    for pname, q in POSES.items():
        d.qpos[:6] = q; d.qvel[:] = 0; mujoco.mj_forward(m, d)
        p = d.xpos[fl] + d.xmat[fl].reshape(3, 3) @ tip; a = d.xaxis[j6].copy()
        if a @ (p - d.xanchor[j5]) < 0:
            a = -a
        arm = d.qfrc_bias[:6].copy()
        rows = {'arm_self_weight': np.abs(arm).round(3).tolist(),
                'hold_1kg': np.abs(arm + need(p + args.payload_offset * a, np.array([0, 0, -G]))).round(3).tolist()}
        for Fn in (5, 10, 20):
            for mu in (0.5, 1.0):
                worst = np.zeros(6)
                for s in (1, -1):                      # 向上抹 / 向下抹，拖曳方向相反
                    f = -Fn * a + s * mu * Fn * up
                    tau = (arm + need(p + 0.5 * args.tool_length * a, np.array([0, 0, -args.tool_mass * G]))
                           + need(p + args.tool_length * a, f))
                    worst = np.maximum(worst, np.abs(tau))
                rows[f'trowel_press_{Fn}N_drag_{mu:.1f}x'] = worst.round(3).tolist()
        out['poses'][pname] = {'q_rad': q, 'mount_face_pos_m': p.round(4).tolist(), 'tool_axis': a.round(3).tolist(),
                               'torques': rows}
        print(f'\n{pname}   J1..J6 (N*m)')
        for k, v in rows.items():
            print(f'  {k:32s} ' + ' '.join(f'{x:6.2f}' for x in v))

    if args.model == 'v2':
        from dummy_loop import v2
        jp = v2.joint_params()
        cont = np.array([j['continuous_Nm'] for j in jp]); peak = np.array([j['peak_Nm'] for j in jp])
        out['v2_ratings_Nm'] = {'continuous': cont.tolist(), 'peak': peak.tolist(),
                                'source': 'dummy_loop/v2.py (reducer datasheet; motor catalog values)'}
        print('\nV2 ratings  continuous ' + ' '.join(f'{x:6.2f}' for x in cont))
        print('            peak       ' + ' '.join(f'{x:6.2f}' for x in peak))
        for pname, pose in out['poses'].items():
            pose['load_over_continuous'] = {k: (np.array(v) / cont).round(2).tolist() for k, v in pose['torques'].items()}
            worst = {k: float(np.max(np.array(v)[:5] / cont[:5])) for k, v in pose['torques'].items()}
            print(f'{pname}: worst J1-J5 load / continuous rating: ' + ', '.join(f'{k}={x:.2f}' for k, x in worst.items()))
    j6 = {}
    for drag in (5, 10):
        j6[f'drag_{drag}N'] = {f'offset_{int(o * 1000)}mm': round(drag * o, 3) for o in (0, 0.005, 0.01, 0.02, 0.05)}
    out['j6_torque_if_mounted_on_shaft'] = {
        'formula': 'tau_J6 = drag_force * lateral offset of blade contact centre from J6 axis',
        'values_Nm': j6,
        'if_mounted_on_J6_housing': 'J6 carries 0; J1..J5 loads unchanged'}
    print('\nJ6 torque if trowel is on the J6 shaft (N*m):')
    for k, v in j6.items():
        print(f'  {k}: ' + '  '.join(f'{kk}={vv}' for kk, vv in v.items()))
    (args.out / 'static_torque.json').write_text(json.dumps(out, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()

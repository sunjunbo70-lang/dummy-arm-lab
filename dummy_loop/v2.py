"""Dummy V2 的运动学与传动参数（单一来源）。

用户实机是 Dummy V2，末端是 J6 电机的裸轴（没有法兰）。这里集中放 V2 的：
  - 固件 DH 参数与正运动学（逐行移植自固件 DOF6Kinematic::SolveFK）；
  - 固件角度 <-> 仿真模型关节角 的换算；
  - V2 固件限位、减速比、方向；
  - 各关节电机 / 减速器 / 皮带配置与额定值。

证据等级：全部是**资料值**（L1），来自 V2 资料包（见 THIRD_PARTY.md「Dummy V2 资料包」）：
  - DH 与限位/减速比：`6.Docs/DH参数/dummy V2 版本 DH参数.png`（V2 固件参数截图）；
  - 传动配置：`4.Model/.../Dummy-B v2装配体.step` 的装配结构；
  - 减速器额定值：`6.Docs/减速器参数/mini11_parameters.JPG`、`mini8_parameters.JPG`；
  - 电机：BOM 只给了外形规格；质量、转子惯量、保持转矩是同规格步进电机的**典型目录值**。
这些都**不是**对本台实机的测量。M3 实测前一律视为候选值，不写入 configs/ 的已标定字段。

坐标约定（模型坐标 = 固件坐标）：
  世界系原点在 J1 轴线上（固件基座系），+X 朝前，+Z 向上，+Y 向左。
  模型关节零位 = 固件 HOME (0, 0, 90, 0, 0, 0)，即「L 形」姿态：大臂竖直，小臂水平朝前。
  模型关节角 q(rad) = deg2rad(固件角 - HOME)，正方向与固件相同。
"""
import numpy as np

# ── 固件 DH（V2 截图：DOF6Kinematic(0.1265, 0.035, 0.146, 0.117, 0.052, 0.0755)）──────────
DH_V2 = dict(L_BS=0.1265, D_BS=0.035, L_AM=0.146, L_FA=0.117, D_EW=0.052, L_WT=0.0755)
# 原版（稚晖君仓库 dummy_robot.cpp）参数，仅用于对照，不用于 V2 模型
DH_ORIGINAL = dict(L_BS=0.109, D_BS=0.035, L_AM=0.146, L_FA=0.115, D_EW=0.052, L_WT=0.072)

HOME_DEG = np.array([0., 0., 90., 0., 0., 0.])
FOLD_DEG = np.array([0., -75., 180., 0., 0., 0.])     # 收纳姿态；V2 装配体 STEP 就画在这个姿态

# V2 固件限位（度，固件角）。2026-09-16 预设复核中 J2 到达 -74.9997，与 V2 下限 -75 相符，
# 支持「实机刷的是 V2 参数」这一推断，但不是证明。M3 逐轴实测后才能当作已标定限位。
FIRMWARE_LOWER_DEG_V2 = np.array([-170., -75., 0., -180., -100., -720.])
FIRMWARE_UPPER_DEG_V2 = np.array([170., 90., 180., 180., 120., 720.])
FIRMWARE_REDUCTION_V2 = np.array([50, 50, 50, 50, 50, 5])
FIRMWARE_INVERSE_V2 = (False, True, True, True, True, True)

# ── 传动配置 ────────────────────────────────────────────────────────────────
# 减速器额定值（N·m，减速比 50，输入 2000 r/min 时额定），来源见模块说明。
REDUCERS = {
    'MINI11-50': dict(ratio=50, rated_Nm=3.5, start_stop_peak_Nm=8.3, avg_max_Nm=5.5, momentary_max_Nm=17.0,
                      mass_kg=0.40, mass_source='datasheet drawing'),
    'MINI8-50': dict(ratio=50, rated_Nm=1.8, start_stop_peak_Nm=3.3, avg_max_Nm=2.3, momentary_max_Nm=6.6,
                     mass_kg=0.20, mass_source='ESTIMATE: datasheet weight field blank'),
}
# 步进电机典型目录值（同规格常见型号），不是本机铭牌数据。
MOTORS = {
    '42-40': dict(mass_kg=0.28, rotor_inertia_kgm2=54e-7, holding_Nm=0.40),
    '42-34': dict(mass_kg=0.22, rotor_inertia_kgm2=34e-7, holding_Nm=0.28),
    '35-28': dict(mass_kg=0.14, rotor_inertia_kgm2=11e-7, holding_Nm=0.12),
    '35-20': dict(mass_kg=0.10, rotor_inertia_kgm2=6e-7, holding_Nm=0.07),
}
# 每个关节：电机、皮带、减速器（按 V2 装配体 STEP 的装配结构读出）。
DRIVETRAIN_V2 = [
    dict(joint='J1', motor='42-34', belt=None, reducer='MINI11-50'),
    dict(joint='J2', motor='42-40', belt='GT2 20T:20T', reducer='MINI11-50'),
    dict(joint='J3', motor='42-34', belt='GT2 20T:20T', reducer='MINI11-50'),
    dict(joint='J4', motor='35-28', belt=None, reducer='MINI11-50'),
    dict(joint='J5', motor='35-20', belt='GT2 20T:20T', reducer='MINI8-50'),
    dict(joint='J6', motor='35-20', belt=None, reducer=None),     # 直驱，输出就是电机裸轴
]
# 已知的资料冲突（M3 需要在实机上确认）
CONFLICTS = [
    'J6 减速比：V2 固件参数截图写 5，但装配体里 J6 是 35-20 电机直驱、没有减速器。'
    '若固件按 5 换算而实际直驱，J6 实际转角可能与指令角相差 5 倍。M3 先用 ≤5° 小角度测 J6。',
    '电机长度：BOM 写 3 个 42-40、2 个 35-28、1 个 35-20（J6）；装配体里是 J1/J3 为 42-34、J2 为 42-40，'
    'J4 为 35-28、J5/J6 为 35-20。以实机铭牌为准。',
    '资料包里的固件源码 dummy_robot.cpp 仍是原版参数（减速比 30、DH 0.109/0.115/0.072），与 V2 截图不同。',
]


def joint_params():
    """逐关节汇总：减速比、输出端可用转矩、反射惯量（电机转子 × 减速比²）。"""
    rows = []
    for d in DRIVETRAIN_V2:
        m = MOTORS[d['motor']]
        red = REDUCERS.get(d['reducer']) if d['reducer'] else None
        ratio = red['ratio'] if red else 1
        motor_side = m['holding_Nm'] * ratio
        if red:
            peak = min(red['start_stop_peak_Nm'], motor_side)
            cont = min(red['rated_Nm'], motor_side)
        else:
            peak = cont = m['holding_Nm']
        rows.append(dict(**d, ratio=ratio, motor_holding_at_output_Nm=round(motor_side, 3),
                         peak_Nm=round(peak, 3), continuous_Nm=round(cont, 3),
                         reflected_inertia_kgm2=float(m['rotor_inertia_kgm2'] * ratio ** 2)))
    return rows


def firmware_to_model(q_deg):
    """固件角（度）-> 模型关节角（rad）。"""
    return np.deg2rad(np.asarray(q_deg, float) - HOME_DEG)


def model_to_firmware(q_rad):
    return np.rad2deg(np.asarray(q_rad, float)) + HOME_DEG


def model_limits_rad():
    return firmware_to_model(FIRMWARE_LOWER_DEG_V2), firmware_to_model(FIRMWARE_UPPER_DEG_V2)


def _dh(c):
    return np.array([[0.0, c['L_BS'], c['D_BS'], -np.pi / 2],
                     [-np.pi / 2, 0.0, c['L_AM'], 0.0],
                     [np.pi / 2, c['D_EW'], 0.0, np.pi / 2],
                     [0.0, c['L_FA'], 0.0, -np.pi / 2],
                     [0.0, 0.0, 0.0, np.pi / 2],
                     [0.0, c['L_WT'], 0.0, 0.0]])


def firmware_fk(q_deg, c=DH_V2):
    """固件 SolveFK 的逐行移植。输入固件角（度），返回各特征点与各级旋转。

    返回 dict：R（R00..R06，每个 3x3）、shoulder、elbow、wrist、end（m，固件基座系）。
    end 是固件的末端点：腕心沿 J6 轴外伸 L_WT。
    """
    D = _dh(c); q = np.deg2rad(np.asarray(q_deg, float)) + D[:, 0]
    R = [np.eye(3)]
    for i in range(6):
        cq, sq, ca, sa = np.cos(q[i]), np.sin(q[i]), np.cos(D[i, 3]), np.sin(D[i, 3])
        R.append(R[-1] @ np.array([[cq, -ca * sq, sa * sq], [sq, ca * cq, -sa * cq], [0, sa, ca]]))
    p_sh = R[1] @ [c['D_BS'], -c['L_BS'], 0]
    p_el = p_sh + R[2] @ [c['L_AM'], 0, 0]
    p_wr = p_el + R[3] @ [-c['D_EW'], 0, c['L_FA']]
    p_end = p_wr + R[6] @ [0, 0, c['L_WT']]
    return dict(R=R, shoulder=p_sh, elbow=p_el, wrist=p_wr, end=p_end,
                j4_point=p_el + R[3] @ [-c['D_EW'], 0, 0])

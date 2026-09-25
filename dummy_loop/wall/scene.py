"""墙面抹涂场景：Dummy V2 机械臂 + J6 减速器 + 3D 打印抹刀座 + 日式尖头抹刀 + 墙 + 固定相机。

实际工具（默认 tool_mount='rigid'）：在原「J6 减速器 + 夹爪电机 + 夹爪」方案上去掉夹爪电机和夹爪，
减速器输出法兰上装 3D 打印抹刀座，卡住抹刀木柄。整体刚性，没有伸缩或回弹。
早期的弹簧滑轨方案保留为 tool_mount='spring'，仅用于对比。

用 MjSpec 在 models/dummy_v2.xml 之上程序化组装，不修改原文件。
所有尺寸都是参数；哪些是假设、从哪里来，写进 SceneConfig.provenance()，
随每条 episode 一起保存。

坐标约定（模型坐标 = 固件基座系，见 docs/ACTION_SPACE.md、docs/hardware/DUMMY_V2.md）：
  世界系原点在 J1 轴线上，+X 朝前，+Z 向上。模型零位 = 固件 HOME（L 形姿态），
  此时小臂水平朝前，J6 轴沿 +X。
  墙位于机械臂正前方，墙面法线 n 指向墙内（+X），u 沿墙水平（+Y），v 向上（+Z）。
  末端是 J6 电机裸轴：工具链从轴端（link6 局部 x = 轴端距腕心）沿 J6 轴向外伸出。
"""
from dataclasses import dataclass, field, asdict
from pathlib import Path
import hashlib, json
import numpy as np
import mujoco

from .. import v2

ROOT = Path(__file__).resolve().parents[2]
ARM_MODEL = ROOT / 'models' / 'dummy_v2.xml'
LINK6 = 'link6'
# link6 -> 工具安装系：安装系 +Y = J6 轴外伸方向（link6 +X），安装系 +X = 刀宽方向（零位时指向 -u = -Y）
MOUNT_R = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])
TOOL_GROUP, WALL_GROUP = 2, 2  # 接触位掩码：只让刀面与墙发生接触


def housing_front_x():
    """J6 电机座前端面在 link5/link6 局部 x 上的位置（m），取自 models/dummy_v2_params.json（由 CAD 量得）。"""
    p = json.loads((ARM_MODEL.parent / 'dummy_v2_params.json').read_text(encoding='utf-8'))
    return p['end_effector']['housing_front_from_wrist_mm'] / 1000.0


def shaft_tip_x(spec_or_model=None):
    """J6 裸轴端面在 link6 局部 x 上的位置（m），取自模型里的 shaft_tip 站点（由 CAD 量得）。"""
    m = mujoco.MjModel.from_xml_path(str(ARM_MODEL)) if spec_or_model is None else spec_or_model
    return float(m.site('shaft_tip').pos[0])


@dataclass
class SceneConfig:
    # 关节范围：默认用 V2 固件限位（换算到模型坐标）。joint_limit_cap_deg 可再加一层
    # 对称的保守包络（例如 M3 之前只允许 ±60°）；None = 不加。
    joint_limit_cap_deg: float = None
    servo_kp: float = 35.0            # 示例值（未辨识）
    servo_kv: float = 4.0
    servo_force_limit: float = None   # None = 用 V2 模型逐轴值（减速器启停峰值 / 电机保持转矩×减速比）
    # 末端工具安装方式：
    #   'rigid'  实际方案（默认）：J6 电机座前端装 J6 减速器（原夹爪方案里的那一节，去掉夹爪电机和夹爪），
    #            减速器输出法兰 → 3D 打印抹刀座（卡住木柄）→ 日式尖头抹刀（木柄与刀面平行，中间一根立柱）。
    #            全刚性，无伸缩。压力由机械臂自身的位置伺服刚度决定；接触检测与压力闭环靠法兰与抹刀座之间的力传感器。
    #   'spring' 早期设计：裸轴转接件 + 弹簧滑轨 + 矩形刀面，保留用于对比。
    tool_mount: str = 'rigid'
    geometry_profile: str = 'legacy'  # lab_20260922 selects measured tool geometry
    lab_geometry: dict = field(default_factory=dict)  # serialized with every scene
    # J6 减速器（照片估计：两块 42 mm 方板夹一段 Φ32 圆柱，总长约 40 mm）。固定在 J6 电机座（link5）上，
    # 只有输出法兰随 J6 转。型号与减速比未确认：j6_reducer_ratio=None 时 J6 仍按 V2 模型的直驱参数。
    j6_reducer_length: float = 0.040  # 电机座前端面 → 输出法兰面
    j6_reducer_plate: float = 0.042   # 方板边长
    j6_reducer_plate_thickness: float = 0.008
    j6_reducer_radius: float = 0.016
    j6_reducer_mass: float = 0.15
    # 减速比未确认（照片上只看得出有这么一节）。给一个候选值 30：J6 输出转矩 = 电机保持转矩 × 减速比，
    # 反射惯量 = 转子惯量 × 减速比²。None = 按 V2 模型的直驱参数（J6 几乎没有力矩，连自身阻尼都拧不动）。
    j6_reducer_ratio: float = 30.0
    j6_reducer_max_Nm: float = 2.5    # 这一节减速器的允许输出转矩上限（假设值）
    # 3D 打印抹刀座：从法兰面到木柄外表面的厚度；夹持位置用「离刀尖的距离」表示
    holder_height: float = 0.012
    holder_mass: float = 0.03
    clamp_from_tip: float = None      # None = 夹在刀面面积形心正上方（J6 轴线穿过刀面形心，拖曳力对 J6 不产生扭矩）
    # 抹刀（照片中的日式尖头抹刀；用户确认实物约为最初估计 240 mm 规格的一半，按 1/2 缩放，量实物后替换）
    trowel_length: float = 0.120      # 刀面总长（刀尖 → 方头）
    trowel_width: float = 0.0275      # 刀面宽
    trowel_tip_length: float = 0.030  # 尖头收窄段长度
    trowel_thickness: float = 0.001
    neck_from_tip: float = 0.0785     # 立柱位置（离刀尖）
    neck_height: float = 0.012        # 刀面背面 → 木柄底面
    handle_from_tip: tuple = (0.0625, 0.1185)   # 木柄沿刀长方向的起止（离刀尖）
    handle_radius: float = 0.0075
    tool_mass: float = 0.06           # 抹刀整体（刚性方案）；弹簧方案也用它作转接件+滑轨+抹刀总质量
    # 弹簧方案用的转接件与矩形刀面
    adapter_length: float = 0.030     # 裸轴端面到滑轨起点
    adapter_radius: float = 0.012
    tool_length: float = 0.10         # 弹簧方案：裸轴端面到刀面（弹簧未压缩时）。刚性方案由上面尺寸算出，见 rigid_tool_length()
    blade_tilt_deg: float = 0.0       # 刀面相对 J6 轴的倾角（绕刀宽方向）。0 = 刀面垂直于 J6 轴
    blade_width: float = 0.08         # 弹簧方案矩形刀面：沿墙水平方向（psi=0 时）
    blade_height: float = 0.05
    blade_thickness: float = 0.003
    # 被动柔顺：弹簧滑轨（仅 tool_mount='spring'）
    spring_k: float = 400.0           # N/m
    spring_preload: float = 2.0       # N
    spring_travel: float = 0.030      # m
    spring_damping: float = 8.0       # N·s/m
    blade_friction: float = 0.6
    # 墙（名义位姿；误差模型会改动真实位姿，控制器只知道名义值）
    wall_distance: float = 0.40       # 墙面到 J1 轴的距离，沿 +X（由 layout 搜索选定，见 provenance）
    wall_yaw_deg: float = 0.0         # 绕竖直轴
    wall_pitch_deg: float = 0.0       # 绕墙面水平轴（墙面前后倾）
    wall_lateral: float = 0.0         # 工作区中心的横向位置（+Y）
    wall_center_z: float = 0.15       # 工作区中心高度（同上；V2 模型重跑布局后由 0.20 改为 0.15）
    wall_size: tuple = (0.80, 0.60)   # 宽, 高
    # 固定相机（D435 类，位置为占位）
    camera_pos: tuple = (-0.05, 0.35, 0.45)
    camera_fovy_deg: float = 58.0
    gravcomp: bool = True             # 理想重力补偿（实机闭环步进位置模式下的近似）

    def provenance(self):
        if self.geometry_profile == 'lab_20260922':
            from .lab_tool import provenance
            return provenance(self)
        return {
            'arm': 'models/dummy_v2.xml (V2 firmware DH kinematics; CAD-derived mass estimates; not identified)',
            'joint_limits': ('V2 firmware limits converted to model coordinates (dummy_loop/v2.py); '
                             'CANDIDATE until M3 measures them' +
                             ('' if self.joint_limit_cap_deg is None else f'; additionally capped at +/-{self.joint_limit_cap_deg} deg')),
            'servo_kp_kv': 'example values, not identified',
            'servo_force': ('per-joint from V2 drivetrain documents (reducer start/stop peak or motor holding x ratio)'
                            if self.servo_force_limit is None else f'uniform override {self.servo_force_limit} N*m'),
            'tool_dimensions': 'PLACEHOLDER: no trowel measured yet',
            'tool_mount': ('RIGID: J6 reducer on the J6 motor housing (gripper motor and gripper removed) -> output flange '
                           '-> 3D-printed holder clamping the wooden handle -> Japanese pointed plastering trowel. '
                           'Dimensions estimated from photos; reducer type/ratio UNCONFIRMED'
                           if self.tool_mount == 'rigid' else
                           'SPRING (earlier design, kept for comparison): adapter on bare shaft + spring slide'),
            'j6_reducer': ('UNCONFIRMED type/ratio; model assumes ratio %s and %.2f N*m output limit'
                           % (self.j6_reducer_ratio, min(0.07 * self.j6_reducer_ratio, self.j6_reducer_max_Nm))
                           if self.j6_reducer_ratio else 'modelled as direct drive (no reducer)'),
            'tool_sensor': ('load cell between flange and holder (hardware_with_sensor); in sim = blade-wall normal force + noise'
                            if self.tool_mount == 'rigid' else 'spring compression (linear pot / hall sensor)'),
            'wall_pose': ('nominal. Chosen by dummy_loop.wall.layout search on the V2 model '
                          '(see experiments/00_initial_debug/records/2026-09-20_v2_model). Re-run after M3.'),
            'camera_pose': 'PLACEHOLDER: fixed D435 not yet mounted (M6)',
        }

    def to_dict(self):
        d = asdict(self)
        d['wall_size'] = list(self.wall_size); d['camera_pos'] = list(self.camera_pos)
        d['handle_from_tip'] = list(self.handle_from_tip)
        return d

    def digest(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:12]


def _quat_from_axes(x, y, z):
    R = np.column_stack([x, y, z]); q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.flatten()); return q


def tool_tilt_matrix(cfg: SceneConfig):
    """刀面坐标系相对 link6 的旋转：绕 link6 局部 x 轴转 blade_tilt_deg。"""
    a = np.deg2rad(cfg.blade_tilt_deg)
    return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])


def tool_frame_matrix(cfg: SceneConfig):
    """刀面坐标系相对 link6 的旋转（控制器用）：安装系旋转 × 刀面倾角。"""
    return MOUNT_R @ tool_tilt_matrix(cfg)


def joint_limits(cfg: SceneConfig):
    """模型坐标下的关节上下限（rad）：V2 固件限位，可选再与对称包络取交集。"""
    lo, hi = v2.model_limits_rad()
    if cfg.joint_limit_cap_deg is not None:
        cap = np.deg2rad(cfg.joint_limit_cap_deg)
        lo, hi = np.maximum(lo, -cap), np.minimum(hi, cap)
    return lo, hi


def blade_area_centroid_from_tip(cfg: SceneConfig):
    """尖头刀面（矩形 + 三角尖）的面积形心离刀尖的距离。"""
    if cfg.geometry_profile == 'lab_20260922':
        from .lab_tool import blade_centroid
        return blade_centroid(cfg)
    L, w, t = cfg.trowel_length, cfg.trowel_width, cfg.trowel_tip_length
    a_rect, c_rect = w * (L - t), (L + t) / 2
    a_tri, c_tri = w * t / 2, 2 * t / 3
    return (a_rect * c_rect + a_tri * c_tri) / (a_rect + a_tri)


def clamp_position(cfg: SceneConfig):
    c = blade_area_centroid_from_tip(cfg) if cfg.clamp_from_tip is None else cfg.clamp_from_tip
    lo, hi = cfg.handle_from_tip
    if not lo <= c <= hi:
        raise ValueError(f'clamp position {c:.3f} m from tip is outside the wooden handle {cfg.handle_from_tip}')
    return c


def rigid_tool_length(cfg: SceneConfig):
    """减速器输出法兰面 → 刀面外表面（沿 J6 轴）。"""
    return cfg.holder_height + 2 * cfg.handle_radius + cfg.neck_height + cfg.trowel_thickness


def blade_outline(cfg: SceneConfig):
    """刀面在刀面系（x 沿刀长、指向刀尖；z 沿刀宽；原点在 J6 轴线上）里的轮廓，供覆盖计算与示教使用。

    返回 dict：x_back（方头端）、x_taper（开始收窄处）、x_tip（刀尖）、half_width。弹簧方案为矩形。
    """
    if cfg.tool_mount != 'rigid':
        return dict(x_back=-cfg.blade_width / 2, x_taper=cfg.blade_width / 2, x_tip=cfg.blade_width / 2,
                    half_width=cfg.blade_height / 2)
    c = clamp_position(cfg)
    return dict(x_back=c - cfg.trowel_length, x_taper=c - cfg.trowel_tip_length, x_tip=c, half_width=cfg.trowel_width / 2)


def blade_span(cfg: SceneConfig):
    """刀面沿刀长（psi=0 时沿墙水平）与沿刀宽（竖直）的总尺寸。"""
    o = blade_outline(cfg)
    return o['x_tip'] - o['x_back'], 2 * o['half_width']


def tool_tilt_quat(cfg: SceneConfig):
    q = np.zeros(4); mujoco.mju_mat2Quat(q, tool_tilt_matrix(cfg).flatten()); return q


def wall_frame(cfg: SceneConfig):
    """返回墙面坐标系 (origin, R)：R 的列依次为 u(沿墙水平)、v(沿墙向上)、n(指向墙内)。"""
    yaw, pitch = np.deg2rad(cfg.wall_yaw_deg), np.deg2rad(cfg.wall_pitch_deg)
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    R0 = np.column_stack([[0., 1, 0], [0, 0, 1.], [1., 0, 0]])    # u=+Y, v=+Z, n=+X（指向墙内）
    R = Rz @ Ry @ R0
    origin = np.array([cfg.wall_distance, cfg.wall_lateral, cfg.wall_center_z])
    return origin, R


def build_spec(cfg: SceneConfig):
    spec = mujoco.MjSpec.from_file(str(ARM_MODEL))
    spec.modelname = 'dummy_wall_trowel_SIM_ASSUMPTIONS_NOT_CALIBRATED'
    lo, hi = joint_limits(cfg)
    for i in range(6):
        spec.joint(f'Joint{i + 1}').range = [lo[i], hi[i]]
    for i, a in enumerate(spec.actuators):
        a.ctrlrange = [lo[i], hi[i]]
        if cfg.servo_force_limit is not None:
            a.forcerange = [-cfg.servo_force_limit, cfg.servo_force_limit]
        a.gainprm[0] = cfg.servo_kp
        a.biasprm[1] = -cfg.servo_kp; a.biasprm[2] = -cfg.servo_kv
    for b in spec.bodies:
        if b.name != 'world':
            b.gravcomp = 1.0 if cfg.gravcomp else 0.0

    if cfg.tool_mount == 'rigid' and cfg.j6_reducer_ratio:
        from .. import v2 as _v2
        motor = _v2.MOTORS[[d for d in _v2.DRIVETRAIN_V2 if d['joint'] == 'J6'][0]['motor']]
        tau = min(motor['holding_Nm'] * cfg.j6_reducer_ratio, cfg.j6_reducer_max_Nm)
        a6 = spec.actuator('servo6'); a6.forcerange = [-tau, tau]
        j6 = spec.joint('Joint6')
        j6.armature = motor['rotor_inertia_kgm2'] * cfg.j6_reducer_ratio ** 2

    link6 = spec.body(LINK6)
    gc = 1.0 if cfg.gravcomp else 0.0
    tip = [s_ for s_ in spec.sites if s_.name == 'shaft_tip'][0].pos[0]
    mq = np.zeros(4); mujoco.mju_mat2Quat(mq, MOUNT_R.flatten())
    if cfg.tool_mount == 'rigid':
        # J6 减速器壳体固定在 J6 电机座（link5）上，沿 J6 轴（link5 局部 +X）从电机座前端面向外。
        link5 = spec.body('link5')
        x0 = housing_front_x()
        pt, L = cfg.j6_reducer_plate_thickness, cfg.j6_reducer_length
        half = [pt / 2, cfg.j6_reducer_plate / 2, cfg.j6_reducer_plate / 2]
        grey = [0.72, 0.74, 0.77, 1]
        link5.add_geom(name='j6_reducer_base', type=mujoco.mjtGeom.mjGEOM_BOX, pos=[x0 + pt / 2, 0, 0], size=half,
                       contype=0, conaffinity=0, rgba=grey, mass=cfg.j6_reducer_mass * 0.3)
        link5.add_geom(name='j6_reducer_body', type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                       fromto=[x0 + pt, 0, 0, x0 + L - pt, 0, 0], size=[cfg.j6_reducer_radius, 0, 0],
                       contype=0, conaffinity=0, rgba=[0.8, 0.81, 0.83, 1], mass=cfg.j6_reducer_mass * 0.5)
        # 输出法兰（随 J6 转）：减速器最外一块方板
        mount = link6.add_body(name='tool_mount', pos=[x0 + L, 0, 0], quat=mq.tolist(), gravcomp=gc)
        mount.add_geom(name='j6_output_flange', type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, -pt / 2, 0],
                       size=[cfg.j6_reducer_plate / 2, pt / 2, cfg.j6_reducer_plate / 2],
                       contype=0, conaffinity=0, rgba=grey, mass=cfg.j6_reducer_mass * 0.2)
        # 安装系里：+Y 沿 J6 轴向外，+X = 刀长方向（指向刀尖），+Z = 刀宽方向。
        o = blade_outline(cfg)
        r, hh = cfg.handle_radius, cfg.holder_height
        holder = mount.add_body(name='trowel', pos=[0, 0, 0], gravcomp=gc)
        holder.add_site(name='load_cell', pos=[0, 0, 0], size=[0.004, 0, 0], rgba=[0.2, 0.6, 1, 1])
        # 3D 打印抹刀座：一块从法兰面伸到木柄的夹块，下部包住木柄（外观近似）
        holder.add_geom(name='holder_geom', type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, (hh + r) / 2, 0],
                        size=[0.012, (hh + r) / 2, r + 0.004], contype=0, conaffinity=0,
                        rgba=[0.96, 0.96, 0.93, 1], mass=cfg.holder_mass)
        # 抹刀本体（刚性固定在抹刀座上）
        tr = holder.add_body(name='trowel_body', pos=[0, 0, 0], quat=tool_tilt_quat(cfg).tolist(), gravcomp=gc)
        yh = hh + r                                   # 木柄轴线
        yb = hh + 2 * r + cfg.neck_height             # 刀面背面
        c = o['x_tip']
        h0, h1 = cfg.handle_from_tip
        tr.add_geom(name='handle_geom', type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                    fromto=[c - h0, yh, 0, c - h1, yh, 0], size=[r, 0, 0],
                    contype=0, conaffinity=0, rgba=[0.72, 0.5, 0.3, 1], mass=cfg.tool_mass * 0.25)
        xn = c - cfg.neck_from_tip
        tr.add_geom(name='neck_geom', type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                    fromto=[xn, yh + r - 0.002, 0, xn, yb, 0], size=[0.004, 0, 0],
                    contype=0, conaffinity=0, rgba=[0.12, 0.12, 0.13, 1], mass=cfg.tool_mass * 0.05)
        blade = tr.add_body(name='blade', pos=[0, yb, 0], gravcomp=gc)
        th, hw = cfg.trowel_thickness, o['half_width']
        steel = [0.78, 0.8, 0.83, 1]
        blade_mass = cfg.tool_mass * 0.7
        rect_len = o['x_taper'] - o['x_back']
        blade.add_geom(name='blade_geom', type=mujoco.mjtGeom.mjGEOM_BOX,
                       pos=[(o['x_back'] + o['x_taper']) / 2, th / 2, 0], size=[rect_len / 2, th / 2, hw],
                       contype=TOOL_GROUP, conaffinity=0, friction=[cfg.blade_friction, 0.005, 0.0001],
                       solref=[0.004, 1.0], solimp=[0.95, 0.99, 0.001, 0.5, 2], condim=3, rgba=steel,
                       mass=blade_mass * 0.85)
        xt, xp = o['x_taper'], o['x_tip']
        verts = [xt, 0, -hw, xt, 0, hw, xt, th, -hw, xt, th, hw,
                 xp, 0, -0.0005, xp, 0, 0.0005, xp, th, -0.0005, xp, th, 0.0005]
        spec.add_mesh(name='trowel_tip', uservert=verts)
        blade.add_geom(name='blade_tip_geom', type=mujoco.mjtGeom.mjGEOM_MESH, meshname='trowel_tip',
                       contype=TOOL_GROUP, conaffinity=0, friction=[cfg.blade_friction, 0.005, 0.0001],
                       solref=[0.004, 1.0], solimp=[0.95, 0.99, 0.001, 0.5, 2], condim=3, rgba=steel,
                       mass=blade_mass * 0.15)
        bpos = [0, th / 2, 0]
    else:
        # 弹簧方案（早期设计，仅对比）：工具安装系原点在裸轴端面，+Y 沿 J6 轴向外。
        mount = link6.add_body(name='tool_mount', pos=[tip, 0, 0], quat=mq.tolist(), gravcomp=gc)
        adapter = mount.add_body(name='tool_adapter', pos=[0, 0, 0], gravcomp=gc)
        adapter.add_geom(name='adapter_geom', type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                         fromto=[0, 0, 0, 0, cfg.adapter_length, 0], size=[cfg.adapter_radius, 0, 0],
                         contype=0, conaffinity=0, rgba=[0.95, 0.55, 0.25, 1], mass=cfg.tool_mass * 0.4)
        handle_len = cfg.tool_length - cfg.adapter_length - cfg.blade_thickness
        if handle_len <= 0.005:
            raise ValueError('tool_length too short for adapter_length')
        trowel = adapter.add_body(name='trowel', pos=[0, cfg.adapter_length, 0], gravcomp=gc)
        trowel.add_joint(name='compliance', type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, -1, 0],
                         range=[0, cfg.spring_travel], limited=mujoco.mjtLimited.mjLIMITED_TRUE,
                         stiffness=cfg.spring_k, springref=-cfg.spring_preload / cfg.spring_k,
                         damping=cfg.spring_damping, armature=0.001)
        trowel.add_geom(name='handle_geom', type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                        fromto=[0, 0, 0, 0, handle_len, 0], size=[0.006, 0, 0],
                        contype=0, conaffinity=0, rgba=[0.3, 0.3, 0.3, 1], mass=cfg.tool_mass * 0.2)
        blade = trowel.add_body(name='blade', pos=[0, handle_len, 0], quat=tool_tilt_quat(cfg).tolist(), gravcomp=gc)
        th = cfg.blade_thickness
        bpos = [0, th / 2, 0]
        blade.add_geom(name='blade_geom', type=mujoco.mjtGeom.mjGEOM_BOX, pos=bpos,
                       size=[cfg.blade_width / 2, th / 2, cfg.blade_height / 2],
                       contype=TOOL_GROUP, conaffinity=0, friction=[cfg.blade_friction, 0.005, 0.0001],
                       solref=[0.004, 1.0], solimp=[0.95, 0.99, 0.001, 0.5, 2],
                       condim=3, rgba=[0.75, 0.78, 0.8, 1], mass=cfg.tool_mass * 0.4)
    blade.add_site(name='tcp', pos=[0, th, 0], size=[0.004, 0, 0], rgba=[1, 0, 0, 1])
    blade.add_site(name='blade_site', pos=bpos, size=[0.003, 0, 0], rgba=[0, 0, 0, 0])
    # 墙
    origin, R = wall_frame(cfg)
    wq = _quat_from_axes(R[:, 0], R[:, 1], R[:, 2])
    wall = spec.worldbody.add_body(name='wall', pos=(origin + R[:, 2] * 0.01).tolist(), quat=wq.tolist())
    wall.add_geom(name='wall_geom', type=mujoco.mjtGeom.mjGEOM_BOX,
                  size=[cfg.wall_size[0] / 2, cfg.wall_size[1] / 2, 0.01],
                  contype=0, conaffinity=WALL_GROUP, friction=[cfg.blade_friction, 0.005, 0.0001],
                  rgba=[0.86, 0.84, 0.78, 1])
    wall.add_site(name='wall_origin', pos=[0, 0, -0.01], size=[0.003, 0, 0], rgba=[0, 0, 0, 0])
    # 相机：看向墙面中心
    cam_pos = np.array(cfg.camera_pos); look = origin
    z = cam_pos - look; z /= np.linalg.norm(z)
    x = np.cross([0, 0, 1.], z); x /= np.linalg.norm(x); y = np.cross(z, x)
    spec.worldbody.add_camera(name='d435', pos=cam_pos.tolist(), quat=_quat_from_axes(x, y, z).tolist(),
                              fovy=cfg.camera_fovy_deg)
    # 传感器：与实机可测量对应
    if cfg.tool_mount == 'spring':   # 压缩量：直线位移 / 霍尔传感器
        spec.add_sensor(name='compression', type=mujoco.mjtSensor.mjSENS_JOINTPOS, objtype=mujoco.mjtObj.mjOBJ_JOINT, objname='compliance')
    else:                            # 法兰与抹刀座之间的力传感器
        spec.add_sensor(name='load_cell', type=mujoco.mjtSensor.mjSENS_FORCE, objtype=mujoco.mjtObj.mjOBJ_SITE, objname='load_cell')
    spec.add_sensor(name='blade_force', type=mujoco.mjtSensor.mjSENS_FORCE, objtype=mujoco.mjtObj.mjOBJ_SITE, objname='blade_site')
    if cfg.geometry_profile == 'lab_20260922':
        from .lab_tool import refine_spec
        refine_spec(spec, cfg)
    elif cfg.geometry_profile != 'legacy':
        raise ValueError(f'Unknown geometry_profile: {cfg.geometry_profile}')
    spec.option.noslip_iterations = 0
    return spec


def build_scene(cfg: SceneConfig = None):
    cfg = cfg or SceneConfig()
    spec = build_spec(cfg)
    model = spec.compile()
    return model, spec


def export_xml(cfg: SceneConfig, path: Path):
    """把生成的场景写成 MJCF，便于在 MuJoCo 查看器里直接打开检查。

    网格路径写成相对输出文件的路径，文件随仓库移动到别的机器仍能打开。
    """
    spec = build_spec(cfg)
    spec.compile()
    xml = spec.to_xml()
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    import os
    import xml.etree.ElementTree as ET
    src_meshdir = ET.parse(ARM_MODEL).getroot().find('compiler').get('meshdir')
    target = (ARM_MODEL.parent / src_meshdir).resolve()
    try:
        meshdir = Path(os.path.relpath(target, path.resolve().parent)).as_posix()
    except ValueError:            # Windows：输出目录与仓库不在同一个盘，只能写绝对路径
        meshdir = target.as_posix()
    import re
    xml, n = re.subn(r'meshdir="[^"]*"', f'meshdir="{meshdir}/"', xml, count=1)
    if n != 1:
        raise RuntimeError('could not rewrite meshdir in exported MJCF')
    path.write_text(xml, encoding='utf-8')
    (path.with_suffix('.config.json')).write_text(json.dumps(
        {'config': cfg.to_dict(), 'provenance': cfg.provenance(), 'digest': cfg.digest()},
        indent=2, ensure_ascii=False), encoding='utf-8')
    return path

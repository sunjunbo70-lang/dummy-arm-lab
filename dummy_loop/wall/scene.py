"""墙面抹涂场景：Dummy V2 机械臂 + 裸轴转接件 + 弹簧柔顺滑轨 + 抹刀 + 墙 + 固定相机。

用 MjSpec 在 models/dummy_v2.xml 之上程序化组装，不修改原文件。
所有尺寸都是参数；哪些是假设、从哪里来，写进 SceneConfig.provenance()，
随每条 episode 一起保存。

坐标约定（模型坐标 = 固件基座系，见 docs/ACTION_SPACE.md、docs/hardware/DUMMY_V2.md）：
  世界系原点在 J1 轴线上，+X 朝前，+Z 向上。模型零位 = 固件 HOME（L 形姿态），
  此时小臂水平朝前，J6 轴沿 +X。
  墙位于机械臂正前方，墙面法线 n 指向墙内（+X），u 沿墙水平（+Y），v 向上（+Z）。
  末端是 J6 电机裸轴（无法兰）：转接件套在轴上，从轴端（link6 局部 x = 轴端距腕心）向外伸出。
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
    # 转接件与抹刀（占位尺寸，拿到实物后替换）
    adapter_length: float = 0.030     # 裸轴端面到滑轨起点（转接件还要向内套住轴，那部分不计入）
    adapter_radius: float = 0.012
    tool_length: float = 0.10         # 裸轴端面到刀面（弹簧未压缩时）
    blade_tilt_deg: float = 0.0       # 刀面相对 J6 轴的倾角（绕刀宽方向）。0 = 刀面垂直于 J6 轴
    blade_width: float = 0.08         # 沿墙水平方向（psi=0 时）
    blade_height: float = 0.05
    blade_thickness: float = 0.003
    tool_mass: float = 0.15
    # 被动柔顺：弹簧滑轨
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
        return {
            'arm': 'models/dummy_v2.xml (V2 firmware DH kinematics; CAD-derived mass estimates; not identified)',
            'joint_limits': ('V2 firmware limits converted to model coordinates (dummy_loop/v2.py); '
                             'CANDIDATE until M3 measures them' +
                             ('' if self.joint_limit_cap_deg is None else f'; additionally capped at +/-{self.joint_limit_cap_deg} deg')),
            'servo_kp_kv': 'example values, not identified',
            'servo_force': ('per-joint from V2 drivetrain documents (reducer start/stop peak or motor holding x ratio)'
                            if self.servo_force_limit is None else f'uniform override {self.servo_force_limit} N*m'),
            'tool_dimensions': 'PLACEHOLDER: no trowel measured yet',
            'tool_mount': 'adapter clamps the bare J6 shaft; origin at the shaft end face measured from the V2 CAD',
            'spring': 'DESIGN CHOICE for passive compliance, not a purchased part',
            'wall_pose': ('nominal. Chosen by dummy_loop.wall.layout search on the V2 model '
                          '(see experiments/2026-09-20_v2_model). Re-run after M3.'),
            'camera_pose': 'PLACEHOLDER: fixed D435 not yet mounted (M6)',
        }

    def to_dict(self):
        d = asdict(self)
        d['wall_size'] = list(self.wall_size); d['camera_pos'] = list(self.camera_pos)
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

    link6 = spec.body(LINK6)
    gc = 1.0 if cfg.gravcomp else 0.0
    tip = [s for s in spec.sites if s.name == 'shaft_tip'][0].pos[0]
    # 工具安装系：原点在裸轴端面，+Y 沿 J6 轴向外。转接件刚性套在轴上。
    mq = np.zeros(4); mujoco.mju_mat2Quat(mq, MOUNT_R.flatten())
    mount = link6.add_body(name='tool_mount', pos=[tip, 0, 0], quat=mq.tolist(), gravcomp=gc)
    adapter = mount.add_body(name='tool_adapter', pos=[0, 0, 0], gravcomp=gc)
    adapter.add_geom(name='adapter_geom', type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                     fromto=[0, 0, 0, 0, cfg.adapter_length, 0], size=[cfg.adapter_radius, 0, 0],
                     contype=0, conaffinity=0, rgba=[0.95, 0.55, 0.25, 1], mass=cfg.tool_mass * 0.4)
    # 抹刀：沿刀具轴的弹簧滑轨。压缩量 = 滑轨关节位置（>=0）。
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
    # 刀面可相对 J6 轴倾斜 blade_tilt_deg（绕刀宽方向 = link6 局部 x）。
    # 刀面中心始终在 J6 轴线上：沿墙拖曳力对 J6 不产生扭矩。
    blade = trowel.add_body(name='blade', pos=[0, handle_len, 0], quat=tool_tilt_quat(cfg).tolist(), gravcomp=gc)
    bpos = [0, cfg.blade_thickness / 2, 0]
    blade.add_geom(name='blade_geom', type=mujoco.mjtGeom.mjGEOM_BOX, pos=bpos,
                    size=[cfg.blade_width / 2, cfg.blade_thickness / 2, cfg.blade_height / 2],
                    contype=TOOL_GROUP, conaffinity=0, friction=[cfg.blade_friction, 0.005, 0.0001],
                    solref=[0.004, 1.0], solimp=[0.95, 0.99, 0.001, 0.5, 2],
                    condim=3, rgba=[0.75, 0.78, 0.8, 1], mass=cfg.tool_mass * 0.4)
    blade.add_site(name='tcp', pos=[0, cfg.blade_thickness, 0], size=[0.004, 0, 0], rgba=[1, 0, 0, 1])
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
    # 传感器：与实机可测量对应（压缩量可用便宜的直线位移/霍尔传感器实测）
    spec.add_sensor(name='compression', type=mujoco.mjtSensor.mjSENS_JOINTPOS, objtype=mujoco.mjtObj.mjOBJ_JOINT, objname='compliance')
    spec.add_sensor(name='blade_force', type=mujoco.mjtSensor.mjSENS_FORCE, objtype=mujoco.mjtObj.mjOBJ_SITE, objname='blade_site')
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

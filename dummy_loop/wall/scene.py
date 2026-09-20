"""墙面抹涂场景：参考机械臂 + 转接件 + 弹簧柔顺滑轨 + 抹刀 + 墙 + 固定相机。

用 MjSpec 在 models/dummy_reference.xml 之上程序化组装，不修改原文件。
所有尺寸都是参数；哪些是假设、从哪里来，写进 SceneConfig.provenance()，
随每条 episode 一起保存。

坐标约定（模型坐标 = 规范坐标 canonical，见 docs/ACTION_SPACE.md）：
  世界系原点在底座，+Z 向上；模型零位时前臂水平指向 -Y。
  墙位于机械臂前方，墙面法线指向机械臂（+Y）。
  J6 连杆局部 +Y 是刀具外伸方向；法兰端面在 link6 局部 y = 0.003 m 处。
"""
from dataclasses import dataclass, field, asdict
from pathlib import Path
import hashlib, json
import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[2]
ARM_MODEL = ROOT / 'models' / 'dummy_reference.xml'
FLANGE_FACE_Y = 0.003          # link6 局部坐标中法兰端面位置（由 link6_1_1.stl 量得）
TOOL_GROUP, WALL_GROUP = 2, 2  # 接触位掩码：只让刀面与墙发生接触


@dataclass
class SceneConfig:
    # 关节范围：实机未标定（configs/ 限位全 null）。默认 ±90° 是几何探索用的假设。
    joint_range_deg: float = 90.0
    servo_kp: float = 35.0            # 沿用参考模型的示例值（未辨识）
    servo_kv: float = 4.0
    servo_force_limit: float = 5.0    # N·m，沿用参考模型
    # 转接件与抹刀（占位尺寸，拿到实物后替换）
    adapter_length: float = 0.030     # 法兰端面到滑轨起点
    adapter_radius: float = 0.012
    tool_length: float = 0.10         # 法兰端面到刀面（弹簧未压缩时）
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
    wall_distance: float = 0.40       # 墙面到 J1 轴的距离，沿 -Y（由 layout 搜索选定，见 provenance）
    wall_yaw_deg: float = 0.0         # 绕竖直轴
    wall_pitch_deg: float = 0.0       # 绕水平轴（墙面前后倾）
    wall_x: float = 0.0
    wall_center_z: float = 0.20       # 工作区中心高度（同上）
    wall_size: tuple = (0.80, 0.60)   # 宽, 高
    # 固定相机（D435 类，位置为占位）
    camera_pos: tuple = (0.35, 0.05, 0.45)
    camera_fovy_deg: float = 58.0
    gravcomp: bool = True             # 沿用参考模型的理想重力补偿

    def provenance(self):
        return {
            'arm': 'models/dummy_reference.xml (reference geometry, not calibrated)',
            'joint_range_deg': 'ASSUMPTION: configs/ profile limits are null until M3',
            'servo_kp_kv_force': 'copied from reference model example values, not identified',
            'tool_dimensions': 'PLACEHOLDER: no trowel measured yet',
            'spring': 'DESIGN CHOICE for passive compliance, not a purchased part',
            'wall_pose': ('nominal. wall_distance=0.40, wall_center_z=0.20, blade_tilt=0 chosen by '
                          'dummy_loop.wall.layout search (best worst-case conditioning over a 12x10 cm patch, '
                          'ASSUMING +/-90 deg joint limits). Re-run the search after M3 measures real limits.'),
            'camera_pose': 'PLACEHOLDER: fixed D435 not yet mounted (M6)',
            'flange_face_y': 'measured from models/meshes/link6_1_1.stl; real arm has a bare J6 shaft instead',
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


def tool_tilt_quat(cfg: SceneConfig):
    q = np.zeros(4); mujoco.mju_mat2Quat(q, tool_tilt_matrix(cfg).flatten()); return q


def wall_frame(cfg: SceneConfig):
    """返回墙面坐标系 (origin, R)：R 的列依次为 u(沿墙水平)、v(沿墙向上)、n(指向墙内)。"""
    yaw, pitch = np.deg2rad(cfg.wall_yaw_deg), np.deg2rad(cfg.wall_pitch_deg)
    n = np.array([0., -1., 0.])                                   # 名义：指向墙内 = -Y
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
    R0 = np.column_stack([[1., 0, 0], [0, 0, 1.], n])             # u=+X, v=+Z, n=-Y
    R = Rz @ Rx @ R0
    origin = np.array([cfg.wall_x, -cfg.wall_distance, cfg.wall_center_z])
    return origin, R


def build_spec(cfg: SceneConfig):
    spec = mujoco.MjSpec.from_file(str(ARM_MODEL))
    spec.modelname = 'dummy_wall_trowel_SIM_ASSUMPTIONS_NOT_CALIBRATED'
    lim = np.deg2rad(cfg.joint_range_deg)
    for j in spec.joints:
        j.range = [-lim, lim]
    for a in spec.actuators:
        a.ctrlrange = [-lim, lim]
        a.forcerange = [-cfg.servo_force_limit, cfg.servo_force_limit]
        a.gainprm[0] = cfg.servo_kp
        a.biasprm[1] = -cfg.servo_kp; a.biasprm[2] = -cfg.servo_kv
    for b in spec.bodies:
        if b.name != 'world':
            b.gravcomp = 1.0 if cfg.gravcomp else 0.0

    link6 = spec.body('link6_1_1')
    gc = 1.0 if cfg.gravcomp else 0.0
    # 转接件：刚性，固定在法兰上
    adapter = link6.add_body(name='tool_adapter', pos=[0, FLANGE_FACE_Y, 0], gravcomp=gc)
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
    meshdir = Path(os.path.relpath((ARM_MODEL.parent / 'meshes').resolve(), path.resolve().parent)).as_posix()
    import re
    xml, n = re.subn(r'meshdir="[^"]*"', f'meshdir="{meshdir}/"', xml, count=1)
    if n != 1:
        raise RuntimeError('could not rewrite meshdir in exported MJCF')
    path.write_text(xml, encoding='utf-8')
    (path.with_suffix('.config.json')).write_text(json.dumps(
        {'config': cfg.to_dict(), 'provenance': cfg.provenance(), 'digest': cfg.digest()},
        indent=2, ensure_ascii=False), encoding='utf-8')
    return path

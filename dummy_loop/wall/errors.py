"""仿真到实机的误差模型。

每一项都对应实机上一个具体、可命名的误差来源。控制器只知道名义值；
误差只作用在「真实世界」一侧（仿真模型、执行器接口、传感器读数）。

  joint_offset_rad   逐轴零位误差：固件零点与规范零点差 δ。执行时真实角 = 指令 + δ，
                     编码器读回 = 真实 - δ（机器自以为在指令位置）。M3 标定要消除的正是它。
  latency_steps      指令延迟若干控制周期（串口排队、固件缓存）。M2 要测的上界。
  q_noise_rad        关节读数噪声。
  compression_noise_m  压缩量传感器噪声。
  wall_dn_m          真实墙面沿名义法线的偏移（正 = 更远）。墙板摆放误差。
  wall_yaw_deg / wall_pitch_deg  真实墙面相对名义的偏转 / 前后倾。
  spring_k_scale     弹簧刚度相对设计值的比例（弹簧公差、3D 打印导轨摩擦）。
  friction_scale     刀面-墙面摩擦系数比例（材料不同）。
  servo_kp_scale     关节伺服刚度比例（实机闭环步进的等效刚度未知）。

未建模（在记录里写明，不假装覆盖）：齿隙/回差、连杆长度误差、减速器柔性、
温漂、材料流动与堆积、墙面不平整。
"""
from dataclasses import dataclass, field, asdict, replace
import numpy as np

from .scene import SceneConfig


@dataclass
class Perturbation:
    joint_offset_rad: tuple = (0.0,) * 6
    latency_steps: int = 0
    q_noise_rad: float = 0.0
    compression_noise_m: float = 0.0
    wall_dn_m: float = 0.0
    wall_yaw_deg: float = 0.0
    wall_pitch_deg: float = 0.0
    spring_k_scale: float = 1.0
    friction_scale: float = 1.0
    servo_kp_scale: float = 1.0

    def true_scene(self, nominal: SceneConfig) -> SceneConfig:
        return replace(nominal,
                       wall_distance=nominal.wall_distance + self.wall_dn_m,
                       wall_yaw_deg=nominal.wall_yaw_deg + self.wall_yaw_deg,
                       wall_pitch_deg=nominal.wall_pitch_deg + self.wall_pitch_deg,
                       spring_k=nominal.spring_k * self.spring_k_scale,
                       blade_friction=nominal.blade_friction * self.friction_scale,
                       servo_kp=nominal.servo_kp * self.servo_kp_scale)

    def to_dict(self):
        d = asdict(self); d['joint_offset_rad'] = [float(x) for x in self.joint_offset_rad]; return d

    @staticmethod
    def sample(rng, scale=1.0):
        """域随机化：按「标定前的合理误差量级」随机抽取。scale 统一放大或缩小。"""
        s = scale
        return Perturbation(
            joint_offset_rad=tuple(rng.normal(0, np.deg2rad(1.0) * s, 6)),
            latency_steps=int(rng.integers(0, 1 + round(2 * s))),
            q_noise_rad=np.deg2rad(0.05) * s,
            compression_noise_m=0.0005 * s,
            wall_dn_m=float(rng.normal(0, 0.005 * s)),
            wall_yaw_deg=float(rng.normal(0, 2.0 * s)),
            wall_pitch_deg=float(rng.normal(0, 2.0 * s)),
            spring_k_scale=float(np.exp(rng.normal(0, 0.15 * s))),
            friction_scale=float(np.exp(rng.normal(0, 0.3 * s))),
            servo_kp_scale=float(np.exp(rng.normal(0, 0.3 * s))))


# 单因素敏感性分析用的扫描表：每个因素几个量级，其余保持名义。
def single_factor_sweeps():
    one = np.eye(6)
    return {
        'wall_dn_mm':        [(v, Perturbation(wall_dn_m=v / 1000)) for v in (-10, -5, 5, 10)],
        'wall_yaw_deg':      [(v, Perturbation(wall_yaw_deg=v)) for v in (2, 5)],
        'wall_pitch_deg':    [(v, Perturbation(wall_pitch_deg=v)) for v in (2, 5)],
        'J2_offset_deg':     [(v, Perturbation(joint_offset_rad=tuple(one[1] * np.deg2rad(v)))) for v in (1, 2)],
        'J3_offset_deg':     [(v, Perturbation(joint_offset_rad=tuple(one[2] * np.deg2rad(v)))) for v in (1, 2)],
        'J5_offset_deg':     [(v, Perturbation(joint_offset_rad=tuple(one[4] * np.deg2rad(v)))) for v in (1, 2)],
        'latency_steps':     [(v, Perturbation(latency_steps=v)) for v in (1, 3)],
        'spring_k_scale':    [(v, Perturbation(spring_k_scale=v)) for v in (0.7, 1.4)],
        'servo_kp_scale':    [(v, Perturbation(servo_kp_scale=v)) for v in (0.5, 2.0)],
        'friction_scale':    [(v, Perturbation(friction_scale=v)) for v in (0.5, 1.7)],
    }

"""墙面材料（砂浆 / 腻子）的降阶模型：墙面高度场 + 刀上带料量。

**这是代理模型，不是流体仿真。** MuJoCo 是刚体引擎，算不了砂浆的流动、屈服应力与触变性。
但强化学习需要「抹得匀不匀、掉没掉料」这种奖励信号，而这类信号只取决于
「刀面扫过之后墙上留下多厚一层」。本模型就只建这一件事：

  墙面切成格子，每格记一个厚度 h（m）。刀面是一块平板，有俯仰角，所以它与墙面的间隙
  在刀宽方向上线性变化。刀面沿墙移动时，真正决定「留下多厚」的是**后缘**
  （运动方向的后方那条边）：前缘抬起让料流到刀下，后缘刮出最终厚度。
    - 后缘经过某格时，该格厚度被推到 min(h, 后缘间隙)：多出来的料被刀刮走，进入刀上带料量；
    - 若该格比后缘间隙薄且刀上有料，就补到后缘间隙（受每步补料速率上限限制）；
    - 刀离墙太远（间隙 > contact_gap）时什么也不发生；
    - 抹到工作区以外的料算浪费；episode 结束时刀上剩的料单独记为「没抹出去」。

刀上能「兜住」多少料（为什么工人要让下边缘先贴墙、再逐渐放平）：
  - 刀还没贴墙时，料靠粘附挂在刀面上，最多 stick_free 厚；
  - 下边缘贴墙后，料被兜在刀面与墙之间的楔形空间里：容量 = 刀长 ×（刀宽·cosθ·间隙 + ½·刀宽²·sinθ·cosθ），
    再加一层 stick_contact 厚的粘附层。θ 越大楔形越大；刀面放平时楔形几乎为零；
  - 刀上的料超过当前容量，多出的部分每步按 spill_rate 从刀边挤出掉落（算浪费）；
  - 俯仰超过 slide_angle 时料会从刀面上滑落（每步 slide_rate）。
  于是：平着拍上去 → 一贴墙料就被挤掉；先斜着让下边缘吃住墙、边走边随料变少而放平 → 掉得最少。
  **这些规则与系数都是假设**（按工人手法的经验解释建的），要用刮板实验标定后才能当真。

材料对刀面的反作用力（挤压 + 拖曳）按屈服应力 × 接触面积估算，通过 xfrc_applied 加到刀面体上，
这样机械臂「能感觉到」砂浆。所有系数都是**假设值**，要用实物刮板实验标定（见 docs/SIMULATION.md）。
"""
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class MaterialConfig:
    cell: float = 0.005               # 格子边长 m
    target_thickness: float = 0.002   # 目标抹灰厚度 m（小抹刀、薄层）
    contact_gap: float = 0.008        # 后缘间隙超过这个值就认为刀没在抹料
    deposit_rate: float = 0.004       # 每步单格最多补多厚（m），相当于料的流动速率上限
    initial_load_scale: float = 1.2   # 每刀带料量 = 本刀目标区域体积 × 该系数
    stick_free: float = 0.008         # 刀未贴墙时，粘附在刀面上的料最多多厚 m
    stick_contact: float = 0.001      # 贴墙后刀面上的粘附层厚度 m
    spill_rate: float = 0.5           # 超出容量的料每步挤出掉落的比例
    slide_angle_deg: float = 30.0     # 俯仰超过此角，料开始从刀面滑落
    slide_rate: float = 0.05          # 超过滑落角后每步滑落的比例
    yield_stress: float = 2500.0      # Pa：刮动材料时的等效法向压强
    drag_ratio: float = 0.6           # 拖曳力 / 法向力
    density: float = 1900.0           # kg/m³，仅用于把体积换算成质量（报告用）

    def to_dict(self):
        return asdict(self)


class MortarField:
    """墙面高度场 + 刀上带料量。坐标用真实墙面系 (u, v)，单位 m。"""

    def __init__(self, cfg: MaterialConfig, region_u, region_v, margin=0.02):
        self.cfg = cfg
        self.u0, self.u1 = region_u[0] - margin, region_u[1] + margin
        self.v0, self.v1 = region_v[0] - margin, region_v[1] + margin
        self.region_u, self.region_v = region_u, region_v
        nu = int(round((self.u1 - self.u0) / cfg.cell))
        nv = int(round((self.v1 - self.v0) / cfg.cell))
        cu = self.u0 + (np.arange(nu) + 0.5) * cfg.cell
        cv = self.v0 + (np.arange(nv) + 0.5) * cfg.cell
        self.cu, self.cv = np.meshgrid(cu, cv)                      # (nv, nu)
        self.area = cfg.cell ** 2
        self.inside = ((self.cu >= region_u[0]) & (self.cu <= region_u[1]) &
                       (self.cv >= region_v[0]) & (self.cv <= region_v[1]))
        self.reset()

    # ------------------------------------------------------------------ 状态
    def reset(self, h0=None, load=None):
        self.h = np.zeros(self.cu.shape) if h0 is None else np.array(h0, float)
        self.load = self.initial_load() if load is None else float(load)
        self.load0 = self.load          # 本刀带料量（奖励里的分母）
        self.supplied = self.load       # 整次作业累计上料量（指标里的分母）
        self.wasted = 0.0          # 抹到工作区外的体积 m³
        self.scraped = 0.0         # 被刮回刀上的体积 m³
        self.deposited = 0.0       # 留在墙上的体积 m³
        self.dropped = 0.0         # 从刀上挤出 / 滑落掉到地上的体积 m³
        self.last_edge = None      # 上一步后缘位置 (v, 间隙)
        return self

    def reload(self, volume):
        """「上料」：把刀上的料补到一份的量（上一刀剩下的接着用，不会越堆越多）。"""
        add = max(float(volume) - self.load, 0.0)
        self.load += add; self.supplied += add; self.load0 = float(volume)
        return self.load

    def initial_load(self):
        c = self.cfg
        target_volume = self.inside.sum() * self.area * c.target_thickness
        return target_volume * c.initial_load_scale

    # ------------------------------------------------------------------ 更新
    def sweep(self, edge_v, edge_gap, half_width, centre_u, moving_up):
        """后缘从上一次位置扫到 (edge_v, edge_gap)，更新高度场。返回本步的 (刮走, 抹上, 浪费) 体积。

        edge_gap: 后缘离墙面的间隙 m（>0 表示还没接触墙面本体）。
        half_width / centre_u: 刀面沿墙水平方向的半长与中心（真实墙面系）。
        """
        c = self.cfg
        prev = self.last_edge
        self.last_edge = (edge_v, edge_gap)
        if prev is None:
            return 0.0, 0.0, 0.0
        v_a, g_a = prev
        v_lo, v_hi = (v_a, edge_v) if edge_v >= v_a else (edge_v, v_a)
        if edge_gap > c.contact_gap and g_a > c.contact_gap:
            return 0.0, 0.0, 0.0
        band = (self.cv >= v_lo - self.cfg.cell / 2) & (self.cv < v_hi + self.cfg.cell / 2) & \
               (np.abs(self.cu - centre_u) <= half_width)
        if not band.any():
            return 0.0, 0.0, 0.0
        # 后缘经过每个格子时的间隙：沿 v 线性插值
        t = np.clip((self.cv - v_a) / (edge_v - v_a), 0, 1) if abs(edge_v - v_a) > 1e-9 else np.zeros_like(self.cv)
        gap = np.clip(g_a + t * (edge_gap - g_a), 0.0, None)
        target = np.where(gap <= c.contact_gap, gap, self.h)        # 太远就不动
        h_old = self.h.copy()
        # 刮：比后缘间隙厚的部分被带走
        cut = band & (h_old > target)
        removed = float(np.sum(h_old[cut] - target[cut]) * self.area)
        # 补：比后缘间隙薄且刀上有料
        fill = band & (h_old < target)
        need = np.minimum(target - h_old, c.deposit_rate)
        want = float(np.sum(need[fill]) * self.area)
        give = min(want, self.load + removed)
        scale = 0.0 if want <= 1e-12 else give / want
        new = h_old.copy()
        new[cut] = target[cut]
        new[fill] = h_old[fill] + need[fill] * scale
        self.h = new
        self.load = self.load + removed - give
        # 抹到工作区以外的料算浪费
        outside = band & (~self.inside)
        waste = float(np.sum(np.maximum(new[outside] - h_old[outside], 0)) * self.area)
        self.wasted += waste
        self.scraped += removed
        self.deposited += give
        return removed, give, waste

    def capacity(self, pitch, lower_gap, blade_len, blade_w):
        """刀上当前能兜住的料（m³）。pitch：刀面俯仰（rad，>0 下缘更贴墙）；lower_gap：下缘离墙间隙 m。"""
        c = self.cfg
        area = blade_len * blade_w
        if lower_gap > c.contact_gap:                          # 还没贴墙：靠粘附挂着
            return area * c.stick_free
        p = max(pitch, 0.0); g = max(lower_gap, 0.0)
        wedge = blade_len * (blade_w * np.cos(p) * g + 0.5 * blade_w ** 2 * np.sin(p) * np.cos(p))
        return area * c.stick_contact + wedge

    def carry(self, pitch, lower_gap, blade_len, blade_w):
        """按当前姿态检查刀上的料兜不兜得住，返回本步掉落的体积 m³。"""
        c = self.cfg
        cap = self.capacity(pitch, lower_gap, blade_len, blade_w)
        drop = c.spill_rate * max(self.load - cap, 0.0)
        if pitch > np.deg2rad(c.slide_angle_deg):
            drop += c.slide_rate * max(self.load - drop, 0.0)
        drop = min(drop, self.load)
        self.load -= drop; self.dropped += drop
        return drop

    def resistance(self, contact_area, speed_dir):
        """材料对刀面的反作用力（世界系方向由调用方给）：法向挤压 + 反向拖曳，单位 N。"""
        c = self.cfg
        fn = c.yield_stress * max(contact_area, 0.0)
        return fn, c.drag_ratio * fn

    # ------------------------------------------------------------------ 指标
    def metrics(self):
        c = self.cfg
        h = self.h[self.inside]
        tgt = c.target_thickness
        covered = float(np.mean(h >= 0.5 * tgt))
        rmse = float(np.sqrt(np.mean((h - tgt) ** 2)))
        return {'coverage': round(covered, 4),
                'mean_thickness_mm': round(float(h.mean()) * 1000, 3),
                'rms_error_mm': round(rmse * 1000, 3),
                'flatness_std_mm': round(float(h.std()) * 1000, 3),
                'wasted_frac': round((self.wasted + self.dropped) / max(self.supplied, 1e-12), 4),
                'dropped_frac': round(self.dropped / max(self.supplied, 1e-12), 4),
                'left_on_tool_frac': round(self.load / max(self.supplied, 1e-12), 4),
                'deposited_g': round(self.deposited * c.density * 1000, 2)}

    def volume_balance(self):
        """体积守恒检查：墙上的 + 刀上的 + 掉落的 = 初始带料量（抹出区外的已经算在墙上的格子里）。"""
        return float(self.h.sum() * self.area + self.load + self.dropped)

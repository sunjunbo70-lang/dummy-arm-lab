"""多刀作业：在机械臂够得到的范围内，一条一条把墙抹完（L1 仿真）。

一刀只能抹一条带（刀长 × 一个行程），整面墙要分成几条带，每条带一刀：

    回到「立起」待命位（刀面斜着、下缘朝墙、离墙 standoff）→ 上料（刀上装一份新料）
    → 执行一刀（策略或脚本控制：下缘先贴墙，边上行边放平）→ 抬刀退回待命位
    → 横移到下一条带，重复。

策略只学「一刀」（dummy_loop/wall/stroke_env.py），带与带之间的回位、横移、上料是固定的脚本动作：
它们不影响抹涂质量，没必要学。评价用整片区域的料层厚度：覆盖率、厚度偏差、平整度、浪费。

**带与带之间的移动默认不仿真**（`simulate_transit=False`：每一刀直接从该条带的待命位开始）。
原因是实测出来的一个限制：锁住刀面 roll 之后，相邻条带的自然逆解往往落在不同的腕部支解上，
横移过去要把 J4/J6 各转一百多度，而 J6 是直驱（力矩很小），中途刀面还会扫到墙。
这说明**换条带需要专门规划过渡轨迹**（或者允许刀面绕法线转一定角度），属于待办，不影响单刀手法的结论。
打开 `simulate_transit=True` 可以复现这个问题。

可达范围：`reachable_bands()` 会逐条带试逆解，够不到的直接剔除，所以「机械臂能活动的范围」是算出来的，
不是拍脑袋定的。
"""
from dataclasses import dataclass, asdict, field as dc_field
import numpy as np

from .material import MortarField
from .stroke_env import StrokeEnv, rollout


@dataclass
class SessionConfig:
    column_centres: tuple = (-0.08, 0.0, 0.08)   # 每条带的刀面中心 u（墙面系）
    band_v: tuple = (0.0, 0.07)                  # 每条带的竖直范围
    simulate_transit: bool = False               # 是否把带与带之间的移动也仿真出来（见下方说明）
    transit_steps: int = 25                      # 带与带之间插值的步数
    reload_steps: int = 10                       # 待命位「等新料上刀」的停留步数
    margin: float = 0.02                         # 区域外多留一圈格子用来统计掉到区外的料

    def to_dict(self):
        d = asdict(self); d['column_centres'] = list(self.column_centres); d['band_v'] = list(self.band_v)
        return d


def area_of(cfg: SessionConfig, half_len):
    """整片作业区域（墙面系）：所有条带并起来。"""
    us = np.array(cfg.column_centres)
    return (float(us.min() - half_len), float(us.max() + half_len)), tuple(cfg.band_v)


def reachable_bands(env: StrokeEnv, cfg: SessionConfig):
    """逐条带检查逆解：待命位、起刀点、行程终点都解得出来才算够得到。"""
    ok, dropped = [], []
    hw = env.outline['half_width']
    v0, v1 = cfg.band_v
    for u in cfg.column_centres:
        poses = [(v0 - env.cfg.v_start_margin + hw, -env.cfg.standoff, np.deg2rad(env.cfg.start_pitch_deg)),
                 (v0 + hw, -0.004, np.deg2rad(env.cfg.start_pitch_deg)),
                 (v1 + env.cfg.v_stop_margin + hw, -0.004, np.deg2rad(5.0))]
        try:
            for v, n, pitch in poses:
                env.ctrl.reset([u + env.xc, v, n], 0.0, np.zeros(6), pitch=pitch)
            ok.append(u)
        except ValueError:
            dropped.append(u)
    return ok, dropped


def run_session(env: StrokeEnv, policy=None, cfg: SessionConfig = None, on_step=None, on_reset=None):
    """跑一整片：返回 (整片指标, 每一刀的指标, 轨迹信息)。policy=None 用手写脚本。"""
    cfg = cfg or SessionConfig()
    half_len = (env.outline['x_tip'] - env.outline['x_back']) / 2
    region_u, region_v = area_of(cfg, half_len)
    field = MortarField(env.mat_cfg, region_u, region_v, margin=cfg.margin)
    field.reset(load=0.0)                       # 整片作业：料由每一刀「上料」时给，不预先装在刀上
    columns, unreachable = reachable_bands(env, cfg)
    strokes, steps_total = [], 0
    for i, u in enumerate(columns):
        env.set_band(u, cfg.band_v, field, reload_load=True)
        obs = env.reset(transit=(cfg.simulate_transit and i > 0),
                        transit_steps=cfg.transit_steps, reload_steps=cfg.reload_steps)
        if on_reset is not None:
            on_reset(i, obs)
        done, ret, n = False, 0.0, 0
        while not done:
            a = env.scripted_action() if policy is None else policy(obs)
            obs, r, done, info = env.step(a)
            ret += r; n += 1
            if on_step is not None:
                on_step(i, obs, a, info)
        steps_total += n
        strokes.append({'column_u': round(float(u), 4), 'steps': n, 'return': round(ret, 3),
                        'max_force_N': round(env.stats['max_force_N'], 2),
                        'aborted': bool(env.stats['aborted'])})
    m = field.metrics()
    m.update({'strokes': len(columns), 'steps': steps_total, 'transit_simulated': bool(cfg.simulate_transit),
              'unreachable_columns': [round(float(u), 4) for u in unreachable],
              'area_u_m': list(region_u), 'area_v_m': list(region_v),
              'area_cm2': round((region_u[1] - region_u[0]) * (region_v[1] - region_v[0]) * 1e4, 1)})
    return m, strokes, field

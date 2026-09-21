"""单次抹涂行程的强化学习环境（L1 仿真）。

一个 episode = 一次抹涂：机械臂从待命位出发 → 刀面下边缘先贴近墙 → 一边向上走一边逐渐放平 →
把刀上的料抹到墙上 → 抬刀 → 回到待命位。奖励看这一刀抹出来的料层：够不够厚、匀不匀、浪费多少。

动作（3 维，每 0.05 s 一次，先限幅再累加到目标位姿）：
    dv      沿墙竖直移动（m/步）
    dn      法向压入 / 退出（m/步）
    dpitch  刀面俯仰变化（rad/步），>0 = 下边缘更贴墙

观测（7 维，全部是实机上能拿到的量：指令 + 力传感器 + 由关节读数正解的位置）：
    刀后缘高度（读数正解，名义墙面系）、指令法向位置、刀面俯仰角、力传感器读数
    （墙面硬接触力 + 材料挤压反力）、
    本步位移、行程进度、刀上剩料的时间估计（= 已走过的行程比例）

材料是降阶高度场模型（dummy_loop/wall/material.py），不是流体仿真；
机械臂、接触力、逆解、误差模型都沿用墙面仿真那一套。
"""
from dataclasses import dataclass, asdict, replace
import numpy as np
import mujoco

from .scene import SceneConfig, build_scene, wall_frame, tool_frame_matrix, joint_limits, blade_outline
from .controller import ActionSpec, EEController, WallFrame
from .errors import Perturbation
from .material import MaterialConfig, MortarField
from .task import TaskConfig

OBS_NAMES = ('edge_v_m', 'target_n_m', 'pitch_rad', 'force_N', 'dv_m', 'progress', 'load_time_frac')


@dataclass
class StrokeConfig:
    horizon: int = 60                 # 步；20 Hz → 3 s
    start_pitch_deg: float = 30.0     # 待命位「立起」的刀面俯仰（上缘离墙、下缘朝墙），每刀都从这里开始
    script_style: str = 'technique'   # 手写示教：'technique' = 斜着贴墙再逐渐放平；'flat' = 贴墙前就放平
    standoff: float = 0.025           # 待命位离墙距离 m
    park_standoff: float = 0.08       # 换条带时先退到离墙这么远再动腕：换支解时刀面会扫一大圈
    v_start_margin: float = 0.005     # 起刀点在工作区下边界之下多少
    v_stop_margin: float = 0.012      # 走出工作区上边界多少就结束
    force_window: tuple = (2.0, 15.0)
    abort_force: float = 40.0         # 超过就判定撞墙，结束并扣分
    max_dv: float = 0.008             # 每步最大竖直位移 m（0.16 m/s）
    max_dn: float = 0.002
    max_dpitch: float = 0.04
    # 奖励权重
    w_coverage: float = 10.0
    w_rms: float = 15.0
    w_waste: float = 5.0
    w_left: float = 2.0
    w_deposit: float = 4.0            # 每步抹上墙的料（按占初始带料量的比例）给的奖励——稀疏奖励下学不动，需要它
    w_waste_step: float = 4.0         # 每步抹到工作区外的料的惩罚
    w_force: float = 0.4              # 每步超出力窗口的惩罚（按 N 计）
    w_action: float = 0.02
    fail_penalty: float = 5.0

    def to_dict(self):
        d = asdict(self); d['force_window'] = list(self.force_window); return d


class StrokeEnv:
    """gym 风格接口：reset() -> obs；step(a) -> (obs, reward, done, info)。不依赖 gym。"""

    def __init__(self, scene: SceneConfig = None, task: TaskConfig = None, material: MaterialConfig = None,
                 stroke: StrokeConfig = None, perturbation: Perturbation = None, seed: int = 0):
        self.scene = scene or SceneConfig()
        # 一刀负责的条带：沿墙 12 cm（≈刀长）× 竖直 7 cm。整面墙由多刀拼成（plaster_session.py）
        self.task = task or TaskConfig(region_u=(-0.06, 0.06), region_v=(0.0, 0.07))
        self.mat_cfg = material or MaterialConfig()
        self.cfg = stroke or StrokeConfig()
        self.pert = perturbation or Perturbation()
        self.rng = np.random.default_rng(seed)
        self.spec = ActionSpec(max_du=1e-9, max_dv=self.cfg.max_dv, max_dn=self.cfg.max_dn,
                               max_dpsi=1e-9, max_dpitch=self.cfg.max_dpitch, with_pitch=True)
        self.true_scene = self.pert.true_scene(self.scene)
        self.model, _ = build_scene(self.true_scene)
        self.data = mujoco.MjData(self.model)
        self.n_sub = int(round(self.spec.dt / self.model.opt.timestep))
        nominal, _ = build_scene(self.scene)
        lo, hi = joint_limits(self.scene)
        # J6 固件限位是 ±720°，逆解可能给出绕好几圈的等价解（刀面姿态一样但要多转几百度）。
        # 抹涂用不到多圈，这里把 J6 收到 ±180°。
        lo, hi = np.array(lo, float), np.array(hi, float)
        lo[5], hi[5] = max(lo[5], -np.pi), min(hi[5], np.pi)
        # 抹涂手法任务锁住 roll（刀长保持水平），否则逆解会把刀面绕刀具轴转过去，下缘就不是整条贴墙
        self.ctrl = EEController(nominal, WallFrame(*wall_frame(self.scene)), lo, hi, self.spec,
                                 tool_R=tool_frame_matrix(self.scene), w_roll=0.1)
        self.true_frame = WallFrame(*wall_frame(self.true_scene))
        ids = lambda t, n: mujoco.mj_name2id(self.model, t, n)
        self.blade_body = ids(mujoco.mjtObj.mjOBJ_BODY, 'blade')
        self.wall_geom = ids(mujoco.mjtObj.mjOBJ_GEOM, 'wall_geom')
        self.blade_geoms = {g for g in (ids(mujoco.mjtObj.mjOBJ_GEOM, 'blade_geom'),
                                        ids(mujoco.mjtObj.mjOBJ_GEOM, 'blade_tip_geom')) if g >= 0}
        self.outline = blade_outline(self.scene)
        self.delta = np.asarray(self.pert.joint_offset_rad, float)
        self.field = MortarField(self.mat_cfg, self.task.region_u, self.task.region_v)
        self.shared_field = False        # 多刀作业时共用一块高度场（dummy_loop/wall/session.py）
        self.u_centre = float(np.mean(self.task.region_u))
        self.xc = (self.outline['x_tip'] + self.outline['x_back']) / 2
        self.obs_dim, self.act_dim = len(OBS_NAMES), 3

    # ------------------------------------------------------------------ 内部
    def _physics(self, q_cmd, external=None):
        lo, hi = self.ctrl.ik.lo, self.ctrl.ik.hi
        self.data.ctrl[:] = np.clip(q_cmd + self.delta, lo, hi)
        self.data.xfrc_applied[:] = 0
        if external is not None:
            self.data.xfrc_applied[self.blade_body, :3] = external
        for _ in range(self.n_sub):
            mujoco.mj_step(self.model, self.data)
        if not np.all(np.isfinite(self.data.qpos)):
            raise RuntimeError('non-finite simulation state')

    def contact_force(self):
        f = np.zeros(6); total = 0.0
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if self.wall_geom in (c.geom1, c.geom2) and ({c.geom1, c.geom2} & self.blade_geoms):
                mujoco.mj_contactForce(self.model, self.data, i, f); total += abs(f[0])
        return total

    def blade_pitch(self):
        """刀面真实俯仰角（rad）：刀宽方向偏离墙面的角度，>0 = 上缘离墙、下缘贴墙。"""
        R = self.data.xmat[self.blade_body].reshape(3, 3)
        return float(-np.arcsin(np.clip(R[:, 2] @ self.true_frame.R[:, 2], -1, 1)))

    def blade_edges(self):
        """返回刀面上下两条边中点在真实墙面系里的坐标（u, v, n）。「中点」取刀面沿刀长方向的中心。"""
        R = self.data.xmat[self.blade_body].reshape(3, 3)
        xc = (self.outline['x_tip'] + self.outline['x_back']) / 2
        p = self.data.xpos[self.blade_body] + R[:, 1] * self.scene.trowel_thickness + R[:, 0] * xc
        hw = self.outline['half_width']
        low = self.true_frame.from_world(p - hw * R[:, 2])
        high = self.true_frame.from_world(p + hw * R[:, 2])
        return low, high

    def _material_step(self, moving_up):
        low, high = self.blade_edges()
        edge = low if moving_up else high
        gap = max(-float(edge[2]), 0.0)
        half = (self.outline['x_tip'] - self.outline['x_back']) / 2
        removed, given, waste = self.field.sweep(float(edge[1]), gap, half, float(edge[0]), moving_up)
        # 刀上的料兜不兜得住（看下缘间隙与俯仰），兜不住的掉落，也算浪费
        dropped = self.field.carry(self.blade_pitch(), max(-float(low[2]), 0.0), 2 * half, 2 * self.outline['half_width'])
        waste = waste + dropped
        # 材料反作用力：被刮动的体积换算成接触面积，按屈服应力给法向推力 + 反向拖曳
        area = (removed / max(self.mat_cfg.deposit_rate, 1e-9)) if removed > 0 else 0.0
        fn, fd = self.field.resistance(min(area, 0.01), None)
        Rw = self.true_frame.R
        force = -fn * Rw[:, 2]                                  # 沿墙法线往外推
        if moving_up:
            force = force - fd * Rw[:, 1]
        else:
            force = force + fd * Rw[:, 1]
        return force, fn, removed, given, waste

    def _observe(self, force, dv):
        q_meas = self.data.qpos[:6] - self.delta
        if self.pert.q_noise_rad:
            q_meas = q_meas + self.rng.normal(0, self.pert.q_noise_rad, 6)
        p, R = self.ctrl.ik.fk(q_meas)                           # 读数正解的刀面位姿（名义墙面系）
        hw = self.outline['half_width']
        edge = self.ctrl.frame.from_world(p - hw * R[:, 2])
        f = force + (self.rng.normal(0, self.pert.force_noise_N) if self.pert.force_noise_N else 0.0)
        progress = (self.step_i + 1) / self.cfg.horizon
        return np.array([edge[1] - self.task.region_v[0], self.ctrl.target[2], self.ctrl.pitch, f / 10.0, dv, progress,
                         self.travelled / max(self.v_span, 1e-9)])

    def tcp_u(self):
        """让刀面中心对准条带中心时 TCP（J6 轴线）应在的 u：psi=0 时刀长方向指向 -u。"""
        return self.u_centre + self.xc

    def stroke_load(self):
        """一刀该带多少料：这一条带的目标体积 × initial_load_scale。"""
        v0, v1 = self.task.region_v
        length = self.outline['x_tip'] - self.outline['x_back']
        return length * (v1 - v0) * self.mat_cfg.target_thickness * self.mat_cfg.initial_load_scale

    def set_band(self, u_centre, band_v, field=None, reload_load=True):
        """切到下一条带（多刀作业用）：换条带中心与竖直范围，可共用同一块高度场。"""
        self.u_centre = float(u_centre)
        self.task = replace(self.task, region_v=tuple(band_v))
        if field is not None:
            self.field = field; self.shared_field = True
        self.q_ready = None
        return self

    def go_ready(self, start, pitch0, transit_steps=25):
        """换条带：抬刀 → 退到停放距离 → 横移 → 回到「立起」待命位。

        腕部支解可能要换（不同条带的自然解不同），换支解时刀面会扫一大圈，所以一定在停放距离上做。
        每一步都检查逆解是否收敛，不收敛就换支解，避免出现「机械臂停在原地、每步都报不可达」的情况。
        """
        c = self.cfg
        park = [start[0], start[1], -c.park_standoff]
        self.transit_to([self.ctrl.target[0], self.ctrl.target[1], -c.park_standoff],
                        np.deg2rad(c.start_pitch_deg), steps=12)
        _, ok = self.transit_to(park, pitch0, steps=transit_steps)
        if not ok or not self.branch_can_stroke(start, pitch0, self.ctrl.q_cmd):
            self.switch_branch(park, pitch0, transit_steps)
        _, ok = self.transit_to(start, pitch0, steps=12)
        if not ok or not self.branch_can_stroke(start, pitch0, self.ctrl.q_cmd):
            self.switch_branch(start, pitch0, transit_steps)
        return self.ctrl.q_cmd

    def switch_branch(self, pose, pitch, steps=25):
        """用多种子逆解换一支解，并按限速在关节空间走过去。够不到就原样返回。"""
        q_from = self.ctrl.q_cmd.copy()
        try:
            q_to = self.ctrl.reset(pose, 0.0, np.zeros(6), pitch=pitch)
        except ValueError:
            self.ctrl.set_state(self.ctrl.target, 0.0, q_from, pitch=self.ctrl.pitch)
            return 0
        n = max(int(np.ceil(np.max(np.abs(q_to - q_from)) / max(self.ctrl.max_joint_step, 1e-9))), steps)
        for i in range(1, n + 1):
            self._physics(q_from + (q_to - q_from) * (i / n))
        self.ctrl.set_state(pose, 0.0, q_to, pitch=pitch)
        return n

    def branch_can_stroke(self, start, pitch0, q_seed):
        """这支逆解能不能把一刀走完：沿「贴墙 → 走到条带顶 → 放平」逐点延续求解，中间断了就算不行。"""
        c, hw = self.cfg, self.outline['half_width']
        v_top = self.task.region_v[1] + c.v_stop_margin + hw
        way = [(start[1], -0.004, pitch0),
               ((start[1] + v_top) / 2, -0.004, (pitch0 + np.deg2rad(5)) / 2),
               (v_top, -0.004, np.deg2rad(5.0))]
        q = np.asarray(q_seed, float).copy()
        for v, n, pitch in way:
            q, ok, _, _ = self.ctrl.ik.solve(self.ctrl.frame.to_world([start[0], v, n]),
                                             self.ctrl.frame.tool_rotation(0.0, pitch), q, iters=80)
            if not ok:
                return False
        return True

    def transit_to(self, target_uvn, pitch, steps=25):
        """沿直线把刀具移到某个位姿（带与带之间的回位 / 横移）：任务空间插值，逐点求逆解。

        逐点求解（每一点都从上一点的关节角出发）有两个理由：一是保证解连续，换支解会让 J6 转一百多度；
        二是路径可预期，不会从墙里穿过去。发出的关节目标按控制器的限速执行，不是瞬移。
        """
        start_t = self.ctrl.target.copy(); start_p = self.ctrl.pitch
        goal_t = np.asarray(target_uvn, float)
        q = self.ctrl.q_cmd.copy()
        n = max(int(steps), 1)
        ok_all = True
        for i in range(1, n + 1):
            f = i / n
            t = start_t + (goal_t - start_t) * f
            p = start_p + (pitch - start_p) * f
            q_next, ok, _, _ = self.ctrl.ik.solve(self.ctrl.frame.to_world(t),
                                                  self.ctrl.frame.tool_rotation(0.0, p), q, iters=60)
            ok_all = ok_all and ok
            step_cap = self.ctrl.max_joint_step
            d = q_next - q
            scale = min(1.0, step_cap / max(np.abs(d).max(), 1e-12))
            q = q + scale * d
            self._physics(q)
        self.ctrl.set_state(goal_t, 0.0, q, pitch=pitch)
        return n, ok_all

    # ------------------------------------------------------------------ 接口
    def reset(self, transit=False, transit_steps=25, reload_steps=10):
        """回到「立起」待命位（刀面斜着、下缘朝墙、离墙 standoff），上一份新料。

        transit=False：直接把机械臂放到待命位（训练时每个 episode 这样开始，省时间）。
        transit=True：从当前位姿按限速走回待命位，再停 reload_steps 步「等新料上刀」（多刀作业用）。
        """
        c, t = self.cfg, self.task
        v_start = t.region_v[0] - c.v_start_margin
        # TCP 在刀面中心；让下缘位于起刀高度
        start = [self.tcp_u(), v_start + self.outline['half_width'], -c.standoff]
        pitch0 = np.deg2rad(c.start_pitch_deg)
        self.stats = {'over_force_steps': 0, 'max_force_N': 0.0, 'unreachable_steps': 0, 'aborted': False}
        if transit:
            self.go_ready(start, pitch0, transit_steps)
            for _ in range(reload_steps):                # 停在待命位等新料上刀
                self._physics(self.ctrl.q_cmd)
        else:
            if getattr(self, 'q_ready', None) is None:   # 待命位的逆解只算一次，之后直接复用
                self.q_ready = self.ctrl.reset(start, 0.0, np.zeros(6), pitch=pitch0)
            else:
                self.ctrl.set_state(start, 0.0, self.q_ready, pitch=pitch0)
            q0 = self.q_ready
            mujoco.mj_resetData(self.model, self.data)
            self.data.qpos[:6] = q0 + self.delta
            self.data.ctrl[:] = q0 + self.delta
            mujoco.mj_forward(self.model, self.data)
            for _ in range(5):
                self._physics(self.ctrl.q_cmd)
        if self.shared_field:
            self.field.reload(self.stroke_load())        # 上料：刀上加一份新料，墙面保留上一刀抹的
        else:
            self.field.reset()
        self.step_i = 0; self.travelled = 0.0
        self.v_span = (t.region_v[1] + c.v_stop_margin) - v_start
        self.last_v = float(self.blade_edges()[0][1]); self.last_force = 0.0
        self.field.last_edge = None
        return self._observe(self.contact_force(), 0.0)

    def step(self, action):
        a = np.asarray(action, float)
        if a.shape != (3,) or not np.all(np.isfinite(a)):
            raise ValueError('action must be 3 finite numbers (dv, dn, dpitch)')
        full = np.array([0.0, a[0], a[1], 0.0, a[2]])
        q_cmd, status, applied = self.ctrl.step(full)
        moving_up = applied[1] >= 0
        force_ext, f_mat, removed, given, waste = self._material_step(moving_up)
        self._physics(q_cmd, external=force_ext)
        # 力传感器读的是刀面受到的法向力：墙面硬接触 + 材料的挤压反力
        F = self.contact_force() + f_mat
        self.last_force = F
        v_now = float(self.blade_edges()[0][1])
        dv = v_now - self.last_v; self.last_v = v_now
        self.travelled += max(dv, 0.0)
        s, c = self.stats, self.cfg
        s['max_force_N'] = max(s['max_force_N'], F)
        s['unreachable_steps'] += status == 'unreachable'
        over = max(0.0, F - c.force_window[1])
        load0 = max(self.field.load0, 1e-12)
        r = (c.w_deposit * (given - waste) / load0 - c.w_waste_step * waste / load0
             - c.w_force * over * self.spec.dt
             - c.w_action * float(np.sum(np.abs(a) / self.spec.limits[[1, 2, 4]])))
        if over > 0:
            s['over_force_steps'] += 1
        self.step_i += 1
        done = False
        if F > c.abort_force:
            s['aborted'] = True; done = True; r -= c.fail_penalty
        elif v_now > self.task.region_v[1] + c.v_stop_margin or self.step_i >= c.horizon:
            done = True
        if done:
            r += self.terminal_reward()
        return self._observe(F, dv), float(r), bool(done), {'status': status, 'force_N': F,
                                                            'deposited_m3': given, 'scraped_m3': removed,
                                                            'wasted_m3': waste}

    def terminal_reward(self):
        c, m = self.cfg, self.field.metrics()
        rms = min(m['rms_error_mm'] / (self.mat_cfg.target_thickness * 1000), 2.0)
        return (c.w_coverage * m['coverage'] - c.w_rms * rms
                - c.w_waste * m['wasted_frac'] - c.w_left * m['left_on_tool_frac'])

    def metrics(self):
        m = dict(self.field.metrics())
        m.update({k: (round(v, 3) if isinstance(v, float) else v) for k, v in self.stats.items()})
        m['terminal_reward'] = round(self.terminal_reward(), 3)
        return m

    def describe(self):
        return {'obs_names': list(OBS_NAMES), 'action_names': ['dv', 'dn', 'dpitch'],
                'action_limits': self.spec.limits[[1, 2, 4]].tolist(), 'dt_s': self.spec.dt,
                'stroke': self.cfg.to_dict(), 'material': self.mat_cfg.to_dict(),
                'scene_digest': self.scene.digest(), 'task': self.task.to_dict(),
                'perturbation': self.pert.to_dict(),
                'evidence_level': 'L1', 'hardware_motion': False,
                'material_model': 'reduced-order height field, NOT a fluid simulation; coefficients are assumptions'}

    # ------------------------------------------------------------------ 脚本基线
    def scripted_action(self):
        """手写示教（只用实机拿得到的量：指令位姿 + 行程）。

        style='technique'（工人手法）：先让**下缘**贴到离墙一个料层厚的位置，再一边上行一边把刀面放平；
        style='flat'：贴墙前就把刀面放平（不用手法），用来检验强化学习能不能自己找回手法。
        """
        c, hw = self.cfg, self.outline['half_width']
        frac = np.clip(self.travelled / max(self.v_span, 1e-9), 0, 1)
        pitch0 = np.deg2rad(c.start_pitch_deg)
        # 贴墙之前不放平：先让下缘吃住墙（flat 风格则相反，先放平再贴墙）
        n_now = self.ctrl.target[2] + hw * np.sin(max(self.ctrl.pitch, 0.0))    # 指令下缘位置
        touched = n_now > -(self.mat_cfg.target_thickness + 0.002)
        if c.script_style == 'flat':
            pitch_goal = 0.0
        elif not touched:
            pitch_goal = pitch0
        else:
            pitch_goal = pitch0 * (1 - frac) + np.deg2rad(5.0) * frac
        dpitch = np.clip(pitch_goal - self.ctrl.pitch, -c.max_dpitch, c.max_dpitch)
        pitch_next = self.ctrl.pitch + dpitch
        # 目标：下缘离墙 = 一个料层厚。TCP 在刀面中心，所以法向目标要加上半刀宽的俯仰投影。
        n_goal = -(self.mat_cfg.target_thickness + hw * np.sin(max(pitch_next, 0.0)))
        dn = 0.35 * (n_goal - self.ctrl.target[2]) - 0.0004 * max(self.last_force - 5.0, 0.0)
        dn = np.clip(dn, -c.max_dn, c.max_dn)
        dv = 0.0 if not touched else c.max_dv * 0.7
        return np.array([dv, dn, dpitch])


def rollout(env: StrokeEnv, policy=None, record=None):
    """跑一个 episode。policy(obs)->action；None 表示用脚本基线。返回 (总回报, 步数, 指标)。"""
    obs = env.reset(); total, steps, done = 0.0, 0, False
    while not done:
        a = env.scripted_action() if policy is None else policy(obs)
        nxt, r, done, info = env.step(a)
        if record is not None:
            record(obs, a, r, info)
        obs = nxt; total += r; steps += 1
    return total, steps, env.metrics()

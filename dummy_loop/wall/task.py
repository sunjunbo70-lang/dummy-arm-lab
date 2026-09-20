"""墙面抹涂覆盖任务。

一次 episode：从墙前的待机位出发，刀面压到墙上，把目标区域「抹一遍」，退回。
评价：目标区域里有多少格被刀面在合适压力下经过（覆盖率），压力是否始终在窗口内，
有没有把弹簧压到底（刚性撞墙），有没有不可达、限速。

观测分成两类，写进 OBS_SPEC：
  hardware   实机上也能拿到（关节读数、由读数正解的 TCP、压缩量——需要装位移传感器）
  privileged 只有仿真知道（真实接触力、真实 TCP、覆盖图）。只能用于评价和调试，
             不得作为策略输入，否则部署到实机时这些输入不存在。
"""
from dataclasses import dataclass, asdict
import time
import numpy as np
import mujoco

from .scene import SceneConfig, build_scene, wall_frame, tool_frame_matrix, joint_limits
from .controller import ActionSpec, EEController, WallFrame
from .errors import Perturbation
from .layout import NATURAL_PLANE_U

_OBS_COMMON = {
    'q_meas_rad':         {'shape': [6], 'units': 'rad', 'availability': 'hardware', 'note': '编码器读数（含零位误差与噪声）'},
    'tcp_belief_uvn_m':   {'shape': [3], 'units': 'm', 'availability': 'hardware', 'note': '读数正解到名义墙面系'},
    'target_uvn_m':       {'shape': [3], 'units': 'm', 'availability': 'hardware', 'note': '控制器当前目标（名义墙面系）'},
    'contact_force_N':    {'shape': [1], 'units': 'N', 'availability': 'privileged', 'note': '刀面-墙面法向接触力合计'},
    'tcp_true_uvn_m':     {'shape': [3], 'units': 'm', 'availability': 'privileged', 'note': '真实 TCP，真实墙面系'},
}
_TOOL_SENSOR = {
    'rigid': ('tool_force_N', {'shape': [1], 'units': 'N', 'availability': 'hardware_with_sensor',
                               'note': '法兰与抹刀座之间的力传感器（沿刀具轴）；实机需加装。仿真 = 刀面法向接触力 + 噪声'}),
    'spring': ('compression_m', {'shape': [1], 'units': 'm', 'availability': 'hardware_with_sensor',
                                 'note': '弹簧压缩量；实机需加装直线位移/霍尔传感器'}),
}


def obs_spec(tool_mount='rigid'):
    key, spec = _TOOL_SENSOR[tool_mount]
    d = dict(_OBS_COMMON); d[key] = spec
    return d


OBS_SPEC = obs_spec('rigid')     # 默认（实际）工具的观测表


@dataclass
class TaskConfig:
    region_u: tuple = (NATURAL_PLANE_U - 0.06, NATURAL_PLANE_U + 0.06)   # 真实墙面系
    region_v: tuple = (-0.05, 0.05)
    cell: float = 0.005
    force_window: tuple = (2.0, 15.0)      # N：低于下限视为没压实，高于上限视为过压
    target_compression: float = 0.012      # m：弹簧方案的设计压缩量 → 名义压力 = 预紧 + k × 压缩
    target_force: float = 6.0              # N：刚性方案的目标压力（力传感器闭环时）
    press_depth: float = 0.001             # m：刚性方案不闭环时，指令刀面「压进」名义墙面的深度。
                                           #    实际压力 = 伺服刚度 × 该深度，完全取决于未辨识的关节刚度
    touch_force: float = 2.0               # N：刚性方案探触时判定接触的力阈值（连续 2 次超过才算，防噪声误触发）
    standoff: float = 0.03                 # m：待机位离墙距离
    settle_steps: int = 10

    def to_dict(self):
        d = asdict(self); d['region_u'] = list(self.region_u); d['region_v'] = list(self.region_v)
        d['force_window'] = list(self.force_window); return d


class WallTask:
    def __init__(self, scene: SceneConfig = None, task: TaskConfig = None, perturbation: Perturbation = None,
                 seed: int = 0, action_spec: ActionSpec = ActionSpec()):
        self.scene = scene or SceneConfig(); self.task = task or TaskConfig()
        self.pert = perturbation or Perturbation(); self.spec = action_spec
        self.rng = np.random.default_rng(seed)
        # 真实世界
        self.true_scene = self.pert.true_scene(self.scene)
        self.model, _ = build_scene(self.true_scene)
        self.data = mujoco.MjData(self.model)
        self.n_sub = int(round(action_spec.dt / self.model.opt.timestep))
        # 控制器只知道名义场景
        nominal_model, _ = build_scene(self.scene)
        self.lo, self.hi = joint_limits(self.scene)
        self.ctrl = EEController(nominal_model, WallFrame(*wall_frame(self.scene)), self.lo, self.hi,
                                 action_spec, tool_R=tool_frame_matrix(self.scene))
        self.true_frame = WallFrame(*wall_frame(self.true_scene))
        ids = lambda t, n: mujoco.mj_name2id(self.model, t, n)
        self.blade_geom = ids(mujoco.mjtObj.mjOBJ_GEOM, 'blade_geom')
        self.blade_geoms = {g for g in (self.blade_geom, ids(mujoco.mjtObj.mjOBJ_GEOM, 'blade_tip_geom')) if g >= 0}
        self.wall_geom = ids(mujoco.mjtObj.mjOBJ_GEOM, 'wall_geom')
        self.tcp_site = ids(mujoco.mjtObj.mjOBJ_SITE, 'tcp')
        self.blade_body = ids(mujoco.mjtObj.mjOBJ_BODY, 'blade')
        self.rigid = self.scene.tool_mount == 'rigid'
        self.sensor_key = _TOOL_SENSOR[self.scene.tool_mount][0]
        self.obs_spec = obs_spec(self.scene.tool_mount)
        self.comp_adr = None if self.rigid else self.model.jnt_qposadr[ids(mujoco.mjtObj.mjOBJ_JOINT, 'compliance')]
        self.delta = np.asarray(self.pert.joint_offset_rad, float)
        t = self.task
        self.nu_cells = int(round((t.region_u[1] - t.region_u[0]) / t.cell))
        self.nv_cells = int(round((t.region_v[1] - t.region_v[0]) / t.cell))
        cu = t.region_u[0] + (np.arange(self.nu_cells) + 0.5) * t.cell
        cv = t.region_v[0] + (np.arange(self.nv_cells) + 0.5) * t.cell
        self.cell_u, self.cell_v = np.meshgrid(cu, cv)
        # 格子中心在世界系中的位置（真实墙面上）
        self.cell_world = np.stack([self.true_frame.to_world([u, v, 0.0]) for u, v in
                                    zip(self.cell_u.ravel(), self.cell_v.ravel())]).reshape(*self.cell_u.shape, 3)
        from .scene import blade_outline
        self.outline = blade_outline(self.scene)

    # ---------------------------------------------------------------- 接口
    def reset(self, start_uv=None, psi=0.0):
        t = self.task
        start_uv = (t.region_u[0] + 0.04, t.region_v[0]) if start_uv is None else start_uv
        q0 = self.ctrl.reset([start_uv[0], start_uv[1], -t.standoff], psi, np.zeros(6))
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:6] = q0 + self.delta
        self.data.ctrl[:] = q0 + self.delta
        mujoco.mj_forward(self.model, self.data)
        self.queue = [np.zeros(4)] * self.pert.latency_steps
        self.covered = np.zeros(self.cell_u.shape, bool)
        self.overforced = np.zeros(self.cell_u.shape, bool)
        self.seq = 0; self.t0_host = time.monotonic()
        self.stats = {'steps': 0, 'unreachable': 0, 'rate_limited': 0, 'contact_steps': 0, 'in_window_steps': 0,
                      'over_steps': 0, 'bottom_out_steps': 0, 'max_force_N': 0.0, 'force_sum': 0.0}
        for _ in range(t.settle_steps):
            self._physics(self.ctrl.q_cmd)
        return self.observe()

    def step(self, action):
        self.queue.append(np.asarray(action, float))
        a = self.queue.pop(0)
        q_cmd, status, applied = self.ctrl.step(a)
        self._physics(q_cmd)
        obs = self.observe()
        self._account(obs, status)
        self.seq += 1
        return obs, {'status': status, 'applied_action': applied, 'q_cmd': q_cmd}

    # ---------------------------------------------------------------- 内部
    def _physics(self, q_cmd):
        self.data.ctrl[:] = np.clip(q_cmd + self.delta, self.lo, self.hi)
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

    def observe(self):
        d = self.data
        q_true = d.qpos[:6].copy()
        q_meas = q_true - self.delta + self.rng.normal(0, self.pert.q_noise_rad, 6) if self.pert.q_noise_rad else q_true - self.delta
        F = self.contact_force()
        if self.rigid:
            comp = 0.0
            sensor = F + (self.rng.normal(0, self.pert.force_noise_N) if self.pert.force_noise_N else 0.0)
        else:
            comp = float(d.qpos[self.comp_adr])
            sensor = comp + (self.rng.normal(0, self.pert.compression_noise_m) if self.pert.compression_noise_m else 0.0)
        tcp_belief = self.ctrl.frame.from_world(self.ctrl.ik.fk(q_meas)[0])
        tcp_true = self.true_frame.from_world(d.site_xpos[self.tcp_site])
        return {'seq': self.seq, 't_sample_s': float(d.time), 't_host_s': time.monotonic() - self.t0_host,
                'q_meas_rad': q_meas, 'tcp_belief_uvn_m': tcp_belief, 'target_uvn_m': self.ctrl.target.copy(),
                self.sensor_key: np.array([sensor]), 'contact_force_N': np.array([F]),
                'tcp_true_uvn_m': tcp_true, '_compression_true': comp}

    def _account(self, obs, status):
        s = self.stats; t = self.task
        s['steps'] += 1; s['unreachable'] += status == 'unreachable'; s['rate_limited'] += status == 'rate_limited'
        F = float(obs['contact_force_N'][0])
        if not self.rigid and obs['_compression_true'] >= self.true_scene.spring_travel - 1e-4:
            s['bottom_out_steps'] += 1
        if F <= 0.05:
            return
        s['contact_steps'] += 1; s['force_sum'] += F; s['max_force_N'] = max(s['max_force_N'], F)
        # 刀面实际轮廓（刚性方案为尖头抹刀：矩形 + 三角尖），在刀面系里判断每个格子是否被覆盖
        R6 = self.data.xmat[self.blade_body].reshape(3, 3)
        o = self.outline
        rel_u = self.cell_world - self.data.xpos[self.blade_body]
        x = rel_u @ R6[:, 0]; z = np.abs(rel_u @ R6[:, 2])
        rect = (x >= o['x_back']) & (x <= o['x_taper']) & (z <= o['half_width'])
        span = max(o['x_tip'] - o['x_taper'], 1e-9)
        tipz = o['half_width'] * np.clip((o['x_tip'] - x) / span, 0, 1)
        tip = (x > o['x_taper']) & (x <= o['x_tip']) & (z <= tipz)
        inside = rect | tip
        if t.force_window[0] <= F <= t.force_window[1]:
            s['in_window_steps'] += 1; self.covered |= inside
        elif F > t.force_window[1]:
            s['over_steps'] += 1; self.overforced |= inside

    def metrics(self):
        s = dict(self.stats)
        cs = max(s['contact_steps'], 1)
        return {'coverage': round(float(self.covered.mean()), 4),
                'overforced_area_frac': round(float(self.overforced.mean()), 4),
                'in_window_frac_of_contact': round(s['in_window_steps'] / cs, 4),
                'mean_contact_force_N': round(s['force_sum'] / cs, 3),
                'max_contact_force_N': round(s['max_force_N'], 3),
                'bottom_out_steps': s['bottom_out_steps'], 'unreachable_steps': s['unreachable'],
                'rate_limited_steps': s['rate_limited'], 'contact_steps': s['contact_steps'], 'steps': s['steps']}

    def describe(self):
        import hashlib
        from .scene import ARM_MODEL
        return {'arm_model': ARM_MODEL.relative_to(ARM_MODEL.parents[1]).as_posix(),
                'arm_model_sha256': hashlib.sha256(ARM_MODEL.read_bytes()).hexdigest(),
                'joint_convention': 'model q = deg2rad(firmware_deg - HOME[0,0,90,0,0,0]), firmware direction',
                'scene_nominal': self.scene.to_dict(), 'scene_nominal_digest': self.scene.digest(),
                'scene_provenance': self.scene.provenance(), 'task': self.task.to_dict(),
                'perturbation': self.pert.to_dict(), 'action_spec': self.spec.to_dict(), 'obs_spec': self.obs_spec,
                'tool_mount': self.scene.tool_mount,
                'nominal_force_at_target_N': (self.task.target_force if self.rigid else
                                              round(self.scene.spring_preload + self.scene.spring_k * self.task.target_compression, 2))}

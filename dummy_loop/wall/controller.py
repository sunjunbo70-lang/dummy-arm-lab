"""末端动作空间与微分逆运动学控制器。

动作空间约定见 docs/ACTION_SPACE.md。要点：
  - 动作表达在「墙面坐标系」里：u 沿墙水平、v 沿墙向上、n 指向墙内，单位 m；
    psi 是刀面绕刀具轴的转角，单位 rad。
  - 动作是每个控制周期（默认 0.05 s = 20 Hz）的增量，先限幅再累加到目标位姿。
  - 刀具轴始终保持与墙面法线平行（硬约束，不是动作的一部分）。
  - 控制器只知道「名义」墙面位姿。真实墙面与名义不符是误差来源之一，由补偿层处理。
  - 控制器在内部维护指令关节角 q_cmd，逆解从 q_cmd 出发，不从测量值出发——
    与实机位置模式一致：实机只接受目标角，跟随误差、下垂不会被这一层自动吃掉。
"""
from dataclasses import dataclass
import numpy as np
import mujoco


@dataclass(frozen=True)
class ActionSpec:
    dt: float = 0.05
    max_du: float = 0.010     # m / 步 → 0.2 m/s
    max_dv: float = 0.010
    max_dn: float = 0.004     # m / 步 → 0.08 m/s，法向更保守
    max_dpsi: float = 0.05    # rad / 步
    names: tuple = ('du', 'dv', 'dn', 'dpsi')
    units: tuple = ('m', 'm', 'm', 'rad')

    @property
    def limits(self):
        return np.array([self.max_du, self.max_dv, self.max_dn, self.max_dpsi])

    def clip(self, a):
        a = np.asarray(a, float)
        if a.shape != (4,) or not np.all(np.isfinite(a)):
            raise ValueError('action must be 4 finite numbers (du, dv, dn, dpsi)')
        return np.clip(a, -self.limits, self.limits)

    def to_dict(self):
        return {'frame': 'wall (u along wall, v up the wall, n into wall)', 'names': list(self.names),
                'units': list(self.units), 'per_step_limits': self.limits.tolist(), 'dt_s': self.dt,
                'kind': 'delta target pose, clipped then accumulated'}


class WallFrame:
    def __init__(self, origin, R):
        self.origin = np.asarray(origin, float); self.R = np.asarray(R, float)

    def to_world(self, p):
        return self.origin + self.R @ np.asarray(p, float)

    def from_world(self, x):
        return self.R.T @ (np.asarray(x, float) - self.origin)

    def tool_rotation(self, psi):
        """刀面目标姿态：列 = (刀宽方向, 刀面法线=墙法线, 两者叉积)，对应刀面系 (x, y, z)。

        psi=0 对应模型零位时 J6 的自然朝向（link6 局部 x 指向 -u）。刀宽仍沿 u 方向，
        只是正负号与零位一致，这样 psi=0 不需要 J6 转 180°。
        """
        u, v, n = self.R[:, 0], self.R[:, 1], self.R[:, 2]
        xw = -(np.cos(psi) * u + np.sin(psi) * v)
        return np.column_stack([xw, n, np.cross(xw, n)])

    def copy(self):
        return WallFrame(self.origin.copy(), self.R.copy())


class ToolIK:
    """刀面位姿逆解：位置 3 维 + 刀面法线 2 维为主约束，刀面绕法线的转角（roll）只做弱约束。

    为什么 roll 只做弱约束：参考模型分析（docs/SIMULATION.md「腕部奇异」）表明，
    同时锁死刀面法线和 roll 对这台臂是 6 约束对 6 关节，在 ±90° 关节范围内沿墙水平移动
    会逼 J4/J6 拧到 ±90° 附近并频繁不可解。放开 roll 后多出 1 个冗余自由度，
    用来远离腕部奇异和关节限位。抹涂对刀面 roll 不敏感，覆盖计算使用实际 roll。

    阻尼最小二乘，关节代价加权，关节限位投影。使用独立 MjData，不扰动仿真状态；
    弹簧滑轨固定在 0（未压缩）。
    """
    W_TILT = 0.1     # 1 rad 法线偏差折合 0.1 m 位置误差
    W_ROLL = 0.01    # roll 权重是法线的 1/10：尽量保持，但让位于可达性
    # 关节运动代价：J1、J4、J6（绕自身轴转的关节）更贵，优先用肩、肘、腕俯仰。
    JOINT_COST = np.array([2.0, 1.0, 1.0, 4.0, 1.0, 4.0])

    def __init__(self, model, lo, hi, tool_R=None):
        self.m = model; self.d = mujoco.MjData(model)
        self.tool_R = np.eye(3) if tool_R is None else np.asarray(tool_R, float)   # 刀面系相对 link6
        self.site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'tcp')
        self.body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'link6_1_1')
        self.lo, self.hi = np.asarray(lo, float), np.asarray(hi, float)
        self.jp = np.zeros((3, model.nv)); self.jr = np.zeros((3, model.nv))

    def fk(self, q):
        """返回 (TCP 位置, 刀面系姿态)。"""
        self.d.qpos[:6] = q; self.d.qpos[6:] = 0
        mujoco.mj_kinematics(self.m, self.d); mujoco.mj_comPos(self.m, self.d)
        return self.d.site_xpos[self.site].copy(), self.d.xmat[self.body].reshape(3, 3) @ self.tool_R

    def error(self, q, p_t, R_t):
        """位置误差、法线偏差（世界系旋转向量，垂直于法线的分量）、roll 偏差（沿法线分量）。"""
        p, R = self.fk(q)
        e = np.zeros(3); q1 = np.zeros(4); q2 = np.zeros(4)
        mujoco.mju_mat2Quat(q1, R_t.flatten()); mujoco.mju_mat2Quat(q2, R.flatten())
        mujoco.mju_subQuat(e, q1, q2)          # 当前刀面系下的旋转向量
        e = R @ e                              # 转到世界系
        n = R_t[:, 1]
        e_roll = (e @ n) * n
        return p_t - p, e - e_roll, e_roll

    def solve(self, p_t, R_t, q0, iters=30, tol_pos=2e-4, tol_tilt=2e-3, cost=None):
        cost = self.JOINT_COST if cost is None else np.asarray(cost, float)
        q = np.clip(np.asarray(q0, float), self.lo, self.hi)
        Wi = 1.0 / cost
        for _ in range(iters):
            ep, et, er = self.error(q, p_t, R_t)
            if np.linalg.norm(ep) < tol_pos and np.linalg.norm(et) < tol_tilt:
                break
            mujoco.mj_jacSite(self.m, self.d, self.jp, self.jr, self.site)
            J = np.r_[self.jp[:, :6], self.W_TILT * self.jr[:, :6]]
            e = np.r_[ep, self.W_TILT * et + self.W_ROLL * er]
            # roll 方向的角速度权重调低：对 J 的角速度行按 (法线方向降权) 投影
            n = R_t[:, 1]; P = np.eye(3) - (1 - self.W_ROLL / self.W_TILT) * np.outer(n, n)
            J[3:] = P @ J[3:]
            JW = J * Wi
            dq = Wi * (J.T @ np.linalg.solve(JW @ J.T + 1e-4 * np.eye(6), e))
            q = np.clip(q + dq, self.lo, self.hi)
        ep, et, er = self.error(q, p_t, R_t)
        ok = np.linalg.norm(ep) < 2e-3 and np.linalg.norm(et) < np.deg2rad(1.5)
        return q, bool(ok), float(np.linalg.norm(ep)), float(np.linalg.norm(et))


class EEController:
    """把墙面系增量动作变成关节目标角。只依赖名义墙面位姿与指令关节角。"""
    NATURAL_COST = np.array([30.0, 1.0, 1.0, 60.0, 1.0, 60.0])

    def __init__(self, model, frame: WallFrame, lo, hi, spec: ActionSpec = ActionSpec(), tool_R=None,
                 max_joint_speed=np.deg2rad(60)):
        self.ik = ToolIK(model, lo, hi, tool_R); self.frame = frame; self.spec = spec
        self.max_joint_step = max_joint_speed * spec.dt   # 关节速度限制，模拟实机限速
        self.target = np.zeros(3); self.psi = 0.0; self.q_cmd = None
        self.last_ok = True

    def reset(self, target_uvn, psi, q_seed):
        """解到初始目标；从多个种子尝试，失败则报错（初始位姿不可达）。"""
        self.target = np.asarray(target_uvn, float).copy(); self.psi = float(psi)
        p, R = self.frame.to_world(self.target), self.frame.tool_rotation(self.psi)
        rng = np.random.default_rng(0)
        seeds = [np.asarray(q_seed, float)] + [self.ik.lo + (self.ik.hi - self.ik.lo) * rng.random(6) for _ in range(20)]
        best = None
        for s in seeds:
            # 先把绕自身轴转的关节（J1/J4/J6）锁得很贵，找「不拧腕」的自然分支，再正常精修。
            q1, _, _, _ = self.ik.solve(p, R, s, iters=300, cost=self.NATURAL_COST)
            q, ok, _, _ = self.ik.solve(p, R, q1, iters=300)
            if ok:
                twist = float(np.max(np.abs(q[[0, 3, 5]])))
                if best is None or twist < best[0]:
                    best = (twist, q)
                if twist < np.deg2rad(20):
                    break
        if best is not None:
            self.q_cmd = best[1]; return best[1].copy()
        raise ValueError(f'initial target {self.target} (wall frame) is not reachable with square-to-wall tool')

    def step(self, action):
        """返回 (q_cmd, status, 实际采用的动作)。status: 'ok' | 'rate_limited' | 'unreachable'。

        rate_limited：逆解需要的关节变化超过速度限制，本周期只走到限速处，
        目标保留，后续周期继续追。unreachable：目标不可达，撤回本次增量。
        """
        a = self.spec.clip(action)
        prev_t, prev_psi = self.target.copy(), self.psi
        self.target = self.target + a[:3]; self.psi += a[3]
        q, ok, ep, et = self.ik.solve(self.frame.to_world(self.target), self.frame.tool_rotation(self.psi), self.q_cmd)
        if not ok:
            self.target, self.psi = prev_t, prev_psi
            self.last_ok = False
            return self.q_cmd.copy(), 'unreachable', np.zeros(4)
        dq = q - self.q_cmd
        scale = min(1.0, self.max_joint_step / max(np.abs(dq).max(), 1e-12))
        self.q_cmd = self.q_cmd + scale * dq
        self.last_ok = True
        return self.q_cmd.copy(), ('ok' if scale >= 1.0 else 'rate_limited'), a

    def commanded_tcp_world(self):
        return self.ik.fk(self.q_cmd)[0]

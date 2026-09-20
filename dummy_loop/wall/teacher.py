"""脚本示教：竖向往返抹涂（boustrophedon）。

只使用控制器自己的目标位姿和名义墙面知识（实机上同样可得），不读特权信息。
竖向行程在这台臂的自然运动平面内（肩、肘、腕俯仰），见 docs/SIMULATION.md。

补偿（可选，均只用实机可得的量）：
  probe   开工前在工作区三个角点沿法线慢速探触，工具传感器超过阈值即记为接触点
          （刚性工具：抹刀座力传感器 > touch_force；弹簧工具：压缩量 > 1 mm），
          拟合真实墙面平面并更新控制器的墙面坐标系。探触在机器自己的（可能有零位误差的）
          运动学里完成，所以同时吸收了局部的标定误差。
  servo   抹涂阶段按工具传感器闭环修正法向深度，限幅：
          刚性工具 dn += k_f (F* - F_meas)；弹簧工具 dn += k (c* - c_meas)。
"""
import numpy as np
from .controller import WallFrame


def _toward(cur, goal, step):
    d = np.asarray(goal) - np.asarray(cur); n = np.linalg.norm(d)
    return d if n <= step else d * step / n


class RasterTeacher:
    def __init__(self, env, speed=0.004, approach_speed=0.002, column_pitch=0.06, probe=False, servo=False,
                 servo_gain=0.3, probe_threshold=0.001, max_corr=0.015, force_gain=0.0001):
        self.env = env; self.speed = speed; self.approach = approach_speed; self.pitch = column_pitch
        self.probe = probe; self.servo = servo; self.k = servo_gain; self.kf = force_gain
        self.rigid = env.rigid; self.key = env.sensor_key
        self.thr = env.task.touch_force if self.rigid else probe_threshold
        self.debounce = 2 if self.rigid else 1
        self.max_corr = max_corr; self.dn_corr = 0.0   # 补偿层累计的法向深度偏置（有上限）
        self.log = {'probe_points_uvn': [], 'wall_correction': None, 'frame_update': None}

    # 向目标点移动（名义墙面系，基于控制器目标）。contact=True 时目标的法向分量
    # 叠加补偿层学到的深度偏置 dn_corr——否则补偿与示教会互相拉扯。
    # 规划基于「自己已经发出的指令累计」self.plan，而不是控制器当前目标：
    # 有指令延迟时控制器目标滞后，拿它做反馈会过冲、在路点附近来回振荡（延迟引起的极限环）。
    # 策略永远知道自己发过什么，这在实机上同样成立。
    def _move_to(self, goal, speed, record, contact=False, max_steps=3000, stall_steps=40):
        stalled = 0
        for _ in range(max_steps):
            g = np.array(goal, float)
            if contact:
                g[2] += self.dn_corr
            d = _toward(self.plan, g, speed)
            if np.linalg.norm(d) < 1e-9:
                return
            _, status = record(np.r_[d, 0.0])
            stalled = stalled + 1 if status == 'unreachable' else 0
            if stalled >= stall_steps:
                raise RuntimeError(f'teacher stalled {stall_steps} steps toward {np.round(g, 4).tolist()} '
                                   '(target unreachable)')
        raise RuntimeError('teacher exceeded step budget for one segment')

    def _probe_wall(self, record):
        env, t = self.env, self.env.task
        u0, u1 = t.region_u; v0, v1 = t.region_v
        pts = [(u0 + 0.03, v0 + 0.02), (u1 - 0.03, v0 + 0.02), ((u0 + u1) / 2, v1 - 0.02)]
        found = []
        for (u, v) in pts:
            self._move_to([u, v, -t.standoff], self.speed, record)
            hit = None; above = 0
            for _ in range(int((t.standoff + 0.03) / 0.001)):
                obs, _ = record(np.array([0, 0, 0.001, 0]))    # 2 cm/s 慢速逼近
                above = above + 1 if obs[self.key][0] > self.thr else 0
                if above >= self.debounce:                        # 连续超过阈值才算接触，防噪声误触发
                    hit = obs['tcp_belief_uvn_m'].copy()                   # 读数正解的 TCP
                    if not self.rigid:
                        hit[2] += obs[self.key][0]                          # 弹簧：刀面实际在更前方，加回压缩量
                    break
            self._move_to([u, v, -t.standoff], self.speed, record)
            if hit is None:
                raise RuntimeError('probe did not find the wall within travel; check wall placement')
            found.append(hit)
        P = np.array(found)
        # 拟合 n = a + b u + c v（名义墙面系）
        A = np.c_[np.ones(3), P[:, 0], P[:, 1]]
        a, b, c = np.linalg.solve(A, P[:, 2])
        R = env.ctrl.frame.R
        n_new = R @ np.array([-b, -c, 1.0]); n_new /= np.linalg.norm(n_new)
        u_new = R[:, 0] - (R[:, 0] @ n_new) * n_new; u_new /= np.linalg.norm(u_new)
        v_new = np.cross(n_new, u_new)
        origin_new = env.ctrl.frame.to_world([0, 0, a])
        old = env.ctrl.frame
        new = WallFrame(origin_new, np.column_stack([u_new, v_new, n_new]))
        # 目标点在新坐标系下保持同一物理位置（控制器目标与示教的指令累计都要换算）
        env.ctrl.target = new.from_world(old.to_world(env.ctrl.target))
        self.plan = new.from_world(old.to_world(self.plan))
        env.ctrl.frame = new
        self.log['frame_update'] = {'at_seq': env.seq, 'origin': new.origin.tolist(), 'R': new.R.tolist()}
        self.log['probe_points_uvn'] = P.round(5).tolist()
        self.log['wall_correction'] = {'dn_m': round(float(a), 5), 'slope_u': round(float(b), 5), 'slope_v': round(float(c), 5)}

    def run(self, on_step=None):
        """执行一整条 episode。on_step(obs, action, info) 可用于录制。返回 (metrics, teacher_log)。"""
        env, t = self.env, self.env.task
        obs = env.reset()
        state = {'obs': obs, 'pressing': False}
        self.dn_corr = 0.0
        self.plan = env.ctrl.target.copy()

        def record(a):
            a = np.asarray(a, float)
            if self.servo and state['pressing']:
                m = state['obs'][self.key][0]
                err = self.kf * (t.target_force - m) if self.rigid else self.k * (t.target_compression - m)
                corr = float(np.clip(err, -0.001, 0.001))
                corr = float(np.clip(self.dn_corr + corr, -self.max_corr, self.max_corr) - self.dn_corr)
                self.dn_corr += corr
                a = a.copy(); a[2] += corr
            a = env.spec.clip(a)
            self.plan = self.plan + a[:3]
            o, info = env.step(a)
            if info['status'] == 'unreachable' and env.pert.latency_steps == 0:
                self.plan = self.plan - a[:3]       # 控制器拒收了这次增量
            if on_step:
                on_step(state['obs'], a, info)
            state['obs'] = o
            return o, info['status']

        if self.probe:
            self._probe_wall(record)
        u0, u1 = t.region_u; v0, v1 = t.region_v
        from .scene import blade_span
        bw, bh = blade_span(env.scene)     # 沿墙水平（刀长）、竖直（刀宽）
        depth = t.press_depth if self.rigid else t.target_compression
        cols = np.arange(u0 + bw / 2 - 0.01, u1 - bw / 2 + 0.01 + 1e-9, self.pitch)
        if len(cols) == 0 or cols[-1] < u1 - bw / 2 - 1e-6:
            cols = np.append(cols, u1 - bw / 2 + 0.01)
        v_lo, v_hi = v0 + bh / 2 - 0.01, v1 - bh / 2 + 0.01
        for k, u in enumerate(cols):
            v_start, v_end = (v_lo, v_hi) if k % 2 == 0 else (v_hi, v_lo)
            self._move_to([u, v_start, -t.standoff], self.speed, record)
            state['pressing'] = False
            self._move_to([u, v_start, depth], self.approach, record, contact=True)   # 压入
            state['pressing'] = True
            self._move_to([u, v_end, depth], self.speed, record, contact=True)         # 抹
            state['pressing'] = False
            self._move_to([u, v_end, -t.standoff], self.approach, record)               # 抬起
        self.log['servo_dn_corr_m'] = round(self.dn_corr, 5)
        return env.metrics(), self.log

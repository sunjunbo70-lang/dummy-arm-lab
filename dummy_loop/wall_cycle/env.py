"""Stroke-level POMDP for a continuous whole-wall work cycle.

One RL step is one complete high-level decision. It accounts for scan, return,
random reload, approach, arbitrary line/arc contact stroke, lift and retreat.
No policy observation contains the simulator's true wall or blade fields.
"""
from dataclasses import dataclass
import numpy as np

from .config import CycleConfig
from .material import MaterialSystem
from .sensor import D435Proxy, coarse_map

MODES = ('DEPOSIT', 'REUSE', 'LEVEL', 'RESCAN', 'FINISH')
ACTION_NAMES = ('mode', 'start_u', 'start_v', 'end_u', 'end_v',
                'blade_cos', 'blade_sin', 'bend', 'force', 'speed', 'load')


@dataclass
class DecodedAction:
    mode: str
    start: tuple
    end: tuple
    blade_angle: float
    bend_m: float
    force_N: float
    speed_m_s: float
    requested_load_ml: float

    def to_dict(self):
        return {'mode': self.mode, 'start_uv_m': list(self.start), 'end_uv_m': list(self.end),
                'blade_angle_deg': float(np.rad2deg(self.blade_angle)), 'bend_m': self.bend_m,
                'force_N': self.force_N, 'speed_m_s': self.speed_m_s,
                'requested_load_ml': self.requested_load_ml}


class WallCycleEnv:
    act_dim = len(ACTION_NAMES)

    def __init__(self, cfg=None, seed=0, initial_mix=True, record=False):
        self.cfg = cfg or CycleConfig()
        self.seed = seed; self.rng = np.random.default_rng(seed)
        self.material = MaterialSystem(self.cfg, seed)
        self.sensor = D435Proxy(self.cfg, seed+1)
        self.initial_mix = initial_mix; self.record = record
        nv, nu2 = self.cfg.wall_shape[0]//2, self.cfg.wall_shape[1]//4
        self.obs_dim = nv*nu2*3 + 8
        self.events = []

    def _scan(self):
        self.scan = self.sensor.scan(self.material.wall)
        self.control_steps += round(self.cfg.scan_frames/self.cfg.camera_fps*self.cfg.control_hz)
        return self.scan

    def _observe(self):
        c = self.cfg; h = coarse_map(self.scan['height']); conf = coarse_map(self.scan['confidence'])
        measured = np.where(conf > .5, h, c.target_m)
        thickness = np.clip(measured/c.target_m, -1, 3)
        error = np.clip((c.target_m-measured)/c.target_m, -2, 2)
        scalar = np.array([
            np.clip(self.material.blade_volume_m3/(c.blade_capacity_ml*1e-6), 0, 1.5),
            self.cycles/c.max_cycles,
            self.control_steps/c.max_steps,
            self.stall_count/c.stall_limit,
            self.reloads/c.max_reload_cycles,
            self.last_mode/(len(MODES)-1),
            float(conf.mean()),
            np.clip(self.last_improvement/.1, -1, 1),
        ])
        return np.r_[thickness.ravel(), error.ravel(), conf.ravel(), scalar]

    def reset(self):
        initial = 'partial' if self.initial_mix and self.rng.random() < .35 else 'bare'
        self.material.reset(initial)
        self.cycles = self.reloads = self.control_steps = self.stall_count = 0
        self.budget = self.cfg.base_steps; self.last_mode = 0; self.last_improvement = 0.0
        self.done = False; self.events = []
        self._scan(); self.previous_cost = self.material.quality_cost()
        self._event('SCAN', {'initial': initial})
        return self._observe()

    def _event(self, phase, extra=None):
        if not self.record:
            return
        self.events.append({'phase': phase, 'cycle': self.cycles, 'control_steps': self.control_steps,
                            'wall': self.material.wall.copy(), 'blade': self.material.blade.copy(),
                            'metrics': self.material.metrics(), **(extra or {})})

    def decode(self, action):
        a = np.clip(np.asarray(action, float), -1, 1)
        mode_i = min(int((a[0]+1)*.5*len(MODES)), len(MODES)-1)
        c = self.cfg
        def uv(x, y):
            return (float(x*c.width_m*.5), float((y+1)*.5*c.height_m))
        start, end = uv(a[1], a[2]), uv(a[3], a[4])
        d = np.asarray(end)-start; length = np.linalg.norm(d)
        if length < c.min_stroke_m:
            ang = np.arctan2(a[4], a[3]+1e-9)
            end = tuple(np.asarray(start) + c.min_stroke_m*np.array([np.cos(ang), np.sin(ang)]))
        elif length > c.max_stroke_m:
            end = tuple(np.asarray(start) + d/length*c.max_stroke_m)
        # Keep endpoints inside the wall; exact swept-tool checks belong to the executor.
        end = (float(np.clip(end[0], -c.width_m/2, c.width_m/2)),
               float(np.clip(end[1], 0, c.height_m)))
        phi = float(np.arctan2(a[6], a[5])) if abs(a[5])+abs(a[6]) > 1e-6 else 0.0
        force = c.force_window[0] + (a[8]+1)*.5*(c.force_window[1]-c.force_window[0])
        speed = c.speed_range_m_s[0] + (a[9]+1)*.5*(c.speed_range_m_s[1]-c.speed_range_m_s[0])
        li = min(int((a[10]+1)*.5*len(c.load_choices_ml)), len(c.load_choices_ml)-1)
        return DecodedAction(MODES[mode_i], start, end, phi, float(a[7]*c.curve_offset_m),
                             float(force), float(speed), float(c.load_choices_ml[li]))

    def encode(self, d: DecodedAction):
        c = self.cfg; a = np.zeros(self.act_dim)
        i = MODES.index(d.mode); a[0] = -1 + (i+.5)*2/len(MODES)
        a[1] = np.clip(d.start[0]/(c.width_m/2), -1, 1)
        a[2] = np.clip(d.start[1]/c.height_m*2-1, -1, 1)
        a[3] = np.clip(d.end[0]/(c.width_m/2), -1, 1)
        a[4] = np.clip(d.end[1]/c.height_m*2-1, -1, 1)
        a[5], a[6] = np.cos(d.blade_angle), np.sin(d.blade_angle)
        a[7] = np.clip(d.bend_m/c.curve_offset_m, -1, 1)
        a[8] = np.clip((d.force_N-c.force_window[0])/(c.force_window[1]-c.force_window[0])*2-1, -1, 1)
        a[9] = np.clip((d.speed_m_s-c.speed_range_m_s[0])/(c.speed_range_m_s[1]-c.speed_range_m_s[0])*2-1, -1, 1)
        li = int(np.argmin(np.abs(np.asarray(c.load_choices_ml)-d.requested_load_ml)))
        a[10] = -1 + (li+.5)*2/len(c.load_choices_ml)
        return a

    def teacher_action(self):
        """Sensor-only spatial heuristic used for behaviour-cloning warm start."""
        c = self.cfg; h = self.scan['height']; conf = self.scan['confidence']
        valid = conf > .5
        est_coverage = np.mean(valid & (h >= c.acceptable_low_m) & (h <= c.acceptable_high_m))
        est_rmse = np.sqrt(np.mean(np.where(valid, (h-c.target_m)**2, 0)))*1000
        if conf.mean() < .82:
            mode = 'RESCAN'
        elif est_coverage >= c.finish_coverage and est_rmse <= c.finish_rmse_mm:
            mode = 'FINISH'
        else:
            under = np.where(valid, np.maximum(c.target_m-h, 0), 0)
            over = np.where(valid, np.maximum(h-c.target_m, 0), 0)
            if over.sum() > under.sum()*.7 and np.mean(over > .0005) > .08:
                mode, weight = 'LEVEL', over
            else:
                estimated_need = float(under.sum()*c.cell_m**2)
                enough_on_tool = self.material.blade_volume_m3 >= min(8e-6, .65*estimated_need)
                mode = 'REUSE' if enough_on_tool or estimated_need < 2e-6 else 'DEPOSIT'
                weight = under
        if mode in ('RESCAN', 'FINISH'):
            return self.encode(DecodedAction(mode, (0,.01), (0,.06), 0, 0, 6, .05, 0))
        total = weight.sum()
        if total <= 1e-12:
            center = np.array([0., c.height_m/2]); direction = np.array([0., 1.])
            cov = np.diag([.001, .002])
        elif mode in ('DEPOSIT', 'REUSE'):
            # A wide missing area is multimodal: its global centroid can sit on an
            # already-finished strip. Select the best blade-width window, then make
            # a vertical stroke. Later repair actions remain free to use any angle.
            column_need = weight.sum(axis=0)
            window = max(1, round(c.blade_length_m/c.cell_m))
            score = np.convolve(column_need, np.ones(window), mode='same')
            center = np.array([self.material.u[int(np.argmax(score))], c.height_m/2])
            direction = np.array([0., 1.])
            cov = np.diag([1e-5, (c.height_m/2)**2])
        else:
            U, V = np.meshgrid(self.material.u, self.material.v)
            center = np.array([(U*weight).sum()/total, (V*weight).sum()/total])
            X = np.c_[U.ravel()-center[0], V.ravel()-center[1]]
            w = weight.ravel()/total
            cov = (X*w[:, None]).T @ X + 1e-7*np.eye(2)
            _, vec = np.linalg.eigh(cov); direction = vec[:, -1]
            # Prefer upward execution for equal axes, while retaining diagonals/horizontal repairs.
            if direction[1] < 0: direction = -direction
        length = min(c.max_stroke_m, max(.055, 2*np.sqrt(max(np.linalg.eigvalsh(cov)[-1], 1e-6))))
        start = center-direction*length/2; end = center+direction*length/2
        start[0] = np.clip(start[0], -c.width_m/2, c.width_m/2); end[0] = np.clip(end[0], -c.width_m/2, c.width_m/2)
        start[1] = np.clip(start[1], 0, c.height_m); end[1] = np.clip(end[1], 0, c.height_m)
        phi = np.arctan2(direction[1], direction[0]) + np.pi/2
        load = 18 if mode == 'DEPOSIT' else 0
        return self.encode(DecodedAction(mode, tuple(start), tuple(end), phi, 0, 8, .06, load))

    def step(self, action):
        if self.done:
            raise RuntimeError('step after episode finished')
        c = self.cfg; d = self.decode(action); self.cycles += 1; self.last_mode = MODES.index(d.mode)
        before = self.material.quality_cost(); before_waste = self.material.dropped_m3+self.material.outside_m3
        steps_before = self.control_steps
        self._event('DECIDE', {'action': d.to_dict()})
        reward = -.02
        if d.mode == 'FINISH':
            m = self.material.metrics(); confident = self.scan['confidence'].mean() >= c.finish_confidence
            success = (m['coverage'] >= c.finish_coverage and m['rmse_mm'] <= c.finish_rmse_mm and
                       m['p95_error_mm'] <= c.finish_p95_mm and confident)
            reward += 100 if success else -20
            self.done = True; self._event('FINISH', {'success': success})
            return self._observe(), reward, True, self.info(success)
        if d.mode == 'RESCAN':
            self._event('SCAN_RETURN'); self._scan(); self._event('SCAN')
            improvement = 0.0
        else:
            if d.mode == 'DEPOSIT' and d.requested_load_ml > 0 and self.reloads < c.max_reload_cycles:
                self.control_steps += 50; self._event('LOAD_APPROACH')
                load = self.material.load_random(d.requested_load_ml); self.reloads += 1
                reward -= .2; self._event('DISPENSE', {'load': load}); self._event('TOOL_INSPECT')
            self.control_steps += 35; self._event('WALL_APPROACH', {'action': d.to_dict()})
            distance = float(np.linalg.norm(np.asarray(d.end)-d.start))
            work_steps = max(1, round(distance/max(d.speed_m_s, 1e-4)*c.control_hz))
            self._event('CONTACT_ACQUIRE')
            def work_frame(k, center, wall, blade):
                if self.record:
                    self.events.append({'phase': 'WORK_STEP', 'cycle': self.cycles,
                                        'sample': k, 'control_steps': self.control_steps +
                                        round((k+1)*work_steps/c.stroke_samples),
                                        'tool_center_uv_m': center, 'action': d.to_dict(),
                                        'wall': wall, 'blade': blade,
                                        'metrics': self.material.metrics()})
            stats = self.material.stroke(d.mode, d.start, d.end, d.blade_angle, d.bend_m,
                                         d.force_N, d.speed_m_s, callback=work_frame)
            self.control_steps += work_steps
            self._event('WORK', {'action': d.to_dict(), 'transfer': stats.__dict__})
            self.control_steps += 55; self._event('LIFT'); self._event('RETREAT'); self._event('SCAN_RETURN')
            self._scan(); self._event('SCAN')
            after = self.material.quality_cost(); improvement = before-after
            waste = (self.material.dropped_m3+self.material.outside_m3)-before_waste
            reward += 30*improvement - 4*waste/max(18e-6, 1e-12)
            if stats.peak_force_N > c.max_force_N:
                reward -= 50; self.done = True
        self.last_improvement = improvement
        if improvement < c.progress_epsilon:
            self.stall_count += 1; reward -= min(.5*2**(self.stall_count-1), 8)
        else:
            self.stall_count = 0
        reward -= .0002*(self.control_steps-steps_before)
        if self.control_steps >= self.budget and self.budget < c.max_steps and improvement >= c.progress_epsilon:
            self.budget = min(self.budget+c.extension_steps, c.max_steps)
        if (self.stall_count >= c.stall_limit or self.cycles >= c.max_cycles or
                self.control_steps >= self.budget or self.control_steps >= c.max_steps):
            self.done = True
        self.previous_cost = self.material.quality_cost()
        return self._observe(), float(reward), self.done, self.info(False)

    def info(self, success=False):
        return {'success': bool(success), 'metrics': self.material.metrics(), 'cycles': self.cycles,
                'reloads': self.reloads, 'control_steps': self.control_steps,
                'budget': self.budget, 'stall_count': self.stall_count,
                'volume_balance_m3': self.material.volume_balance()}

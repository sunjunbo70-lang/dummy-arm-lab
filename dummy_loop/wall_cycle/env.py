"""Stroke-level POMDP for a continuous whole-wall work cycle.

One RL step is one complete high-level decision. It accounts for scan, return,
random reload, approach, arbitrary line/arc contact stroke, lift and retreat.
No policy observation contains the simulator's true wall or blade fields.
"""
from dataclasses import dataclass
import numpy as np

from .config import CycleConfig
from .material import MaterialSystem
from .mortar import MortarSystem, StrokeStats
from .sensor import D435Proxy, coarse_map, coarse_shape

MODES = ('DEPOSIT', 'REUSE', 'LEVEL', 'RESCAN', 'FINISH')
ACTION_NAMES_V2 = ('mode', 'start_u', 'start_v', 'end_u', 'end_v',
                   'blade_cos', 'blade_sin', 'bend', 'force', 'speed', 'load')
# v0.3: the pitch profile of the stroke -- tilted at contact (lower edge first), then flattened.
ACTION_NAMES = ACTION_NAMES_V2 + ('pitch_start', 'pitch_end')


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
    pitch_start: float = 0.0      # rad, v0.3 only
    pitch_end: float = 0.0

    def to_dict(self):
        return {'mode': self.mode, 'start_uv_m': list(self.start), 'end_uv_m': list(self.end),
                'blade_angle_deg': float(np.rad2deg(self.blade_angle)), 'bend_m': self.bend_m,
                'force_N': self.force_N, 'speed_m_s': self.speed_m_s,
                'requested_load_ml': self.requested_load_ml,
                'pitch_start_deg': float(np.rad2deg(self.pitch_start)),
                'pitch_end_deg': float(np.rad2deg(self.pitch_end))}


class WallCycleEnv:
    def __init__(self, cfg=None, seed=0, initial_mix=True, record=False, executor=None):
        self.cfg = cfg or CycleConfig()
        self.v3 = self.cfg.physics != 'v0.2'
        self.action_names = ACTION_NAMES if self.v3 else ACTION_NAMES_V2
        self.act_dim = len(self.action_names)
        # Optional arm executor (dummy_loop/wall_cycle/arm.py): checks every stroke on the
        # real Dummy V2 MuJoCo model (IK along the path, joint limits, arm-wall clearance).
        self.executor = executor
        self.teacher_style = 'technique'
        # v0.3 teacher settings that are NEUTRAL to the pitch question (same for both styles);
        # chosen by the average over both styles, see docs/changes/2026-09-22_wall_cycle_v0.3.md C14
        self.teacher_force_N = 1.0
        self.teacher_load_ml = 12.0
        self.teacher_edge_margin_m = 0.02
        self.seed = seed; self.rng = np.random.default_rng(seed)
        self.material = (MortarSystem if self.cfg.physics != 'v0.2' else MaterialSystem)(self.cfg, seed)
        self.sensor = D435Proxy(self.cfg, seed+1)
        self.initial_mix = initial_mix; self.record = record
        nv, nu2 = coarse_shape(self.cfg.wall_shape)
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
        self.done = False; self.events = []; self.unreachable = 0; self.projected = 0
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
        fw = c.force_window_v3 if self.v3 else c.force_window
        force = fw[0] + (a[8]+1)*.5*(fw[1]-fw[0])
        speed = c.speed_range_m_s[0] + (a[9]+1)*.5*(c.speed_range_m_s[1]-c.speed_range_m_s[0])
        li = min(int((a[10]+1)*.5*len(c.load_choices_ml)), len(c.load_choices_ml)-1)
        p0 = p1 = 0.0
        if self.v3:
            pmax = np.deg2rad(c.max_pitch_deg)
            p0, p1 = float((a[11]+1)*.5*pmax), float((a[12]+1)*.5*pmax)
        return DecodedAction(MODES[mode_i], start, end, phi, float(a[7]*c.curve_offset_m),
                             float(force), float(speed), float(c.load_choices_ml[li]), p0, p1)

    def encode(self, d: DecodedAction):
        c = self.cfg; a = np.zeros(self.act_dim)
        i = MODES.index(d.mode); a[0] = -1 + (i+.5)*2/len(MODES)
        a[1] = np.clip(d.start[0]/(c.width_m/2), -1, 1)
        a[2] = np.clip(d.start[1]/c.height_m*2-1, -1, 1)
        a[3] = np.clip(d.end[0]/(c.width_m/2), -1, 1)
        a[4] = np.clip(d.end[1]/c.height_m*2-1, -1, 1)
        a[5], a[6] = np.cos(d.blade_angle), np.sin(d.blade_angle)
        a[7] = np.clip(d.bend_m/c.curve_offset_m, -1, 1)
        fw = c.force_window_v3 if self.v3 else c.force_window
        a[8] = np.clip((d.force_N-fw[0])/(fw[1]-fw[0])*2-1, -1, 1)
        a[9] = np.clip((d.speed_m_s-c.speed_range_m_s[0])/(c.speed_range_m_s[1]-c.speed_range_m_s[0])*2-1, -1, 1)
        li = int(np.argmin(np.abs(np.asarray(c.load_choices_ml)-d.requested_load_ml)))
        a[10] = -1 + (li+.5)*2/len(c.load_choices_ml)
        if self.v3:
            pmax = np.deg2rad(c.max_pitch_deg)
            a[11] = np.clip(d.pitch_start/pmax*2-1, -1, 1)
            a[12] = np.clip(d.pitch_end/pmax*2-1, -1, 1)
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
        if self.v3:
            load = self.teacher_load_ml if mode == 'DEPOSIT' else 0
            mgn = self.teacher_edge_margin_m
            if mgn > 0:
                start = np.array(start, float); end = np.array(end, float)
                for pt in (start, end):
                    pt[0] = np.clip(pt[0], -c.width_m / 2 + mgn, c.width_m / 2 - mgn)
                    pt[1] = np.clip(pt[1], mgn, c.height_m - mgn)
        if not self.v3:
            return self.encode(DecodedAction(mode, tuple(start), tuple(end), phi, 0, 8, .06, load))
        # v0.3 teacher (only a warm start; PPO is meant to improve on it).
        # It knows nothing about the mortar model. 'flat' is the default and the question of
        # the experiment is whether RL finds a better pitch/force than this; 'technique' is the
        # worker's recipe (stand the blade up, flatten), used as a comparison warm start.
        force = self.teacher_force_N
        if mode == 'LEVEL':
            p0 = p1 = np.deg2rad(5)
        elif self.teacher_style == 'flat':
            p0 = p1 = np.deg2rad(3)
        else:
            p0, p1 = np.deg2rad(25), np.deg2rad(8)
        d = DecodedAction(mode, tuple(start), tuple(end), phi, 0, force, .06, load, p0, p1)
        # stay within what the arm can do: if the executor rejects the stroke, move it inwards
        # (towards the square centre) and finally drop the tilt
        # (first keep the style's pitch and shorten the stroke, only then drop the tilt, so the
        # 'technique' teacher does not silently degrade into the 'flat' one)
        if self.executor is not None and hasattr(self.executor, 'plan'):
            centre = np.array([0.0, c.height_m / 2])
            # 'technique' stands the blade up as far as the arm allows at this start point
            # (J5 limits upward lower-edge-first tilts near the bottom, see change log C14)
            if mode != 'LEVEL' and self.teacher_style != 'flat':
                profiles = [(np.deg2rad(a_), np.deg2rad(b_)) for a_, b_ in ((25, 8), (20, 6), (10, 4))] + [(0.0, 0.0)]
            else:
                profiles = [(p0, p1), (0.0, 0.0)]
            for pp in profiles:
                for shrink in (0.0, 0.15, 0.3):
                    s_ = tuple(np.asarray(start) + (centre - start) * shrink)
                    e_ = tuple(np.asarray(end) + (centre - end) * shrink)
                    cand = DecodedAction(mode, s_, e_, phi, 0, force, .06, load, *pp)
                    ok = self.executor.plan(cand).ok
                    self.executor.stats['planned'] -= 1
                    if ok:
                        return self.encode(cand)
                    self.executor.stats['rejected'] -= 1
        return self.encode(d)

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
        plan = self.executor.plan(d) if (self.executor is not None and d.mode != 'RESCAN') else None
        if plan is not None and not plan.ok and self.v3:
            # Safety layer (action projection): execute the nearest stroke the arm CAN do --
            # less tilt first, then pulled towards the square centre -- with a small penalty.
            # Nothing infeasible is ever executed; the episode is not ended by one bad proposal.
            # (v0.3 first training run without it: ~half the strokes rejected, episodes ended
            # after ~8 cycles by the stall rule; see change log C16.)
            proj = self._project(d)
            if proj is not None:
                d, plan = proj
                self.projected += 1
                reward -= 0.5
                self._event('PROJECTED', {'action': d.to_dict()})
        if plan is not None and not plan.ok:
            # The real arm cannot execute this stroke (IK, joint limit or arm hits the wall).
            # Nothing touches the wall; the attempt costs time and a penalty.
            self.unreachable += 1
            self._event('UNREACHABLE', {'action': d.to_dict(), 'reason': plan.reason})
            self.control_steps += 20
            reward -= 2.0
            improvement = 0.0
        elif d.mode == 'RESCAN':
            self._event('SCAN_RETURN'); self._scan(); self._event('SCAN')
            improvement = 0.0
        else:
            if plan is not None and self.record:
                self._event('ARM_PLAN', {'q_approach': plan.q_approach, 'q_lift': plan.q_lift})
            if d.mode == 'DEPOSIT' and d.requested_load_ml > 0 and self.reloads < c.max_reload_cycles:
                self.control_steps += 50; self._event('LOAD_APPROACH')
                load = self.material.load_random(d.requested_load_ml); self.reloads += 1
                reward -= .2; self._event('DISPENSE', {'load': load}); self._event('TOOL_INSPECT')
            self.control_steps += 35; self._event('WALL_APPROACH', {'action': d.to_dict()})
            distance = float(np.linalg.norm(np.asarray(d.end)-d.start))
            work_steps = max(1, round(distance/max(d.speed_m_s, 1e-4)*c.control_hz))
            self._event('CONTACT_ACQUIRE')
            def work_frame(k, center, wall, blade, q=None):
                if self.record:
                    ev = {'phase': 'WORK_STEP', 'cycle': self.cycles,
                          'sample': k, 'control_steps': self.control_steps +
                          round((k+1)*work_steps/c.stroke_samples),
                          'tool_center_uv_m': center, 'action': d.to_dict(),
                          'pitch_deg': float(np.rad2deg(getattr(self.material, 'last_pitch', 0.0))),
                          'wall': wall, 'blade': blade, 'metrics': self.material.metrics()}
                    if q is not None:                      # co-simulation: actual joint angles
                        ev['q'] = q
                    elif plan is not None and getattr(plan, 'q_work', None):
                        ev['q'] = plan.q_work[min(k, len(plan.q_work)-1)]
                    self.events.append(ev)
            if not self.v3:
                stats = self.material.stroke(d.mode, d.start, d.end, d.blade_angle, d.bend_m,
                                             d.force_N, d.speed_m_s, callback=work_frame)
            elif plan is not None and getattr(self.executor, 'dynamic', False):
                # MuJoCo co-simulation: the arm's actual pose drives the mortar model
                stats = self.executor.execute(plan, d, self.material, StrokeStats(), callback=work_frame)
            else:
                # nominal path (training with the reach table, or no arm at all)
                stats = self.material.stroke(d.mode, d.start, d.end, d.blade_angle, d.bend_m,
                                             d.force_N, d.speed_m_s, callback=work_frame,
                                             pitch_start=d.pitch_start, pitch_end=d.pitch_end)
            self.last_stroke = {k: v for k, v in dict(stats.__dict__ if not isinstance(stats, dict) else stats).items()
                                if not isinstance(v, list)}
            self.control_steps += work_steps
            self._event('WORK', {'action': d.to_dict(), 'transfer': self.last_stroke})
            self.control_steps += 55; self._event('LIFT'); self._event('RETREAT'); self._event('SCAN_RETURN')
            self._scan(); self._event('SCAN')
            after = self.material.quality_cost(); improvement = before-after
            waste = (self.material.dropped_m3+self.material.outside_m3)-before_waste
            reward += 30*improvement - 4*waste/max(18e-6, 1e-12)
            if (stats.peak_force_N or 0.0) > c.max_force_N:
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

    def _project(self, d):
        c = self.cfg
        centre = np.array([0.0, c.height_m / 2])
        for f_tilt in (2 / 3, 1 / 3, 0.0):
            for shrink in (0.0, 0.2, 0.4):
                if f_tilt == 1.0 and shrink == 0.0:
                    continue
                s_ = tuple(np.asarray(d.start) + (centre - np.asarray(d.start)) * shrink)
                e_ = tuple(np.asarray(d.end) + (centre - np.asarray(d.end)) * shrink)
                cand = DecodedAction(d.mode, s_, e_, d.blade_angle, d.bend_m, d.force_N, d.speed_m_s,
                                     d.requested_load_ml, d.pitch_start * f_tilt, d.pitch_end * f_tilt)
                plan = self.executor.plan(cand)
                if plan.ok:
                    return cand, plan
        return None

    def info(self, success=False):
        return {'success': bool(success), 'metrics': self.material.metrics(), 'cycles': self.cycles,
                'reloads': self.reloads, 'control_steps': self.control_steps,
                'budget': self.budget, 'stall_count': self.stall_count,
                'unreachable_strokes': getattr(self, 'unreachable', 0),
                'projected_strokes': getattr(self, 'projected', 0),
                'volume_balance_m3': self.material.volume_balance()}

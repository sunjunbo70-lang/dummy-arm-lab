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

# v0.8 local teacher: candidate strokes (length x direction x centre) and the score-grid cells
# each one sweeps, as a 0/1 matrix. Pure geometry, so it is built once per process and
# work-area shape. Swept cells = blade footprint (blade_length across the path, blade_width
# along it) over a straight path -- a binary heuristic kernel, not a transfer model.
_LOCAL_LENGTHS = (0.06, 0.10, 0.14)
_LOCAL_STEP_M = 0.01
_LOCAL_DIRS = np.array([(0., 1.), (1., 0.), (1., 1.), (-1., 1.)]) / np.array([1., 1., np.sqrt(2), np.sqrt(2)])[:, None]
_CANDIDATES = {}


def _stroke_footprint(U, V, centre, direction, length, blade_len, blade_w):
    ru, rv = U - centre[0], V - centre[1]
    along = ru * direction[0] + rv * direction[1]
    perp = -ru * direction[1] + rv * direction[0]
    return (np.abs(along) <= length / 2 + blade_w / 2) & (np.abs(perp) <= blade_len / 2)


def local_candidates(cfg, u_score, v_score):
    # dense float32 (~7k x 1.6k, ~44 MB): numpy only -- scipy is not in the lab .venv-loop
    key = (cfg.width_m, cfg.height_m, cfg.blade_length_m, cfg.blade_width_m, len(u_score), len(v_score),
           float(u_score[0]), float(v_score[0]))
    if key not in _CANDIDATES:
        U, V = np.meshgrid(u_score, v_score)
        us = np.arange(-cfg.width_m / 2, cfg.width_m / 2 + 1e-9, _LOCAL_STEP_M)
        vs = np.arange(0.0, cfg.height_m + 1e-9, _LOCAL_STEP_M)
        CU, CV = np.meshgrid(us, vs); centres = np.c_[CU.ravel(), CV.ravel()]
        rows, meta = [], []
        for L in _LOCAL_LENGTHS:
            for dvec in _LOCAL_DIRS:
                ru = U.ravel()[None] - centres[:, :1]; rv = V.ravel()[None] - centres[:, 1:]
                along = ru * dvec[0] + rv * dvec[1]; perp = -ru * dvec[1] + rv * dvec[0]
                m = (np.abs(along) <= L / 2 + cfg.blade_width_m / 2) & (np.abs(perp) <= cfg.blade_length_m / 2)
                keep = m.any(axis=1)
                rows.append(m[keep]); meta += [(c_, dvec, L) for c_ in centres[keep]]
        M = np.vstack(rows).astype(np.float32)
        norm = np.array([1.0 + 0.15 * L / 0.06 for _, _, L in meta])
        cells = M.sum(axis=1).astype(float)
        _CANDIDATES[key] = (M, meta, norm, np.array([c_ for c_, _, _ in meta]), cells)
    return _CANDIDATES[key]

MODES = ('DEPOSIT', 'REUSE', 'LEVEL', 'RESCAN', 'FINISH')
ACTION_NAMES_V2 = ('mode', 'start_u', 'start_v', 'end_u', 'end_v',
                   'blade_cos', 'blade_sin', 'bend', 'force', 'speed', 'load')
# v0.3: the pitch profile of the stroke -- tilted at contact (lower edge first), then flattened.
ACTION_NAMES = ACTION_NAMES_V2 + ('pitch_start', 'pitch_end')
ACTION_NAMES_V5 = ACTION_NAMES + ('carry_face_up',)


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
    carry_face_up: float = 0.0    # dot(material-face normal, world up), v0.5

    def to_dict(self):
        return {'mode': self.mode, 'start_uv_m': list(self.start), 'end_uv_m': list(self.end),
                'blade_angle_deg': float(np.rad2deg(self.blade_angle)), 'bend_m': self.bend_m,
                'force_N': self.force_N, 'speed_m_s': self.speed_m_s,
                'requested_load_ml': self.requested_load_ml,
                'pitch_start_deg': float(np.rad2deg(self.pitch_start)),
                'pitch_end_deg': float(np.rad2deg(self.pitch_end)),
                'carry_face_up': float(self.carry_face_up)}


class WallCycleEnv:
    def __init__(self, cfg=None, seed=0, initial_mix=True, record=False, executor=None):
        self.cfg = cfg or CycleConfig()
        self.v3 = self.cfg.physics != 'v0.2'
        self.v6 = self.cfg.physics == 'v0.6'
        self.v5 = self.cfg.physics in ('v0.5', 'v0.6')
        self.action_names = (ACTION_NAMES if self.v6 else
                             (ACTION_NAMES_V5 if self.v5 else (ACTION_NAMES if self.v3 else ACTION_NAMES_V2)))
        self.act_dim = len(self.action_names)
        # Optional arm executor (dummy_loop/wall_cycle/arm.py): checks every stroke on the
        # real Dummy V2 MuJoCo model (IK along the path, joint limits, arm-wall clearance).
        self.executor = executor
        self.teacher_style = 'technique'
        # v0.3 teacher settings that are NEUTRAL to the pitch question (same for both styles);
        # chosen by the average over both styles, see experiments/v0.3/r0/design/2026-09-22_wall_cycle_v0.3.md C14
        self.teacher_force_N = 1.0
        self.teacher_load_ml = 12.0
        # v0.7: was a hardcoded 0.02 (2 cm); now reads cfg.teacher_edge_margin_m (default 0.003)
        # now that overtravel past the scored square is allowed and not penalised as waste.
        self.teacher_edge_margin_m = self.cfg.teacher_edge_margin_m
        self.seed = seed; self.rng = np.random.default_rng(seed)
        self.material = (MortarSystem if self.cfg.physics != 'v0.2' else MaterialSystem)(self.cfg, seed)
        self.sensor = D435Proxy(self.cfg, seed+1)
        self.initial_mix = initial_mix; self.record = record
        nv, nu2 = coarse_shape(self.cfg.wall_shape)
        self.obs_dim = nv*nu2*3 + (10 if self.v5 else 8) + (nv*nu2 + 2 if self.cfg.obs_memory else 0)
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
        if self.v5:
            material_scale = max(self.material.supplied_m3 + self.material.initial_m3, 18e-6)
            scalar = np.r_[scalar, self.last_carry_face_up,
                           np.clip(self.material.carry_loss_m3/material_scale, 0, 1)]
        if c.obs_memory:
            # v0.8: what the local teacher remembers is observed too, so BC can reproduce it:
            # decayed map of recent non-improving strokes and strokes since the sensor score
            # last improved (the teacher's plateau stop).
            memory = np.r_[np.clip(coarse_map(self.fail_map), 0, 3).ravel() / 3,
                           min((self.cycles - self.best_score_cycle) / max(c.teacher_plateau_n, 1), 2.0),
                           np.clip(self.best_score, -1, 1)]
            return np.r_[thickness.ravel(), error.ravel(), conf.ravel(), scalar, memory]
        return np.r_[thickness.ravel(), error.ravel(), conf.ravel(), scalar]

    def sensor_score(self):
        """Sensor-only quality estimate on the scored square: coverage - rmse_mm/10."""
        c = self.cfg; rows, cols = self.material._score_rows, self.material._score_cols
        h, conf = self.scan['height'][rows, cols], self.scan['confidence'][rows, cols]
        valid = conf > .5
        cov = np.mean(valid & (h >= c.acceptable_low_m) & (h <= c.acceptable_high_m))
        rmse = np.sqrt(np.mean(np.where(valid, (h - c.target_m) ** 2, 0))) * 1000
        return float(cov - rmse / 10.0)

    def _update_memory(self, d, executed, improvement):
        """v0.8: decay the fail map and add the swept footprint of a stroke that was actually
        attempted (executed or rejected as unreachable) but did not improve the true quality
        cost by progress_epsilon; then refresh the sensor-score plateau tracker."""
        c = self.cfg
        self.fail_map *= c.fail_decay
        if d is not None and d.mode not in ('RESCAN', 'FINISH') and (not executed or improvement < c.progress_epsilon):
            # mark a disc (fail_radius_m) around the stroke's midpoint: the teacher avoids
            # re-aiming there, not the whole area the blade swept (that suppressed too much)
            mid = (np.asarray(d.start, float) + np.asarray(d.end, float)) / 2
            U, V = np.meshgrid(self.material.u, self.material.v)
            self.fail_map += (U - mid[0]) ** 2 + (V - mid[1]) ** 2 <= c.fail_radius_m ** 2
        score = self.sensor_score()
        if score > self.best_score + c.teacher_plateau_eps:
            self.best_score, self.best_score_cycle = score, self.cycles

    def reset(self):
        # Cached MuJoCo executors retain q_last as an IK continuation seed within an
        # episode. Reset it at the episode boundary so a fixed seed is independent of
        # which episode was evaluated immediately before it.
        if self.executor is not None and hasattr(self.executor, 'q_scan') and hasattr(self.executor, 'q_last'):
            self.executor.q_last = np.asarray(self.executor.q_scan, float).copy()
            if hasattr(self.executor, 'rng'):
                self.executor.rng = np.random.default_rng(self.seed)
            if self.v6 and hasattr(self.executor, 'reset'):
                self.executor.reset()
        initial = 'partial' if self.initial_mix and self.rng.random() < .35 else 'bare'
        self.material.reset(initial)
        self.cycles = self.reloads = self.control_steps = self.stall_count = 0
        self.progress_history = []
        self.budget = self.cfg.base_steps; self.last_mode = 0; self.last_improvement = 0.0
        self.last_carry_face_up = 0.0
        if self.v5:
            lo, hi = self.cfg.interface_wall_share_range
            self.material.p.lift_wall_fraction = (float(self.rng.uniform(lo, hi))
                                                   if self.cfg.randomize_interface
                                                   else self.cfg.lift_wall_fraction)
        self.done = False; self.events = []; self.unreachable = 0; self.projected = 0
        self.fail_map = np.zeros(self.cfg.wall_shape)
        self.loss_ledger = {'carry_m3': 0.0, 'other_drop_m3': 0.0, 'outside_m3': 0.0}
        self.end_reason = None
        self._scan(); self.previous_cost = self.material.quality_cost()
        self.best_score, self.best_score_cycle = self.sensor_score(), 0
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
        p0 = p1 = 0.0; carry = 0.0
        if self.v3:
            pmax = np.deg2rad(c.max_pitch_deg)
            p0, p1 = float((a[11]+1)*.5*pmax), float((a[12]+1)*.5*pmax)
        if self.v5 and not self.v6:
            carry = float(a[13])
        elif self.v6:
            carry = 1.0
        return DecodedAction(MODES[mode_i], start, end, phi, float(a[7]*c.curve_offset_m),
                             float(force), float(speed), float(c.load_choices_ml[li]), p0, p1, carry)

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
        if self.v5 and not self.v6:
            a[13] = np.clip(d.carry_face_up, -1, 1)
        return a

    def teacher_action(self):
        """Sensor-only spatial heuristic used for behaviour-cloning warm start."""
        c = self.cfg
        if self.v6 and c.teacher_targeting == 'local':
            return self._teacher_local()
        # v0.7: aim only at the scored sub-region (material._score_rows/_score_cols); the
        # simulation grid may be larger (overtravel margin, see area.py/material.py), and the
        # teacher should not chase coverage/RMSE targets in cells that are never scored.
        rows, cols = self.material._score_rows, self.material._score_cols
        h, conf = self.scan['height'][rows, cols], self.scan['confidence'][rows, cols]
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
            if self.v5:
                # Balanced skill teacher. An anisotropic defect follows its principal
                # direction; broad/ambiguous regions cycle through four useful families.
                U, V = np.meshgrid(self.material.u[cols], self.material.v[rows])
                center = np.array([(U*weight).sum()/total, (V*weight).sum()/total])
                X = np.c_[U.ravel()-center[0], V.ravel()-center[1]]
                w = weight.ravel()/total
                cov = (X*w[:, None]).T @ X + 1e-7*np.eye(2)
                eig, vec = np.linalg.eigh(cov); principal = vec[:, -1]
                curriculum = np.array(((0., 1.), (1., 0.), (1., 1.), (-1., 1.)), float)
                direction = curriculum[(self.cycles + self.seed) % len(curriculum)]
                direction /= np.linalg.norm(direction)
                if eig[-1]/max(eig[0], 1e-8) > 2.0:
                    direction = principal / np.linalg.norm(principal)
                if direction[1] < 0: direction = -direction
            else:
                column_need = weight.sum(axis=0)
                window = max(1, round(c.blade_length_m/c.cell_m))
                score = np.convolve(column_need, np.ones(window), mode='same')
                center = np.array([self.material.u[cols][int(np.argmax(score))], c.height_m/2])
                direction = np.array([0., 1.])
                cov = np.diag([1e-5, (c.height_m/2)**2])
        else:
            U, V = np.meshgrid(self.material.u[cols], self.material.v[rows])
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
        carry = 1.0 if self.v5 and self.teacher_style != 'legacy_transport' else -1.0
        d = DecodedAction(mode, tuple(start), tuple(end), phi, 0, force, .06, load, p0, p1, carry)
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
                    cand = DecodedAction(mode, s_, e_, phi, 0, force, .06, load, *pp, carry)
                    ok = self.executor.plan(cand).ok
                    self.executor.stats['planned'] -= 1
                    if ok:
                        return self.encode(cand)
                    self.executor.stats['rejected'] -= 1
        return self.encode(d)

    def _teacher_local(self):
        """v0.8 teacher: pick the candidate stroke whose swept footprint covers the most local
        deficit (or, if larger, twice the local excess -> LEVEL), instead of aiming every stroke
        at the global centroid of the deficit map. Uses only the scan, the blade load and the
        observed memory (fail map, plateau counter). Weights (0.5 excess penalty, level weight,
        fail damping, length normalisation) are heuristics, not physical constants."""
        c = self.cfg
        rows, cols = self.material._score_rows, self.material._score_cols
        h, conf = self.scan['height'][rows, cols], self.scan['confidence'][rows, cols]
        valid = conf > .5
        est_coverage = np.mean(valid & (h >= c.acceptable_low_m) & (h <= c.acceptable_high_m))
        est_rmse = np.sqrt(np.mean(np.where(valid, (h-c.target_m)**2, 0)))*1000
        idle = lambda mode: self.encode(DecodedAction(mode, (0, .01), (0, .06), 0, 0, 6, .05, 0))
        if conf.mean() < .82:
            return idle('RESCAN')
        if est_coverage >= c.finish_coverage and est_rmse <= c.finish_rmse_mm:
            return idle('FINISH')
        if self.cycles - self.best_score_cycle >= c.teacher_plateau_n:
            return idle('FINISH')               # no measured progress for n strokes: stop, do not overwork
        under = np.where(valid, np.maximum(c.acceptable_low_m + 0.0002 - h, 0), 0).ravel()
        over = np.where(valid, np.maximum(h - c.acceptable_high_m, 0), 0).ravel()
        M, meta, norm, centres, cells = local_candidates(c, self.material.u[cols], self.material.v[rows])
        # fail-map value at each candidate's centre (nearest simulation cell)
        iu = np.clip(np.round((centres[:, 0] + c.width_m / 2) / c.cell_m - .5).astype(int), 0, self.fail_map.shape[1] - 1)
        iv = np.clip(np.round(centres[:, 1] / c.cell_m - .5).astype(int), 0, self.fail_map.shape[0] - 1)
        damp = 1.0 - c.teacher_fail_damp * np.clip(self.fail_map[iv, iu], 0, 1)
        s_dep = (M @ under - 0.5 * (M @ over)) / norm * damp
        s_lev = c.teacher_level_weight * (M @ over) / norm * damp
        if s_lev.max() > s_dep.max():
            mode = 'LEVEL'; centre, direction, L = meta[int(np.argmax(s_lev))]
        else:
            need = float(under.sum() * c.cell_m ** 2)
            enough = self.material.blade_volume_m3 >= min(8e-6, .65 * need)
            mode = 'REUSE' if enough or need < 2e-6 else 'DEPOSIT'
            centre, direction, L = meta[int(np.argmax(s_dep))]
        direction = np.array(direction, float)
        if mode == 'LEVEL':
            # push the excess towards the nearest side, i.e. into the overtravel margin
            hw = c.width_m / 2
            dist = {(-1, 0): centre[0] + hw, (1, 0): hw - centre[0], (0, -1): centre[1], (0, 1): c.height_m - centre[1]}
            if direction @ np.array(min(dist, key=dist.get), float) < 0:
                direction = -direction
        elif direction[1] < 0:
            direction = -direction
        start = np.array(centre) - direction * L / 2; end = np.array(centre) + direction * L / 2
        mgn = self.teacher_edge_margin_m
        for pt in (start, end):
            pt[0] = np.clip(pt[0], -c.width_m / 2 + mgn, c.width_m / 2 - mgn)
            pt[1] = np.clip(pt[1], mgn, c.height_m - mgn)
        phi = np.arctan2(direction[1], direction[0]) + np.pi / 2
        load = self.teacher_load_ml if mode == 'DEPOSIT' else 0
        force = self.teacher_force_N
        profiles = ([(np.deg2rad(5), np.deg2rad(5)), (0.0, 0.0)] if mode == 'LEVEL' else
                    [(np.deg2rad(a_), np.deg2rad(b_)) for a_, b_ in ((25, 8), (20, 6), (10, 4))] + [(0.0, 0.0)])
        if self.executor is not None and hasattr(self.executor, 'plan'):
            # explicit priority: every tilt profile in place first, only then pull the stroke
            # towards the square centre (v0.7 teacher shrank before trying the next tilt)
            ctr = np.array([0.0, c.height_m / 2])
            for shrink in (0.0, 0.15, 0.3, 0.5):
                for pp in profiles:
                    cand = DecodedAction(mode, tuple(start + (ctr - start) * shrink), tuple(end + (ctr - end) * shrink),
                                         phi, 0, force, .06, load, *pp, 1.0)
                    ok = self.executor.plan(cand).ok
                    self.executor.stats['planned'] -= 1
                    if ok:
                        return self.encode(cand)
                    self.executor.stats['rejected'] -= 1
        return self.encode(DecodedAction(mode, tuple(start), tuple(end), phi, 0, force, .06, load,
                                         *profiles[0], 1.0))

    def step(self, action):
        if self.done:
            raise RuntimeError('step after episode finished')
        c = self.cfg; d = self.decode(action); self.cycles += 1; self.last_mode = MODES.index(d.mode)
        before = self.material.quality_cost(); before_waste = self.material.dropped_m3+self.material.outside_m3
        led0 = (self.material.dropped_m3, self.material.outside_m3, getattr(self.material, 'carry_loss_m3', 0.0))
        coverage_before = self.material.metrics()['coverage']
        steps_before = self.control_steps
        self._event('DECIDE', {'action': d.to_dict()})
        reward = -.02
        if d.mode == 'FINISH':
            m = self.material.metrics()
            rows, cols = self.material._score_rows, self.material._score_cols
            confident = self.scan['confidence'][rows, cols].mean() >= c.finish_confidence
            success = (m['coverage'] >= c.finish_coverage and m['rmse_mm'] <= c.finish_rmse_mm and
                       m['p95_error_mm'] <= c.finish_p95_mm and confident)
            reward += 100 if success else -c.finish_fail_penalty
            self.done = True; self.end_reason = 'finish'; self._event('FINISH', {'success': success})
            return self._observe(), reward, True, self.info(success)
        plan = self.executor.plan(d) if (self.executor is not None and d.mode != 'RESCAN') else None
        if plan is not None and not plan.ok and self.v3:
            # Safety layer (action projection): execute the nearest stroke the arm CAN do --
            # less tilt first, then pulled towards the square centre -- with a small penalty.
            # Nothing infeasible is ever executed; the episode is not ended by one bad proposal.
            # (v0.3 first training run without it: ~half the strokes rejected, episodes ended
            # after ~8 cycles by the stall rule; see change log C16.)
            # v0.7: the flat -0.5 was cheaper than a genuine risky stroke, so PPO learned to
            # propose infeasible actions and let this safety layer bail it out cheaply (P2-A
            # reward-hacking). Cost now scales with how much tilt/reach was given up, plus a
            # per-episode count penalty so repeatedly leaning on this layer keeps getting worse.
            proj = self._project(d)
            if proj is not None:
                d, plan, f_tilt, shrink = proj
                self.projected += 1
                reward -= (c.project_base_penalty * (1 + (1 - f_tilt) + 2 * shrink) +
                          c.project_count_penalty * self.projected)
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
                if self.v5:
                    self.control_steps += 20; self._event('LOAD')
                    self._event('SCOOP')
                    dynamic_v6 = self.v6 and plan is not None and getattr(self.executor, 'dynamic', False)
                    load = ({'deferred_to_executor': True} if dynamic_v6 else
                            self.material.feed(d.requested_load_ml, c.feed_normal_force_N,
                                               c.feed_scoop_depth_m, c.feed_scoop_distance_m,
                                               c.feed_scoop_speed_m_s))
                    self._event('LIFT_FROM_FEED', {'load': load})
                else:
                    self.control_steps += 50; self._event('LOAD_APPROACH')
                    load = self.material.load_random(d.requested_load_ml)
                    self._event('DISPENSE', {'load': load})
                self.reloads += 1; reward -= .2; self._event('TOOL_INSPECT')
            carry_before = self.material.dropped_m3
            if self.v5:
                carry_score = (float(getattr(plan, 'carry_face_up_min', 1.0))
                               if self.v6 and plan is not None else d.carry_face_up)
                self.last_carry_face_up = carry_score
                self._event('CARRY', {'face_up_score': carry_score})
                if not (self.v6 and plan is not None and getattr(self.executor, 'dynamic', False)):
                    self.material.transport(carry_score, c.carry_duration_s)
                self.control_steps += round(c.carry_duration_s*c.control_hz)
                self._event('PRECONTACT'); self._event('ROTATE_TO_WALL')
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
                if self.v6 and self.record and stats.get('trajectory'):
                    self._event('ARM_TRACE', {'trajectory': stats['trajectory']})
            else:
                # nominal path (training with the reach table, or no arm at all)
                stats = self.material.stroke(d.mode, d.start, d.end, d.blade_angle, d.bend_m,
                                             d.force_N, d.speed_m_s, callback=work_frame,
                                             pitch_start=d.pitch_start, pitch_end=d.pitch_end)
            self.last_stroke = {k: v for k, v in dict(stats.__dict__ if not isinstance(stats, dict) else stats).items()
                                if not isinstance(v, list)}
            self.control_steps += work_steps
            self._event('WORK', {'action': d.to_dict(), 'transfer': self.last_stroke})
            self.control_steps += 55
            if self.v5:
                self._event('SEPARATE'); self._event('RECOVER_RETURN')
            else:
                self._event('LIFT'); self._event('RETREAT')
            self._event('SCAN_RETURN')
            self._scan(); self._event('SCAN')
            after = self.material.quality_cost(); improvement = before-after
            coverage_after = self.material.metrics()['coverage']
            if c.loss_accounting == 'v0.8':
                # three ledgers, each charged per 18 mL: transport loss (the material's own
                # carry_loss_m3 counter), every other drop (feed overflow, air drop, wall slump)
                # and material pushed off the simulated grid
                d_carry = getattr(self.material, 'carry_loss_m3', 0.0) - led0[2]
                d_other = (self.material.dropped_m3 - led0[0]) - d_carry
                d_out = self.material.outside_m3 - led0[1]
                out_coef = c.waste_coef if c.outside_coef is None else c.outside_coef
                reward += 30*improvement - (c.waste_coef*d_other + out_coef*d_out + c.carry_loss_coef*d_carry)/18e-6
                carry_loss = 0.0          # already charged above
            else:
                carry_loss = self.material.dropped_m3-carry_before if self.v5 else 0.0
                waste = (self.material.dropped_m3+self.material.outside_m3)-before_waste-carry_loss
                reward += 30*improvement - c.waste_coef*waste/max(18e-6, 1e-12)
            # v0.7: explicit potential-based coverage shaping. Does not change the optimal
            # policy (it's a difference of the same potential at two successive states), but
            # gives PPO an earlier, denser signal than waiting for quality_cost/FINISH alone.
            reward += c.coverage_shaping_coef * (coverage_after - coverage_before)
            if self.v5 and c.loss_accounting != 'v0.8':
                material_scale = max(self.material.supplied_m3 + self.material.initial_m3, 18e-6)
                reward -= c.carry_loss_coef*carry_loss/material_scale
            # v0.7: graded penalty instead of a fixed -50 cliff at exactly max_force_N. The old
            # cliff made one risky stroke catastrophic relative to the cheap safety-layer path
            # (see _project() above), pushing PPO towards low-force, low-coverage strokes.
            over_force = max(0.0, (stats.peak_force_N or 0.0) - c.max_force_N)
            if over_force > 0:
                reward -= min(c.force_over_penalty_cap,
                              c.force_over_coef * (over_force / c.max_force_N) ** 2)
            if (stats.peak_force_N or 0.0) > c.force_terminate_mult * c.max_force_N:
                self.done = True; self.end_reason = 'force'
        self.last_improvement = improvement
        # loss ledgers (reported for every accounting mode; transport loss = material counter)
        d_carry = getattr(self.material, 'carry_loss_m3', 0.0) - led0[2]
        self.loss_ledger['carry_m3'] += d_carry
        self.loss_ledger['other_drop_m3'] += (self.material.dropped_m3 - led0[0]) - d_carry
        self.loss_ledger['outside_m3'] += self.material.outside_m3 - led0[1]
        if c.obs_memory or c.teacher_targeting == 'local':
            executed = not (plan is not None and not plan.ok) and d.mode != 'RESCAN'
            self._update_memory(d, executed, improvement)
        if self.v6:
            self.progress_history.append(float(improvement))
            self.progress_history = self.progress_history[-max(1, c.stall_window):]
            stalled = (len(self.progress_history) >= c.stall_window and
                       sum(max(x, 0.0) for x in self.progress_history) < c.progress_epsilon)
        else:
            stalled = improvement < c.progress_epsilon
        if stalled:
            self.stall_count += 1; reward -= min(.5*2**(self.stall_count-1), 8)
        else:
            self.stall_count = 0
        reward -= c.time_coef*(self.control_steps-steps_before)
        if self.control_steps >= self.budget and self.budget < c.max_steps and improvement >= c.progress_epsilon:
            self.budget = min(self.budget+c.extension_steps, c.max_steps)
        stall_done = self.cycles >= c.min_cycles_before_stall and self.stall_count >= c.stall_limit
        if (stall_done or self.cycles >= c.max_cycles or
                self.control_steps >= self.budget or self.control_steps >= c.max_steps):
            self.done = True
        if self.done and self.end_reason is None:
            self.end_reason = ('stall' if stall_done else 'cycles' if self.cycles >= c.max_cycles else 'budget')
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
                                     d.requested_load_ml, d.pitch_start * f_tilt, d.pitch_end * f_tilt,
                                     d.carry_face_up)
                plan = self.executor.plan(cand)
                if plan.ok:
                    return cand, plan, f_tilt, shrink
        return None

    def info(self, success=False):
        return {'success': bool(success), 'metrics': self.material.metrics(), 'cycles': self.cycles,
                'reloads': self.reloads, 'control_steps': self.control_steps,
                'budget': self.budget, 'stall_count': self.stall_count,
                'unreachable_strokes': getattr(self, 'unreachable', 0),
                'projected_strokes': getattr(self, 'projected', 0),
                'end_reason': getattr(self, 'end_reason', None),
                'loss_ledger_m3': dict(getattr(self, 'loss_ledger', {})),
                'volume_balance_m3': self.material.volume_balance(),
                'carry_loss_m3': float(getattr(self.material, 'carry_loss_m3', 0.0)),
                'wall_share': float(getattr(getattr(self.material, 'p', None),
                                            'lift_wall_fraction', self.cfg.lift_wall_fraction))}

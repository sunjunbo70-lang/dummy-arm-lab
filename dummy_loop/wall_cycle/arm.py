"""Execute whole-cycle strokes on the real Dummy V2 MuJoCo model (kinematic check).

The v0.2 experiment trained in a pure 2-D material world: nothing checked whether the
arm could reach a stroke, and the replay solved IK afterwards. Here every stroke the
manager proposes is planned on the same scene the single-stroke experiment uses
(models/dummy_v2.xml + rigid J6-reducer trowel + wall):

    level at the retreat distance -> stand the blade up while closing in -> touch with
    the trailing (lower) edge first -> work samples along the line/arc with the pitch
    ramp -> lift -> retreat

Each waypoint is solved by continuation IK (ToolIK, roll weakly constrained, J6 within
+/-180 deg). A stroke is rejected if any waypoint fails, or if any part of the arm other
than the blade comes within `clearance_m` of the wall (mj_geomDistance; the arm's visual
geoms have contype=0 so ordinary contacts would never report them).

plan() is the KINEMATIC feasibility check. execute() then runs the stroke through MuJoCo
DYNAMICS as a co-simulation with the mortar model (mortar.py), the same coupling the
single-stroke experiment uses:

    each sample: MuJoCo steps the arm (position servos, V2 torque limits, gravity comp.)
                 with the mortar reaction applied to the blade (xfrc_applied)
              -> the ACTUAL blade pose is read back (centre, pitch, trailing-edge gap)
              -> the mortar model advances at that actual pose and returns its support
                 force and drag for the next sample
              -> an admittance force loop moves the commanded stand-off so the measured
                 normal force (mortar support + steel-on-wall contact) tracks the target

So servo compliance, the weak J6 reducer, tracking lag and impacts all show up in the
coating and in the measured force. Moves between strokes (to the scan / loading poses)
are not simulated dynamically. Evidence level L1.
"""
from dataclasses import dataclass, field
import numpy as np
import mujoco

from ..wall.scene import SceneConfig, build_scene, wall_frame, tool_frame_matrix, joint_limits, blade_outline
from ..wall.controller import ToolIK, WallFrame
from .config import CycleConfig


@dataclass
class StrokePlan:
    ok: bool
    reason: str = ''
    q_approach: list = field(default_factory=list)
    q_work: list = field(default_factory=list)
    q_lift: list = field(default_factory=list)
    q_feed: list = field(default_factory=list)
    q_carry: list = field(default_factory=list)
    q_rotate: list = field(default_factory=list)
    carry_face_up_min: float = 1.0
    approach_distances_m: list = field(default_factory=list)


def scene_config(cfg: CycleConfig) -> SceneConfig:
    """Wall frame origin = centre of the work square (wall_center_z = square centre height)."""
    overrides = dict(wall_distance=cfg.scene_wall_distance_m, wall_lateral=cfg.area_centre_u_m,
                     wall_center_z=cfg.area_centre_z_m, wall_size=(1.2, 1.0))
    if cfg.tool_profile == 'lab_20260922':
        from ..wall.lab_tool import scene_config as lab_scene_config
        return lab_scene_config(**overrides)
    return SceneConfig(**overrides)


class ArmExecutor:
    dynamic = True          # WallCycleEnv runs execute() (MuJoCo co-simulation) for accepted strokes

    def __init__(self, cfg: CycleConfig, work_points=10, clearance_m=0.005, seed=0):
        self.cfg = cfg
        self.scene = scene_config(cfg)
        self.model, _ = build_scene(self.scene)
        self.data = mujoco.MjData(self.model)
        lo, hi = joint_limits(self.scene)
        lo, hi = np.array(lo, float), np.array(hi, float)
        lo[5], hi[5] = max(lo[5], -np.pi), min(hi[5], np.pi)
        self.ik = ToolIK(self.model, lo, hi, tool_frame_matrix(self.scene), w_roll=0.1)
        self.frame = WallFrame(*wall_frame(self.scene))
        self.outline = blade_outline(self.scene)
        self.hw = self.outline['half_width']
        self.xc = (self.outline['x_tip'] + self.outline['x_back']) / 2
        self.blade_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'blade')
        self.n_sub_min = 5
        gid = lambda n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, n)
        self.wall = gid('wall_geom')
        tool = {gid(n) for n in ('blade_geom', 'blade_tip_geom', 'neck_geom')} - {-1}
        self.arm_geoms = [g for g in range(self.model.ngeom) if g not in tool | {self.wall, gid('floor')}]
        self.blade_geoms = tool - {gid('neck_geom')}
        self.work_points = work_points
        self.clearance = clearance_m
        self.rng = np.random.default_rng(seed)
        self.q_scan = self._solve_seeded(self._uvn(0.0, cfg.height_m / 2, -cfg.wall_clearance_m), 0.0, 0.0)
        self.q_last = self.q_scan
        self.q_feed = np.clip(np.asarray(cfg.feed_pose_q_rad, float), self.ik.lo, self.ik.hi)
        self.stats = {'planned': 0, 'rejected': 0}

    def reset(self):
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:6] = self.q_scan
        mujoco.mj_forward(self.model, self.data)
        self.q_last = self.q_scan.copy()

    # --------------------------------------------------------------- geometry
    def _uvn(self, u, v, n):
        """Work-square coordinates (u centred, v from the square's bottom) -> wall frame."""
        return np.array([u, v - self.cfg.height_m / 2, n])

    def _pose(self, uvn, psi, pitch):
        """Target for the IK tool point such that the BLADE CENTRE sits at uvn. The model's tool
        point ('tcp') is 7 mm from the blade centre along the blade; the mortar model works with
        the blade centre, so the offset is removed here."""
        R = self.frame.tool_rotation(psi, pitch)
        return self.frame.to_world(uvn) - R[:, 0] * self.xc, R

    def blade_pose(self, q):
        """Blade face-centre and semantic rotation. +local-y is the material-face normal."""
        q=np.asarray(q,float).copy(); oldq=self.data.qpos.copy(); oldv=self.data.qvel.copy()
        self.data.qpos[:] = 0; self.data.qpos[:6] = q
        mujoco.mj_forward(self.model, self.data)
        R = self.data.xmat[self.blade_body].reshape(3, 3).copy()
        p = self.data.xpos[self.blade_body].copy() + R[:, 1] * self.scene.trowel_thickness + R[:, 0] * self.xc
        self.data.qpos[:]=oldq;self.data.qvel[:]=oldv;mujoco.mj_forward(self.model,self.data)
        return p, R

    @staticmethod
    def _slerp_R(A, B, t):
        """Small dependency-free SO(3) interpolation used by deterministic motion skills."""
        D = A.T @ B
        c = np.clip((np.trace(D) - 1) / 2, -1.0, 1.0); a = float(np.arccos(c))
        if a < 1e-8:
            return A.copy()
        K = (D - D.T) / (2 * np.sin(a))
        return A @ (np.eye(3) + np.sin(t*a)*K + (1-np.cos(t*a))*(K@K))

    def _solve_pose(self, blade_centre, R, seed, max_jump_deg=20):
        p = np.asarray(blade_centre) - R[:, 0] * self.xc
        q, ok, _, _ = self.ik.solve(p, R, seed, iters=150)
        if not ok or not self.clear_of_wall(q):
            return None
        if np.rad2deg(np.max(np.abs(q-seed))) > max_jump_deg:
            return None
        return q

    def _solve_pose_seeded(self, blade_centre, R, preferred):
        seeds=[np.asarray(preferred,float),np.zeros(6)]
        seeds += [self.ik.lo+(self.ik.hi-self.ik.lo)*self.rng.random(6) for _ in range(24)]
        p=np.asarray(blade_centre)-R[:,0]*self.xc
        best=None; score=np.inf
        for seed in seeds:
            q,ok,_,_=self.ik.solve(p,R,seed,iters=250)
            if ok and self.clear_of_wall(q):
                s=float(np.max(np.abs(q-preferred)))
                if s<score: best,score=q,s
        return best

    def clear_of_wall(self, q):
        oldq=self.data.qpos.copy();oldv=self.data.qvel.copy()
        self.data.qpos[:] = 0; self.data.qpos[:6] = q
        mujoco.mj_forward(self.model, self.data)
        ok=all(mujoco.mj_geomDistance(self.model, self.data, g, self.wall, 0.05, None) >= self.clearance
               for g in self.arm_geoms)
        self.data.qpos[:]=oldq;self.data.qvel[:]=oldv;mujoco.mj_forward(self.model,self.data)
        return ok

    def _solve(self, uvn, psi, pitch, seed, iters=60):
        p, R = self._pose(uvn, psi, pitch)
        q, ok, _, _ = self.ik.solve(p, R, seed, iters=iters)
        return q, ok

    def _step(self, uvn, psi, pitch, q, max_jump_deg=None):
        """Continue from q; if that fails, reseed. With max_jump_deg set (contact phases),
        a reseeded solution may not jump more than that on any joint -- otherwise the arm
        would switch wrist branch while touching the wall."""
        q1, ok = self._solve(uvn, psi, pitch, q, iters=100)
        if ok and self.clear_of_wall(q1):
            return q1
        q2 = self._solve_seeded(uvn, psi, pitch, extra=(q,))
        if q2 is None:
            return None
        if max_jump_deg is not None and np.rad2deg(np.max(np.abs(q2 - q))) > max_jump_deg:
            return None
        return q2

    def _solve_seeded(self, uvn, psi, pitch, extra=()):
        seeds = list(extra) + [np.zeros(6)] + [self.ik.lo + (self.ik.hi - self.ik.lo) * self.rng.random(6)
                                               for _ in range(12)]
        for s in seeds:
            q, ok = self._solve(uvn, psi, pitch, s, iters=150)
            if ok and self.clear_of_wall(q):
                return q
        return None

    def contact_n(self, pitch, gap=0.0):
        """Wall-normal coordinate of the tool point (blade-face centre) that puts the trailing
        edge at `gap` from the wall. The tool point is the face centre, so a tilted blade must
        stand off by half-width x sin(pitch) -- otherwise its trailing edge is inside the wall
        (at 30 deg: 6.9 mm; the first draft of this file got that wrong)."""
        return -(gap + self.hw * np.sin(abs(pitch)))

    @staticmethod
    def blade_psi(phi):
        """wall_cycle blade angle (long axis from +u) -> WallFrame psi (psi=0: long axis along -u)."""
        return float((phi - np.pi + np.pi) % (2 * np.pi) - np.pi)

    def signed_pitch(self, psi, direction, pitch):
        """Tilt so that the edge BEHIND the motion is pressed (lower edge first for an upward stroke)."""
        zb = self.frame.R.T @ self.frame.tool_rotation(psi, 0.0)[:, 2]     # blade width axis, wall frame
        s = 1.0 if float(zb[:2] @ direction) >= 0 else -1.0
        return s * pitch

    # --------------------------------------------------------------- planning
    def plan(self, d):
        """d: DecodedAction. Returns a StrokePlan; ok=False if the real arm cannot do it."""
        c = self.cfg; self.stats['planned'] += 1
        p0, p2 = np.asarray(d.start, float), np.asarray(d.end, float)
        vec = p2 - p0; L = np.linalg.norm(vec)
        direction = vec / L if L > 1e-9 else np.array([0.0, 1.0])
        normal = np.array([-direction[1], direction[0]])
        pc = (p0 + p2) / 2 + normal * d.bend_m
        psi = self.blade_psi(d.blade_angle)
        s0 = self.signed_pitch(psi, direction, d.pitch_start)
        s1 = self.signed_pitch(psi, direction, d.pitch_end)
        stand = -c.approach_clearance_m
        # 1) approach, planned BACKWARDS from the wall: solve the contact pose first (seeded),
        #    then walk outwards by continuation -- stand-off with the blade stood up, then the
        #    level retreat pose. Walking out from a known good pose is far more reliable than
        #    guessing a pose 8 cm off the wall near the edge of the reach. The retreat distance
        #    shrinks to 4 cm where 8 cm is out of reach (4 cm still clears a blade at 35 deg).
        q_c = self._solve_seeded(self._uvn(*p0, self.contact_n(s0)), psi, s0, extra=(self.q_last, self.q_scan))
        if q_c is None:
            return self._reject('contact pose unreachable or arm hits wall')
        q = q_c; path = []
        for f in np.linspace(0, 1, 5)[1:]:                      # back off to the stand-off, tilted
            q = self._step(self._uvn(*p0, self.contact_n(s0) + (stand - self.contact_n(s0)) * f), psi, s0, q,
                           max_jump_deg=20)
            if q is None:
                return self._reject('cannot stand off in front of the start point')
        q_stand = q
        q_ret = None
        for dist in (c.wall_clearance_m, 0.06, 0.04):
            qq = q_stand
            for f in np.linspace(0, 1, 5)[1:]:                  # level the blade while backing off
                # max_jump: without it the solver may reseed onto the other wrist branch (J6 turned
                # ~180 deg); the weak J6 then never catches up and the whole stroke runs with the
                # blade turned round (seen in the first co-simulation: constant 14 mm offset)
                qq = self._step(self._uvn(*p0, stand + (-dist - stand) * f), psi, s0 * (1 - f), qq,
                                max_jump_deg=20)
                if qq is None:
                    break
            if qq is not None:
                q_ret = qq; break
        if q_ret is None:
            return self._reject('no retreat pose in front of the start point')
        qa = [q_ret, q_stand, q_c]
        q_feed_path=[]; q_carry=[]; q_rotate=[]; carry_min=1.0; approach_d=[]
        if c.physics == 'v0.6':
            # The user-validated loading posture has the material face horizontal
            # (dot(face normal,+Z)=0.9955). Build a Cartesian, face-up transfer to a
            # rotation waypoint; only there rotate towards the wall and approach once.
            feed_p, feed_R = self.blade_pose(self.q_feed)
            face_up = float(feed_R[:, 1] @ np.array([0.0, 0.0, 1.0]))
            if face_up < np.cos(np.deg2rad(c.face_up_target_deg)):
                return self._reject(f'feed pose is not face-up ({np.rad2deg(np.arccos(face_up)):.1f} deg)')
            q_feed_path=[self.q_feed.copy()]
            q_pre=None; pre_p=None; rot_dist=None
            for dist in c.rotation_clearance_range_m:
                # Carry high and face-up to a point close to the wall. The V2 cannot
                # keep this orientation at low wall cells; rotation therefore happens
                # at a high, reachable waypoint and the short descent is wall-facing.
                centre = self.frame.to_world(self._uvn(p0[0], c.height_m / 2, -dist))
                centre[2] = max(0.45, feed_p[2])
                q_end=self._solve_pose_seeded(centre,feed_R,self.q_feed)
                trial=[]; good=q_end is not None
                if good:
                    for t in np.linspace(0,1,25)[1:]:
                        q=(1-t)*self.q_feed+t*q_end
                        _,Rt=self.blade_pose(q)
                        score=float(Rt[:,1]@[0,0,1])
                        if score<np.cos(np.deg2rad(c.face_up_hard_deg)) or not self.clear_of_wall(q):
                            good=False; break
                        trial.append(q.copy())
                if good:
                    q_pre, pre_p, rot_dist = q_end, centre, dist; q_carry=trial; break
            if q_pre is None:
                return self._reject('no face-up wall rotation prepose')
            for q in q_carry:
                _, RR=self.blade_pose(q); carry_min=min(carry_min,float(RR[:,1]@[0,0,1]))
            # Rotate only after reaching the high near-wall corridor. Interpolate to the
            # already collision-checked retreat posture and audit the ACTUAL blade normal;
            # it may turn from up to vertical but must never point down.
            q=q_pre.copy()
            for t in np.linspace(0,1,25)[1:]:
                qn=(1-t)*q_pre+t*q_ret
                if not self.clear_of_wall(qn): return self._reject('near-wall rotation collision')
                _,Rt=self.blade_pose(qn)
                if float(Rt[:,1]@[0,0,1]) < -1e-3:
                    return self._reject('material face turns downward during near-wall rotation')
                q=qn; q_rotate.append(q.copy())
            # Continue from the retreat posture to stand-off and contact exactly once.
            q_stand2=self._step(self._uvn(*p0,stand),psi,s0,q,max_jump_deg=25)
            if q_stand2 is None: return self._reject('monotonic stand-off approach breaks')
            q_c2=self._step(self._uvn(*p0,self.contact_n(s0)),psi,s0,q_stand2,max_jump_deg=20)
            if q_c2 is None: return self._reject('monotonic contact approach breaks')
            q_stand,q_c=q_stand2,q_c2
            qa=[q_rotate[-1],q_stand,q_c]
            approach_d=[rot_dist,c.approach_clearance_m,abs(self.contact_n(s0))]
        q = q_c
        # 2) work: sample the same Bezier curve as the material model, with the pitch ramp
        n_s = c.stroke_samples
        qw_key, ts = [], np.linspace(0, 1, self.work_points)
        for t in ts:
            centre = (1-t)**2*p0 + 2*(1-t)*t*pc + t**2*p2
            pt = s0 + (s1 - s0) * t
            q = self._step(self._uvn(*centre, self.contact_n(pt)), psi, pt, q, max_jump_deg=20)
            if q is None:
                return self._reject(f'work path breaks at t={t:.2f}')
            qw_key.append(q)
        qw_key = np.array(qw_key)
        t_all = np.linspace(0, 1, n_s)
        q_work = np.array([np.interp(t_all, ts, qw_key[:, j]) for j in range(6)]).T
        # 3) lift and retreat
        ql = []
        q = self._step(self._uvn(*p2, stand), psi, s1, q, max_jump_deg=20)
        if q is None:
            return self._reject('lift unreachable')
        ql.append(q); q_up = q
        for dist in (c.wall_clearance_m, 0.06, 0.04):
            q = self._step(self._uvn(*p2, -dist), psi, 0.0, q_up, max_jump_deg=20)
            if q is not None:
                break
        if q is None:
            return self._reject('retreat unreachable')
        ql.append(q)
        self.q_last = q
        plan = StrokePlan(True, '', [x.copy() for x in qa], [x.copy() for x in q_work], [x.copy() for x in ql],
                          [x.copy() for x in q_feed_path], [x.copy() for x in q_carry],
                          [x.copy() for x in q_rotate], carry_min, approach_d)
        plan.meta = dict(p0=p0, p2=p2, pc=pc, psi=psi, s0=s0, s1=s1, direction=direction)
        return plan

    def _reject(self, why):
        self.stats['rejected'] += 1
        return StrokePlan(False, why)

    # --------------------------------------------------------------- dynamics (co-simulation)
    def blade_state(self, e_w):
        """Actual blade from MuJoCo, in work-square coords: centre (u, v), pitch (>0 = trailing
        edge closer to the wall), trailing-edge gap, and the wall-frame edge points."""
        R = self.data.xmat[self.blade_body].reshape(3, 3)
        th = self.scene.trowel_thickness
        p = self.data.xpos[self.blade_body] + R[:, 1] * th + R[:, 0] * self.xc
        a = self.frame.from_world(p - self.hw * R[:, 2])
        b = self.frame.from_world(p + self.hw * R[:, 2])
        e3 = np.array([e_w[0], e_w[1], 0.0])
        trail, lead = (a, b) if (a - b) @ e3 < 0 else (b, a)
        centre = self.frame.from_world(p)
        pitch = float(np.arcsin(np.clip((trail[2] - lead[2]) / (2 * self.hw), -1, 1)))
        gap = max(-float(trail[2]), 0.0)
        return np.array([centre[0], centre[1] + self.cfg.height_m / 2]), pitch, gap

    def rigid_contact_N(self):
        f = np.zeros(6); total = 0.0
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if self.wall in (c.geom1, c.geom2) and ({c.geom1, c.geom2} & self.blade_geoms):
                mujoco.mj_contactForce(self.model, self.data, i, f); total += abs(f[0])
        return total

    def _run(self, q_target, seconds, force_world=None):
        n = max(self.n_sub_min, int(round(seconds / self.model.opt.timestep)))
        lo, hi = self.ik.lo, self.ik.hi
        self.data.ctrl[:] = np.clip(q_target, lo, hi)
        self.data.xfrc_applied[:] = 0
        if force_world is not None:
            self.data.xfrc_applied[self.blade_body, :3] = force_world
        peak = 0.0
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)
            peak = max(peak, self.rigid_contact_N())
        return peak

    def _move_time(self, q, joint_speed_deg_s=60.0, settle_s=0.15):
        """Time for a point-to-point move: the largest joint change at 60 deg/s plus settling."""
        dq = np.rad2deg(np.max(np.abs(np.asarray(q) - self.data.qpos[:6])))
        return dq / joint_speed_deg_s + settle_s

    def execute(self, plan, d, mortar, stats, callback=None, k_force=0.00008):
        """Run a planned stroke through MuJoCo dynamics coupled with the mortar model."""
        c = self.cfg; m = plan.meta
        R_w = self.frame.R
        trace=[]
        def mark(phase):
            _,R=self.blade_pose(self.data.qpos[:6])
            trace.append({'phase':phase,'q':self.data.qpos[:6].copy(),
                          'face_up_score':float(R[:,1]@np.array([0.,0.,1.])),
                          'blade':mortar.blade.copy(),'wall':mortar.wall.copy(),
                          'metrics':mortar.metrics()})
        if c.physics == 'v0.6':
            # Preserve one continuous MuJoCo state across the complete skill. Loading is
            # permitted only at the verified face-up pose; transport loss uses measured
            # orientation after every dynamically executed waypoint.
            if d.mode == 'DEPOSIT' and d.requested_load_ml > 0:
                self._run(self.q_feed,self._move_time(self.q_feed,joint_speed_deg_s=20.0,settle_s=0.8));mark('FEED_ALIGN_UP')
                score=trace[-1]['face_up_score']
                if score < np.cos(np.deg2rad(c.face_up_hard_deg)):
                    raise RuntimeError(f'executed feed pose is not face-up: score={score:.4f}')
                stats['feed']=mortar.feed(d.requested_load_ml,c.feed_normal_force_N,
                                          c.feed_scoop_depth_m,c.feed_scoop_distance_m,
                                          c.feed_scoop_speed_m_s)
                mark('FEED_SCOOP')
            for q in plan.q_carry:
                dt=self._move_time(q);self._run(q,dt);mark('CARRY_FACE_UP')
                mortar.transport(trace[-1]['face_up_score'],dt,stats=stats)
            for q in plan.q_rotate:
                dt=self._move_time(q);self._run(q,dt);mark('ROTATE_TO_WALL')
                mortar.transport(trace[-1]['face_up_score'],dt,stats=stats)
        else:
            # Historical v0.3/v0.5 behaviour, retained for reproducibility.
            self.data.qpos[:] = 0; self.data.qvel[:] = 0; self.data.qpos[:6] = plan.q_approach[0]
            mujoco.mj_forward(self.model, self.data)
        mortar.begin_stroke(d.blade_angle, m['direction'])
        mortar.air(d.pitch_start, stats)                       # blade stood up in front of the wall
        peak = 0.0
        for q in plan.q_approach[1:]:
            peak = max(peak, self._run(q, self._move_time(q))); mark('WALL_APPROACH')
        L = float(np.linalg.norm(m['p2'] - m['p0']))
        duration = max(L / max(d.speed_m_s, 1e-4), 0.05)
        dt = duration / c.stroke_samples
        e_w = mortar.e_w
        # initial stand-off from the quasi-static force balance on the current load
        from .mortar import solve_gap
        g_cmd, _ = solve_gap(mortar._rows(), abs(m['s0']), d.force_N, c.blade_cell_m, mortar.p)
        q = plan.q_approach[-1].copy()
        f_mortar = np.zeros(3); qs, forces, gaps, track = [], [], [], []; ik_fail = 0
        for k, t in enumerate(np.linspace(0, 1, c.stroke_samples)):
            centre = (1-t)**2*m['p0'] + 2*(1-t)*t*m['pc'] + t**2*m['p2']
            pitch = m['s0'] + (m['s1'] - m['s0']) * t
            p_t, R_t = self._pose(self._uvn(*centre, self.contact_n(pitch, g_cmd)), m['psi'], pitch)
            # continuation IK from the last command. Accept the result even if it has not met the
            # 2 mm tolerance yet: keeping the old command (first draft) made the arm trail the
            # path by ~3 samples (a constant 14 mm) whatever the servo gains were.
            q, ok, _, _ = self.ik.solve(p_t, R_t, q, iters=60)
            ik_fail += (not ok)
            f_rigid = self._run(q, dt, f_mortar)
            actual_centre, actual_pitch, gap = self.blade_state(e_w)
            res = mortar.contact(actual_centre, abs(actual_pitch), d.speed_m_s, gap_m=gap, stats=stats)
            f_meas = res.support_N + self.rigid_contact_N()
            # mortar pushes the blade out of the wall (-n) and drags against the motion
            motion = R_w[:, 0] * e_w[0] + R_w[:, 1] * e_w[1]
            f_mortar = -res.support_N * R_w[:, 2] - res.drag_N * motion
            # admittance: too much force -> stand further off
            g_cmd = float(np.clip(g_cmd + k_force * (f_meas - d.force_N), -0.002, 0.010))
            qs.append(self.data.qpos[:6].copy()); forces.append(f_meas); gaps.append(gap)
            track.append(float(np.linalg.norm(actual_centre - centre)))
            peak = max(peak, f_rigid + res.support_N)
            if callback is not None:
                callback(k, actual_centre.copy(), mortar.wall.copy(), mortar.blade.copy(), self.data.qpos[:6].copy())
            mark('WORK_STEP')
        mortar.end_stroke(stats)
        for q in plan.q_lift:
            self._run(q, self._move_time(q)); mark('SEPARATE')
        mortar.air(0.0, stats)
        if c.physics == 'v0.6':
            dt=self._move_time(self.q_scan,joint_speed_deg_s=30.0,settle_s=0.4)
            self._run(self.q_scan,dt);mark('SCAN_RETURN')
            mortar.transport(trace[-1]['face_up_score'],dt,stats=stats)
        self.q_last = self.data.qpos[:6].copy()
        stats['peak_force_N'] = peak
        stats['force_mean_N'] = float(np.mean(forces)); stats['force_target_N'] = d.force_N
        stats['force_rmse_N'] = float(np.sqrt(np.mean((np.asarray(forces) - d.force_N) ** 2)))
        stats['tracking_max_mm'] = 1000 * max(track)
        stats['tracking_mm'] = [1000 * x for x in track]
        stats['ik_not_converged'] = ik_fail
        stats['forces_N'] = forces
        stats['gaps_mm'] = [1000 * g for g in gaps]
        stats['q_work'] = qs
        stats['trajectory'] = trace
        if c.physics == 'v0.6':
            carry = [x['face_up_score'] for x in trace
                     if x['phase'] in ('FEED_ALIGN_UP', 'FEED_SCOOP', 'CARRY_FACE_UP')]
            rotate = [x['face_up_score'] for x in trace if x['phase'] == 'ROTATE_TO_WALL']
            stats['carry_face_up_min'] = float(min(carry or [1.0]))
            stats['rotation_face_up_min'] = float(min(rotate or [1.0]))
            stats['face_down_frames'] = int(sum(x['face_up_score'] < -1e-3 for x in trace
                                                if x['phase'] in ('FEED_ALIGN_UP', 'FEED_SCOOP',
                                                                  'CARRY_FACE_UP', 'ROTATE_TO_WALL')))
            stats['approach_reversal_count'] = int(sum(
                b > a + 5e-4 for a, b in zip(plan.approach_distances_m,
                                             plan.approach_distances_m[1:])))
        return stats

"""Continuous-time v0.9 executor. Historical ArmExecutor is left unchanged."""
import numpy as np
import mujoco
from ..wall_cycle.arm import ArmExecutor

class TimedArmExecutor(ArmExecutor):
    def __init__(self,*args,trace_enabled=False,**kwargs):
        super().__init__(*args,**kwargs)
        self.trace_enabled=trace_enabled;self.transport_dt_s=.02;self.trace_dt_s=.05
        # Proxy sensing does not require the legacy vertical scan posture.
        # Use the verified face-up staging posture between strokes.
        self.q_scan=self.q_feed.copy();self.q_last=self.q_scan.copy()
        self.dense_trace=[];self._phase='IDLE';self._mortar=None;self._stats=None
        self._last_trace_s=-1.;self.actual_elapsed_s=0.
        self.transport_samples=0;self.loaded_tilt_samples=0

    def reset(self):
        super().reset()
        self.dense_trace=[];self._phase="IDLE";self._mortar=None;self._stats=None
        self._last_trace_s=-1.;self.actual_elapsed_s=0.
        self.transport_samples=0;self.loaded_tilt_samples=0

    def sample(self,phase=None):
        if not self.trace_enabled or self._mortar is None:return
        R=self.data.xmat[self.blade_body].reshape(3,3).copy();m=self._mortar
        f={'time_s':float(self.data.time),'phase':phase or self._phase,'q':self.data.qpos[:6].copy(),
           'qvel':self.data.qvel[:6].copy(),'face_up_score':float(R[:,1]@np.array([0.,0.,1.])),
           'wall':m.wall.copy(),'blade':m.blade.copy(),'metrics':m.metrics()}
        if self.dense_trace and abs(self.dense_trace[-1]['time_s']-f['time_s'])<1e-10:self.dense_trace[-1]=f
        else:self.dense_trace.append(f)
        self._last_trace_s=self.data.time

    def _run(self,q_target,seconds,force_world=None):
        n=max(self.n_sub_min,int(round(seconds/self.model.opt.timestep)))
        start_ctrl=self.data.qpos[:6].copy();target_ctrl=np.clip(q_target,self.ik.lo,self.ik.hi)
        self.data.xfrc_applied[:]=0
        if force_world is not None:self.data.xfrc_applied[self.blade_body,:3]=force_world
        transport=self._phase in ('FEED_TRANSIT','FEED_ALIGN_UP','CARRY_FACE_UP','ROTATE_TO_WALL','SCAN_RETURN')
        elapsed=0.;peak=0.;t0=float(self.data.time)
        stride=max(1,int(round(self.transport_dt_s/self.model.opt.timestep)))
        for j in range(n):
            u=(j+1)/n;blend=u*u*(3-2*u)
            self.data.ctrl[:]=start_ctrl+(target_ctrl-start_ctrl)*blend
            mujoco.mj_step(self.model,self.data);elapsed+=self.model.opt.timestep
            peak=max(peak,self.rigid_contact_N())
            if (j+1)%stride==0 or j==n-1:
                if transport and self._mortar is not None:
                    R=self.data.xmat[self.blade_body].reshape(3,3);score=float(R[:,1]@np.array([0.,0.,1.]))
                    self.transport_samples+=1
                    self.loaded_tilt_samples+=int(score<np.cos(np.deg2rad(15)) and self._mortar.blade_volume_m3>1e-7)
                    self._mortar.transport(score,elapsed,dt_s=self.transport_dt_s,stats=self._stats)
                elapsed=0.
                if self.data.time-self._last_trace_s>=self.trace_dt_s-1e-9:self.sample()
        self.actual_elapsed_s+=float(self.data.time)-t0
        self.sample()
        return peak

    def execute(self, plan, d, mortar, stats, callback=None, k_force=0.00008):
        """Run a planned stroke through MuJoCo dynamics coupled with the mortar model."""
        c = self.cfg; m = plan.meta
        R_w = self.frame.R
        trace=[]
        self._mortar=mortar; self._stats=stats; self.dense_trace=[]
        self._last_trace_s=-1.;self.sample("START")
        def mark(phase):
            _,R=self.blade_pose(self.data.qpos[:6])
            self.sample(phase)
            trace.append({'time_s':float(self.data.time),'phase':phase,'q':self.data.qpos[:6].copy(),
                          'face_up_score':float(R[:,1]@np.array([0.,0.,1.])),
                          'blade':mortar.blade.copy(),'wall':mortar.wall.copy(),
                          'metrics':mortar.metrics()})
        if c.physics == 'v0.6':
            # Preserve one continuous MuJoCo state across the complete skill. Loading is
            # permitted only at the verified face-up pose; transport loss uses measured
            # orientation after every dynamically executed waypoint.
            if True: # every carry path starts at q_feed, including reuse/level
                # v0.7: was one _run() straight to q_feed with a single mark() -- physically
                # simulated but only its endpoint ever recorded, so replays showed a teleport.
                # q_feed_path (built once in __init__) gives this a dense, checked trajectory.
                # v0.8: only the final (verified face-up) waypoint is FEED_ALIGN_UP; the frames on
                # the way from the scan pose are FEED_TRANSIT. The blade can still carry residual
                # material here, so with feed_transit_physics the measured orientation drives the
                # same transport model as CARRY/ROTATE/SCAN_RETURN (v0.6/v0.7 skipped it).
                feed_path=self._dense_path(self.data.qpos[:6].copy(),self.q_feed,n=30)
                if feed_path is None:raise RuntimeError("No continuous feed-entry path")
                n_feed = len(feed_path)
                for k_feed, q in enumerate(feed_path):
                    dt = self._move_time(q, joint_speed_deg_s=20.0, settle_s=0.05)
                    self._phase="FEED_TRANSIT"
                    self._run(q, dt)
                    mark('FEED_ALIGN_UP' if k_feed == n_feed - 1 else 'FEED_TRANSIT')

                score=trace[-1]['face_up_score']
                if score < np.cos(np.deg2rad(c.face_up_hard_deg)):
                    raise RuntimeError(f'executed feed pose is not face-up: score={score:.4f}')
                if d.mode == 'DEPOSIT' and d.requested_load_ml > 0:
                    stats['feed']=mortar.feed(d.requested_load_ml,c.feed_normal_force_N,
                                              c.feed_scoop_depth_m,c.feed_scoop_distance_m,
                                              c.feed_scoop_speed_m_s)
                    mark('FEED_SCOOP')
            for q in plan.q_carry:
                self._phase='CARRY_FACE_UP'
                dt=self._move_time(q);self._run(q,dt);mark('CARRY_FACE_UP')

            for q in plan.q_rotate:
                self._phase='ROTATE_TO_WALL'
                dt=self._move_time(q);self._run(q,dt);mark('ROTATE_TO_WALL')

        else:
            # Historical v0.3/v0.5 behaviour, retained for reproducibility.
            self.data.qpos[:] = 0; self.data.qvel[:] = 0; self.data.qpos[:6] = plan.q_approach[0]
            mujoco.mj_forward(self.model, self.data)
        mortar.begin_stroke(d.blade_angle, m['direction'])
        mortar.air(d.pitch_start, stats)                       # blade stood up in front of the wall
        peak = 0.0
        for q in plan.q_approach[1:]:
            self._phase='WALL_APPROACH'
            peak = max(peak, self._run(q, self._move_time(q))); mark('WALL_APPROACH')
        L = float(np.linalg.norm(m['p2'] - m['p0']))
        duration = max(L / max(d.speed_m_s, 1e-4), 0.05)
        dt = duration / c.stroke_samples
        e_w = mortar.e_w
        # initial stand-off from the quasi-static force balance on the current load
        from ..wall_cycle.mortar import solve_gap
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
            self._phase='WORK_STEP'
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
            self._phase='SEPARATE'
            self._run(q, self._move_time(q)); mark('SEPARATE')
        mortar.air(0.0, stats)
        if c.physics == 'v0.6':
            # v0.7: was one _run() straight to q_scan with a single mark() -- same teleport
            # issue as the feed-alignment move above. plan.q_scan_path (computed per stroke in
            # plan(), since the retreat pose it starts from varies) gives this a dense trajectory.
            for q in (plan.q_scan_path or [self.q_scan]):
                dt=self._move_time(q,joint_speed_deg_s=30.0,settle_s=0.05)
                self._phase='SCAN_RETURN'
                self._run(q,dt);mark('SCAN_RETURN')

        self.q_last = self.data.qpos[:6].copy()
        stats['feed_path_degraded'] = self.feed_path_degraded
        stats['peak_force_N'] = peak
        stats['force_mean_N'] = float(np.mean(forces)); stats['force_target_N'] = d.force_N
        stats['force_rmse_N'] = float(np.sqrt(np.mean((np.asarray(forces) - d.force_N) ** 2)))
        stats['tracking_max_mm'] = 1000 * max(track)
        stats['tracking_mm'] = [1000 * x for x in track]
        stats['ik_not_converged'] = ik_fail
        stats['forces_N'] = forces
        stats['gaps_mm'] = [1000 * g for g in gaps]
        stats['q_work'] = qs
        stats['trajectory'] = self.dense_trace if self.trace_enabled else trace
        stats['time_s']=float(self.data.time)
        stats['transport_samples']=self.transport_samples
        stats['loaded_tilt_samples']=self.loaded_tilt_samples
        if c.physics == 'v0.6':
            carry = [x['face_up_score'] for x in trace
                     if x['phase'] in ('FEED_ALIGN_UP', 'FEED_SCOOP', 'CARRY_FACE_UP')]
            rotate = [x['face_up_score'] for x in trace if x['phase'] == 'ROTATE_TO_WALL']
            stats['carry_face_up_min'] = float(min(carry or [1.0]))
            stats['rotation_face_up_min'] = float(min(rotate or [1.0]))
            stats['face_down_frames'] = int(sum(x['face_up_score'] < -1e-3 for x in trace
                                                if x['phase'] in ('FEED_ALIGN_UP', 'FEED_SCOOP',
                                                                  'CARRY_FACE_UP', 'ROTATE_TO_WALL')))
            # v0.8 diagnostic: residual material carried through the feed transit while tilted
            # past the hard face-up limit (0.1 mL threshold is a diagnostic, not a calibrated one).
            cos_hard = np.cos(np.deg2rad(c.face_up_hard_deg)); A_b = mortar.blade_cell_area
            stats['feed_transit_loaded_tilted_frames'] = int(sum(
                x['face_up_score'] < cos_hard and x['blade'].sum() * A_b > 1e-7
                for x in trace if x['phase'] == 'FEED_TRANSIT'))
            stats['approach_reversal_count'] = int(sum(
                b > a + 5e-4 for a, b in zip(plan.approach_distances_m,
                                             plan.approach_distances_m[1:])))
        return stats


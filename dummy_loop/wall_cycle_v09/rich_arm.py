"""Rich CONTACT executor, based on the frozen legacy planner and timed controller.
The same Contact.at(t) defines geometry, orientation, target force and speed in both backends.
"""
import numpy as np
from ..wall_cycle.arm import StrokePlan
from .timed_arm import TimedArmExecutor

class RichArm(TimedArmExecutor):
    def plan(self, d):
        """d: DecodedAction. Returns a StrokePlan; ok=False if the real arm cannot do it."""
        c = self.cfg; self.stats['planned'] += 1
        p0, p2 = np.asarray(d.start, float), np.asarray(d.end, float)
        vec = p2 - p0; L = np.linalg.norm(vec)
        direction = vec / L if L > 1e-9 else np.array([0.0, 1.0])
        normal = np.array([-direction[1], direction[0]])
        pc = (p0 + p2) / 2 + normal * d.bend_m
        psi = self.blade_psi(d.blade_angle)
        s0 = self.signed_pitch(psi, d.contact.at(0)[1], d.pitch_start)
        s1 = self.signed_pitch(self.blade_psi(d.contact.at(1)[2]), d.contact.at(1)[1], d.pitch_end)
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
            centre,tangent,phi,pt,_,_=d.contact.at(t)
            psi=self.blade_psi(phi);pt=self.signed_pitch(psi,tangent,pt)
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
        # v0.7: dense, collision-checked path from this stroke's actual retreat pose back to
        # the scan pose (the retreat pose varies stroke by stroke, unlike q_feed_path above, so
        # this has to be planned per stroke rather than once). A failure here must reject and
        # let the caller re-plan/project -- degrading silently to a single-frame jump would
        # bring back the same "teleport" this is meant to fix, just for a different transition.
        q_scan_path = []
        if c.physics == 'v0.6':
            q_scan_path = self._dense_path(ql[-1], self.q_scan, n=20)
            if q_scan_path is None:
                return self._reject('no dense path back to scan pose')
        plan = StrokePlan(True, '', [x.copy() for x in qa], [x.copy() for x in q_work], [x.copy() for x in ql],
                          [x.copy() for x in q_feed_path], [x.copy() for x in q_carry],
                          [x.copy() for x in q_rotate], carry_min, approach_d,
                          [x.copy() for x in q_scan_path])
        plan.meta = dict(p0=p0, p2=p2, pc=pc, psi=psi, s0=s0, s1=s1, direction=direction)
        return plan

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
        sampled=np.array([d.contact.at(t)[0] for t in np.linspace(0,1,c.stroke_samples)])
        L=float(np.linalg.norm(np.diff(sampled,axis=0),axis=1).sum())
        duration = max(L / max(d.speed_m_s, 1e-4), 0.05)
        dt = duration / c.stroke_samples
        e_w = mortar.e_w
        # initial stand-off from the quasi-static force balance on the current load
        from ..wall_cycle.mortar import solve_gap
        g_cmd, _ = solve_gap(mortar._rows(), abs(m['s0']), d.force_N, c.blade_cell_m, mortar.p)
        q = plan.q_approach[-1].copy()
        xy,tangent,phi,pitch,force,speed=d.contact.at(0)
        initial_psi=self.blade_psi(phi);initial_pitch=self.signed_pitch(initial_psi,tangent,pitch)
        p_target,R_target=self._pose(self._uvn(*xy,self.contact_n(initial_pitch,g_cmd)),initial_psi,initial_pitch)
        q_acquire,ok,_,_=self.ik.solve(p_target,R_target,q,iters=150)
        if ok:
            self._phase='CONTACT_ACQUIRE';self._run(q_acquire,.35);q=q_acquire;mark('CONTACT_ACQUIRE')
        f_mortar = np.zeros(3); qs, forces, gaps, track = [], [], [], []; ik_fail = 0; target_forces=[]; joint_errors=[]; saturation=[]; xy_errors=[]
        for k, t in enumerate(np.linspace(0, 1, c.stroke_samples)):
            centre,tangent,phi,pitch,force,speed=d.contact.at(t)
            psi=self.blade_psi(phi);pitch=self.signed_pitch(psi,tangent,pitch)
            mortar.e_x=np.array([np.cos(phi),np.sin(phi)])
            mortar.e_w=np.array([-np.sin(phi),np.cos(phi)])*(-1 if mortar._flip else 1)
            e_w=mortar.e_w
            step_distance=np.linalg.norm(sampled[k]-sampled[max(k-1,0)])
            dt=max(step_distance/speed,.01)
            p_t, R_t = self._pose(self._uvn(*centre, self.contact_n(pitch, g_cmd)), psi, pitch)
            # continuation IK from the last command. Accept the result even if it has not met the
            # 2 mm tolerance yet: keeping the old command (first draft) made the arm trail the
            # path by ~3 samples (a constant 14 mm) whatever the servo gains were.
            q, ok, _, _ = self.ik.solve(p_t, R_t, q, iters=60)
            ik_fail += (not ok)
            self._phase='WORK_STEP'
            f_rigid = self._run(q, dt, f_mortar)
            actual_centre, actual_pitch, gap = self.blade_state(e_w)
            res = mortar.contact(actual_centre, abs(actual_pitch), speed, gap_m=gap, stats=stats)
            f_meas = res.support_N + self.rigid_contact_N()
            # mortar pushes the blade out of the wall (-n) and drags against the motion
            motion = R_w[:, 0] * e_w[0] + R_w[:, 1] * e_w[1]
            f_mortar = -res.support_N * R_w[:, 2] - res.drag_N * motion
            # admittance: too much force -> stand further off
            g_cmd = float(np.clip(g_cmd + k_force * (f_meas - force), -0.002, 0.010))
            qs.append(self.data.qpos[:6].copy()); forces.append(f_meas); gaps.append(gap)
            track.append(float(np.linalg.norm(actual_centre - centre)));target_forces.append(force);joint_errors.append(np.rad2deg(self.data.qpos[:6]-q).tolist());xy_errors.append((actual_centre-centre).tolist());saturation.append((np.abs(self.data.actuator_force)>=.99*np.max(np.abs(self.model.actuator_forcerange),axis=1)).tolist())
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
        stats['force_mean_N'] = float(np.mean(forces)); stats['force_target_N'] = float(np.mean(target_forces))
        stats['force_rmse_N'] = float(np.sqrt(np.mean((np.asarray(forces) - np.asarray(target_forces)) ** 2)))
        stats['tracking_max_mm'] = 1000 * max(track)
        stats['tracking_mm'] = [1000 * x for x in track]
        stats['ik_not_converged'] = ik_fail
        stats['forces_N'] = forces
        stats['gaps_mm'] = [1000 * g for g in gaps]
        stats['q_work'] = qs
        stats['joint_errors_deg']=joint_errors;stats['saturation']=saturation;stats['xy_errors_m']=xy_errors;stats['target_forces_N']=target_forces
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

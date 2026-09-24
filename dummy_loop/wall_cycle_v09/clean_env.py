"""A0 observation-clean bridge. No raw material counters in actor observation."""
import time
import numpy as np
from ..wall_cycle.env import WallCycleEnv, MODES
from ..wall_cycle.sensor import coarse_map
from ..wall_cycle.recipes import make_config

OBS_SCHEMA='v09.a0.clean.v1'

def config():
    c=make_config('v0.8');c.score_width_m=c.score_height_m=.2
    c.min_cycles_before_stall=1000000 # overridden below using sensor observations only
    c.teacher_plateau_n=50
    c.base_steps=c.max_steps # fixed allowance, no hidden-quality budget extension
    return c

class CleanEnv(WallCycleEnv):
    def reset(self):
        self.estimated_load_ml=0.;self.requested_load_total_ml=0.;self.observed_improvement=0.
        self.sensor_history=[];self.step_diagnostics=[];self.sensor_stall_count=0
        o=super().reset();self._last_sensor_score=self.sensor_score()
        return o

    def _observe(self):
        c=self.cfg;h=coarse_map(self.scan['height']);conf=coarse_map(self.scan['confidence'])
        measured=np.where(conf>.5,h,c.target_m)
        scalars=[np.clip(getattr(self,'estimated_load_ml',0)/c.blade_capacity_ml,0,1.5),
            self.cycles/c.max_cycles,self.control_steps/c.max_steps,getattr(self,"sensor_stall_count",0)/2,
            self.reloads/c.max_reload_cycles,self.last_mode/(len(MODES)-1),float(conf.mean()),
            np.clip(getattr(self,'observed_improvement',0)/.1,-1,1),self.last_carry_face_up,
            np.clip(getattr(self,'requested_load_total_ml',0)/c.feed_board_volume_ml,0,1)]
        memory=np.r_[np.clip(coarse_map(self.fail_map),0,3).ravel()/3,
             min((self.cycles-self.best_score_cycle)/max(c.teacher_plateau_n,1),2),np.clip(self.best_score,-1,1)]
        return np.r_[np.clip(measured/c.target_m,-1,3).ravel(),
             np.clip((c.target_m-measured)/c.target_m,-2,2).ravel(),conf.ravel(),scalars,memory].astype(np.float32)

    def _update_memory(self,d,executed,improvement):
        score=self.sensor_score();observed=score-getattr(self,'_last_sensor_score',score)
        self.observed_improvement=float(observed)
        # Base function receives ONLY sensor-estimated improvement, never true cost delta.
        super()._update_memory(d,executed,observed)
        self._last_sensor_score=score

    def teacher_action(self):
        # Same local rule but loader state is the explicitly estimated quantity.
        return self._teacher_clean()

    def step(self,a):
        t=time.perf_counter();d=self.decode(a)
        before=self.material.quality_cost();before_loss=np.array([self.material.carry_loss_m3,self.material.dropped_m3,self.material.outside_m3])
        steps=self.control_steps;reloads=self.reloads;bad=self.unreachable;proj=self.projected
        oldscan=self.scan['height'].copy()
        _,_,done,info=super().step(a)
        delta=float(((self.scan['height']-oldscan)*self.cfg.cell_m**2).sum()*1e6)
        if self.reloads>reloads:
            self.requested_load_total_ml+=d.requested_load_ml
            self.estimated_load_ml=min(self.cfg.blade_capacity_ml,self.estimated_load_ml+d.requested_load_ml)
        self.estimated_load_ml=float(np.clip(self.estimated_load_ml-delta,0,self.cfg.blade_capacity_ml))
        # An imperfect bookkeeping estimator; no correction against true blade load.
        self.sensor_history.append(self.sensor_score())
        stall=False
        if self.cycles>=120 and self.cycles%60==0 and len(self.sensor_history)>=120:
            s=self.sensor_history
            stall=max(s[-60:])-max(s[-120:-60])<.005 and s[-1]-s[-120]<.005
        if done and self.end_reason=='stall': done=False;self.done=False;self.end_reason=None
        self.sensor_stall_count=int(stall)
        if stall: done=True;self.done=True;self.end_reason='stall'
        cost=self.material.quality_cost();loss=np.array([self.material.carry_loss_m3,self.material.dropped_m3,self.material.outside_m3])-before_loss
        reward=30*(before-cost)-(4*loss[0]+4*(loss[1]-loss[0])+loss[2])/18e-6
        reward-=.02+self.cfg.time_coef*(self.control_steps-steps)+2*(self.unreachable-bad)+.5*(self.projected-proj)
        if done:
            success=bool(info.get('success'))
            reward+=100*success-10*cost
            if self.end_reason=='finish':self.end_reason='success' if success else 'give_up'
        info=self.info(bool(info.get('success')))
        info['profiling']={'env_step_s':time.perf_counter()-t}
        info['observed_improvement']=self.observed_improvement
        info['observation_schema']=OBS_SCHEMA
        return self._observe(),float(reward),done,info

    def _teacher_clean(self):
        """v0.8 teacher: pick the candidate stroke whose swept footprint covers the most local
        deficit (or, if larger, twice the local excess -> LEVEL), instead of aiming every stroke
        at the global centroid of the deficit map. Uses only the scan, the blade load and the
        observed memory (fail map, plateau counter). Weights (0.5 excess penalty, level weight,
        fail damping, length normalisation) are heuristics, not physical constants."""
        from ..wall_cycle.env import DecodedAction, local_candidates
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
            enough = (self.estimated_load_ml*1e-6) >= min(8e-6, .65 * need)
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

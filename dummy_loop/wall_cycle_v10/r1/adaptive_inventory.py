"""v5 candidate: transactional step-doubling with latent inventory error control.
Local estimates are NOT a proven global error bound; expanded G0 still required.
No extrapolation, clipping, deletion of material or relaxed audit thresholds.
"""
import copy
import numpy as np
from .split_inventory import SplitPressure

class IntegrationFailure(RuntimeError):
    pass

class AdaptivePressure:
    physics_version='v10r1.area_pressure_candidate.v5_inventory_adaptive'
    def __init__(self,model,budget_mm=.005,reference_length=.26,max_depth=14,max_trials=20000):
        if not isinstance(model,SplitPressure):raise TypeError('Expected v4 state')
        if not np.isfinite([budget_mm,reference_length]).all() or min(budget_mm,reference_length)<=0:
            raise ValueError('Invalid tolerance')
        self.model=copy.deepcopy(model);self.budget_mm=budget_mm;self.reference_length=reference_length
        self.max_depth=max_depth;self.max_trials=max_trials
        self.stats=dict(trials=0,accepted=0,rejected=0,max_depth=0,min_accepted_travel_m=None)

    @staticmethod
    def distance(model,a,b):
        radius=.5*np.linalg.norm(np.array(model.blade.shape)*model.bc)
        return float(np.linalg.norm(b[:2]-a[:2])+radius*abs(b[2]-a[2]))

    @staticmethod
    def error_mm(a,b):
        # Include latent blade/bead differences, plus their actual projected lift.
        x,y=copy.deepcopy(a),copy.deepcopy(b)
        x.lift(x.params.lift_wall_fraction);y.lift(y.params.lift_wall_fraction)
        errors=[np.max(abs(a.wall-b.wall))/a.wc**2,
                np.max(abs(a.blade-b.blade))/a.bc**2,
                np.max(abs(a.bead-b.bead))/a.bc**2,
                np.max(abs(x.wall-y.wall))/a.wc**2,
                abs(a.dropped-b.dropped)/a.wc**2,
                abs(a.outside-b.outside)/a.wc**2]
        return float(max(errors)*1000)

    @staticmethod
    def step(model,target):
        return model.contact(target[:2],float(target[2]),float(target[3]),float(target[4]))

    def contact(self,center,angle,pitch,force):
        target=np.array([*center,angle,pitch,force],dtype=float)
        if target.shape!=(5,) or not np.isfinite(target).all() or min(pitch,force)<0:
            raise ValueError('Invalid contact')
        initial=copy.deepcopy(self.model)
        if initial.pose is None:
            result=self.step(initial,target);self.model=initial;return result
        start=np.array([*initial.pose[0],initial.pose[1],*initial.last_controls])
        # Identical pose/control is an identity, avoiding zero-length geometric roundoff.
        if np.array_equal(start,target):return dict(gap_m=float(initial.g0))
        stats=copy.deepcopy(self.stats);local_trials=0
        def advance(model,a,b,depth):
            nonlocal local_trials
            local_trials+=1;stats['trials']+=1;stats['max_depth']=max(stats['max_depth'],depth)
            if local_trials>self.max_trials:raise IntegrationFailure('Trial budget exhausted; state unchanged')
            mid=(a+b)/2
            coarse=copy.deepcopy(model);self.step(coarse,b)
            fine=copy.deepcopy(model);self.step(fine,mid);self.step(fine,b)
            travel=self.distance(model,a,b)
            allowed=self.budget_mm*max(travel,1e-12)/self.reference_length
            error=self.error_mm(coarse,fine)
            finite=all(np.isfinite(z).all() for z in (fine.wall,fine.blade,fine.bead))
            positive=min(fine.wall.min(),fine.blade.min(),fine.bead.min())>=-1e-18
            if finite and positive and np.isfinite(error) and error<=allowed:
                stats['accepted']+=1
                old=stats['min_accepted_travel_m'];stats['min_accepted_travel_m']=travel if old is None else min(old,travel)
                return fine
            stats['rejected']+=1
            if depth>=self.max_depth:raise IntegrationFailure('Local error tolerance unresolved; state unchanged')
            first=advance(model,a,mid,depth+1)
            return advance(first,mid,b,depth+1)
        accepted=advance(initial,start,target,0)
        self.model=accepted;self.stats=stats
        return dict(gap_m=float(accepted.g0))

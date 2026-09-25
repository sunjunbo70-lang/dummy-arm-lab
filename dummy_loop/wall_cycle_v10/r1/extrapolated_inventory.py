"""Conservative step-doubling extrapolation experiment, not frozen physics.
Reject extrapolation if any inventory or irreversible increment becomes negative.
Never clip/renormalize material to hide numerical errors.
"""
import copy
import numpy as np
from .ordered_inventory import OrderedPressure

class ExtrapolatedPressure(OrderedPressure):
    physics_version='v10r1.area_pressure_candidate.v3_extrapolation'

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.accepted_extrapolations=0
        self.fallback_steps=0
        self.last_controls=None
        self.rejection_counts={} 
        self.minimum_proposal={}

    def contact(self,center,angle,pitch,force):
        if self.pose is None:
            info=OrderedPressure.contact(self,center,angle,pitch,force)
            self.last_controls=(float(pitch),float(force))
            return info
        center=np.asarray(center,dtype=float)
        old_center,old_angle=self.pose
        old_pitch,old_force=self.last_controls
        coarse=copy.deepcopy(self);fine=copy.deepcopy(self)
        OrderedPressure.contact(coarse,center,angle,pitch,force)
        OrderedPressure.contact(fine,(old_center+center)*.5,(old_angle+angle)*.5,(old_pitch+pitch)*.5,(old_force+force)*.5)
        info=OrderedPressure.contact(fine,center,angle,pitch,force)
        fields=('wall','blade','bead','outside','dropped')
        proposal={k:2*getattr(fine,k)-getattr(coarse,k) for k in fields}
        valid=all(np.isfinite(v).all() and np.all(v>=0) for v in proposal.values())
        valid=valid and proposal['outside']>=self.outside and proposal['dropped']>=self.dropped
        if valid:
            for k,v in proposal.items():setattr(self,k,v)
            self.gaps=np.maximum(0.,2*fine.gaps-coarse.gaps)
            self.g0=max(0.,2*fine.g0-coarse.g0)
            self.accepted_extrapolations+=1
        else:
            for k,v in proposal.items():
                self.minimum_proposal[k]=min(self.minimum_proposal.get(k,0.),float(np.min(v)))
                if np.any(v<0):self.rejection_counts[k]=self.rejection_counts.get(k,0)+1
            for k in ('outside','dropped'):
                if proposal[k]<getattr(self,k):self.rejection_counts[k+'_decrease']=self.rejection_counts.get(k+'_decrease',0)+1
            for k in fields:setattr(self,k,getattr(fine,k))
            self.gaps,self.g0=fine.gaps,fine.g0
            self.fallback_steps+=1
        self.pose,self.mapping=fine.pose,fine.mapping
        self.last_controls=(float(pitch),float(force))
        # Last support is the fine substep's diagnostic, not a new force solve.
        info['extrapolation_accepted']=bool(valid)
        return info


"""Pressure bisection precision diagnostic; same support law and bracket."""
import numpy as np
from numba import njit
from .compiled_material import support
from .error_control_pressure import ErrorControlPressure
@njit(cache=True)
def precise_gap_solve(B,pitch,force,cell,tau,min_gap,max_gap):
    rise=(np.arange(B.shape[0])+.5)*cell*np.sin(max(pitch,0.))
    if force<=0:return max_gap,0.
    s=support(B,min_gap+rise,cell,tau)
    if s<force:return 0.,s
    s=support(B,max_gap+rise,cell,tau)
    if s>=force:return max_gap,s
    lo=min_gap;hi=max_gap
    for _ in range(40):
        mid=(lo+hi)*.5
        if support(B,mid+rise,cell,tau)>=force:lo=mid
        else:hi=mid
    return lo,support(B,lo+rise,cell,tau)

class PreciseGapPressure(ErrorControlPressure):
 physics_version='v10r1.area_pressure_candidate.v8_9_precise_gap_error_control'
 def _pressure(self,B,pitch,force):
  p=self.params
  g,s=precise_gap_solve(B,pitch,force,self.bc,p.tau_y,p.min_gap_m,p.max_gap_m)
  return g,g+(np.arange(B.shape[0])+.5)*self.bc*np.sin(pitch),s

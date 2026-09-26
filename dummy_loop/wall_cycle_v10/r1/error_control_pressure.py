"""Diagnostic whole-contact step doubling with explicit bounded local error.
Not frozen physics; no extrapolation, clipping or acceptance-gate change.
"""
import copy
import numpy as np
from .stationary_boundary_pressure import StationaryBoundaryPressure
class ErrorControlPressure(StationaryBoundaryPressure):
 physics_version='v10r1.area_pressure_candidate.v8_8_contact_step_doubling'
 error_per_cell_travel_mm=1e-4
 max_contact_depth=6
 def contact(self,center,angle,pitch,force):
  if self.pose is None:return super().contact(center,angle,pitch,force)
  target=(np.asarray(center,float),float(angle),float(pitch),float(force))
  state,info,nodes,depth,error=self._refine(target,0)
  self.__dict__.update(state.__dict__)
  self.boundary_stats['contact_error_nodes']=self.boundary_stats.get('contact_error_nodes',0)+nodes
  self.boundary_stats['contact_error_depth']=max(self.boundary_stats.get('contact_error_depth',0),depth)
  self.boundary_stats['contact_estimated_error_sum_mm']=self.boundary_stats.get('contact_estimated_error_sum_mm',0.)+error
  return info
 def _refine(self,target,depth):
  center,angle,pitch,force=target
  distance=np.linalg.norm(center-self.pose[0])+.5*np.linalg.norm(np.array(self.blade.shape)*self.bc)*abs(angle-self.pose[1])
  coarse=copy.deepcopy(self);info=StationaryBoundaryPressure.contact(coarse,*target)
  if distance==0:return coarse,info,1,depth,0.
  mid=((self.pose[0]+center)/2,(self.pose[1]+angle)/2,*(.5*(np.asarray(self.last_controls)+[pitch,force])))
  fine=copy.deepcopy(self);StationaryBoundaryPressure.contact(fine,*mid);info=StationaryBoundaryPressure.contact(fine,*target)
  error=max(float(np.max(abs(getattr(fine,k)-getattr(coarse,k))))/self.bc**2*1000 for k in ('wall','blade','bead'))
  tolerance=self.error_per_cell_travel_mm*distance/self.bc+1e-12
  if error<=tolerance:return fine,info,3,depth,error
  if depth>=self.max_contact_depth:raise RuntimeError(f'Contact error budget exhausted: error={error}, tolerance={tolerance}, depth={depth}')
  half,_,n1,d1,e1=self._refine(mid,depth+1)
  end,info,n2,d2,e2=half._refine(target,depth+1)
  return end,info,3+n1+n2,max(d1,d2),e1+e2

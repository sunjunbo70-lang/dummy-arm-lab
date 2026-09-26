"""v8 boundary-exchange + existing SSPRK2 pressure candidate. Not frozen physics."""
import numpy as np
from .split_inventory import SplitPressure
from .contact_inventory import ContactInventory
from .adaptive_boundary_flow import adaptive
from .coupled_boundary_exchange import exchange
from .overlap import planar_overlap

class BoundaryPressure(SplitPressure):
 physics_version='v10r1.area_pressure_candidate.v8_adaptive_boundary'
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs)
  if self.wall.shape!=(100,100) or self.blade.shape!=(6,24) or self.bc!=.005 or self.wc!=.005 or tuple(self.origin)!=(-.25,0.):raise ValueError('Candidate currently requires standard audit grids')
  self.boundary_stats=dict(nodes=0,area_correction_m2=0.,moves=0)
 def move(self,center,angle,exit_height):
  if self.pose is None:return ContactInventory.move(self,center,angle,exit_height)
  center=np.asarray(center,float);angle=float(angle)
  old=self.mapping;new=planar_overlap(center,angle,self.blade.shape,self.bc,self.wall.shape,self.wc,np.array(self.origin))
  def free(m):
   a=self.wc**2-np.bincount(m[1],weights=m[2],minlength=self.wall.size)
   if a.min()<-2.5e-17:raise ValueError('Geometry exceeds roundoff budget')
   return np.where(a<2.5e-17,0.,np.where(abs(a-self.wc**2)<2.5e-17,self.wc**2,a))
  area,target=free(old),free(new)
  maps,stats=adaptive(self.pose[0],center,self.pose[1],angle)
  e,x=[np.bincount(m[1],weights=m[2],minlength=self.wall.size) for m in maps]
  through=np.minimum(e,x);empty=(area==0)&(target==0)
  if (through[empty]>2.5e-17).any():raise ValueError('Unresolved zero-area transient boundary flow')
  through[empty]=0.;change=target-area
  desired=[through+np.maximum(-change,0.),through+np.maximum(change,0.)]
  corrected=[];correction=0.
  for m,raw,want in zip(maps,(e,x),desired):
   if ((raw==0)&(want>0)).any():raise ValueError('Missing boundary material recipient; no endpoint patching')
   factor=np.divide(want,raw,out=np.zeros_like(raw),where=raw>0)
   corrected.append((m[0],m[1],m[2]*factor[m[1]],m[3]));correction+=float(abs(want-raw).sum())
  gap=np.broadcast_to(exit_height,self.blade.shape).ravel()
  wall,blade,lost,info=exchange(self.wall.ravel(),self.blade.ravel(),area,target,self.bc**2,*corrected,gap)
  self.wall=wall.reshape(self.wall.shape);self.blade=blade.reshape(self.blade.shape);self.outside+=lost
  self.pose=(center,angle);self.mapping=new
  self.boundary_stats['nodes']+=stats['nodes'];self.boundary_stats['area_correction_m2']+=correction;self.boundary_stats['moves']+=1

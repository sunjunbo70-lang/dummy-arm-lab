"""Candidate applying existing gravity cap before wall pickup and at deposition.
Changes within-step ordering; requires full independent numerical/physical audit.
"""
import numpy as np
from .bounded_edge_pressure import BoundedEdgePressure
from .adaptive_boundary_flow import adaptive
from .overlap import planar_overlap
from .saturated_wall_exchange import exchange
from .precise_gap_pressure import PreciseGapPressure
class SaturatedWallPressure(PreciseGapPressure):
 physics_version='v10r1.area_pressure_candidate.v8_10_continuous_wall_cap'
 def move(self,center,angle,exit_height):
  if self.pose is None:
   center=np.asarray(center,float);angle=float(angle)
   m=planar_overlap(center,angle,self.blade.shape,self.bc,self.wall.shape,self.wc,np.array(self.origin))
   bi,wi,areas,_=m
   covered=np.bincount(wi,weights=areas,minlength=self.wall.size)
   cell_area=self.wc**2
   if covered.max()>cell_area+2.5e-17:raise ValueError('Initial coverage exceeds geometric roundoff budget')
   # Compute the wall donor total ONCE, then distribute it to blade cells.
   # Independently summing rounded requested pickups can overdraw a full cell.
   covered_bounded=np.where(abs(covered-cell_area)<2.5e-17,cell_area,covered)
   picked=self.wall.ravel()*(covered_bounded/cell_area)
   share=np.divide(picked,covered,out=np.zeros_like(picked),where=covered>0)
   gain=np.bincount(bi,weights=areas*share[wi],minlength=self.blade.size)
   self.wall=(self.wall.ravel()-picked).reshape(self.wall.shape)
   self.blade+=gain.reshape(self.blade.shape)
   self.pose=(center,angle);self.mapping=m
   return
  center=np.asarray(center,float);angle=float(angle)
  old=self.mapping;new=planar_overlap(center,angle,self.blade.shape,self.bc,self.wall.shape,self.wc,np.array(self.origin))
  def free(m):
   a=self.wc**2-np.bincount(m[1],weights=m[2],minlength=self.wall.size)
   if a.min()<-2.5e-17:raise ValueError('Geometry exceeds roundoff budget')
   return np.where(a<2.5e-17,0.,np.where(abs(a-self.wc**2)<2.5e-17,self.wc**2,a))
  area,target=free(old),free(new)
  limit=self.params.tau_y/(self.params.rho*9.81)
  excess=np.maximum(self.wall.ravel()-limit*area,0.)
  self.wall-=excess.reshape(self.wall.shape);self.dropped+=float(excess.sum())
  maps,stats=adaptive(self.pose[0],center,self.pose[1],angle)
  e,x=[np.bincount(m[1],weights=m[2],minlength=self.wall.size) for m in maps]
  through=np.minimum(e,x);empty=(area==0)&(target==0)
  if (through[empty]>2.5e-17).any():raise ValueError('Unresolved zero-area transient boundary flow')
  through[empty]=0.;change=target-area
  desired=[through+np.maximum(-change,0.),through+np.maximum(change,0.)]
  corrected=[];correction=0.
  for m,raw,want in zip(maps,(e,x),desired):
   if ((raw==0)&(want>0)).any():raise ValueError('Missing boundary material recipient; no endpoint patching')
   factor=np.divide(want,raw,out=np.zeros_like(raw,dtype=float),where=raw>0)
   corrected.append((m[0],m[1],m[2]*factor[m[1]],m[3]));correction+=float(abs(want-raw).sum())
  gap=np.broadcast_to(exit_height,self.blade.shape).ravel()
  wall,blade,lost,info=exchange(self.wall.ravel(),self.blade.ravel(),area,target,self.bc**2,*corrected,gap,max_wall_height=limit)
  self.wall=wall.reshape(self.wall.shape);self.blade=blade.reshape(self.blade.shape);self.outside+=lost;self.dropped+=info["shed_m3"]
  self.pose=(center,angle);self.mapping=new
  self.boundary_stats['nodes']+=stats['nodes'];self.boundary_stats['area_correction_m2']+=correction;self.boundary_stats['moves']+=1

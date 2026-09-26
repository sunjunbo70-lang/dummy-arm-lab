"""Candidate v8.2: exclude roundoff edge overlaps into fully covered cells.
No solver tolerance relaxation; removed overlap redistributed per donor column.
"""
import numpy as np
from .boundary_pressure import BoundaryPressure
from .overlap import planar_overlap,deposit

class BoundedEdgePressure(BoundaryPressure):
 physics_version='v10r1.area_pressure_candidate.v8_2_edge_support'
 def _edge_deposit(self,volumes,center,angle,side):
  e=np.array([np.cos(angle),np.sin(angle)])
  edge=np.asarray(center)+side*(self.blade.shape[0]+1)*self.bc*.5*e
  m=planar_overlap(edge,angle,(1,self.blade.shape[1]),self.bc,self.wall.shape,self.wc,np.asarray(self.origin))
  bi,wi,areas,outside=m
  covered=np.bincount(self.mapping[1],weights=self.mapping[2],minlength=self.wall.size)
  forbidden=(self.wc**2-covered[wi])<2.5e-17
  removed=np.bincount(bi,weights=np.where(forbidden,areas,0.),minlength=self.blade.shape[1])
  if (removed>2.5e-17).any():raise ValueError('Edge deposit overlap exceeds geometric roundoff budget')
  kept=np.where(forbidden,0.,areas)
  total=np.bincount(bi,weights=kept,minlength=self.blade.shape[1])+outside
  if (total<=0).any():raise ValueError('Edge donor has no recipient')
  factor=self.bc**2/total
  wall,lost=deposit(volumes[None,:],(bi,wi,kept*factor[bi],outside*factor),self.bc,self.wall.shape)
  self.wall+=wall;self.outside+=lost
  self.boundary_stats['edge_removed_area_m2']=self.boundary_stats.get('edge_removed_area_m2',0.)+float(removed.sum())

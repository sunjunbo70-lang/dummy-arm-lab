"""Diagnostic instantaneous wall-cap deposition with explicit shed ledger.
Sparse integrated boundary maps must already satisfy geometric area closure.
Wall analytic transfer couples to midpoint, gap-capped blade exit density.
"""
import numpy as np
from .moving_wall_exchange import transfer
VERSION='v10r1.coupled_boundary_exchange.saturated_wall.v2'

def exchange(wall,blade,area0,area1,blade_area,entry,exit,gap,max_iter=80,tol=1e-18,max_wall_height=None):
 wall=np.asarray(wall,float);blade=np.asarray(blade,float)
 area0=np.asarray(area0,float);area1=np.asarray(area1,float)
 nb=len(blade);nw=len(wall);ba=np.broadcast_to(np.asarray(blade_area,float),blade.shape)
 gap=np.broadcast_to(np.asarray(gap,float),blade.shape)
 if any(not np.isfinite(z).all() or (z<0).any() for z in (wall,blade,area0,area1,ba,gap)) or (ba<=0).any():raise ValueError('Invalid state')
 if wall.ndim!=1 or blade.ndim!=1 or area0.shape!=wall.shape or area1.shape!=wall.shape:raise ValueError('Invalid shapes')
 def check(m):
  bi,wi,a,out=m;bi=np.asarray(bi);wi=np.asarray(wi);a=np.asarray(a,float);out=np.asarray(out,float)
  if bi.shape!=wi.shape or a.shape!=bi.shape or out.shape!=blade.shape or (bi<0).any() or (bi>=nb).any() or (wi<0).any() or (wi>=nw).any() or not np.isfinite(np.r_[a,out]).all() or (a<0).any() or (out<0).any():raise ValueError('Invalid map')
  return bi,wi,a,out
 ib,iw,ia,io=check(entry);ob,ow,oa,oo=check(exit)
 incoming=np.bincount(iw,weights=ia,minlength=nw)
 outgoing=np.bincount(ow,weights=oa,minlength=nw)
 exit_blade=np.bincount(ob,weights=oa,minlength=nb)+oo
 if (exit_blade>ba).any():raise ValueError('Blade exit exceeds one cell area: subdivide geometry step')
 # Within this CFL bound midpoint depletion cannot make a blade cell negative.
 received=np.zeros(nb);last=None
 for iteration in range(max_iter):
  density=np.minimum((blade+.5*received)/(ba+.5*exit_blade),gap)
  wall_density=density if max_wall_height is None else np.minimum(density,max_wall_height)
  deposited=np.bincount(ow,weights=oa*wall_density[ob],minlength=nw)
  remaining,picked,_=transfer(wall,area0,incoming,outgoing,deposited,final_area=area1)
  per_area=np.divide(picked,incoming,out=np.zeros(nw),where=incoming>0)
  updated=np.bincount(ib,weights=ia*per_area[iw],minlength=nb)
  residual=float(abs(updated-received).max(initial=0.))
  last=(remaining,updated,density,residual,iteration+1)
  if residual<=tol:break
  received=updated
 else:raise RuntimeError('Coupled boundary fixed-point budget exhausted')
 remaining,received,density,residual,iterations=last
 final_blade=blade+received-density*exit_blade
 lost=float(density@oo)
 if (final_blade<0).any():raise FloatingPointError('Negative blade stock')
 shed=0. if max_wall_height is None else float(oa @ np.maximum(density[ob]-max_wall_height,0.))
 return remaining,final_blade,lost,dict(iterations=iterations,residual_m3=residual,shed_m3=shed)

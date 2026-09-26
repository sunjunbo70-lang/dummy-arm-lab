"""Numba implementation of the existing instantaneous boundary-area integral.
Same equations and spatial splits, not a new material law.
"""
import math
import numpy as np
from numba import njit

@njit(cache=True)
def kernel(center,angle,velocity,omega,bs,bc,ws,wc,origin):
 c=math.cos(angle);s=math.sin(angle)
 cap=2*(bs[0]+bs[1])*(2*int(math.ceil(bc/wc))+12)
 kinds=np.empty(cap,np.int64);bis=np.empty(cap,np.int64);wis=np.empty(cap,np.int64);rates=np.empty(cap);outside=np.zeros((2,bs[0]*bs[1]));count=0
 for axis in range(2):
  if axis==0:nx,ny,tx,ty,n,nt=c,s,-s,c,bs[0],bs[1]
  else:nx,ny,tx,ty,n,nt=-s,c,c,s,bs[1],bs[0]
  for side in (-1,1):
   ox=side*nx;oy=side*ny
   for j in range(nt):
    ax=center[0]+side*n*bc/2*nx+(j-nt/2)*bc*tx
    ay=center[1]+side*n*bc/2*ny+(j-nt/2)*bc*ty
    dx=bc*tx;dy=bc*ty
    q0=(velocity[0]-omega*(ay-center[1]))*ox+(velocity[1]+omega*(ax-center[0]))*oy
    slope=omega*(-dy*ox+dx*oy)
    cuts=np.empty(2*int(math.ceil(bc/wc))+12);cuts[0]=0.;cuts[1]=1.;nc=2
    if abs(slope)>1e-20:
     z=-q0/slope
     if 0<z<1:cuts[nc]=z;nc+=1
    for kaxis in range(2):
     aa=ax if kaxis==0 else ay;dd=dx if kaxis==0 else dy
     if abs(dd)>1e-20:
      low=min(aa,aa+dd);high=max(aa,aa+dd)
      for k in range(int(math.floor((low-origin[kaxis])/wc))+1,int(math.ceil((high-origin[kaxis])/wc))):
       z=(origin[kaxis]+k*wc-aa)/dd
       if 0<z<1:cuts[nc]=z;nc+=1
    ordered=np.sort(cuts[:nc])
    bi=((0 if side<0 else bs[0]-1)*bs[1]+j) if axis==0 else j*bs[1]+(0 if side<0 else bs[1]-1)
    for k in range(nc-1):
     lo=ordered[k];hi=ordered[k+1]
     if hi<=lo:continue
     t=(lo+hi)/2;signed=q0+slope*t
     if signed==0:continue
     kind=0 if signed>0 else 1;rate=abs(signed)*(hi-lo)*bc
     col=int(math.floor((ax+t*dx-origin[0])/wc));row=int(math.floor((ay+t*dy-origin[1])/wc))
     if 0<=row<ws[0] and 0<=col<ws[1]:
      if count>=cap:raise ValueError('Boundary buffer exceeded')
      kinds[count]=kind;bis[count]=bi;wis[count]=row*ws[1]+col;rates[count]=rate;count+=1
     else:outside[kind,bi]+=rate
 return kinds[:count],bis[:count],wis[:count],rates[:count],outside

def instantaneous(center,angle,velocity,omega,bs=(6,24),bc=.005,ws=(100,100),wc=.005,origin=(-.25,0.)):
 center=np.asarray(center,float);velocity=np.asarray(velocity,float)
 if center.shape!=(2,) or velocity.shape!=(2,) or not np.isfinite(np.r_[center,velocity,angle,omega,bc,wc]).all() or min(bc,wc)<=0:raise ValueError('Invalid geometry')
 kinds,bi,wi,rates,out=kernel(center,angle,velocity,omega,bs,bc,ws,wc,origin)
 maps=[{},{}]
 for k,b,w,r in zip(kinds,bi,wi,rates):
  key=(int(b),int(w));maps[k][key]=maps[k].get(key,0.)+r
 return maps,[out[0],out[1]]

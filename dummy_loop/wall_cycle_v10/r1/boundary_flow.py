"""Boundary-normal area flux for translating/rotating rectangular tools.
Spatial integration along each edge is exact for linear rigid-body velocity;
time integration uses composite two-point Gauss. Geometry only, not material.
"""
import numpy as np
GEOMETRY_VERSION='v10r1.rigid_boundary_flux.v1'

def instantaneous(center,angle,velocity,omega,bs=(6,24),bc=.005,ws=(100,100),wc=.005,origin=(-.25,0.)):
    center=np.asarray(center,float);velocity=np.asarray(velocity,float)
    if center.shape!=(2,) or velocity.shape!=(2,) or not np.isfinite(np.r_[center,velocity,angle,omega,bc,wc]).all() or min(bc,wc)<=0:raise ValueError('Invalid geometry')
    e=np.array([np.cos(angle),np.sin(angle)]);f=np.array([-np.sin(angle),np.cos(angle)])
    result=[{},{}];outside=[np.zeros(np.prod(bs)),np.zeros(np.prod(bs))]
    for axis,count,nt,normal,tangent in ((0,bs[0],bs[1],e,f),(1,bs[1],bs[0],f,e)):
      for side in (-1,1):
       outward=side*normal
       for j in range(nt):
        local=side*count*bc/2*normal+(j+.5-nt/2)*bc*tangent
        a=center+local-.5*bc*tangent;d=bc*tangent
        va=velocity+omega*np.array([-(a-center)[1],(a-center)[0]])
        slope=float(omega*np.array([-d[1],d[0]])@outward);q0=float(va@outward)
        cuts=[0.,1.]
        if abs(slope)>1e-20:
         z=-q0/slope
         if 0<z<1:cuts.append(z)
        for ax in (0,1):
         if abs(d[ax])>1e-20:
          low,high=sorted((a[ax],a[ax]+d[ax]))
          for k in range(int(np.floor((low-origin[ax])/wc))+1,int(np.ceil((high-origin[ax])/wc))):
           z=(origin[ax]+k*wc-a[ax])/d[ax]
           if 0<z<1:cuts.append(z)
        cuts=sorted(set(cuts))
        bi=((0 if side<0 else bs[0]-1)*bs[1]+j) if axis==0 else j*bs[1]+(0 if side<0 else bs[1]-1)
        for lo,hi in zip(cuts,cuts[1:]):
         t=(lo+hi)/2;signed=q0+slope*t
         if signed==0:continue
         kind=0 if signed>0 else 1 # expanding footprint => material entry
         rate=abs(signed)*(hi-lo)*bc;xy=a+t*d
         col,row=np.floor((xy-origin)/wc).astype(int)
         if 0<=row<ws[0] and 0<=col<ws[1]:
          key=(bi,int(row*ws[1]+col));result[kind][key]=result[kind].get(key,0.)+rate
         else:outside[kind][bi]+=rate
    return result,outside

def integrate(begin,end,angle0,angle1,intervals=64,**kwargs):
    if not isinstance(intervals,int) or intervals<1:raise ValueError('Invalid intervals')
    begin=np.asarray(begin,float);end=np.asarray(end,float);velocity=end-begin;omega=angle1-angle0
    sums=[{},{}];out=None
    for i in range(intervals):
     for node in (-1/np.sqrt(3),1/np.sqrt(3)):
      t=(i+.5+.5*node)/intervals;maps,external=instantaneous(begin+t*velocity,angle0+t*omega,velocity,omega,**kwargs)
      if out is None:out=[np.zeros_like(v) for v in external]
      for kind in (0,1):
       for key,val in maps[kind].items():sums[kind][key]=sums[kind].get(key,0.)+val/(2*intervals)
       out[kind]+=external[kind]/(2*intervals)
    answer=[]
    for d,external in zip(sums,out):
     keys=sorted(d);answer.append((np.array([k[0] for k in keys],dtype=np.int64),np.array([k[1] for k in keys],dtype=np.int64),np.array([d[k] for k in keys]),external))
    return answer

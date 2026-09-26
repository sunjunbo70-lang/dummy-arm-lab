"""Bounded contact trajectory sampler; no IK, hardware or physical acceptance.
Control-polygon length bounds translation; unwrapped angle bounds tool sweep.
"""
from dataclasses import dataclass
import numpy as np
@dataclass
class ContactSamples:
 parameter: np.ndarray
 xy: np.ndarray
 angle: np.ndarray
 pitch: np.ndarray
 force: np.ndarray
 speed: np.ndarray
 time_s: np.ndarray
 swept_bound_m: np.ndarray

def sample_contact(contact, *, max_swept_m, tool_radius_m, max_angular_speed_rad_s, max_depth=20):
 if min(max_swept_m,max_angular_speed_rad_s)<=0 or tool_radius_m<0:raise ValueError('Invalid sampling limits')
 arrays=[np.asarray(getattr(contact,k),float) for k in ('points','angles','pitches','forces','speeds')]
 if [x.shape for x in arrays]!=[(4,2),(2,),(3,),(3,),(3,)] or not all(np.isfinite(x).all() for x in arrays) or (arrays[-1]<=0).any():raise ValueError('Invalid contact')
 angle_delta=float(np.arctan2(np.sin(arrays[1][1]-arrays[1][0]),np.cos(arrays[1][1]-arrays[1][0])))
 leaves=[]
 def split(points,a,b,depth):
  bound=float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum()+tool_radius_m*abs(angle_delta)*(b-a))
  if bound<=max_swept_m:leaves.append((b,bound));return
  if depth>=max_depth:raise RuntimeError('Contact sampling depth exhausted')
  x=(points[:-1]+points[1:])/2;y=(x[:-1]+x[1:])/2;z=(y[0]+y[1])/2;t=(a+b)/2
  split(np.array([points[0],x[0],y[0],z]),a,t,depth+1)
  split(np.array([z,y[1],x[2],points[3]]),t,b,depth+1)
 split(arrays[0],0.,1.,0)
 ts=np.r_[0.,[v[0] for v in leaves]];values=[contact.at(float(t)) for t in ts]
 xy=np.array([v[0] for v in values]);angles=np.array([v[2] for v in values]);pitch=np.array([v[3] for v in values]);force=np.array([v[4] for v in values]);speed=np.array([v[5] for v in values])
 # Trapezoidal reciprocal-speed time is a quadrature approximation, not dynamics.
 translation=np.linalg.norm(np.diff(xy,axis=0),axis=1);dt=translation*.5*(1/speed[:-1]+1/speed[1:])
 dt=np.maximum(dt,abs(np.diff(angles))/max_angular_speed_rad_s)
 return ContactSamples(ts,xy,angles,pitch,force,speed,np.r_[0.,np.cumsum(dt)],np.array([v[1] for v in leaves]))

def check_continuation(previous,next_contact,atol=1e-10):
 a=previous.at(1.);b=next_contact.at(0.)
 # Endpoint position, attitude, pressure and speed must be inherited; tangent may turn.
 angle_error=abs(np.arctan2(np.sin(a[2]-b[2]),np.cos(a[2]-b[2])))
 return bool(np.allclose(a[0],b[0],atol=atol,rtol=0) and angle_error<=atol and np.allclose(a[3:],b[3:],atol=atol,rtol=0))

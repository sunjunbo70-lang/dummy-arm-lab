"""Versioned 19-dimensional CONTACT geometry; SI units throughout."""
from dataclasses import dataclass
import numpy as np
from ..wall_cycle.env import DecodedAction
SCHEMA='v09r12.contact19.v1'
OPS=('LOAD','CONTACT','RESCAN','FINISH_REQUEST')
@dataclass
class Contact:
 points: np.ndarray
 angles: np.ndarray
 pitches: np.ndarray
 forces: np.ndarray
 speeds: np.ndarray
 def at(self,t):
  p=self.points;s=1-t
  xy=s**3*p[0]+3*s*s*t*p[1]+3*s*t*t*p[2]+t**3*p[3]
  tangent=3*s*s*(p[1]-p[0])+6*s*t*(p[2]-p[1])+3*t*t*(p[3]-p[2])
  angle=self.angles[0]+t*np.arctan2(np.sin(self.angles[1]-self.angles[0]),np.cos(self.angles[1]-self.angles[0]))
  return xy,tangent,float(angle),float(np.interp(t,[0,.5,1],self.pitches)),float(np.interp(t,[0,.5,1],self.forces)),float(np.interp(t,[0,.5,1],self.speeds))
 def legacy(self):
  a=self.at(0);b=self.at(1)
  d=DecodedAction('REUSE',tuple(a[0]),tuple(b[0]),a[2],0.,a[4],a[5],0.,a[3],b[3],1.)
  d.contact=self
  return d

def decode(x,cfg,buffer=.03):
 x=np.asarray(x,float);assert x.shape==(19,)
 half=.1+buffer;p0=x[:2]*half+np.array([0,cfg.height_m/2]);p3=x[2:4]*half+np.array([0,cfg.height_m/2]);delta=p3-p0
 points=np.array([p0,p0+delta/3+x[4:6]*.03,p0+2*delta/3+x[6:8]*.03,p3])
 return Contact(points,x[8:10]*np.pi,(x[10:13]+1)*.5*np.deg2rad(cfg.max_pitch_deg),.5+(x[13:16]+1)*7.25,.02+(x[16:19]+1)*.05)

def encode_legacy(d,cfg,buffer=.03):
 p0=np.array(d.start);p3=np.array(d.end);v=p3-p0;v=v/max(np.linalg.norm(v),1e-9);pc=(p0+p3)/2+np.array([-v[1],v[0]])*d.bend_m
 p1=p0+2/3*(pc-p0);p2=p3+2/3*(pc-p3);mid=np.array([0,cfg.height_m/2]);half=.1+buffer
 return np.clip(np.r_[(p0-mid)/half,(p3-mid)/half,(p1-p0-(p3-p0)/3)/.03,(p2-p0-2*(p3-p0)/3)/.03,np.full(2,((d.blade_angle+np.pi)%(2*np.pi)-np.pi)/np.pi),np.array([d.pitch_start,(d.pitch_start+d.pitch_end)/2,d.pitch_end])/np.deg2rad(cfg.max_pitch_deg)*2-1,np.full(3,(d.force_N-.5)/7.25-1),np.full(3,(d.speed_m_s-.02)/.05-1)],-1,1).astype(np.float32)

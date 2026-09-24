"""Original five operation families and quadratic paths, common v0.9 physics.
DEPOSIT is a load/contact macro: both substeps consume real work/time budgets.
No label-specific material equation is used. A0/A1 report physical work separately.
"""
import numpy as np
import torch
from .learner import MultiHybridPolicy
from .environment import Env
from .trajectory_action import encode_legacy
from ..wall_cycle.env import DecodedAction

class A0Policy(MultiHybridPolicy):
 def __init__(self,obs_dim):
  super().__init__(obs_dim,5,(0,1,2));mask=torch.zeros(5,19);mask[:3,:12]=1;mask[1:3,9]=0;self.parameter_mask=mask
 def teacher_modes(self,op,target):
  return torch.where(op<3,torch.round(torch.atan2(target[:,5],target[:,4])*4/torch.pi).long()%8,0)

class A0Env(Env):
 def valid_ops(self):return np.array([self.reloads<60 and self.load_requested<600,True,True,True,True],bool)
 def step(self,action):
  op,p=action;p=np.asarray(p,float)
  if op>=3:return self._substep((2 if op==3 else 3,np.zeros(19)))
  c=self.cfg;half=.1+self.buffer;start=np.array([p[0]*half,.25+p[1]*half]);end=np.array([p[2]*half,.25+p[3]*half]);phi=np.arctan2(p[5],p[4]);force=.5+(p[7]+1)*7.25;speed=.02+(p[8]+1)*.05;pitch=(p[10:12]+1)*np.deg2rad(35)/2
  d=DecodedAction('REUSE',tuple(start),tuple(end),phi,p[6]*.03,force,speed,0,*pitch,1);a=encode_legacy(d,c,self.buffer);reward=0.;traces=[];sub=[]
  if op==0:
   choices=np.asarray(c.load_choices_ml);amount=float(choices[min(int((p[9]+1)*.5*len(choices)),len(choices)-1)]);load=np.zeros(19);load[0]=(amount-6)/9-1
   o,r,done,info=self._substep((0,load));reward+=r;traces.extend(self.last_trace);sub.append(info.copy())
   if done:info['macro_substeps']=sub.copy();return o,r,done,info
  o,r,done,info=self._substep((1,a));traces.extend(self.last_trace);self.last_trace=traces;info['macro_substeps']=sub;return o,reward+r,done,info
 def teacher(self,strong=False,no_plateau=False):
  op,a=Env.teacher(self,strong,no_plateau);amount=None
  if op==0:
   amount=6+(a[0]+1)*9;estimate=self.estimate_ml;self.estimate_ml=30.
   try:op,a=Env.teacher(self,strong,no_plateau)
   finally:self.estimate_ml=estimate
   if op!=1:return 3,np.zeros(19,np.float32)
  if op>=2:return (3 if op==2 else 4),np.zeros(19,np.float32)
  x=np.zeros(19,np.float32);x[:4]=a[:4];x[4:6]=[np.cos(a[8]*np.pi),np.sin(a[8]*np.pi)];x[7]=a[13];x[8]=a[16];x[10]=a[10];x[11]=a[12]
  if amount is not None:
   choices=np.asarray(self.cfg.load_choices_ml);index=np.argmin(abs(choices-amount));x[9]=-1+(index+.5)*2/len(choices)
  return (0 if amount is not None else 1),x
 def _substep(self,a):
  # Env's four-operation validity mask must be used inside the atomic substep.
  self.valid_ops=lambda:Env.valid_ops(self)
  try:return Env.step(self,a)
  finally:del self.valid_ops

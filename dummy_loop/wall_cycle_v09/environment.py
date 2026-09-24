"""Unified four-operation task, measured-height observations and conservative material ledger."""
import time
import numpy as np
from .config import config
from .trajectory_action import decode,encode_legacy
from .reward_quality import quality,reward
from ..wall_cycle.mortar import MortarSystem,StrokeStats
from ..wall_cycle.sensor import D435Proxy
from ..wall_cycle.env import local_candidates,DecodedAction

TASKS=('bare','rough','edge_ridge','corner','high_low','finish')
OBS_SCHEMA='v09r12.measured-height.v3'

class Env:
 def __init__(self,seed=0,reach=None,backend='nominal',task=None,buffer=.03,edge_term=True,straight=False,old_stop=False):
  self.cfg=config();self.seed=seed;self.reach=reach;self.backend=backend;self.task=task;self.buffer=buffer;self.edge_term=edge_term;self.straight=straight;self.old_stop=old_stop
  self.arm=None
  if backend=='cosim':
   from .rich_arm import RichArm
   self.arm=RichArm(self.cfg,trace_enabled=True)
  self.reset()
 def reset(self):
  c=self.cfg;ss=np.random.SeedSequence(self.seed).spawn(3);self.rng=np.random.default_rng(ss[0]);self.material=MortarSystem(c,int(ss[1].generate_state(1)[0]));self.sensor=D435Proxy(c,int(ss[2].generate_state(1)[0]));self.material.reset()
  self.kind=self.task or str(self.rng.choice(TASKS));m=self.material;rows,cols=m._score_rows,m._score_cols;h=m.wall[rows,cols];y,x=np.mgrid[-1:1:complex(h.shape[0]),-1:1:complex(h.shape[1])]
  if self.kind=='rough':h[:]=np.clip(.002+.001*np.sin(x*9)*np.cos(y*7),0,.006)
  if self.kind=='edge_ridge':h[:]=.002+.003*np.exp(-((x-.9)/.12)**2)
  if self.kind=='corner':h[:]=.0015+.004*np.exp(-((x-.85)**2+(y-.85)**2)/.08)
  if self.kind=='high_low':h[:]=np.clip(.002+.002*np.tanh(x*8),0,.004)
  if self.kind=='finish':h[:]=.002+.0003*np.sin(x*12)*np.cos(y*10)
  h[:]=np.clip(h*self.rng.uniform(.85,1.15),0,.007)
  m.initial_m3=float(m.wall.sum()*c.cell_m**2);m.p.lift_wall_fraction=float(self.rng.uniform(.6,.9))
  self.initial_wall=m.wall.copy();self.initial_params={'wall_share':m.p.lift_wall_fraction};self.scan=self.sensor.scan(m.wall)
  self.decisions=self.reloads=self.steps=0;self.seconds=0.;self.load_requested=0.;self.since_load=0.;self.estimate_ml=0.;self.history=[];self.fail_map=np.zeros(c.wall_shape);self.last_op=2;self.last_executed=True;self.last_improvement=0.;self.done=False;self.reason=None;self.best_sensor=-np.inf;self.best_step=0;self.stall_windows=0;self.budget=c.base_steps;self.last_trace=[]
  if self.arm:self.arm.reset()
  return self.observe()
 def sensor_score(self):
  m=self.material;h=self.scan['height'][m._score_rows,m._score_cols];cf=self.scan['confidence'][m._score_rows,m._score_cols];valid=cf>.5
  return float(np.mean(valid&(h>=.0015)&(h<=.0025))-np.sqrt(np.mean(np.where(valid,(h-.002)**2,0)))*100)
 def observe(self):
  # All map channels originate from the measurement, never the material truth.
  h=self.scan['height'];cf=self.scan['confidence'];valid=self.scan['valid'];measured=np.where(valid,h,0)
  def pool(a):return a.reshape(25,4,25,4).mean((1,3))
  m=self.material;r,s=m._score_rows,m._score_cols;roi=measured[r,s];conf=cf[r,s];edge=m._edge_mask;gy,gx=np.gradient(roi)
  scalars=np.array([self.decisions/2000,self.steps/100000,self.reloads/60,self.load_requested/600,self.since_load/2,self.estimate_ml/30,self.last_op/3,self.last_executed,self.last_improvement,self.stall_windows/2,(self.decisions-self.best_step)/120])
  return np.r_[pool(measured/.002).ravel(),pool(cf).ravel(),pool(valid.astype(float)).ravel(),(roi[edge]/.002),conf[edge],gx[edge]/.002,gy[edge]/.002,(roi/.002).ravel(),conf.ravel(),(gx/.002).ravel(),(gy/.002).ravel(),pool(self.fail_map).ravel(),scalars].astype(np.float32)
 def valid_ops(self):return np.array([self.reloads<60 and self.load_requested<600,True,True,True],bool)
 def metrics(self):
  m=self.material;q=m.metrics();q.update(quality(m,self.edge_term));q['carry_loss_ml']=m.carry_loss_m3*1e6;q['outside_ml']=m.outside_m3*1e6;q['volume_error_m3']=m.initial_m3+m.supplied_m3-m.wall.sum()*m.wall_cell_area-m.blade_volume_m3-m.dropped_m3-m.outside_m3
  return q
 def step(self,action):
  start=time.perf_counter();op,params=action;op=int(op);params=np.asarray(params,float);c=self.cfg;m=self.material;before=quality(m,self.edge_term)['J'];loss0=np.array([m.carry_loss_m3,m.dropped_m3-m.carry_loss_m3,m.outside_m3]);oldscore=self.sensor_score();oldscan=self.scan['height'].copy();invalid=not self.valid_ops()[op];duration=1.;executed=False;self.last_trace=[];details={}
  if self.arm:self.arm.dense_trace=[];self.arm._mortar=m;self.arm._stats=StrokeStats()
  if not invalid and op==0:
   amount=float(6+(params[0]+1)*9);duration=2.
   if self.arm:
    ex=self.arm;ex._mortar=m;ex._stats=StrokeStats();ex.dense_trace=[];ex._phase='FEED_TRANSIT';path=ex._dense_path(ex.data.qpos[:6],ex.q_feed,30)
    if path is None:invalid=True
    else:
     t0=ex.data.time
     for q in path:ex._run(q,ex._move_time(q))
     normal=ex.data.xmat[ex.blade_body].reshape(3,3)[:,1]@[0,0,1]
     if normal<np.cos(np.deg2rad(15)):invalid=True
     duration=ex.data.time-t0+2
   if not invalid:
    m.feed(amount,c.feed_normal_force_N,c.feed_scoop_depth_m,c.feed_scoop_distance_m,c.feed_scoop_speed_m_s);self.reloads+=1;self.load_requested+=amount;self.estimate_ml=min(30,self.estimate_ml+amount);self.since_load=0;executed=True
   if self.arm:
    if not invalid:self.arm._phase='FEED_SCOOP';self.arm._run(self.arm.q_feed,2.)
    self.arm.sample('FEED_SCOOP');self.last_trace=list(self.arm.dense_trace)
  elif not invalid and op==1:
   contact=decode(params,c,self.buffer)
   if self.straight:contact.points[1]=contact.points[0]+(contact.points[3]-contact.points[0])/3;contact.points[2]=contact.points[0]+2*(contact.points[3]-contact.points[0])/3
   positions=np.array([contact.at(t)[0] for t in np.linspace(0,1,c.stroke_samples)]);length=np.linalg.norm(np.diff(positions,axis=0),axis=1).sum()
   invalid=length<.005 or length>.5
   if self.reach is not None and not invalid:invalid=not self.reach.accepts(contact)
   if self.arm and not invalid:
    ex=self.arm;d=contact.legacy();plan=ex.plan(d);invalid=not plan.ok
    if not invalid:
     t0=ex.data.time;stats=ex.execute(plan,d,m,StrokeStats());duration=ex.data.time-t0;self.last_trace=list(ex.dense_trace);details={k:stats[k] for k in ('tracking_max_mm','force_rmse_N','ik_not_converged','face_down_frames','forces_N','gaps_mm','tracking_mm','joint_errors_deg','saturation','xy_errors_m','target_forces_N')};executed=True
     if stats.get('peak_force_N',0)>c.force_terminate_mult*c.max_force_N:self.done=True;self.reason='force'
   elif not self.arm and not invalid:
    a=contact.at(0);m.begin_stroke(a[2],a[1]);stats=StrokeStats();duration=c.carry_duration_s+2
    # Explicit near-wall turn, nominal normal changes from up to wall-facing.
    for n in np.linspace(1,0,20):m.transport(float(n),.05,dt_s=.02,stats=stats)
    for k,t in enumerate(np.linspace(0,1,c.stroke_samples)):
     xy,tangent,phi,pitch,force,speed=contact.at(t);m.e_x=np.array([np.cos(phi),np.sin(phi)]);m.e_w=np.array([-np.sin(phi),np.cos(phi)])*(-1 if m._flip else 1)
     m.contact(xy,pitch,speed,force_N=force,stats=stats)
     if k:duration+=np.linalg.norm(positions[k]-positions[k-1])/speed
    m.end_stroke(stats)
    for n in np.linspace(0,1,20):m.transport(float(n),.05,dt_s=.02,stats=stats)
    executed=True
   self.since_load+=float(length)
   if executed:self.estimate_ml=max(0.,self.estimate_ml-length*c.blade_length_m*c.target_m*1e6*.6)
  elif op==3 and not invalid:self.done=True;self.reason='give_up'
  if self.arm and (op in (2,3) or invalid):
   self.arm._phase='DECISION_WAIT';self.arm._run(self.arm.data.ctrl[:6].copy(),duration);self.last_trace=list(self.arm.dense_trace)
  self.decisions+=1;self.last_op=op;self.last_executed=executed;self.seconds+=duration;self.steps+=max(1,round(duration*c.control_hz));self.scan=self.sensor.scan(m.wall)
  observed=self.sensor_score();self.last_improvement=observed-oldscore;self.history.append(observed);self.fail_map*=.8
  # Estimated inventory follows commanded loading and executed work, not whole-wall scan bias.
  if op==1 and (invalid or self.last_improvement<0):
   xy=contact.at(.5)[0];u,v=np.meshgrid(m.u,m.v);self.fail_map+=((u-xy[0])**2+(v-xy[1])**2<.025**2)
  if observed>self.best_sensor+.005:self.best_sensor=observed;self.best_step=self.decisions
  window=20 if self.old_stop else 60
  if self.decisions>=window*2 and self.decisions%window==0:
   improve=max(self.history[-window:])-max(self.history[-2*window:-window]);self.stall_windows=self.stall_windows+1 if improve<.01 else 0
   if self.stall_windows>=2 and self.decisions>=100:self.done=True;self.reason=self.reason or 'stall'
  if self.steps>=self.budget and self.budget<c.max_steps and self.decisions-self.best_step<120:self.budget=min(c.max_steps,self.budget+5000)
  if self.steps>=self.budget or self.decisions>=2000:self.done=True;self.reason=self.reason or 'budget'
  q=self.metrics();success=q['coverage']>=.95 and q['edge_coverage']>=.95 and q['rmse_mm']<=.5 and q['p95_error_mm']<=1
  if self.done and success:self.reason='success'
  losses=np.array([m.carry_loss_m3,m.dropped_m3-m.carry_loss_m3,m.outside_m3])-loss0
  r=reward(before,q['J'],losses,duration,invalid,self.done,success if self.done else False)
  info={'metrics':q,'task':self.kind,'seed':self.seed,'decisions':self.decisions,'control_steps':self.steps,'sim_seconds':self.seconds,'end_reason':self.reason,'invalid':bool(invalid),'executed':executed,'load_requested_ml':self.load_requested,'loss_delta_m3':losses.tolist(),'environment_s':time.perf_counter()-start,**details}
  return self.observe(),r,self.done,info
 def teacher(self,strong=False,no_plateau=False):
  c=self.cfg;m=self.material;r,s=m._score_rows,m._score_cols;h=self.scan['height'][r,s];cf=self.scan['confidence'][r,s];valid=cf>.5;zero=np.zeros(19,np.float32)
  if not no_plateau and self.decisions>60 and self.decisions-self.best_step>50:return 3,zero
  if cf.mean()<.82:return 2,zero
  under=np.where(valid,np.maximum(.0017-h,0),0).ravel().astype(np.float32);over=np.where(valid,np.maximum(h-.0025,0),0).ravel().astype(np.float32)
  if under.sum()*c.cell_m**2>2e-6 and self.estimate_ml<5 and self.valid_ops()[0]:zero[0]=(18-6)/9-1;return 0,zero
  M,meta,norm,centres,cells=local_candidates(c,m.u[s],m.v[r]);dep=(M@under-.5*(M@over))/norm;lev=2*(M@over)/norm;values=np.maximum(dep,lev);order=np.argsort(values)[::-1]
  for idx in order[:100 if strong else 24]:
   centre,direction,L=meta[idx];direction=np.array(direction,float);start=np.array(centre)-direction*L/2;end=np.array(centre)+direction*L/2;phi=np.arctan2(direction[1],direction[0])+np.pi/2;level=lev[idx]>dep[idx];p0,p1=(5,5) if level else (25,8)
   profiles=[(4,4),(8,4),(0,0)] if level else [(4,4),(8,4),(10,4),(0,0)]
   for p0,p1 in profiles:
    d=DecodedAction('REUSE',tuple(start),tuple(end),phi,0,2,.06,0,np.deg2rad(p0),np.deg2rad(p1),1);x=encode_legacy(d,c,self.buffer)
    if self.reach is None or self.reach.accepts(decode(x,c,self.buffer)):return 1,x
  return 2,zero

"""Spawn-safe CPU workers, with explicit environment snapshots."""
import os,pickle,time
from pathlib import Path
import numpy as np
from .environment import Env,TASKS
from .reach import Reach

def collect_teacher(job):
 index,seed,task,reach_path,out,buffer,*limits=job;limit=limits[0] if limits else 2000;reach=Reach(reach_path);e=Env(seed,reach=reach,task=task,buffer=buffer);obs=[];ops=[];params=[];masks=[];valid=[];log=[];t=time.perf_counter();initial=e.initial_wall.copy();initial_J=e.metrics()["J"]
 while not e.done and e.decisions<limit:
  o=e.observe();mask=e.valid_ops();action=e.teacher();n,r,done,info=e.step(action);obs.append(o);ops.append(action[0]);params.append(action[1]);masks.append(mask);valid.append(not info['invalid']);log.append(info)
 p=Path(out)/f'{index:04d}.npz';np.savez_compressed(p,obs=np.array(obs),op=np.array(ops),params=np.array(params),mask=np.array(masks),valid=np.array(valid),initial_wall=initial,final_wall=e.material.wall,seed=seed,task=task)
 return {'index':index,'seed':seed,'task':task,'samples':len(obs),'valid':sum(valid),'executed_contacts':sum(x['executed'] for x,a in zip(log,ops) if a==1),'contact_requests':ops.count(1),'elapsed_s':time.perf_counter()-t,'final':log[-1],'best_coverage':max(x['metrics']['coverage'] for x in log),'quality_improved':bool(min(x['metrics']['J'] for x in log)<initial_J-.01)}

def worker(pipe,seed,reach_path,kwargs):
 reach=Reach(reach_path);episode=0;rng=np.random.default_rng(seed+900000);whole=.5
 def make():
  task=str(rng.choice(TASKS[:2] if rng.random()<whole else TASKS[2:]));options=dict(kwargs);space=options.pop('action_space','A1')
  if space=='A0':
   from .a0_control import A0Env
   return A0Env(seed+episode,reach=reach,task=task,**options)
  return Env(seed+episode,reach=reach,task=task,**options)
 e=make();pipe.send((e.observe(),e.valid_ops()))
 while True:
  cmd,arg=pipe.recv()
  if cmd=='close':break
  if cmd=='snapshot':pipe.send(pickle.dumps((e,episode,rng,whole)));continue
  if cmd=='restore':e,episode,rng,whole=pickle.loads(arg);pipe.send((e.observe(),e.valid_ops()));continue
  if cmd=='curriculum':whole=arg;pipe.send(True);continue
  if cmd=='step':
   o,r,done,info=e.step(arg)
   if done:episode+=1;e=make();o=e.observe()
   pipe.send((o,e.valid_ops(),r,done,info))
 pipe.close()

class Vec:
 def __init__(self,n,seed,reach_path,kwargs=None):
  import multiprocessing as mp
  self.pipes=[];self.procs=[]
  for i in range(n):
   parent,child=mp.get_context('spawn').Pipe();p=mp.get_context('spawn').Process(target=worker,args=(child,seed+i*100000,str(reach_path),kwargs or {}));p.start();child.close();self.pipes.append(parent);self.procs.append(p)
  initial=[p.recv() for p in self.pipes];self.obs=np.array([x[0] for x in initial]);self.mask=np.array([x[1] for x in initial])
 def step(self,ops,params):
  for p,o,a in zip(self.pipes,ops,params):p.send(('step',(o,a)))
  rows=[p.recv() for p in self.pipes[:len(ops)]];self.obs[:len(rows)]=np.array([x[0] for x in rows]);self.mask[:len(rows)]=np.array([x[1] for x in rows]);r=np.zeros(len(self.pipes));done=np.zeros(len(self.pipes),bool);r[:len(rows)]=[x[2] for x in rows];done[:len(rows)]=[x[3] for x in rows];return r,done,[x[4] for x in rows]
 def command(self,cmd,arg=None):
  for p in self.pipes:p.send((cmd,arg))
  return [p.recv() for p in self.pipes]
 def close(self):
  for p in self.pipes:p.send(('close',None))
  for p in self.procs:p.join(30)

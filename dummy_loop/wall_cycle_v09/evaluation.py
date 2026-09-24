"""Shared deterministic evaluation; nominal and full execution are labelled separately."""
import json
from pathlib import Path
import numpy as np
from .environment import Env,TASKS
from .reach import Reach

def episode(model,seed,reach_path,backend='nominal',task=None,teacher=None,record=None,kwargs=None,max_decisions=None,snapshot=None):
 options=dict(kwargs or {});space=options.pop('action_space','A1');EnvClass=Env
 if space=='A0':
  from .a0_control import A0Env
  EnvClass=A0Env
 e=EnvClass(seed,Reach(reach_path),backend,task or TASKS[seed%6],**options);initial=e.initial_wall.copy();
 if e.arm:e.arm.trace_enabled=record is not None
 if snapshot is not None:
  path=Path(snapshot);path.parent.mkdir(parents=True,exist_ok=True)
  np.savez_compressed(path,initial_wall=initial,initial_measured_height=e.scan['height'],initial_confidence=e.scan['confidence'],initial_valid=e.scan['valid'],sensor_rng=json.dumps(e.sensor.rng.bit_generator.state),task_rng=json.dumps(e.rng.bit_generator.state),material_parameters=json.dumps(e.initial_params),seed=seed,task=e.kind)
 initial_J=e.metrics()["J"];rows=[];frames=[];ret=0
 while not e.done and (max_decisions is None or e.decisions<max_decisions):
  if model is None:action=(3,np.zeros(19)) if teacher=='FINISH' else e.teacher(strong=teacher=='T+',no_plateau=teacher=='T_budget')
  else:
   import torch
   device=next(model.parameters()).device
   with torch.no_grad(),torch.random.fork_rng(devices=[device.index or 0] if device.type=='cuda' else []):op,a,_,_,_=model.sample(torch.tensor(e.observe(),device=device).unsqueeze(0),torch.tensor(e.valid_ops(),device=device).unsqueeze(0),deterministic=True)
   action=(int(op.item()),a[0].cpu().numpy())
  _,r,_,info=e.step(action);ret+=r;rows.append({'op':action[0],'parameters':action[1].tolist(),'reward':r,**info})
  if record is not None:frames.extend(e.last_trace)
 result={'seed':seed,'task':e.kind,'backend':backend,'return':ret,'decisions':e.decisions,'end_reason':e.reason or ('controlled_horizon' if max_decisions is not None else None),'invalid_fraction':float(np.mean([x['invalid'] for x in rows])),'final':rows[-1]['metrics'],'best_J':min(x['metrics']['J'] for x in rows),'initial_J':initial_J,'initial_material_parameters':e.initial_params,'max_tracking_mm':max([x.get('tracking_max_mm',0) for x in rows]),'face_down_frames':sum(x.get('face_down_frames',0) for x in rows),'contact_requests':sum(x['op'] in ((0,1,2) if space=='A0' else (1,)) for x in rows),'executed_contacts':sum(x['op'] in ((0,1,2) if space=='A0' else (1,)) and x['executed'] for x in rows)}
 if record is not None:
  path=Path(record);path.mkdir(parents=True,exist_ok=False);(path/'steps.json').write_text(json.dumps(rows,indent=2),encoding='utf8');(path/'manifest.json').write_text(json.dumps({'config':e.cfg.to_dict(),'backend':backend,'seed':seed,'result':result,'kwargs':kwargs or {}},indent=2),encoding='utf8');np.savez_compressed(path/'states.npz',initial_wall=initial,final_wall=e.material.wall)
  if frames:np.savez_compressed(path/'trajectory.npz',time_s=np.array([x['time_s'] for x in frames]),q=np.array([x['q'] for x in frames]),qvel=np.array([x['qvel'] for x in frames]),phase=np.array([x['phase'] for x in frames]),wall=np.array([x['wall'] for x in frames]),blade=np.array([x['blade'] for x in frames]))
 return result

def evaluate(model,reach_path,seeds,backend='nominal',teacher=None,kwargs=None):
 rows=[episode(model,s,reach_path,backend,teacher=teacher,kwargs=kwargs) for s in seeds]
 return {'episodes':rows,'mean_J':float(np.mean([x['final']['J'] for x in rows])),'mean_coverage':float(np.mean([x['final']['coverage'] for x in rows])),'mean_edge_coverage':float(np.mean([x['final']['edge_coverage'] for x in rows])),'mean_rmse_mm':float(np.mean([x['final']['rmse_mm'] for x in rows]))}


def controlled_gate(model,reach_path,kwargs=None):
 """Frozen controlled assessment: 20 states per task, at most 12 decisions.
 Full-episode performance is reported separately, not an imitation quality bar.
 """
 rows=[episode(model,60000+i,reach_path,task=TASKS[i%6],kwargs=kwargs,max_decisions=12) for i in range(120)]
 requests=sum(x['contact_requests'] for x in rows);executed=sum(x['executed_contacts'] for x in rows);improves=int(sum(x['best_J']<x['initial_J']-.01 for x in rows))
 return {'protocol':'120 balanced development states, 12 decisions each, no forced terminal reward','episodes':rows,'contact_requests':requests,'executed_contacts':executed,'executable_fraction':executed/max(1,requests),'quality_improved_scenes':improves,'pass':bool(requests>0 and executed/requests>=.8 and improves>0)}

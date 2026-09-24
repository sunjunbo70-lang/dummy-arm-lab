"""Evaluation or checkpoint-driven full co-simulation, saved before optional native replay."""
import argparse,json,subprocess,sys,os
from pathlib import Path
import torch
from .learner import HybridPolicy,MultiHybridPolicy
from .evaluation import episode
from .provenance import capture

def load_model(state):
 if state.get('action_space',state.get('kwargs',{}).get('action_space'))=='A0':
  from .a0_control import A0Policy
  return A0Policy(state['obs_dim'])
 return (MultiHybridPolicy if state.get('components',1)==8 else HybridPolicy)(state['obs_dim'])

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path);p.add_argument('--teacher',choices=['T','T+','T_budget','FINISH']);p.add_argument('--reach',required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--seed',type=int,default=60080);p.add_argument('--episodes',type=int,default=1);p.add_argument('--backend',choices=['nominal','cosim'],default='cosim');p.add_argument('--record',action='store_true');p.add_argument('--task',choices=['bare','rough','edge_ridge','corner','high_low','finish']);p.add_argument('--view',action='store_true');a=p.parse_args()
 if bool(a.checkpoint)==bool(a.teacher):p.error('Choose exactly one checkpoint or teacher')
 if a.view and (not a.record or a.backend!='cosim'):p.error('View requires recorded cosim')
 a.out.mkdir(parents=True,exist_ok=False);capture(a.out/'config');torch.set_num_threads(2);model=None;kwargs={}
 if a.checkpoint:
  state=torch.load(a.checkpoint,weights_only=False,map_location='cpu');model=load_model(state);model.load_state_dict(state['model']);model.eval();kwargs=state.get('kwargs',{})
 subprocess.Popen([r'C:\Users\28017\anaconda3\python.exe','-m','dummy_loop.wall_cycle_v09.telemetry','--out',str(a.out/'resources.jsonl'),'--pid',str(os.getpid())],creationflags=0x08000000)
 rows=[]
 for seed in range(a.seed,a.seed+a.episodes):
  result=episode(model,seed,a.reach,a.backend,task=a.task,teacher=a.teacher,record=a.out/str(seed) if a.record else None,kwargs=kwargs);rows.append(result)
  with (a.out/'episodes.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
  print(seed,result['final']['J'],result['executed_contacts'],result['contact_requests'],flush=True)
 (a.out/'summary.json').write_text(json.dumps({'episodes':rows,'complete':True,'backend':a.backend,'checkpoint':str(a.checkpoint),'teacher':a.teacher},indent=2),encoding='utf8')
 if a.view:subprocess.run([sys.executable,'-m','dummy_loop.wall_cycle_v09.replay',str(a.out/str(a.seed)/'trajectory.npz')],check=True)
if __name__=='__main__':main()

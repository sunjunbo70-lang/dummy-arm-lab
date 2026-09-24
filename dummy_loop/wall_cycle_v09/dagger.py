"""Pre-RL DAgger only: label learner-visited states, preserve failures."""
import argparse,json,time,os,subprocess
from pathlib import Path
import numpy as np
from .environment import Env,TASKS
from .reach import Reach

def main():
 import torch
 from .learner import HybridPolicy,MultiHybridPolicy
 from .evaluate_run import load_model
 from .a0_control import A0Env
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--reach',required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--seed',type=int,default=60620);p.add_argument('--episodes',type=int,default=40);a=p.parse_args();a.out.mkdir(exist_ok=False);state=torch.load(a.checkpoint,weights_only=False,map_location='cpu');model=load_model(state);model.load_state_dict(state['model']);torch.set_num_threads(2);reach=Reach(a.reach);rng=np.random.default_rng(a.seed)
 from .provenance import capture
 capture(a.out/'config')
 subprocess.Popen([r'C:\Users\28017\anaconda3\python.exe','-m','dummy_loop.wall_cycle_v09.telemetry','--out',str(a.out/'resources.jsonl'),'--pid',str(os.getpid())],creationflags=0x08000000)
 for ep in range(a.episodes):
  e=(A0Env if state.get('action_space')=='A0' else Env)(a.seed+ep,reach,task=TASKS[ep%6]);X=[];OP=[];P=[];M=[];initial=e.initial_wall.copy()
  while not e.done:
   o=e.observe();mask=e.valid_ops();label=e.teacher();X.append(o);OP.append(label[0]);P.append(label[1]);M.append(mask)
   with torch.no_grad():op,pa,_,_,_=model.sample(torch.tensor(o).unsqueeze(0),torch.tensor(mask).unsqueeze(0),deterministic=True)
   action=(op.item(),pa[0].numpy()) if rng.random()<.7 else label
   _,_,_,info=e.step(action)
  np.savez_compressed(a.out/f'{ep:04d}.npz',obs=np.array(X),op=np.array(OP),params=np.array(P),mask=np.array(M),valid=np.ones(len(X),bool),initial_wall=initial,final_wall=e.material.wall,seed=e.seed,task=e.kind)
  with (a.out/'episodes.jsonl').open('a') as f:f.write(json.dumps({'episode':ep,'samples':len(X),'final':info})+'\n')
  print(ep,len(X),info['metrics']['coverage'],flush=True)
if __name__=='__main__':main()

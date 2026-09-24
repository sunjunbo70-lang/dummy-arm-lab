"""Paired scene evaluation with independent initial-state artifacts; no training."""
import argparse,json,os,subprocess,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
_MODEL=None
_KW=None

def initialise(checkpoint):
 global _MODEL,_KW
 import torch
 torch.set_num_threads(1);_KW={}
 if checkpoint:
  from .evaluate_run import load_model
  s=torch.load(checkpoint,weights_only=False,map_location='cpu');_MODEL=load_model(s);_MODEL.load_state_dict(s['model']);_MODEL.eval();_KW=s.get('kwargs',{})

def work(job):
 from .evaluation import episode
 seed,reach,backend,teacher,out=job
 return episode(_MODEL,seed,reach,backend,teacher=teacher,kwargs=_KW,snapshot=Path(out)/'initial_states'/f'{seed}.npz')

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint');p.add_argument('--teacher',choices=['T','T+','T_budget','FINISH']);p.add_argument('--reach',required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--episodes',type=int,default=60);p.add_argument('--backend',choices=['nominal','cosim'],default='nominal');p.add_argument('--workers',type=int,default=8);a=p.parse_args()
 if bool(a.checkpoint)==bool(a.teacher):p.error('Choose checkpoint or teacher')
 a.out.mkdir(parents=True,exist_ok=False)
 from .provenance import capture
 capture(a.out/'config');manifest={'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},'checkpoint_sha256':hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest() if a.checkpoint else None,'status':'running'};(a.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
 subprocess.Popen([r'C:\Users\28017\anaconda3\python.exe','-m','dummy_loop.wall_cycle_v09.telemetry','--out',str(a.out/'resources.jsonl'),'--pid',str(os.getpid())],creationflags=0x08000000)
 rows=[];errors=[]
 with ProcessPoolExecutor(a.workers,initializer=initialise,initargs=(a.checkpoint,)) as pool:
  jobs={pool.submit(work,(s,a.reach,a.backend,a.teacher,str(a.out))):s for s in range(a.seed,a.seed+a.episodes)}
  for f in as_completed(jobs):
   try:
    row=f.result();rows.append(row)
    with (a.out/'episodes.jsonl').open('a') as log:log.write(json.dumps(row)+'\n')
    print(len(rows),a.episodes,row['seed'],row['final']['J'],flush=True)
   except Exception as e:
    errors.append({'seed':jobs[f],'error':repr(e)});(a.out/'errors.json').write_text(json.dumps(errors,indent=2))
 summary={'complete':not errors and len(rows)==a.episodes,'requested':a.episodes,'completed':len(rows),'errors':errors,'by_task':{}}
 for task in ['all','bare','rough','edge_ridge','corner','high_low','finish']:
  rr=[r for r in rows if task=='all' or r['task']==task]
  if not rr:continue
  summary['by_task'][task]={'n':len(rr),**{k:float(np.mean([r['final'][k] for r in rr])) for k in ['J','coverage','edge_coverage','rmse_mm','p95_error_mm','roughness_mm','carry_loss_ml','outside_ml']},'improved_from_initial':int(sum(r['final']['J']<r['initial_J']-.01 for r in rr)),'final_success':int(sum(r['final']['coverage']>=.95 and r['final']['edge_coverage']>=.95 and r['final']['rmse_mm']<=.5 and r['final']['p95_error_mm']<=1 for r in rr)),'contact_executable_fraction':sum(r['executed_contacts'] for r in rr)/max(1,sum(r['contact_requests'] for r in rr))}
 (a.out/'summary.json').write_text(json.dumps(summary,indent=2));manifest['status']='completed' if summary['complete'] else 'incomplete';(a.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
 if not summary['complete']:raise SystemExit(2)
if __name__=='__main__':main()

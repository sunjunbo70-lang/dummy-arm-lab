"""P0 audit: matched actions at two material integration time steps, no hardware."""
import argparse,json,time,traceback,os,subprocess,sys
from pathlib import Path
import numpy as np
from dataclasses import asdict
from .clean_env import CleanEnv,config
from .timed_arm import TimedArmExecutor

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--decisions',type=int,default=8);p.add_argument('--seed',type=int,default=60000);a=p.parse_args()
 out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
 (out/'manifest.json').write_text(json.dumps({'stage':'P0 preflight, not training','pid':os.getpid(),'seed':a.seed,'config':asdict(config()),'decisions':a.decisions},indent=2),encoding='utf8')
 monitor=subprocess.Popen([r'C:\Users\28017\anaconda3\python.exe','-m','dummy_loop.wall_cycle_v09.telemetry','--out',str(out/'resources.jsonl'),'--pid',str(os.getpid()),'--interval','5'],creationflags=0x08000000)
 results=[];actions=[]
 try:
  for dt in (.02,.01):
   started=time.perf_counter();c=config();ex=TimedArmExecutor(c,trace_enabled=True);ex.transport_dt_s=dt;e=CleanEnv(c,a.seed,executor=ex);o=e.reset()
   before=o.copy();e.material.blade[:]+=.001;e.last_improvement=777;e.stall_count=77
   assert np.array_equal(before,e._observe()),'hidden-state leak'
   e=CleanEnv(c,a.seed,executor=ex);e.reset();frames=[];rows=[]
   for k in range(a.decisions):
    if dt==.02:actions.append(e.teacher_action())
    action=actions[k];t=time.perf_counter();o,r,done,info=e.step(action)
    row={'decision':k,'action':action.tolist(),'reward':r,'done':done,'wall_s':time.perf_counter()-t,**info};rows.append(row)
    with (out/f'steps_{dt}.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(row,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x))+'\n')
    frames.extend(ex.dense_trace);ex.dense_trace=[]
    print(dt,k,info['metrics']['coverage'],flush=True)
    if done:break
   if frames:
    times=np.array([f['time_s'] for f in frames]);q=np.array([f['q'] for f in frames]);keep=np.r_[True,np.diff(times)>1e-9];times=times[keep];q=q[keep]
    np.savez_compressed(out/f'trajectory_{dt}.npz',time_s=times,q=q,wall=np.array([f['wall'] for f in frames])[keep],blade=np.array([f['blade'] for f in frames])[keep],phase=np.array([f['phase'] for f in frames])[keep],face_up=np.array([f['face_up_score'] for f in frames])[keep])
    maxjump=float(np.rad2deg(np.abs(np.diff(q,axis=0))).max());maxgap=float(np.diff(times).max())
   else:maxjump=maxgap=None
   results.append({'dt':dt,'elapsed_s':time.perf_counter()-started,'final':e.info(False),'transport_samples':ex.transport_samples,'loaded_tilt_samples':ex.loaded_tilt_samples,'max_frame_jump_deg':maxjump,'max_frame_gap_s':maxgap})
  diff=abs(results[0]['final']['metrics']['coverage']-results[1]['final']['metrics']['coverage'])
  report={'stage':'P0 only','observation_mutation_check':True,'runs':results,'coverage_dt_difference':diff,'physics_freeze_approved':False,'next':'Review convergence, pose constraints and reward contract before training.'}
  (out/'report.json').write_text(json.dumps(report,indent=2,default=str),encoding='utf8')
 except Exception:
  (out/'failure.txt').write_text(traceback.format_exc(),encoding='utf8');raise
if __name__=='__main__':main()

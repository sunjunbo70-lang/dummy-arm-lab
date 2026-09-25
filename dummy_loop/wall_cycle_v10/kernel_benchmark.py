"""Sustained fair-workload kernel benchmark with lightweight process telemetry."""
import argparse,json,os,subprocess,time
from pathlib import Path
import numpy as np
import torch
from .tensor_kernels import solve_gap,CapturedGap
from ..wall_cycle.mortar import solve_gap as numpy_gap,MortarParams

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
 monitor=subprocess.Popen(['C:/Users/28017/anaconda3/python.exe','-m','dummy_loop.wall_cycle_v09.telemetry','--out',str(a.out/'resources.jsonl'),'--pid',str(os.getpid()),'--interval','5'],creationflags=subprocess.CREATE_NO_WINDOW)
 rng=np.random.default_rng(271100);data=rng.uniform(.001,.009,(256,6,24));pitch=rng.uniform(0,.14,256);force=rng.uniform(.5,3,256);params=MortarParams();report={'scope':'pressure solve microbenchmark; NOT full material environment or training','gpu':torch.cuda.get_device_name(0),'cpu_tensor_threads':1,'warmup_excluded':True,'results':[]}
 for b in [1,16,64,256]:
  cpu=[torch.tensor(x[:b],dtype=torch.float32) for x in (data,pitch,force)];gpu=[x.cuda() for x in cpu]
  captured=CapturedGap(*gpu)
  functions={'numpy_one_core':lambda:[numpy_gap(data[i],pitch[i],force[i],.005,params) for i in range(b)],'torch_cpu_one_core':lambda:solve_gap(*cpu),'torch_cuda':lambda:solve_gap(*gpu),'cuda_graph':lambda:captured(*gpu)}
  for name,fn in functions.items():
   for _ in range(3):fn()
   torch.cuda.synchronize();durations=[];start=time.perf_counter()
   # 3 x 3.4 seconds per backend/batch: >=120 seconds total sustained workload.
   medians=[]
   for rep in range(3):
    ts=[];end=time.perf_counter()+3.4
    while time.perf_counter()<end:
     if name in ('torch_cuda','cuda_graph'):torch.cuda.synchronize()
     t=time.perf_counter();fn()
     if name in ('torch_cuda','cuda_graph'):torch.cuda.synchronize()
     ts.append(time.perf_counter()-t)
    medians.append(float(np.median(ts)));durations.extend(ts)
   row={'batch':b,'backend':name,'replicate_median_ms':[x*1000 for x in medians],'median_ms':float(np.median(durations)*1000),'p95_ms':float(np.percentile(durations,95)*1000),'calls':len(durations),'elapsed_s':time.perf_counter()-start};report['results'].append(row);print(row,flush=True)
 (a.out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf8')
if __name__=='__main__':main()


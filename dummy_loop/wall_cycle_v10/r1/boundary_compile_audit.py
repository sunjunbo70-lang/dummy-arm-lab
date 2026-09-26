import argparse,json,time
from pathlib import Path
import numpy as np
from .boundary_flow import instantaneous as reference
from .compiled_boundary_flow import instantaneous as compiled
from .adaptive_boundary_flow import adaptive
from ..p0 import snapshot

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--context',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False);head=snapshot(out)
 rng=np.random.default_rng(62006);cases=[(rng.uniform([-.3,-.03],[.3,.53]),rng.uniform(-3,3),rng.uniform(-.3,.3,2),rng.uniform(-2,2)) for _ in range(200)]
 compiled(*cases[0]);timings={};results={}
 for name,fn in [('reference',reference),('compiled',compiled)]:
  start=time.perf_counter();results[name]=[fn(*x) for x in cases];timings[name]=time.perf_counter()-start
 maximum=0.
 for (x,xo),(y,yo) in zip(results['reference'],results['compiled']):
  for xd,yd in zip(x,y):maximum=max(maximum,max((abs(xd.get(k,0)-yd.get(k,0)) for k in set(xd)|set(yd)),default=0.))
  maximum=max(maximum,float(abs(np.asarray(xo)-yo).max()))
 c=json.loads(Path(a.context).read_text())['context'];maps=[];adaptive_times=[]
 for backend in (False,True):
  start=time.perf_counter();m,stats=adaptive(c['begin'],c['end'],c['alpha'],c['beta'],compiled=backend);adaptive_times.append(time.perf_counter()-start);maps.append(m)
 delta=0.
 for x,y in zip(*maps):
  xd={(int(b),int(w)):float(v) for b,w,v in zip(*x[:3])};yd={(int(b),int(w)):float(v) for b,w,v in zip(*y[:3])}
  delta=max(delta,max((abs(xd.get(k,0)-yd.get(k,0)) for k in set(xd)|set(yd)),default=0.),float(abs(x[3]-y[3]).max()))
 result=dict(source_head=head,rate_max_abs_error_m2_per_time=maximum,rate_seconds=timings,rate_speedup=timings['reference']/timings['compiled'],adaptive_seconds=adaptive_times,adaptive_speedup=adaptive_times[0]/adaptive_times[1],adaptive_max_area_difference_m2=delta,scope='warm CPU geometry component only, not end-to-end RL',training_started=False)
 (out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()

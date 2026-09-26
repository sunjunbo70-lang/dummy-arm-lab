"""Matched-time divergence localization; no change to solver or thresholds."""
import argparse,json,os,copy,time
from pathlib import Path
import numpy as np
from .stationary_boundary_pressure import StationaryBoundaryPressure
from ..p0 import make_scene,snapshot

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False);head=snapshot(out)
 source=out.parent/'expanded_bounded_edge_001/single_076_recipes.json';q=json.loads(source.read_text())['recipes'][0];base,_=make_scene(16)
 models=[StationaryBoundaryPressure(base.wall*base.wall_cell_area,base.blade*base.blade_cell_area,params=copy.deepcopy(base.p)) for _ in range(2)];n=3344
 (out/'manifest.json').write_text(json.dumps(dict(pid=os.getpid(),source_head=head,steps=[n,2*n],training_started=False)))
 def step(s,t):
  s.contact(np.array(q['begin'])*(1-t)+np.array(q['end'])*t,*[q[k][0]*(1-t)+q[k][1]*t for k in ('angle','pitch','force')])
 start=time.perf_counter();previous=None;max_jump=0.;top=[]
 for i in range(n+1):
  step(models[0],i/n)
  if i:step(models[1],(2*i-1)/(2*n))
  step(models[1],i/n)
  x,y=models;delta=(x.bead-y.bead)/x.bc**2*1000
  jump=0. if previous is None else float(abs(delta-previous).max());previous=delta.copy()
  row=dict(step=i,t=i/n,gaps=[x.g0,y.g0],bead_max_mm=float(abs(delta).max()),bead_delta_mm=delta.tolist(),blade_max_mm=float(abs(x.blade-y.blade).max()/x.bc**2*1000),wall_max_mm=float(abs(x.wall-y.wall).max()/x.wc**2*1000),bead_jump_mm=jump,seconds=time.perf_counter()-start)
  with (out/'trace.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  if jump>max_jump:
   max_jump=jump;np.savez_compressed(out/'largest_jump_latest.npz',coarse_wall=x.wall,fine_wall=y.wall,coarse_blade=x.blade,fine_blade=y.blade,coarse_bead=x.bead,fine_bead=y.bead,step=i)
  top.append(row);top=sorted(top,key=lambda z:z['bead_jump_mm'],reverse=True)[:10]
 (out/'summary.json').write_text(json.dumps(dict(top_jumps=top,training_started=False),indent=2))
if __name__=='__main__':main()

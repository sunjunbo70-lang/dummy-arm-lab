"""Small v7 geometry convergence probes; diagnostic only, not gate scenes."""
import argparse,json,time
from pathlib import Path
import numpy as np
from .coupled_midpoint_inventory import CoupledMidpointPressure
from .contact_inventory import ContactInventory
from ..p0 import make_scene,snapshot

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--source',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False);head=snapshot(out)
 r=json.loads((Path(a.source)/'single_013_recipes.json').read_text())['recipes'][0]
 rows=[]
 for mode in ('fixed_angle','pure_rotation','combined'):
  states=[];timings=[];ledger=[]
  for intervals in (200,400,800):
   base,_=make_scene(13);m=CoupledMidpointPressure(base.wall*base.wall_cell_area,base.blade*base.blade_cell_area);total=m.total();start=time.perf_counter()
   begin=np.array(r['begin']);end=begin if mode=='pure_rotation' else np.array(r['end']);a0,a1=r['angle'];a1=a0 if mode=='fixed_angle' else a1
   for t in np.linspace(0,1,intervals+1):m.move(begin+t*(end-begin),a0+t*(a1-a0),.002)
   ContactInventory.lift(m,base.p.lift_wall_fraction);states.append(m.wall);timings.append(time.perf_counter()-start);ledger.append(float(abs(m.total()-total)*1e6))
  errors=[float(np.max(abs(x-y))/.005**2*1000) for x,y in zip(states,states[1:])]
  row=dict(mode=mode,intervals=[200,400,800],max_differences_mm=errors,observed_order=float(np.log2(errors[0]/errors[1])) if min(errors)>0 else None,seconds=timings,ledger_error_ml=ledger)
  rows.append(row)
  with (out/'cases.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  print(json.dumps(row),flush=True)
 (out/'summary.json').write_text(json.dumps(dict(source_head=head,rows=rows,scope='geometry isolation only',full_gate=False,training_started=False),indent=2))
if __name__=='__main__':main()

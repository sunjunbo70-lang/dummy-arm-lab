"""Spatial closure audit of rigid boundary flux; no material pass claim."""
import argparse,json,time
from pathlib import Path
import numpy as np
from .boundary_flow import integrate,GEOMETRY_VERSION
from .overlap import planar_overlap
from ..p0 import snapshot

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--out',required=True);p.add_argument('--intervals',type=int,nargs='+',default=[1024,2048,4096]);a=p.parse_args()
 out=Path(a.out);out.mkdir(parents=True,exist_ok=False);head=snapshot(out)
 r=json.loads((Path(a.source)/'single_013_recipes.json').read_text())['recipes'][0];begin=np.array(r['begin']);end=np.array(r['end']);alpha,beta=r['angle']
 def coverage(x,angle):
  m=planar_overlap(x,angle,(6,24),.005,(100,100),.005,np.array([-.25,0.]));return np.bincount(m[1],weights=m[2],minlength=10000)
 expected=coverage(end,beta)-coverage(begin,alpha);rows=[]
 for n in a.intervals:
  start=time.perf_counter();entry,exit=integrate(begin,end,alpha,beta,intervals=n)
  delta=np.bincount(entry[1],weights=entry[2],minlength=10000)-np.bincount(exit[1],weights=exit[2],minlength=10000);err=delta-expected
  row=dict(intervals=n,seconds=time.perf_counter()-start,max_cell_error_m2=float(abs(err).max()),net_area_error_m2=float(abs(err.sum())),uniform_2mm_equivalent_max_error_mm=float(abs(err).max()/.005**2*2))
  rows.append(row);print(json.dumps(row),flush=True)
  np.savez_compressed(out/f'closure_{n}.npz',expected=expected,integrated=delta)
 (out/'summary.json').write_text(json.dumps(dict(source_head=head,geometry_version=GEOMETRY_VERSION,rows=rows,scope='geometry only; uniform thickness conversion is illustrative, not actual material',training_started=False,full_material_gate=False),indent=2))
if __name__=='__main__':main()

"""Chronological zero-gap pickup diagnostic with explicit area closure correction.
Not full material physics; blade receives picked volume but never deposits.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from .moving_wall_exchange import transfer,VERSION
from .boundary_flow import integrate
from .overlap import planar_overlap
from ..p0 import snapshot

def free_area(center,angle):
 m=planar_overlap(np.array(center),angle,(6,24),.005,(100,100),.005,np.array([-.25,0.]))
 a=.005**2-np.bincount(m[1],weights=m[2],minlength=10000)
 if a.min() < -2.5e-17:raise ValueError('Coverage outside roundoff allowance')
 return np.maximum(a,0.)

def run(begin,end,a0,a1,n):
 start=time.perf_counter();area=free_area(begin,a0);wall=area*.002;initial=wall.sum();picked=0.;correction=0.;max_residual=0.;negative_raw=0
 for i in range(n):
  t0=i/n;t1=(i+1)/n;x0=begin+t0*(end-begin);x1=begin+t1*(end-begin)
  entry,exit=integrate(x0,x1,a0+t0*(a1-a0),a0+t1*(a1-a0),intervals=2)
  e=np.bincount(entry[1],weights=entry[2],minlength=10000);x=np.bincount(exit[1],weights=exit[2],minlength=10000)
  target=free_area(x1,a0+t1*(a1-a0));change=target-area
  residual=(x-e)-change;max_residual=max(max_residual,float(abs(residual).max()));negative_raw+=int(((area+x-e)<-2.5e-17).sum())
  # Explicitly enforce geometric conservation against exact endpoint capacity.
  # Preserve min(entry,exit), correct the larger flow. This is a candidate
  # projection, not evidence that raw boundary quadrature is accurate.
  through=np.minimum(e,x);ec=through+np.maximum(-change,0.);xc=through+np.maximum(change,0.)
  correction+=float(abs(ec-e).sum()+abs(xc-x).sum())
  live=(area>0)|(target>0)
  # Avoid cancellation in reconstructing A1: analytic transfer receives
  # consistent rates, but exact-close events can differ by roundoff.
  # Use target directly through an explicit final-area argument.
  w,p,b=transfer(wall[live],area[live],ec[live],xc[live],np.zeros(live.sum()),final_area=target[live])
  wall[live]=w;picked+=float(p.sum());area=target
 return dict(steps=n,picked_ml=picked*1e6,ledger_error_ml=abs(wall.sum()+picked-initial)*1e6,minimum_volume=float(wall.min()),max_raw_area_residual_m2=max_residual,cumulative_area_correction_m2=correction,negative_raw_capacity_cells=negative_raw,seconds=time.perf_counter()-start),wall

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--source');p.add_argument('--cases',nargs='+');p.add_argument('--steps',nargs='+',type=int,default=[32,64,128]);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False);head=snapshot(out);rows=[]
 cases=[('axial',np.array([.02,.25]),0.,0.),('diagonal',np.array([.02,.27]),0.,0.),('rotation',np.array([0.,.25]),.31,.61)]
 cases=[(name,np.array([0.,.25]),end,alpha,beta) for name,end,alpha,beta in cases]
 if a.source:
  recipe=json.loads((Path(a.source)/'single_013_recipes.json').read_text())['recipes'][0]
  cases.append(('case13',np.array(recipe['begin']),np.array(recipe['end']),*recipe['angle']))
 for name,begin,end,alpha,beta in cases:
  if a.cases and name not in a.cases:continue
  previous=None
  for n in a.steps:
   row,wall=run(begin,end,alpha,beta,n);row['case']=name
   if previous is not None:row['max_wall_change_mm']=float(abs(wall-previous).max()/.005**2*1000)
   previous=wall;rows.append(row);print(json.dumps(row),flush=True);np.savez_compressed(out/f'{name}_{n}.npz',wall=wall)
 (out/'summary.json').write_text(json.dumps(dict(source_head=head,version=VERSION,scope='zero-gap pickup only with explicit area closure projection, not G0',rows=rows,training_started=False),indent=2))
if __name__=='__main__':main()

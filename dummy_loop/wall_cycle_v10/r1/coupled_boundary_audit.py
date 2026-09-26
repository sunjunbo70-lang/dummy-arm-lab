"""Chronological coupled boundary diagnostic, excludes pressure/internal flow/lift."""
import argparse,json,time
from pathlib import Path
import numpy as np
from .moving_pickup_audit import free_area
from .overlap import planar_overlap
from .boundary_flow import integrate
from .coupled_boundary_exchange import exchange,VERSION
from ..p0 import snapshot

def run(begin,end,alpha,beta,n,adaptive_mode=False):
 def area_at(x,a):
  raw=free_area(x,a)
  return np.where(raw<2.5e-17,0.,np.where(abs(raw-.005**2)<2.5e-17,.005**2,raw)) # explicit geometric roundoff snap, no positive floor
 area=area_at(begin,alpha);wall=.002*area;blade=np.full(144,.001*.005**2);total=wall.sum()+blade.sum();lost=0.;correction=0.;maxiter=0;support_area=0.;quadrature_nodes=0
 start=time.perf_counter()
 for i in range(n):
  u=i/n;v=(i+1)/n
  if adaptive_mode:
   from .adaptive_boundary_flow import adaptive
   maps,diagnostic=adaptive(begin+u*(end-begin),begin+v*(end-begin),alpha+u*(beta-alpha),alpha+v*(beta-alpha))
   quadrature_nodes+=diagnostic['nodes']
  else:maps=integrate(begin+u*(end-begin),begin+v*(end-begin),alpha+u*(beta-alpha),alpha+v*(beta-alpha),intervals=2)
  target=area_at(begin+v*(end-begin),alpha+v*(beta-alpha));raw=[np.bincount(m[1],weights=m[2],minlength=10000) for m in maps]
  e,x=raw;change=target-area;through=np.minimum(e,x);empty=(area==0)&(target==0)
  if (through[empty]>2.5e-17).any():raise ValueError(f'step {i}: unresolved zero-area throughflow')
  through[empty]=0.
  desired=[through+np.maximum(-change,0.),through+np.maximum(change,0.)]
  corrected=[]
  for kind,(m,old,new) in enumerate(zip(maps,raw,desired)):
   missing=(old==0)&(new>0)
   if missing.any():
    if new[missing].max()>2.5e-13:raise ValueError(f'step {i}: missing support exceeds diagnostic area budget')
    t=v if kind==0 else u
    endpoint=planar_overlap(begin+t*(end-begin),alpha+t*(beta-alpha),(6,24),.005,(100,100),.005,np.array([-.25,0.]))
    take=missing[endpoint[1]]
    bi=endpoint[0][take];wi=endpoint[1][take];weights=endpoint[2][take]
    norm=np.bincount(wi,weights=weights,minlength=10000)
    if (norm[missing]<=0).any():raise ValueError('No endpoint support either')
    weights=weights/norm[wi]*new[wi]
    support_area+=float(new[missing].sum())
    m=(np.r_[m[0],bi],np.r_[m[1],wi],np.r_[m[2],weights],m[3])
    old=old.copy();old[missing]=new[missing]
   ratio=np.divide(new,old,out=np.zeros_like(old),where=old>0)
   corrected.append((m[0],m[1],m[2]*ratio[m[1]],m[3]));correction+=float(abs(new-old).sum())
  wall,blade,dl,info=exchange(wall,blade,area,target,.005**2,*corrected,.001)
  lost+=dl;area=target;maxiter=max(maxiter,info['iterations'])
 return dict(status='completed',steps=n,seconds=time.perf_counter()-start,ledger_error_ml=float(abs(wall.sum()+blade.sum()+lost-total)*1e6),minimum_volume=float(min(wall.min(),blade.min(),lost)),max_iterations=maxiter,geometric_area_correction_m2=correction,endpoint_support_area_m2=support_area,quadrature_nodes=quadrature_nodes),wall,blade

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--adaptive',action='store_true');p.add_argument('--cases',nargs='+');p.add_argument('--source');p.add_argument('--steps',nargs='+',type=int,default=[128,256]);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False);head=snapshot(out);rows=[]
 cases=[('axial',[0.,.25],[.02,.25],0.,0.),('diagonal',[0.,.25],[.02,.27],0.,0.),('rotation',[0.,.25],[0.,.25],.31,.61)]
 if a.source:
  recipe=json.loads((Path(a.source)/'single_013_recipes.json').read_text())['recipes'][0]
  cases.append(('case13',recipe['begin'],recipe['end'],*recipe['angle']))
 for name,begin,end,alpha,beta in cases:
  if a.cases and name not in a.cases:continue
  previous=None
  for n in a.steps:
   try:
    row,wall,blade=run(np.array(begin),np.array(end),alpha,beta,n,a.adaptive)
    if previous is not None:row['max_wall_difference_mm']=float(abs(wall-previous).max()/.005**2*1000)
    previous=wall;np.savez_compressed(out/f'{name}_{n}.npz',wall=wall,blade=blade)
   except (ValueError,RuntimeError,FloatingPointError) as exc:row=dict(steps=n,status='failed',reason=str(exc))
   row['case']=name;rows.append(row);print(json.dumps(row),flush=True)
 (out/'summary.json').write_text(json.dumps(dict(source_head=head,version=VERSION,adaptive=a.adaptive,rows=rows,training_started=False,full_material_gate=False,scope='finite blade boundary exchange, no pressure/bead/lift'),indent=2))
if __name__=='__main__':main()

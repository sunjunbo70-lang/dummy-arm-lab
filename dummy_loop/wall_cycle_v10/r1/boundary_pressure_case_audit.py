"""Reproduce original case13 scene/controls with full candidate pressure and lift.
A diagnostic pair of resolutions, not the 100/50-case material gate.
"""
import argparse,copy,json,time,os
from pathlib import Path
import numpy as np
from .boundary_pressure import BoundaryPressure
from ..p0 import make_scene,snapshot

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--source',required=True);p.add_argument('--steps',type=int,nargs='+',default=[256,512]);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False);head=snapshot(out)
 recipe=json.loads((Path(a.source)/'single_013_recipes.json').read_text())['recipes'][0];base,scene=make_scene(13)
 (out/'manifest.json').write_text(json.dumps(dict(pid=os.getpid(),source_head=head,physics_version=BoundaryPressure.physics_version,steps=a.steps,recipe=recipe,scene=scene,training_started=False),indent=2))
 rows=[];previous=None
 for n in a.steps:
  s=BoundaryPressure(base.wall*base.wall_cell_area,base.blade*base.blade_cell_area,params=copy.deepcopy(base.p));total=s.total();start=time.perf_counter();row=dict(steps=n);i=-1
  try:
   for i,t in enumerate(np.linspace(0,1,n+1)):
    center=(1-t)*np.array(recipe['begin'])+t*np.array(recipe['end'])
    angle,pitch,force=[float((1-t)*recipe[k][0]+t*recipe[k][1]) for k in ('angle','pitch','force')]
    s.contact(center,angle,pitch,force)
    if i%64==0:
     with (out/f'progress_{n}.jsonl').open('a') as f:f.write(json.dumps(dict(step=i,seconds=time.perf_counter()-start,ledger_error_ml=float(abs(s.total()-total)*1e6)))+'\n')
   np.savez_compressed(out/f'prelift_{n}.npz',wall=s.wall,blade=s.blade,bead=s.bead)
   s.lift(s.params.lift_wall_fraction)
   row.update(status='completed',ledger_error_ml=float(abs(s.total()-total)*1e6),minimum_volume=float(min(s.wall.min(),s.blade.min(),s.bead.min())),boundary_stats=s.boundary_stats)
   if previous is not None:row['max_wall_difference_mm']=float(abs(s.wall-previous).max()/.005**2*1000)
   previous=s.wall.copy();np.savez_compressed(out/f'final_{n}.npz',wall=s.wall,blade=s.blade,bead=s.bead)
  except (RuntimeError,ValueError,FloatingPointError) as exc:
   row.update(status='failed',failed_step=i,reason=str(exc),min_wall_m3=float(s.wall.min()),min_blade_m3=float(s.blade.min()))
   np.savez_compressed(out/f'failed_{n}.npz',wall=s.wall,blade=s.blade,bead=s.bead)
  row['seconds']=time.perf_counter()-start;rows.append(row);print(json.dumps(row),flush=True)
  with (out/'cases.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
 (out/'summary.json').write_text(json.dumps(dict(source_head=head,rows=rows,training_started=False,full_material_gate=False),indent=2))
if __name__=='__main__':main()

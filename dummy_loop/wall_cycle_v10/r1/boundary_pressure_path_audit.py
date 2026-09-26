import argparse,json,time
from pathlib import Path
import numpy as np
from .boundary_pressure import BoundaryPressure
from ..p0 import snapshot

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False);head=snapshot(out);rows=[];previous=None
 yy,xx=np.mgrid[:100,:100];initial=(.002+.001*np.exp(-((xx-48)**2+(yy-50)**2)/40))*.005**2
 for n in (16,32,64):
  s=BoundaryPressure(initial,np.full((6,24),.002*.005**2));total=s.total();start=time.perf_counter();row=dict(steps=n)
  try:
   for i,t in enumerate(np.linspace(0,1,n+1)):s.contact([-.02+.005*t,.25+.002*t],.3+.1*t,.2+.1*t,2.+2*t)
   s.lift(.5)
   row.update(status='completed',ledger_error_ml=float(abs(s.total()-total)*1e6),minimum_volume=float(min(s.wall.min(),s.blade.min(),s.bead.min())),boundary_stats=s.boundary_stats)
   if previous is not None:row['max_wall_difference_mm']=float(abs(s.wall-previous).max()/.005**2*1000)
   previous=s.wall.copy();np.savez_compressed(out/f'path_{n}.npz',wall=s.wall,blade=s.blade,bead=s.bead)
  except (RuntimeError,ValueError,FloatingPointError) as exc:row.update(status='failed',failed_step=i,reason=str(exc))
  row['seconds']=time.perf_counter()-start;rows.append(row);print(json.dumps(row),flush=True)
 (out/'summary.json').write_text(json.dumps(dict(source_head=head,physics_version=BoundaryPressure.physics_version,rows=rows,training_started=False,full_material_gate=False),indent=2))
if __name__=='__main__':main()

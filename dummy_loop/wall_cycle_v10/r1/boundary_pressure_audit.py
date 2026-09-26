import argparse,json
from pathlib import Path
import numpy as np
from .boundary_pressure import BoundaryPressure
from ..p0 import snapshot

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False);head=snapshot(out);rows=[]
 for thickness in (0.,.002):
  s=BoundaryPressure(np.full((100,100),thickness*.005**2),np.full((6,24),.002*.005**2));total=s.total();row=dict(initial_wall_thickness_m=thickness);events=[]
  try:
   for i in range(3):
    s.contact([i*.0001,.25],.3+i*.0001,.2,2.)
    events.append(dict(step=i,wall_min_m3=float(s.wall.min()),blade_min_m3=float(s.blade.min()),ledger_error_ml=float(abs(s.total()-total)*1e6)))
   s.lift(.5);row.update(status='completed',ledger_error_ml=float(abs(s.total()-total)*1e6))
  except (ValueError,RuntimeError,FloatingPointError) as exc:row.update(status='failed',reason=str(exc),wall_min_m3=float(s.wall.min()),blade_min_m3=float(s.blade.min()))
  row['events']=events;rows.append(row);print(json.dumps(row),flush=True)
 (out/'summary.json').write_text(json.dumps(dict(source_head=head,physics_version=BoundaryPressure.physics_version,rows=rows,training_started=False,full_material_gate=False),indent=2))
if __name__=='__main__':main()

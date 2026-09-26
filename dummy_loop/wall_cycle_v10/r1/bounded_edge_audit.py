import argparse,json,os
from pathlib import Path
from . import expanded_boundary_audit as audit
from .bounded_edge_pressure import BoundedEdgePressure
from ..p0 import snapshot

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False);head=snapshot(out)
 (out/'manifest.json').write_text(json.dumps(dict(pid=os.getpid(),source_head=head,physics_version=BoundedEdgePressure.physics_version,indices=[1,18,21],training_started=False)))
 audit.BoundaryPressure=BoundedEdgePressure;rows=[]
 for i in [1,18,21]:
  try:r=audit.run_case((i,False,1,str(out)))
  except Exception as e:r=dict(index=i,passed=False,error=repr(e))
  rows.append(r)
  with (out/'cases.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
 (out/'summary.json').write_text(json.dumps(rows,indent=2))
if __name__=='__main__':main()

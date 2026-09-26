import argparse,json,os,traceback
from pathlib import Path
from . import expanded_boundary_audit as audit
from .stationary_boundary_pressure import StationaryBoundaryPressure
from ..p0 import snapshot

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--steps',type=int,default=4);a=p.parse_args();o=Path(a.out);o.mkdir(parents=True,exist_ok=False);h=snapshot(o)
 (o/'manifest.json').write_text(json.dumps(dict(pid=os.getpid(),source_head=h,steps=a.steps,physics_version=StationaryBoundaryPressure.physics_version,training_started=False)))
 audit.BoundaryPressure=StationaryBoundaryPressure
 try:r=audit.run_case((0,True,a.steps,str(o)))
 except Exception as e:r=dict(error=repr(e),traceback=traceback.format_exc())
 (o/'summary.json').write_text(json.dumps(r,indent=2))
if __name__=='__main__':main()

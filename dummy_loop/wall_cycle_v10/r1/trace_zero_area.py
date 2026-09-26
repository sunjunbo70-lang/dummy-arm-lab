"""Capture zero-area exchange inputs without changing the audited solver."""
import argparse,json,traceback
from pathlib import Path
import numpy as np
from . import boundary_pressure as bp
from .expanded_boundary_audit import run_case
from ..p0 import snapshot

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--index',type=int,default=21);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False);snapshot(out)
 original=bp.exchange
 def capture(wall,blade,area,target,blade_area,entry,exit,gap,**kw):
  bad=(area==0)&(wall>0)
  if bad.any():
   np.savez_compressed(out/'failure_inputs.npz',wall=wall,blade=blade,area=area,target=target,entry_bi=entry[0],entry_wi=entry[1],entry_area=entry[2],exit_bi=exit[0],exit_wi=exit[1],exit_area=exit[2],gap=gap)
   (out/'zero_area.json').write_text(json.dumps(dict(cells=np.flatnonzero(bad).tolist(),volumes_m3=wall[bad].tolist(),total_ml=float(wall[bad].sum()*1e6),maximum_m3=float(wall[bad].max())),indent=2))
  return original(wall,blade,area,target,blade_area,entry,exit,gap,**kw)
 bp.exchange=capture
 try:r=run_case((a.index,False,1,str(out)))
 except Exception as exc:r=dict(error=repr(exc),traceback=traceback.format_exc())
 (out/'summary.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
if __name__=='__main__':main()

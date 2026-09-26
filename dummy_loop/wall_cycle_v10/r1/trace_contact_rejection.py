"""Instrument unchanged v8.9 rejection; never relax acceptance or adopt physics."""
import copy,json,pickle,argparse,time
from pathlib import Path
import numpy as np
from .precise_gap_pressure import PreciseGapPressure
from .stationary_boundary_pressure import StationaryBoundaryPressure
from ..p0 import make_scene,snapshot

class TracedPressure(PreciseGapPressure):
 trace_out=None
 def _refine(self,target,depth):
  try:return super()._refine(target,depth)
  except RuntimeError as exc:
   if not str(exc).startswith('Contact error budget exhausted'):raise
   center,angle,pitch,force=target
   mid=((self.pose[0]+center)/2,(self.pose[1]+angle)/2,*(.5*(np.asarray(self.last_controls)+[pitch,force])))
   coarse=copy.deepcopy(self);StationaryBoundaryPressure.contact(coarse,*target)
   fine=copy.deepcopy(self);StationaryBoundaryPressure.contact(fine,*mid);StationaryBoundaryPressure.contact(fine,*target)
   row=dict(depth=depth,reason=str(exc),start_controls=self.last_controls,target_controls=[pitch,force],start_gap=self.g0,coarse_gap=coarse.g0,fine_gap=fine.g0,components={})
   for k in ('wall','blade','bead'):
    a=getattr(coarse,k);b=getattr(fine,k);d=(b-a)/self.bc**2*1000;idx=np.unravel_index(abs(d).argmax(),d.shape)
    row['components'][k]=dict(max_diff_mm=float(abs(d).max()),index=list(map(int,idx)),coarse_m3=float(a[idx]),fine_m3=float(b[idx]),sum_diff_ml=float((b-a).sum()*1e6))
   out=Path(self.trace_out)
   (out/f'depth_{depth}.json').write_text(json.dumps(row,indent=2))
   if depth==self.max_contact_depth:
    with (out/'rejected_state.pkl').open('xb') as f:pickle.dump(dict(state=self,target=target),f)
    np.savez_compressed(out/'rejected_comparison.npz',**{f'{label}_{k}':getattr(state,k) for label,state in [('start',self),('coarse',coarse),('fine',fine)] for k in ('wall','blade','bead')})
   raise

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--source',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False);head=snapshot(out)
 recipe=json.loads((Path(a.source)/'single_076_recipes.json').read_text())['recipes'][0];base,_=make_scene(16)
 s=TracedPressure(base.wall*base.wall_cell_area,base.blade*base.blade_cell_area,params=copy.deepcopy(base.p));s.trace_out=str(out);start=time.perf_counter()
 try:
  for i,t in enumerate(np.linspace(0,1,3345)):
   center=(1-t)*np.array(recipe['begin'])+t*np.array(recipe['end']);angle,pitch,force=[float((1-t)*recipe[k][0]+t*recipe[k][1]) for k in ('angle','pitch','force')];s.contact(center,angle,pitch,force)
  result=dict(status='unexpectedly_completed')
 except RuntimeError as exc:result=dict(status='rejection_captured',step=i,reason=str(exc))
 result.update(seconds=time.perf_counter()-start,head=head,physics_version=s.physics_version,training_started=False)
 (out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()

"""Replay a saved expanded-audit single case with stage-resolved inventories."""
import argparse,copy,json,time
from pathlib import Path
import numpy as np
from .split_inventory import SplitPressure
from ..p0 import make_scene,snapshot

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--out',required=True);p.add_argument('--index',type=int,default=13);a=p.parse_args()
    out=Path(a.out);out.mkdir(exist_ok=False,parents=True);head=snapshot(out)
    recipe=json.loads((Path(a.source)/f'single_{a.index:03d}_recipes.json').read_text())['recipes'][0]
    (out/'manifest.json').write_text(json.dumps(dict(source=a.source,index=a.index,recipe=recipe,source_head=head,physics=SplitPressure.physics_version,scope='stage diagnosis only, no training'),indent=2))
    saved=[];times=[]
    for spacing in (.00005,.000025):
        base,_=make_scene(a.index%60)
        m=SplitPressure(base.wall*base.wall_cell_area,base.blade*base.blade_cell_area,params=copy.deepcopy(base.p))
        initial=m.total();states={};start=time.perf_counter()
        def capture(name):
            states[name]=dict(wall=m.wall.copy(),blade=m.blade.copy(),bead=m.bead.copy(),dropped=float(m.dropped),outside=float(m.outside),total=float(m.total()),ledger_error_ml=float(abs(m.total()-initial)*1e6),gap_m=float(m.g0))
        begin=np.array(recipe['begin']);end=np.array(recipe['end']);a0,a1=recipe['angle'];p0,p1=recipe['pitch'];f0,f1=recipe['force']
        radius=.5*np.linalg.norm(np.array(m.blade.shape)*m.bc)
        count=max(2,int(np.ceil((np.linalg.norm(end-begin)+radius*abs(a1-a0))/spacing))+1)
        for i,t in enumerate(np.linspace(0,1,count)):
            m.contact(begin+t*(end-begin),a0+t*(a1-a0),float(p0+t*(p1-p0)),float(f0+t*(f1-f0)))
            if i==0:capture('first_contact')
        capture('before_lift');m.lift(m.params.lift_wall_fraction);capture('after_lift')
        np.savez_compressed(out/f'spacing_{spacing:.6f}.npz',**{stage+'_'+k:v for stage,s in states.items() for k,v in s.items()})
        saved.append(states);times.append(time.perf_counter()-start)
    rows=[]
    for stage in saved[0]:
        x,y=saved[0][stage],saved[1][stage];diff=x['wall']-y['wall'];loc=np.unravel_index(np.abs(diff).argmax(),diff.shape)
        rows.append(dict(stage=stage,max_wall_diff_mm=float(abs(diff).max()/.005**2*1000),max_cell=list(map(int,loc)),wall_l1_ml=float(abs(diff).sum()*1e6),blade_l1_ml=float(abs(x['blade']-y['blade']).sum()*1e6),bead_l1_ml=float(abs(x['bead']-y['bead']).sum()*1e6),dropped_diff_ml=abs(x['dropped']-y['dropped'])*1e6,ledger_error_ml=max(x['ledger_error_ml'],y['ledger_error_ml'])))
    result=dict(rows=rows,seconds=times,training_started=False)
    (out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()

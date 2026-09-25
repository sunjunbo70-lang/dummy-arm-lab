"""Component isolation, NOT substitute physics or gate. Case13 reproducible inputs."""
import argparse,copy,json,time
from pathlib import Path
import numpy as np
from .split_inventory import SplitPressure
from .ordered_inventory import OrderedInventory
from ..p0 import make_scene,snapshot

def run_flux(recipe):
    base,_=make_scene(13);s=SplitPressure(base.wall*base.wall_cell_area,base.blade*base.blade_cell_area,params=copy.deepcopy(base.p))
    s.contact(recipe['begin'],recipe['angle'][0],recipe['pitch'][0],recipe['force'][0])
    initial_B=s.blade/s.bc**2;initial_Q=s.bead/s.bc**2
    # Prescribed internal flow clock; no geometry pickup/exit/deposit during solve.
    duration=40.;couette=.25;rows=[];states=[]
    for steps in (1000,2000,4000):
        B=initial_B.copy();q=initial_Q.copy();trail=np.zeros_like(q);start=time.perf_counter();h=duration/steps
        for i in range(steps):
            t=(i+.5)/steps;pitch=np.interp(t,[0,1],recipe['pitch']);force=np.interp(t,[0,1],recipe['force'])
            db,dq,dt=s._flux(B,q,pitch,force,couette)
            pb=B+h*db;pq=q+h*dq
            db2,dq2,dt2=s._flux(pb,pq,pitch,force,couette)
            B=.5*B+.5*(pb+h*db2);q=.5*q+.5*(pq+h*dq2);trail+=.5*h*(dt+dt2)
        state=np.r_[B.ravel(),q,trail];states.append(state)
        rows.append(dict(steps=steps,seconds=time.perf_counter()-start,ledger_error_ml=float(abs(state.sum()-initial_B.sum()-initial_Q.sum())*s.bc**2*1e6),min_thickness_m=float(state.min())))
    errors=[float(np.max(abs(a-b))*1000) for a,b in zip(states,states[1:])]
    return dict(rows=rows,max_component_difference_mm=errors,scope='prescribed flux clock only; not actual full trajectory'),states

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);head=snapshot(out)
    r=json.loads((Path(a.source)/'single_013_recipes.json').read_text())['recipes'][0]
    (out/'manifest.json').write_text(json.dumps(dict(source_head=head,case=13,scope='diagnostic isolation, no production physics change',training_started=False),indent=2))
    flux,states=run_flux(r);(out/'flux.json').write_text(json.dumps(flux,indent=2));np.savez_compressed(out/'flux_states.npz',**{f'n{n}':s for n,s in zip((1000,2000,4000),states)});print(json.dumps(dict(flux=flux)),flush=True)
    models=[];rows=[]
    for spacing in (.00005,.000025):
        base,_=make_scene(13);m=OrderedInventory(base.wall*base.wall_cell_area,base.blade*base.blade_cell_area);total=m.total();start=time.perf_counter()
        begin=np.array(r['begin']);end=np.array(r['end']);a0,a1=r['angle'];radius=.5*np.linalg.norm(np.array(m.blade.shape)*m.bc)
        count=int(np.ceil((np.linalg.norm(end-begin)+radius*abs(a1-a0))/spacing))+1
        for t in np.linspace(0,1,count):m.move(begin+t*(end-begin),a0+t*(a1-a0),.002)
        before=m.wall.copy();m.lift(base.p.lift_wall_fraction)
        models.append((before,m.wall.copy(),m.blade.copy()))
        row=dict(spacing_m=spacing,seconds=time.perf_counter()-start,ledger_error_ml=float(abs(m.total()-total)*1e6));rows.append(row)
        np.savez_compressed(out/f'geometry_{spacing:.6f}.npz',before_lift=before,after_lift=m.wall,blade=m.blade)
        with (out/'progress.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    result=dict(flux=flux,geometry=dict(rows=rows,before_lift_max_diff_mm=float(abs(models[0][0]-models[1][0]).max()/.005**2*1000),after_lift_max_diff_mm=float(abs(models[0][1]-models[1][1]).max()/.005**2*1000),blade_l1_diff_ml=float(abs(models[0][2]-models[1][2]).sum()*1e6),scope='constant 2mm exit gap, no constitutive flux/slump/bead; not full physics'),training_started=False,full_gate=False)
    (out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()

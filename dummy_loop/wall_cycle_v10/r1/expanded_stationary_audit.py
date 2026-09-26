"""Expanded v8.4 stationary identity material audit. No robot, transport pose or observation gate claim."""
import argparse,copy,json,os,subprocess,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from .stationary_boundary_pressure import StationaryBoundaryPressure
from ..p0 import make_scene,snapshot
from ...wall_cycle_v09.reward_quality import quality

def run_case(job):
    index,sequence,steps,out=job;out=Path(out)
    name=f'{"sequence" if sequence else "single"}_{index:03d}'
    base,scene=make_scene(index%60);rng=np.random.default_rng(282000+index+1000*sequence)
    models=[StationaryBoundaryPressure(base.wall*base.wall_cell_area,base.blade*base.blade_cell_area,params=copy.deepcopy(base.p)) for _ in range(2)]
    feeders=[copy.deepcopy(base),copy.deepcopy(base)]
    initial=[m.total() for m in models];supplied=[0.,0.]
    spacings=[.00005,.000025];prev=None;logs=[];recipes=[];start_time=time.perf_counter()
    def compare(k,op):
        metrics=[]
        for m in models:
            base.wall=m.wall/m.wc**2;metrics.append(quality(base))
        a,b=models
        row=dict(step=k,op=op,max_wall_diff_mm=float(np.max(abs(a.wall-b.wall))/a.wc**2*1000),rmse_diff_mm=abs(metrics[0]['rmse_mm']-metrics[1]['rmse_mm']),coverage_diff=abs(metrics[0]['coverage']-metrics[1]['coverage']),ledger_error_ml=max(abs(m.total()-v-q)*1e6 for m,v,q in zip(models,initial,supplied)),min_volume_m3=float(min(min(m.wall.min(),m.blade.min(),m.bead.min()) for m in models)))
        logs.append(row)
        with (out/f'{name}_steps.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    for k in range(steps if sequence else 1):
        if sequence and k%20==0:
            for j,m in enumerate(models):
                m.lift(m.params.lift_wall_fraction)
                feed=feeders[j];feed.blade=m.blade/m.bc**2;feed._bead[:]=0.;old_drop=feed.dropped_m3;old_supply=feed.supplied_m3;c=feed.cfg
                feed.feed(12,c.feed_normal_force_N,c.feed_scoop_depth_m,c.feed_scoop_distance_m,c.feed_scoop_speed_m_s)
                m.blade=feed.blade*m.bc**2;m.dropped+=feed.dropped_m3-old_drop;supplied[j]+=feed.supplied_m3-old_supply
            prev=None;recipes.append(dict(step=k,op='load',requested_ml=12));compare(k,'load');continue
        continuation=sequence and k%5 in (2,3) and models[0].pose is not None
        begin=np.array(models[0].pose[0]) if continuation else rng.uniform([-.11,.14],[.11,.36])
        end=rng.uniform([-.11,.14],[.11,.36])
        angle0=float(models[0].pose[1]) if continuation else float(rng.uniform(-np.pi,np.pi))
        angle1=angle0+rng.uniform(-.8,.8)
        if (index+k)%10==0:end=begin.copy() # pure rotation, still integrated by swept radius
        pitch=rng.uniform(0,.5,2);force=rng.uniform(.5,15,2)
        if continuation:pitch[0],force[0]=models[0].last_controls
        recipes.append(dict(step=k,op='continue' if continuation else 'contact',begin=begin.tolist(),end=end.tolist(),angle=[angle0,angle1],pitch=pitch.tolist(),force=force.tolist()))
        for m,spacing in zip(models,spacings):
            if not continuation:m.lift(m.params.lift_wall_fraction)
            radius=.5*np.linalg.norm(np.array(m.blade.shape)*m.bc)
            length=np.linalg.norm(end-begin)+radius*abs(angle1-angle0)
            count=max(2,int(np.ceil(length/spacing))+1)
            for t in np.linspace(0,1,count):m.contact(begin+t*(end-begin),angle0+t*(angle1-angle0),float(pitch[0]+t*(pitch[1]-pitch[0])),float(force[0]+t*(force[1]-force[0])))
            if not sequence or k%5 not in (1,2):m.lift(m.params.lift_wall_fraction)
        compare(k,'continue' if continuation else 'contact')
    for m in models:m.lift(m.params.lift_wall_fraction)
    compare(steps if sequence else 1,'final_lift')
    names=['max_wall_diff_mm','rmse_diff_mm','coverage_diff','ledger_error_ml']
    worst={key:max(r[key] for r in logs) for key in names}
    passed=worst['max_wall_diff_mm']<=(.05 if sequence else .01) and worst['rmse_diff_mm']<=.02 and worst['coverage_diff']<=.005 and worst['ledger_error_ml']<=.01 and min(r['min_volume_m3'] for r in logs)>=-1e-18
    result=dict(index=index,sequence=sequence,steps=steps if sequence else 1,passed=bool(passed),seconds=time.perf_counter()-start_time,**worst)
    (out/f'{name}_recipes.json').write_text(json.dumps(dict(scene=scene,recipes=recipes),indent=2))
    np.savez_compressed(out/f'{name}_final.npz',coarse_wall=models[0].wall,fine_wall=models[1].wall,coarse_blade=models[0].blade,fine_blade=models[1].blade)
    (out/f'{name}.json').write_text(json.dumps(result,indent=2));return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--workers',type=int,default=12);p.add_argument('--singles',type=int,default=100);p.add_argument('--sequences',type=int,default=50);p.add_argument('--steps',type=int,default=100);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);head=snapshot(out)
    (out/'manifest.json').write_text(json.dumps(dict(status='started',pid=os.getpid(),source_head=head,physics_version=StationaryBoundaryPressure.physics_version,singles=a.singles,sequences=a.sequences,steps=a.steps,spacings_m=[.00005,.000025],workers=a.workers,scope='material subsystem; no full environment/robot/transport claim',training_started=False),indent=2))
    jobs=[(i,False,1,str(out)) for i in range(a.singles)]+[(i,True,a.steps,str(out)) for i in range(a.sequences)]
    rows=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        pending={pool.submit(run_case,j):j for j in jobs}
        for f in as_completed(pending):
            try:r=f.result()
            except Exception as e:
                j=pending[f];r=dict(index=j[0],sequence=j[1],passed=False,error=repr(e))
            rows.append(r)
            with (out/'cases.jsonl').open('a') as stream:stream.write(json.dumps(r)+'\n')
            print(json.dumps(r),flush=True)
    (out/'summary.json').write_text(json.dumps(dict(status='completed',cases=rows,passed=sum(r['passed'] for r in rows),total=len(rows),subset_passed=all(r['passed'] for r in rows),required_budget_complete=(a.singles==100 and a.sequences==50 and a.steps==100),material_numerical_gate=(a.singles==100 and a.sequences==50 and a.steps==100 and all(r['passed'] for r in rows)),full_environment_gate=False,training_started=False),indent=2))
if __name__=='__main__':main()

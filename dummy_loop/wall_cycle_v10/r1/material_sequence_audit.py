"""100 single strokes and 50 x 100 material-action sequences; no quality-success gate."""
import argparse,copy,concurrent.futures,json,math,time
from pathlib import Path
import numpy as np
from .compiled_material import integrate_material
from .material import DisplacementMortar
from ..p0 import make_scene,measures,snapshot

def stroke(m,r,spacing,continuation=False):
    start,end,phi0,phi1,pitch,force,speed=r
    n=int(np.ceil(np.linalg.norm(end-start)/spacing))+1;alpha=np.linspace(0.,1.,n)
    points=start+(end-start)*alpha[:,None];phis=phi0+(phi1-phi0)*alpha
    pitches=pitch[0]+(pitch[1]-pitch[0])*alpha;forces=force[0]+(force[1]-force[0])*alpha;speeds=np.full(n,speed)
    if not continuation:m.begin_stroke(phi0,end-start)
    integrate_material(m,points,phis,pitches,forces,speeds)

def make_recipe(rng,prev=None):
    start=rng.uniform([-.11,.14],[.11,.36]) if prev is None else prev
    end=rng.uniform([-.11,.14],[.11,.36]);direction=end-start
    phi=np.arctan2(direction[1],direction[0])-np.pi/2
    return (start,end,phi,phi+rng.uniform(-.15,.15),rng.uniform(0,.5,2),rng.uniform(.5,15.,2),rng.uniform(.02,.12))

def run_case(job):
    i,sequence,out=job;rng=np.random.default_rng(281000+i+1000*sequence)
    m,_=make_scene(i%60);m.__class__=DisplacementMortar;initial=m.wall.copy();a=copy.deepcopy(m);b=copy.deepcopy(m)
    spacing=[.000009765625,.0000048828125];t0=time.perf_counter();log=[];prev=None
    for k in range(100 if sequence else 1):
        if sequence and k%20==0:
            for item in (a,b):item.feed(12,item.cfg.feed_normal_force_N,item.cfg.feed_scoop_depth_m,item.cfg.feed_scoop_distance_m,item.cfg.feed_scoop_speed_m_s)
            op='load'
        else:
            continuation=sequence and k%5 in (2,3) and prev is not None
            recipe=make_recipe(rng,prev if continuation else None)
            for item,s in zip((a,b),spacing):
                stroke(item,recipe,s,continuation)
                if not sequence or k%5 not in (1,2):item.end_stroke()
            prev=recipe[1];op='continue' if continuation else 'contact'
        aa=measures(a,initial);bb=measures(b,initial)
        log.append(dict(step=k,op=op,max_wall_diff_mm=float(np.max(np.abs(a.wall-b.wall))*1000),rmse_diff_mm=abs(aa['rmse_mm']-bb['rmse_mm']),coverage_diff=abs(aa['coverage']-bb['coverage']),ledger_ml=max(abs(aa['ledger_error_ml']),abs(bb['ledger_error_ml']))))
    a.end_stroke();b.end_stroke();aa=measures(a,initial);bb=measures(b,initial)
    maxwall=max([float(np.max(abs(a.wall-b.wall))*1000)]+[r['max_wall_diff_mm'] for r in log]);rmse=max([abs(aa['rmse_mm']-bb['rmse_mm'])]+[r['rmse_diff_mm'] for r in log]);cov=max([abs(aa['coverage']-bb['coverage'])]+[r['coverage_diff'] for r in log]);ledger=max([abs(aa['ledger_error_ml']),abs(bb['ledger_error_ml'])]+[r['ledger_ml'] for r in log])
    result=dict(index=i,sequence=sequence,max_wall_diff_mm=maxwall,rmse_diff_mm=rmse,coverage_diff=cov,ledger_error_ml=ledger,passed=bool(maxwall<=(.05 if sequence else .01) and rmse<=.02 and cov<=.005 and ledger<=.01),seconds=time.perf_counter()-t0,steps=log,scope='material subsystem; no robot/observation claim')
    Path(out,f"{'sequence' if sequence else 'single'}_{i:03d}.json").write_text(json.dumps(result,indent=2));return {k:v for k,v in result.items() if k!='steps'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=12);p.add_argument('--singles',type=int,default=100);p.add_argument('--sequences',type=int,default=50);args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False);snapshot(args.out)
    rows=[];jobs=[(i,False,str(args.out)) for i in range(args.singles)]+[(i,True,str(args.out)) for i in range(args.sequences)]
    (args.out/'manifest.json').write_text(json.dumps(dict(status='running',cases=len(jobs),physics=DisplacementMortar.physics_version,training_started=False),indent=2))
    with concurrent.futures.ProcessPoolExecutor(args.workers) as pool,(args.out/'cases.jsonl').open('w') as f:
        for result in pool.map(run_case,jobs,chunksize=1):
            rows.append(result);f.write(json.dumps(result)+'\n');f.flush();print(f"{len(rows)}/{len(jobs)} passed={result['passed']} maxdiff={result['max_wall_diff_mm']:.5f}mm",flush=True)
    (args.out/'summary.json').write_text(json.dumps(dict(cases=rows,passed=sum(r['passed'] for r in rows),total=len(rows),material_numerical_gate=all(r['passed'] for r in rows),full_environment_gate=False),indent=2))
    (args.out/'completion.json').write_text(json.dumps(dict(status='completed',material_numerical_gate=all(r['passed'] for r in rows),training_started=False)))
if __name__=='__main__':main()

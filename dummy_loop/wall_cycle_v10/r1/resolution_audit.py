"""Resolve sampling convergence without changing physical parameters or quality gates."""
import argparse, concurrent.futures, json, math, time
from pathlib import Path
import numpy as np
from ..p0 import make_scene, measures, snapshot
from ...wall_cycle.mortar import StrokeStats

def case(job):
    row, spacing, out, variant = job
    m,_=make_scene(row['scene']['index']); initial=m.wall.copy()
    if variant in ('stable_index','displacement'):
        from .material import StableIndexMortar,DisplacementMortar
        m.__class__=StableIndexMortar if variant=='stable_index' else DisplacementMortar
    a=np.array(row['start']);b=np.array(row['end']);pitch=math.radians(row['pitch_deg'])
    n=int(math.ceil(np.linalg.norm(b-a)/spacing))+1
    m.begin_stroke(row['phi'],b-a);m.air(pitch);t0=time.perf_counter()
    for t in np.linspace(0,1,n): m.contact(a+(b-a)*t,pitch,.06,force_N=row['force_N'])
    m.end_stroke();m.air(0.)
    path=Path(out)/f"scene{row['scene']['index']:02d}_{spacing:.9f}.npz"
    np.savez_compressed(path,wall=m.wall,blade=m.blade)
    return dict(scene=row['scene']['index'],spacing=spacing,n=n,seconds=time.perf_counter()-t0,metrics=measures(m,initial),path=str(path))

def main():
    p=argparse.ArgumentParser();p.add_argument('--finest',action='store_true');p.add_argument('--variant',choices=['reference','stable_index','displacement'],default='reference');p.add_argument('--out',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--workers',type=int,default=12);p.add_argument('--scenes',type=int,default=12)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False);head=snapshot(args.out)
    selected=json.loads(args.source.read_text())['best']
    indices=np.linspace(0,len(selected)-1,args.scenes,dtype=int)
    rows=[selected[i] for i in indices];spacings=([.0000390625,.00001953125,.000009765625,.0000048828125] if args.finest else [.000625,.0003125,.00015625,.000078125])
    (args.out/'protocol.json').write_text(json.dumps(dict(source=head,scenes=[r['scene']['index'] for r in rows],spacings=spacings,variant=args.variant,physics_changed=args.variant!='reference',training_started=False),indent=2))
    results=[]
    with concurrent.futures.ProcessPoolExecutor(args.workers) as pool,(args.out/'cases.jsonl').open('w') as f:
        for r in pool.map(case,[(r,s,str(args.out),args.variant) for r in rows for s in spacings],chunksize=1):
            results.append(r);f.write(json.dumps(r)+'\n');f.flush();print(f"{len(results)}/{len(rows)*len(spacings)}",flush=True)
    comparisons=[]
    for i in [r['scene']['index'] for r in rows]:
        rr=sorted([r for r in results if r['scene']==i],key=lambda r:r['spacing'],reverse=True)
        for a,b in zip(rr,rr[1:]):
            wa=np.load(a['path'])['wall'];wb=np.load(b['path'])['wall'];ma=a['metrics'];mb=b['metrics']
            diff=float(np.max(np.abs(wa-wb))*1000);jrel=abs(ma['J']-mb['J'])/max(abs(mb['J']),1e-9);rmse=abs(ma['rmse_mm']-mb['rmse_mm'])
            comparisons.append(dict(scene=i,coarse=a['spacing'],fine=b['spacing'],max_wall_diff_mm=diff,J_relative_diff=jrel,rmse_diff_mm=rmse,passed=bool(diff<=.01 and rmse<=.02 and (jrel<=.01 or abs(ma['J']-mb['J'])<=.001))))
    summary=dict(comparisons=comparisons,finest_passed=sum(r['passed'] for r in comparisons if r['fine']==min(spacings)),scenes=len(rows),full_G0=False,training_started=False)
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k!='comparisons'}),flush=True)
if __name__=='__main__':main()

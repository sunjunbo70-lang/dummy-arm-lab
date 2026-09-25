"""Parallel resolution study of frozen v2 candidate; not complete G0."""
import argparse,json,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from .split_inventory import SplitPressure as OrderedPressure

def case(task):
    seed,out=task;out=Path(out);rng=np.random.default_rng(3300+seed)
    wall=rng.uniform(.0005,.004,(100,100))*.005**2
    blade=rng.uniform(0,.004,(6,24))*.005**2
    angle=rng.uniform(-3,3);delta=rng.uniform(-.8,.8)
    arrays={};rows=[]
    for n in (101,201,401):
        s=OrderedPressure(wall,blade);total=s.total();start=time.perf_counter()
        for t in np.linspace(0,1,n):s.contact([.04*t*np.cos(angle),.25+.04*t*np.sin(angle)],angle+t*delta,.15+.1*t,3.+2*t)
        arrays[f'before_{n}']=s.wall.copy()/s.wc**2
        arrays[f'blade_{n}']=s.blade.copy();arrays[f'bead_{n}']=s.bead.copy()
        s.lift(.75);arrays[f'after_{n}']=s.wall/s.wc**2
        rows.append(dict(samples=n,seconds=time.perf_counter()-start,ledger_error_mL=abs(s.total()-total)*1e6,min_volume_m3=float(min(s.wall.min(),s.blade.min()))))
    d1=float(np.max(abs(arrays['after_101']-arrays['after_201']))*1000)
    d2=float(np.max(abs(arrays['after_201']-arrays['after_401']))*1000)
    row=dict(seed=seed,levels=rows,max_diff_101_201_mm=d1,max_diff_201_401_mm=d2,empirical_order=float(np.log2(d1/d2)) if d1>0 and d2>0 else None,local_single_thickness_tolerance_pass=d2<=.01)
    np.savez_compressed(out/f'case_{seed:02d}.npz',**arrays)
    (out/f'case_{seed:02d}.json').write_text(json.dumps(row,indent=2))
    return row

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--workers',type=int,default=6);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    (out/'manifest.json').write_text(json.dumps(dict(physics_version=OrderedPressure.physics_version,workers=a.workers,samples=[101,201,401],seeds=list(range(6)),status='started',scope='isolated six-case convergence; no G0 pass claim'),indent=2))
    start=time.perf_counter();rows=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(case,(seed,str(out))) for seed in range(6)]
        for f in as_completed(futures):
            row=f.result();rows.append(row);print(json.dumps(row),flush=True)
    summary=dict(status='completed',physics_version=OrderedPressure.physics_version,full_G0_passed=False,training_started=False,wall_seconds=time.perf_counter()-start,rows=sorted(rows,key=lambda x:x['seed']))
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
if __name__=='__main__':main()


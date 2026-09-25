"""Isolated inventory convergence diagnostic, NOT full material G0."""
import argparse,json,time
from pathlib import Path
import numpy as np
from .contact_inventory import ContactInventory,PHYSICS_VERSION

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    rows=[]
    for seed in range(12):
        rng=np.random.default_rng(1200+seed)
        wall=rng.uniform(.0005,.004,(100,100))*.005**2
        blade=rng.uniform(0,.004,(6,24))*.005**2
        angle=rng.uniform(-np.pi,np.pi); delta=rng.uniform(-.8,.8)
        start=np.array([0.,.25]);end=start+.06*np.array([np.cos(angle),np.sin(angle)])
        results=[]
        for n in [51,101,201]:
            s=ContactInventory(wall,blade);initial=s.total();t0=time.perf_counter()
            for t in np.linspace(0,1,n):s.move(start+t*(end-start),angle+t*delta,.001)
            s.lift(.75)
            results.append(s.wall/.005**2)
            rows.append(dict(seed=seed,samples=n,seconds=time.perf_counter()-t0,ledger_error_mL=abs(s.total()-initial)*1e6,min_volume_m3=float(min(s.wall.min(),s.blade.min()))))
        np.savez_compressed(out/f'case_{seed:02d}.npz',coarse=results[0],medium=results[1],fine=results[2])
        rows[-1]['coarse_medium_max_mm']=float(np.max(abs(results[0]-results[1]))*1000)
        rows[-1]['medium_fine_max_mm']=float(np.max(abs(results[1]-results[2]))*1000)
        (out/'progress.json').write_text(json.dumps(rows,indent=2))
    summary=dict(physics_version=PHYSICS_VERSION,scope='isolated inventory; fixed exit gap; no pressure/extrusion/gravity; not G0',cases=12,rows=rows,training_started=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({'max_ledger_mL':max(x['ledger_error_mL'] for x in rows),'max_medium_fine_mm':max(x.get('medium_fine_max_mm',0) for x in rows),'improved_cases':sum(x.get('medium_fine_max_mm',1)<x.get('coarse_medium_max_mm',0) for x in rows)},indent=2))
if __name__=='__main__':main()

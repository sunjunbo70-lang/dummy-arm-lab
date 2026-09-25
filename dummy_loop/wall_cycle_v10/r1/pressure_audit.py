"""Coupled candidate diagnostic. Not full G0 or robot G1."""
import argparse,json,time
from pathlib import Path
import numpy as np
from .pressure_inventory import PressureInventory

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False);rows=[]
    for seed in range(6):
        rng=np.random.default_rng(3300+seed)
        wall=rng.uniform(.0005,.004,(100,100))*.005**2
        blade=rng.uniform(0,.004,(6,24))*.005**2
        angle=rng.uniform(-3,3);delta=rng.uniform(-.8,.8)
        results=[]
        for n in (101,201,401):
            s=PressureInventory(wall,blade);initial=s.total();start=time.perf_counter()
            for t in np.linspace(0,1,n):
                s.contact([.04*t*np.cos(angle),.25+.04*t*np.sin(angle)],angle+t*delta,.15+.1*t,3.+2*t)
            s.lift(.75);results.append(s.wall/.005**2)
            rows.append(dict(seed=seed,samples=n,seconds=time.perf_counter()-start,ledger_error_mL=abs(s.total()-initial)*1e6,min_volume_m3=float(min(s.wall.min(),s.blade.min(),s.bead.min()))))
        rows[-1]['coarse_medium_max_mm']=float(np.max(abs(results[0]-results[1]))*1000)
        rows[-1]['medium_fine_max_mm']=float(np.max(abs(results[1]-results[2]))*1000)
        np.savez_compressed(out/f'case_{seed:02d}.npz',coarse=results[0],medium=results[1],fine=results[2])
    summary=dict(physics_version=PressureInventory.physics_version,adopted=False,full_G0_passed=False,scope='six coupled diagnostic paths; not full material/env gate',rows=rows)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(dict(max_ledger_mL=max(r['ledger_error_mL'] for r in rows),max_medium_fine_mm=max(r.get('medium_fine_max_mm',0) for r in rows),improved=sum(r.get('medium_fine_max_mm',1)<r.get('coarse_medium_max_mm',0) for r in rows)),indent=2))
if __name__=='__main__':main()

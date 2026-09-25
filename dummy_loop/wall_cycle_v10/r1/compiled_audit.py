"""Finite, saved reference/compiled checks and warmed throughput measurements."""
import argparse,copy,json,time
from pathlib import Path
import numpy as np
from .material import DisplacementMortar
from .compiled_material import integrate_material
from ..p0 import make_scene,measures,snapshot

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);snapshot(a.out)
    rows=[]
    for i in range(12):
        m,scene=make_scene(i*4);m.__class__=DisplacementMortar
        direction=np.array(scene['direction']);center=np.array(scene['center']);phi=np.arctan2(direction[1],direction[0])-np.pi/2
        m.begin_stroke(phi,direction);m.air(.1);other=copy.deepcopy(m)
        n=1001;points=center+np.linspace(-.06,.06,n)[:,None]*direction
        phis=np.full(n,phi);pitches=np.linspace(.05,.2,n);forces=np.linspace(.5,3.,n);speeds=np.full(n,.06)
        t=time.perf_counter()
        for xy,pi,fo in zip(points,pitches,forces):m.contact(xy,pi,.06,force_N=fo)
        reference=time.perf_counter()-t
        t=time.perf_counter();integrate_material(other,points,phis,pitches,forces,speeds);compiled=time.perf_counter()-t
        m.end_stroke();other.end_stroke()
        diff=max(float(abs(m.wall-other.wall).max()),float(abs(m.blade-other.blade).max()))
        rows.append(dict(scene=i*4,max_difference_mm=diff*1000,ledger_difference_ml=abs(m.dropped_m3-other.dropped_m3)*1e6,reference_s=reference,compiled_s=compiled,warm=i>0,passed=bool(diff<1e-10)))
        print(rows[-1],flush=True)
    (a.out/'summary.json').write_text(json.dumps(dict(cases=rows,passed=all(r['passed'] for r in rows),note='candidate equivalence only; not full G0'),indent=2))
if __name__=='__main__':main()

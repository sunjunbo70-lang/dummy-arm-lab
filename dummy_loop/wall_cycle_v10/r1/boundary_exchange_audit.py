"""Frozen-capacity exchange audit on real case13 geometry; NOT a path simulation."""
import argparse,json,time
from pathlib import Path
import numpy as np
from .boundary_flow import instantaneous
from .boundary_exchange import exchange,EXCHANGE_VERSION
from .overlap import planar_overlap
from ..p0 import snapshot

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',required=True);ap.add_argument('--out',required=True);ap.add_argument('--roundoff-zero',action='store_true')
    args=ap.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=False);head=snapshot(out)
    r=json.loads((Path(args.source)/'single_013_recipes.json').read_text())['recipes'][0]
    begin=np.array(r['begin']);end=np.array(r['end']);a,b=r['angle'];rows=[]
    rng=np.random.default_rng(62002)
    for i in range(32):
        t=(i+.5)/32;center=begin+t*(end-begin);angle=a+t*(b-a)
        m=planar_overlap(center,angle,(6,24),.005,(100,100),.005,np.array([-.25,0.]))
        free=.005**2-np.bincount(m[1],weights=m[2],minlength=10000)
        # Polygon clipping can leave round-off near zero; explicitly reject,
        # rather than injecting arbitrary capacity floors into this audit.
        start=time.perf_counter()
        maps,external=instantaneous(center,angle,end-begin,b-a)
        wall=np.maximum(free,0)*rng.uniform(0,.004,10000)
        blade=.005**2*rng.uniform(0,.004,144)
        row=dict(index=i,negative_free_cells=int((free<0).sum()),minimum_free_area=float(free.min()))
        if args.roundoff_zero:
            if free.min() < -1e-12*.005**2:
                raise ValueError('Area error exceeds roundoff allowance')
            row['roundoff_area_adjustment_m2']=float(-free[free<0].sum())
            free=np.maximum(free,0.) # zero only negative roundoff, no positive floor
        try:
            full=exchange(wall,blade,free,.005**2,maps,external,1/256)
            half=exchange(wall,blade,free,.005**2,maps,external,1/512)
            half=exchange(half[0],half[1],free,.005**2,maps,external,1/512,lost=half[2])
            state=np.r_[full[0],full[1],full[2]]
            row.update(status='passed',ledger_error_ml=float(abs(state.sum()-wall.sum()-blade.sum())*1e6),
                       minimum_volume=float(state.min()),semigroup_max_volume_error=float(abs(state-np.r_[half[0],half[1],half[2]]).max()))
        except (ValueError,FloatingPointError) as exc:row.update(status='rejected',reason=str(exc))
        row['seconds']=time.perf_counter()-start;rows.append(row)
    summary=dict(source_head=head,exchange_version=EXCHANGE_VERSION,scope='32 independent frozen geometry states; not chronological trajectory, pressure or G0',
                 roundoff_zero=args.roundoff_zero,passed=sum(x['status']=='passed' for x in rows),rejected=sum(x['status']=='rejected' for x in rows),rows=rows,training_started=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
if __name__=='__main__':main()

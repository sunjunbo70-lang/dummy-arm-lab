"""Analytic uniform-wall pickup benchmark: endpoint vs continuous swept boundary.
No pressure or extrusion; exit gap zero. Diagnostic, not production physics.
"""
import argparse,json
from pathlib import Path
import numpy as np
from .coupled_midpoint_inventory import CoupledMidpointPressure
from ..p0 import snapshot

def measure(dx,dy,n):
    h=.002;bc=.005;length=6*bc;width=24*bc
    s=CoupledMidpointPressure(np.full((100,100),h*bc**2),np.zeros((6,24)))
    total=s.total();s.move([0.,.25],0.,0.);initial=s.blade.sum()
    for t in np.linspace(0,1,n+1)[1:]:s.move([t*dx,.25+t*dy],0.,0.)
    picked=s.blade.sum()-initial
    exact=h*(width*abs(dx)+length*abs(dy))
    # For dx,dy>0, old\new misses entry-then-exit corner slivers per step.
    predicted=h*abs(dx*dy)/n
    return dict(dx=dx,dy=dy,steps=n,picked_ml=float(picked*1e6),continuous_expected_ml=exact*1e6,deficit_ml=float((exact-picked)*1e6),predicted_endpoint_deficit_ml=predicted*1e6,ledger_error_ml=float(abs(s.total()-total)*1e6))

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False);head=snapshot(out)
    rows=[measure(dx,dy,n) for dx,dy in ((.02,0.),(.02,.02)) for n in (10,20,40,80)]
    result=dict(source_head=head,rows=rows,scope='analytic uniform-wall zero-exit pickup; no production physics change',training_started=False)
    (out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()

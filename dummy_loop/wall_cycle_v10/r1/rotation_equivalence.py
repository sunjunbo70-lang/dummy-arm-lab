"""Audit compiled moving/orienting footprint against explicit Python reference."""
import argparse,copy,json,time
from pathlib import Path
import numpy as np
from .material import DisplacementMortar
from .compiled_material import integrate_material
from .material_sequence_audit import make_recipe
from ..p0 import make_scene,snapshot

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);snapshot(a.out);rows=[]
    for i in range(30):
        rng=np.random.default_rng(281000+i);m,_=make_scene(i%60);m.__class__=DisplacementMortar
        start,end,phi0,phi1,pi,fo,speed=make_recipe(rng);n=301;t=np.linspace(0,1,n)
        pts=start+(end-start)*t[:,None];ph=phi0+(phi1-phi0)*t;pitch=pi[0]+(pi[1]-pi[0])*t;force=fo[0]+(fo[1]-fo[0])*t
        m.begin_stroke(phi0,end-start);b=copy.deepcopy(m)
        for xy,phi,pit,f in zip(pts,ph,pitch,force):
            m.e_x=np.array([np.cos(phi),np.sin(phi)]);m.e_w=np.array([-np.sin(phi),np.cos(phi)])*(-1 if m._flip else 1)
            m.contact(xy,pit,speed,force_N=f)
        integrate_material(b,pts,ph,pitch,force,np.full(n,speed));m.end_stroke();b.end_stroke()
        diff=max(np.max(abs(m.wall-b.wall)),np.max(abs(m.blade-b.blade)));rows.append(dict(index=i,max_diff_mm=float(diff*1000),passed=bool(diff<=1e-10)))
    (a.out/'summary.json').write_text(json.dumps(dict(cases=rows,passed=sum(r['passed'] for r in rows)),indent=2));print('compiled rotated reference passed',sum(r['passed'] for r in rows),'/',len(rows))
if __name__=='__main__':main()

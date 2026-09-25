"""Localize remaining error by orientation and footprint, without accepting new physics."""
import argparse,copy,json
from pathlib import Path
import numpy as np
from .material import DisplacementMortar
from .material_sequence_audit import make_recipe,stroke
from ..p0 import make_scene,snapshot

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);snapshot(a.out);rows=[]
    for index in [0,6,17]:
        m,_=make_scene(index);m.__class__=DisplacementMortar;recipe=make_recipe(np.random.default_rng(281000+index))
        for variant in ['rotating','fixed_angle','aligned_axis']:
            r=list(copy.deepcopy(recipe))
            if variant=='fixed_angle':r[3]=r[2]
            if variant=='aligned_axis':
                length=np.linalg.norm(r[1]-r[0]);r[0]=np.array([0.,.25-length/2]);r[1]=np.array([0.,.25+length/2]);r[2]=r[3]=0.
            pair=[copy.deepcopy(m),copy.deepcopy(m)]
            for model,spacing in zip(pair,[.000009765625,.0000048828125]):stroke(model,r,spacing);model.end_stroke()
            delta=(pair[0].wall-pair[1].wall)*1000;roi=delta[m._score_rows,m._score_cols]
            rows.append(dict(index=index,variant=variant,max_mm=float(abs(delta).max()),roi_max_mm=float(abs(roi).max()),pixels_above_01mm=int((abs(delta)>.01).sum()),absolute_difference_ml=float(abs(delta).sum()/1000*m.wall_cell_area*1e6)))
            np.savez_compressed(a.out/f'{index}_{variant}.npz',coarse=pair[0].wall,fine=pair[1].wall,initial=m.wall)
    (a.out/'summary.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
if __name__=='__main__':main()

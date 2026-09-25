"""Bounded diagnostic of v5 against a previously failed single. Not full G0."""
import argparse,copy,json,time
from pathlib import Path
import numpy as np
from .split_inventory import SplitPressure
from .adaptive_inventory import AdaptivePressure
from ..p0 import make_scene,snapshot

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--source',required=True);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);head=snapshot(out)
    r=json.loads((Path(a.source)/'single_013_recipes.json').read_text())['recipes'][0]
    (out/'manifest.json').write_text(json.dumps(dict(physics_version=AdaptivePressure.physics_version,source_head=head,case=13,budgets_mm=[.01,.005],max_trial_budget_per_run=20000,training_started=False,scope='bounded diagnostic, not gate'),indent=2))
    start=np.array([*r['begin'],r['angle'][0],r['pitch'][0],r['force'][0]])
    end=np.array([*r['end'],r['angle'][1],r['pitch'][1],r['force'][1]])
    results=[];models=[]
    for budget in (.01,.005):
        base,_=make_scene(13);m=AdaptivePressure(SplitPressure(base.wall*base.wall_cell_area,base.blade*base.blade_cell_area,params=copy.deepcopy(base.p)),budget_mm=budget)
        total=m.model.total();tic=time.perf_counter();row=dict(budget_mm=budget,status='running')
        try:
            for i,t in enumerate(np.linspace(0,1,27)):
                m.max_trials=20000-m.stats['trials']
                q=start+t*(end-start);m.contact(q[:2],q[2],q[3],q[4])
                with (out/'progress.jsonl').open('a') as f:f.write(json.dumps(dict(budget_mm=budget,segment=i,stats=m.stats,elapsed_s=time.perf_counter()-tic))+'\n')
            m.model.lift(m.model.params.lift_wall_fraction);models.append(m.model)
            np.savez_compressed(out/f'budget_{budget}.npz',wall=m.model.wall,blade=m.model.blade,bead=m.model.bead)
            row['status']='completed'
        except Exception as e:row.update(status='failed',error=repr(e))
        row.update(seconds=time.perf_counter()-tic,stats=m.stats,ledger_error_ml=float(abs(m.model.total()-total)*1e6))
        results.append(row)
        with (out/'results.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    result=dict(results=results,training_started=False,full_gate=False)
    if len(models)==2:result['max_wall_diff_mm']=float(np.max(abs(models[0].wall-models[1].wall))/.005**2*1000)
    (out/'summary.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__':main()

import argparse,json,time
from pathlib import Path
import numpy as np
from .coupled_boundary_audit import run
from .adaptive_boundary_flow import QuadratureBudgetError
from ..p0 import snapshot

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--source',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False);head=snapshot(out)
 recipe=json.loads((Path(a.source)/'single_013_recipes.json').read_text())['recipes'][0];start=time.perf_counter()
 try:
  row,_,_=run(np.array(recipe['begin']),np.array(recipe['end']),*recipe['angle'],128,True)
  result=dict(status='unexpected_completion',row=row)
 except QuadratureBudgetError as exc:result=dict(status='failure_traced',message=str(exc),context=exc.context)
 result.update(source_head=head,seconds=time.perf_counter()-start,training_started=False)
 (out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()

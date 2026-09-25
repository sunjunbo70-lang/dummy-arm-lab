"""Bounded diagnostic sweep after the initial P0 gate failed; not a new acceptance gate."""
import argparse,concurrent.futures,json,time
from pathlib import Path
import numpy as np
from .p0 import run_case

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 config={'purpose':'diagnosis only; preserve failed fixed-profile gate','scenes':list(range(48)),'loads_ml':[0],'lengths_m':[.06,.12,.20],'pitches_deg':[0,2,4,8,15,25,35],'forces_N':[.5,1,2,4,8,15],'spacing_m':.00125}
 (a.out/'protocol.json').write_text(json.dumps(config,indent=2),encoding='utf8')
 jobs=[(i,0.,L,.00125,pitch,force) for i in range(48) for L in config['lengths_m'] for pitch in config['pitches_deg'] for force in config['forces_N']];best={};t=time.perf_counter();counts={};n=0
 with concurrent.futures.ProcessPoolExecutor(max_workers=8) as pool,(a.out/'cases.jsonl').open('w',encoding='utf8',buffering=1) as f:
  for row in pool.map(run_case,jobs,chunksize=9):
   n+=1;f.write(json.dumps(row)+'\n');i=row['scene']['index'];counts[i]=counts.get(i,0)+int(row['material_gate'])
   if i not in best or (row['material_gate'], -row['after']['J'])>(best[i]['material_gate'],-best[i]['after']['J']):best[i]=row
   if n%504==0:print('sweep',n,'/',len(jobs),'seconds',round(time.perf_counter()-t,1),flush=True)
 summary={'cases':n,'elapsed_s':time.perf_counter()-t,'edge_with_candidate':sum(counts[i]>0 for i in range(24)),'high_with_candidate':sum(counts[i]>0 for i in range(24,48)),'best':list(best.values()),'physical_validation':'pending','training_started':False}
 (a.out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8');print({k:v for k,v in summary.items() if k!='best'},flush=True)
if __name__=='__main__':main()

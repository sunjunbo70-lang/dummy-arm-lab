"""CPU worker throughput comparison on the same recorded action stream."""
import argparse,json,time
from pathlib import Path
import numpy as np
from .workers import Vec

def main():
 p=argparse.ArgumentParser();p.add_argument('--reach',required=True);p.add_argument('--demo',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();z=np.load(a.demo);results=[]
 for n in (8,12,16,23):
  v=Vec(n,9000000,a.reach);t=time.perf_counter();steps=0
  for k in range(32):
   idx=k%len(z['op']);v.step(np.full(n,z['op'][idx]),np.tile(z['params'][idx],(n,1)));steps+=n
  elapsed=time.perf_counter()-t;v.close();r={'workers':n,'decisions':steps,'elapsed_s':elapsed,'decisions_per_s':steps/elapsed};results.append(r);print(r,flush=True)
 a.out.write_text(json.dumps(results,indent=2),encoding='utf8')
if __name__=='__main__':main()

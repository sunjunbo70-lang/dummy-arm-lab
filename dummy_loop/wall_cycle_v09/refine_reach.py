"""Add a measured four-degree kinematic layer; preserve the original table."""
import argparse,json,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from .reach import row
from .provenance import capture

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(exist_ok=False);capture(a.out/'config');z=np.load(a.source);layer=np.zeros(z['ok'].shape[:3]+(1,2),bool);start=time.time()
 with ProcessPoolExecutor(12) as pool:
  for f in as_completed([pool.submit(row,(iv,float(v),z['u'],[4])) for iv,v in enumerate(z['v'])]):
   iv,data=f.result();layer[iv]=data;print(iv,float(data.mean()),flush=True)
 ok=np.concatenate([z['ok'][:,:,:,:1],layer,z['ok'][:,:,:,1:]],axis=3);meta=json.loads(str(z['meta']));meta.update({'source':str(a.source),'added_pitch_degrees':4,'elapsed_s':time.time()-start,'lookup':'nearest grid, screening only; full executor replans'})
 np.savez_compressed(a.out/'reach.npz',ok=ok,u=z['u'],v=z['v'],pitch=np.deg2rad([0,4,8,20,35]),meta=json.dumps(meta))
if __name__=='__main__':main()

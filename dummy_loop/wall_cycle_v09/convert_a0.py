"""Map the exact A1 straight teacher samples to original quadratic A0 controls."""
import argparse,json
from pathlib import Path
import numpy as np
from .config import config

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(exist_ok=False);cfg=config();counts={'paired_load_contact':0,'unpaired_load_excluded':0,'samples':0}
 for f in sorted(a.source.glob('*.npz')):
  archive=np.load(f);z={k:archive[k] for k in archive.files};archive.close();xs=[];ops=[];ps=[];valid=[];i=0
  while i<len(z['op']):
   op=int(z['op'][i]);j=i;newop={1:1,2:3,3:4}.get(op);amount=None
   if op==0:
    if i+1>=len(z['op']) or z['op'][i+1]!=1:counts['unpaired_load_excluded']+=1;i+=1;continue
    j=i+1;newop=0;amount=6+(z['params'][i,0]+1)*9;counts['paired_load_contact']+=1
   x=np.zeros(19,np.float32)
   if newop<3:
    b=z['params'][j];x[:4]=b[:4];x[4:6]=[np.cos(b[8]*np.pi),np.sin(b[8]*np.pi)];x[6]=0;x[7]=b[13];x[8]=b[16];x[10]=b[10];x[11]=b[12]
    if amount is not None:
     index=np.argmin(abs(np.asarray(cfg.load_choices_ml)-amount));x[9]=-1+(index+.5)*2/len(cfg.load_choices_ml)
   xs.append(z['obs'][i]);ops.append(newop);ps.append(x);valid.append(bool(z['valid'][i] and z['valid'][j]));i=j+1
  counts['samples']+=len(xs);np.savez_compressed(a.out/f.name,obs=np.array(xs),op=np.array(ops),params=np.array(ps),mask=np.ones((len(xs),5),bool),valid=np.array(valid),initial_wall=z['initial_wall'],final_wall=z['final_wall'],seed=z['seed'],task=z['task'])
 (a.out/'manifest.json').write_text(json.dumps({'source':str(a.source),'mapping':'A0 five operation families, quadratic path; DEPOSIT pairs existing LOAD+CONTACT; unpaired LOAD has no equivalent and is explicitly excluded','counts':counts},indent=2))
if __name__=='__main__':main()

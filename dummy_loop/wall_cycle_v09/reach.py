"""Parallel full-planner reach samples; preserves intermediate row results."""
import argparse,json,time,os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from .config import config
from .trajectory_action import Contact
_EX=None

def row(job):
 global _EX
 from .rich_arm import RichArm
 iv,v,us,*options=job
 pitches=options[0] if options else [0,8,20,35]
 if _EX is None:_EX=RichArm(config())
 out=np.zeros((len(us),8,len(pitches),2),bool)
 for iu,u in enumerate(us):
  for ip in range(8):
   phi=ip*np.pi/4;normal=np.array([-np.sin(phi),np.cos(phi)])
   for ib,pitch in enumerate(np.deg2rad(pitches)):
    for sense,sign in enumerate([1,-1]):
     _EX.rng=np.random.default_rng(100000+iv*10000+iu*100+ip*8+ib*2+sense);_EX.q_last=_EX.q_scan.copy()
     a=np.array([u,v])-normal*sign*.01;b=np.array([u,v])+normal*sign*.01
     contact=Contact(np.array([a,a+(b-a)/3,a+2*(b-a)/3,b]),np.array([phi,phi]),np.full(3,pitch),np.full(3,6.),np.full(3,.06))
     out[iu,ip,ib,sense]=_EX.plan(contact.legacy()).ok
 return iv,out

class Reach:
 # Nearest-grid screening only: feasibility is not monotonic in pitch.
 # Full MuJoCo execution still checks the entire IK trajectory.
 def __init__(self,path):
  z=np.load(path);self.ok=z['ok'];self.u=z['u'];self.v=z['v'];self.pitch=z['pitch'];self.metadata=json.loads(str(z['meta']))
 def accepts(self,contact):
  for t in np.linspace(0,1,12):
   xy,d,phi,pitch,_,_=contact.at(t)
   if not self.u[0]<=xy[0]<=self.u[-1] or not self.v[0]<=xy[1]<=self.v[-1]:return False
   iu=abs(self.u-xy[0]).argmin();iv=abs(self.v-xy[1]).argmin();ip=int(round((phi%(2*np.pi))/(np.pi/4)))%8;ib=int(abs(self.pitch-pitch).argmin());sense=int(d@np.array([-np.sin(phi),np.cos(phi)])<0)
   if not self.ok[iv,iu,ip,ib,sense]:return False
  return True

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=12);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 us=np.linspace(-.15,.15,16);vs=us+.25;start=time.time();ok=np.zeros((len(vs),len(us),8,len(pitches),2),bool)
 with ProcessPoolExecutor(a.workers) as pool:
  jobs=[pool.submit(row,(iv,float(v),us)) for iv,v in enumerate(vs)]
  for f in as_completed(jobs):
   iv,data=f.result();ok[iv]=data;np.save(a.out/f'row_{iv:02d}.npy',data);print('row',iv,'fraction',data.mean(),'seconds',time.time()-start,flush=True)
 meta={'config':config().to_dict(),'schema':'v09r12.reach.v1','step_m':.02,'seed_rule':'100000+iv*10000+iu*100+ip*8+ib*2+sense','elapsed_s':time.time()-start}
 np.savez_compressed(a.out/'reach.npz',ok=ok,u=us,v=vs,pitch=np.deg2rad([0,8,20,35]),meta=json.dumps(meta))
if __name__=='__main__':main()

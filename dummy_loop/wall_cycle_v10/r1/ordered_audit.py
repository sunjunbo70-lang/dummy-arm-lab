"""Isolated v2 entry-before-exit audit with separate bead/blade lift deposition."""
import argparse,json,time
from pathlib import Path
import numpy as np
from .ordered_inventory import OrderedPressure as PressureInventory

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);report=[]
    for seed in range(6):
        rng=np.random.default_rng(3300+seed)
        wall=rng.uniform(.0005,.004,(100,100))*.005**2
        blade=rng.uniform(0,.004,(6,24))*.005**2
        angle=rng.uniform(-3,3);delta=rng.uniform(-.8,.8)
        pair=[];stored={}
        for n in [201,401]:
            s=PressureInventory(wall,blade);snapshots=[];start=time.perf_counter()
            for i,t in enumerate(np.linspace(0,1,n)):
                info=s.contact([.04*t*np.cos(angle),.25+.04*t*np.sin(angle)],angle+t*delta,.15+.1*t,3.+2*t)
                if i%((n-1)//20)==0:
                    snapshots.append((s.wall.copy()/s.wc**2,s.blade.copy()/s.bc**2,s.bead.copy(),info['gap_m'],s.dropped))
            before=s.wall.copy()/s.wc**2; s._edge_deposit(s.bead,s.pose[0],s.pose[1],1); s.bead[:]=0.; bead_wall=s.wall.copy()/s.wc**2; s.lift(.75);after=s.wall/s.wc**2
            pair.append((snapshots,before,after,after-before,time.perf_counter()-start,bead_wall-before,after-bead_wall))
            stored[f'wall_before_{n}']=before;stored[f'wall_after_{n}']=after
        x,y=pair
        phases=[]
        for k,(u,v) in enumerate(zip(x[0],y[0])):
            phases.append(dict(phase=k/20,wall_max_mm=float(np.max(abs(u[0]-v[0]))*1000),blade_max_mm=float(np.max(abs(u[1]-v[1]))*1000),bead_l1_mL=float(abs(u[2]-v[2]).sum()*1e6),gap_diff_mm=abs(u[3]-v[3])*1000,drop_diff_mL=abs(u[4]-v[4])*1e6))
        row=dict(seed=seed,phases=phases,before_lift_max_mm=float(np.max(abs(x[1]-y[1]))*1000),lift_increment_diff_max_mm=float(np.max(abs(x[3]-y[3]))*1000),after_lift_max_mm=float(np.max(abs(x[2]-y[2]))*1000),seconds=[x[4],y[4]])
        row["bead_deposit_diff_max_mm"]=float(np.max(abs(x[5]-y[5]))*1000); row["blade_deposit_diff_max_mm"]=float(np.max(abs(x[6]-y[6]))*1000); report.append(row);np.savez_compressed(out/f'case_{seed:02d}.npz',**stored)
    (out/'summary.json').write_text(json.dumps(dict(physics_version=PressureInventory.physics_version,full_G0_passed=False,rows=report),indent=2))
    print(json.dumps([{k:v for k,v in r.items() if k!='phases'} for r in report],indent=2))
if __name__=='__main__':main()


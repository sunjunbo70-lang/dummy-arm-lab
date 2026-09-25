"""Validate diagnostic candidates on native MuJoCo; never connects hardware."""
import argparse,json,math,time,concurrent.futures
from pathlib import Path
import numpy as np
from .p0 import make_scene,measures
from ..wall_cycle_v09.rich_arm import RichArm
from ..wall_cycle_v09.trajectory_action import Contact
from ..wall_cycle.mortar import StrokeStats

def execute(job):
 row,out,record=job;index=row['scene']['index'];m,_=make_scene(index);c=m.cfg;c.stroke_samples=row['contact_samples'];initial=m.wall.copy();ex=RichArm(c,trace_enabled=record);ex.reset()
 p0=np.array(row['start']);p3=np.array(row['end']);d=p3-p0;phi=row['phi'];pitch=math.radians(row['pitch_deg']);a=Contact(np.array([p0,p0+d/3,p0+2*d/3,p3]),np.full(2,phi),np.full(3,pitch),np.full(3,row['force_N']),np.full(3,.06)).legacy()
 result={'scene_index':index,'length_m':row['length_m'],'pitch_deg':row['pitch_deg'],'force_N':row['force_N'],'nominal_material_gate':row['material_gate'],'before':measures(m,initial),'evidence':'MuJoCo simulation, no hardware'}
 try:
  plan=ex.plan(a);result['plan_ok']=bool(plan.ok);result['rejection']=str(getattr(plan,'reason',''))
  if plan.ok:
   stats=ex.execute(plan,a,m,StrokeStats());result['after']=measures(m,initial)
   result.update({k:stats.get(k) for k in ['tracking_max_mm','force_rmse_N','face_down_frames','peak_force_N','time_s','ik_not_converged']})
   reduction=(result['before']['excess_ml']-result['after']['excess_ml'])/max(result['before']['excess_ml'],1e-12)
   result['excess_reduction_fraction']=reduction;result['material_gate']=bool(reduction>=.2 and result['after']['damage_fraction_initial_good']<=.02)
   if record and ex.dense_trace:
    dest=Path(out)/f"scene{index:02d}_L{round(row['length_m']*100):02d}";dest.mkdir(parents=True,exist_ok=False);frames=ex.dense_trace
    np.savez_compressed(dest/'trajectory.npz',time_s=np.array([x['time_s'] for x in frames]),q=np.array([x['q'] for x in frames]),phase=np.array([x['phase'] for x in frames]),wall=np.array([x['wall'] for x in frames]),blade=np.array([x['blade'] for x in frames]))
    (dest/'manifest.json').write_text(json.dumps({'config':c.to_dict(),'kwargs':{'buffer':.03},'result':result},indent=2),encoding='utf8');result['replay']=str(dest/'trajectory.npz')
 except Exception as e:result['error']=repr(e)
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--selection-file',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 # Before physical results: ten scene IDs x all three lengths, best nominal profile per cell.
 best={}
 for line in (a.source/'cases.jsonl').read_text().splitlines():
  r=json.loads(line);i=r['scene']['index']
  if i>=10:continue
  k=(i,r['length_m'])
  if k not in best or (r['material_gate'],-r['after']['J'])>(best[k]['material_gate'],-best[k]['after']['J']):best[k]=r
 if a.selection_file:
  explicit=json.loads(a.selection_file.read_text());best={(r['scene']['index'],r['length_m']):r for r in explicit}
 (a.out/'selected.json').write_text(json.dumps(list(best.values()),indent=2),encoding='utf8')
 rows=[];t=time.perf_counter()
 with concurrent.futures.ProcessPoolExecutor(max_workers=3) as pool,(a.out/'cases.jsonl').open('w',encoding='utf8',buffering=1) as f:
  for r in pool.map(execute,[(r,str(a.out/'replay'),r['scene']['index'] in [0,1]) for r in best.values()]):
   rows.append(r);f.write(json.dumps(r)+'\n');print('native',len(rows),'/',len(best),'plan',r.get('plan_ok'),'gate',r.get('material_gate'),'seconds',round(time.perf_counter()-t,1),flush=True)
 summary={'cases':len(rows),'plan_ok':sum(r.get('plan_ok',False) for r in rows),'material_gate':sum(r.get('material_gate',False) for r in rows),'errors':sum('error'in r for r in rows),'elapsed_s':time.perf_counter()-t,'note':'diagnostic subset, not independent policy evaluation','training_started':False}
 (a.out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8');print(summary,flush=True)
if __name__=='__main__':main()


"""Diagnostic interventions; altered adhesion/gap are NOT calibrated training physics."""
import argparse,json,math,time,concurrent.futures
from pathlib import Path
import numpy as np
from .p0 import make_scene,measures,run_case
from ..wall_cycle.mortar import StrokeStats

def branch(job):
 row,spacing,intervention=job;i=row['scene']['index'];m,_=make_scene(i);initial=m.wall.copy();before=measures(m,initial)
 a=np.array(row['start']);b=np.array(row['end']);n=int(math.ceil(row['length_m']/spacing))+1;pitch=math.radians(row['pitch_deg']);stats=StrokeStats();m.begin_stroke(row['phi'],b-a);m.air(pitch,stats)
 for t in np.linspace(0,1,n):
  kwargs={'gap_m':.002} if intervention=='oracle_gap_2mm' else {'force_N':row['force_N']}
  m.contact(a+(b-a)*t,pitch,.06,stats=stats,**kwargs)
 if intervention=='lift_share_1_diagnostic':m.p.lift_wall_fraction=1.
 m.end_stroke(stats);m.air(0.,stats);after=measures(m,initial)
 return {'scene_index':i,'spacing_m':spacing,'intervention':intervention,'before':before,'after':after,'J_delta':after['J']-before['J'],'warning':'interventions are causal diagnostics only, not adopted physical parameters'}

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 selected=json.loads((a.source/'summary.json').read_text())['best'];jobs=[(r,step,mode) for r in selected for step in [.0025,.00125,.000625] for mode in ['reference','lift_share_1_diagnostic','oracle_gap_2mm']]
 rows=[]
 with concurrent.futures.ProcessPoolExecutor(max_workers=8) as pool,(a.out/'cases.jsonl').open('w',encoding='utf8') as f:
  for r in pool.map(branch,jobs,chunksize=3):rows.append(r);f.write(json.dumps(r)+'\n')
 stats=[]
 for mode in ['reference','lift_share_1_diagnostic','oracle_gap_2mm']:
  rr=[r for r in rows if r['intervention']==mode and r['spacing_m']==.00125 and r['scene_index']<24]
  stats.append({'mode':mode,'edge_mean_delta_J':float(np.mean([r['J_delta'] for r in rr])),'edge_mean_damage':float(np.mean([r['after']['damage_fraction_initial_good'] for r in rr]))})
 convergence=[]
 for i in range(48):
  rr=sorted([r for r in rows if r['intervention']=='reference' and r['scene_index']==i],key=lambda x:x['spacing_m']);fine,coarse=rr[0],rr[1]
  convergence.append({'scene':i,'J_relative_diff':abs(fine['after']['J']-coarse['after']['J'])/max(abs(fine['after']['J']),1e-9),'rmse_diff_mm':abs(fine['after']['rmse_mm']-coarse['after']['rmse_mm'])})
 summary={'cases':len(rows),'intervention_summary':stats,'convergence_1_25_to_0_625mm':convergence,'converged_cases':int(sum(r['J_relative_diff']<.01 and r['rmse_diff_mm']<.02 for r in convergence)),'total_convergence_cases':48,'training_physics_changed':False}
 (a.out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8');print({k:v for k,v in summary.items() if k!='convergence_1_25_to_0_625mm'},flush=True)
if __name__=='__main__':main()


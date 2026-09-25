"""Check whether edge-centered paths, rather than model physics alone, cause damage."""
import argparse,json,math,concurrent.futures,time
from pathlib import Path
import numpy as np
from .p0 import make_scene,measures
from ..wall_cycle.mortar import StrokeStats

def run(job):
 row,offset,reverse=job;i=row['scene']['index'];m,scene=make_scene(i);initial=m.wall.copy();side=i%4;center=np.array(scene['center']);axis=0 if side<2 else 1;center[axis]+=(1 if side%2 else -1)*offset
 direction=np.array(scene['direction'])*reverse;length=row['length_m'];start=center-direction*length/2;end=center+direction*length/2;phi=float(math.atan2(direction[1],direction[0])-math.pi/2);pitch=math.radians(row['pitch_deg']);n=int(math.ceil(length/.00125))+1;before=measures(m,initial);m.begin_stroke(phi,direction);m.air(pitch)
 for t in np.linspace(0,1,n):m.contact(start+(end-start)*t,pitch,.06,force_N=row['force_N'])
 m.end_stroke();m.air(0.);after=measures(m,initial);reduction=(before['excess_ml']-after['excess_ml'])/max(before['excess_ml'],1e-12)
 return {'scene_index':i,'scene':scene,'offset_m':offset,'reverse':reverse,'length_m':length,'pitch_deg':row['pitch_deg'],'force_N':row['force_N'],'phi':phi,'start':start.tolist(),'end':end.tolist(),'contact_samples':n,'before':before,'after':after,'excess_reduction_fraction':reduction,'material_gate':bool(reduction>=.2 and after['damage_fraction_initial_good']<=.02),'inside_buffer':bool(np.all(abs(np.array([start,end])-np.array([0,.25]))<=.13000001))}

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);cells={}
 for line in (a.source/'cases.jsonl').read_text().splitlines():
  r=json.loads(line);i=r['scene']['index']
  if i<24:cells.setdefault((i,r['length_m']),[]).append(r)
 selected=[r for rows in cells.values() for r in sorted(rows,key=lambda x:x['after']['J'])[:3]]
 jobs=[(r,d,rev) for r in selected for d in [0.,.02,.035] for rev in [1,-1]]
 (a.out/'protocol.json').write_text(json.dumps({'diagnostic_only':True,'selected_by':'three lowest nominal J profiles per edge/length','offsets_m':[0,.02,.035],'directions':[1,-1],'cases':len(jobs)},indent=2),encoding='utf8');best={};rows=[];t=time.perf_counter()
 with concurrent.futures.ProcessPoolExecutor(max_workers=8) as pool,(a.out/'cases.jsonl').open('w',encoding='utf8',buffering=1) as f:
  for r in pool.map(run,jobs,chunksize=6):
   rows.append(r);f.write(json.dumps(r)+'\n');i=r['scene_index'];score=(r['material_gate'] and r['inside_buffer'],-r['after']['J'])
   if i not in best or score>(best[i]['material_gate'] and best[i]['inside_buffer'],-best[i]['after']['J']):best[i]=r
   if len(rows)%216==0:print('offset',len(rows),'/',len(jobs),flush=True)
 s={'cases':len(rows),'scenes_with_candidate':sum(r['material_gate'] and r['inside_buffer'] for r in best.values()),'scenes':24,'best':list(best.values()),'elapsed_s':time.perf_counter()-t,'note':'expanded diagnostic, not a replacement of the failed frozen P0 gate'};(a.out/'summary.json').write_text(json.dumps(s,indent=2),encoding='utf8');print({k:v for k,v in s.items() if k!='best'},flush=True)
if __name__=='__main__':main()

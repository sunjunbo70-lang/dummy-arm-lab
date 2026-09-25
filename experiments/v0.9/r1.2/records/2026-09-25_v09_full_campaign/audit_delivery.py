import json, hashlib
from pathlib import Path
import numpy as np
B=Path(r"D:\VLA\dummy_arm\experiments\v0.9\r1.2\runs\v09_r12\campaign_005_a0_log_recovery")
result={'evaluations':{},'records':{}}
for d in sorted((B/'evaluation').iterdir()):
 if not (d/'summary.json').exists():continue
 s=json.loads((d/'summary.json').read_text());rows=[json.loads(l) for l in (d/'episodes.jsonl').read_text().splitlines()]
 result['evaluations'][d.name]={'complete':s['complete'],'count':len(rows),'unique_seeds':len({r['seed'] for r in rows}),'max_abs_volume_error_m3':max(abs(r['final']['volume_error_m3']) for r in rows),'max_tracking_mm':max(r.get('max_tracking_mm',0) for r in rows),'face_down_frames':sum(r.get('face_down_frames',0) for r in rows)}
for p in sorted((B/'records').glob('*/*/trajectory.npz')):
 with np.load(p) as z:
  t=z['time_s'];q=z['q'];dt=np.diff(t);dq=np.abs(np.diff(q,axis=0))
  result['records'][p.parent.parent.name]={'frames':len(t),'time_s':float(t[-1]),'nondecreasing_time':bool(np.all(dt>=0)),'finite':bool(all(np.isfinite(z[k]).all() for k in ['q','qvel','wall','blade','time_s'])),'max_sample_joint_delta_deg':float(np.rad2deg(dq).max()),'max_zero_dt_joint_delta_deg':float(np.rad2deg(dq[dt==0]).max()) if np.any(dt==0) else 0,'max_dt_s':float(dt.max()),'phases':np.unique(z['phase']).tolist()}
freeze=json.loads((B/'frozen_before_test.json').read_text());result['checkpoint_hashes_match']=all(hashlib.sha256(Path(p['checkpoint']).read_bytes()).hexdigest()==freeze['checkpoints'][k] for k,p in freeze['selected'].items())
(B/'delivery_audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
# Native MuJoCo rendered final states, same camera and frame as replay.
import mujoco
from PIL import Image,ImageDraw
from dummy_loop.wall_cycle.view import Stage
from dummy_loop.wall_cycle.config import CycleConfig
from dummy_loop.wall_cycle_v09.replay import boundaries
panels=[]
for name in ['T','BC1','RL1_seed33']:
 p=next((B/'records'/name).glob('*/trajectory.npz'));m=json.loads((p.parent/'manifest.json').read_text());st=Stage(CycleConfig(**m['config']))
 with np.load(p) as z:
  st.pose(z['q'][-1]);cam=mujoco.MjvCamera();st.camera(cam)
  with mujoco.Renderer(st.model,height=480,width=640,max_geom=20000) as renderer:
   renderer.update_scene(st.data,camera=cam);st.draw(renderer.scene,z['wall'][-1]*1000);st.draw_blade(renderer.scene,z['blade'][-1]*1000,0);boundaries(st,renderer.scene,.03)
   im=Image.fromarray(renderer.render());ImageDraw.Draw(im).text((10,10),name+' final / seed60090',fill='white');panels.append(im)
canvas=Image.new('RGB',(1920,480))
for i,im in enumerate(panels):canvas.paste(im,(640*i,0))
canvas.save(B/'native_replay_preview.png')

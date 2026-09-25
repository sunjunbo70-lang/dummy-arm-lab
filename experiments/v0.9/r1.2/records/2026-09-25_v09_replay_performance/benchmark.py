import time,json
from pathlib import Path
import numpy as np,mujoco
from dummy_loop.wall_cycle_v09.replay import load_record,sample_pose,boundaries
from dummy_loop.wall_cycle.view import Stage
from dummy_loop.wall_cycle.config import CycleConfig
B=Path(r"D:\VLA\dummy_arm\experiments\v0.9\r1.2\runs\v09_r12\campaign_005_a0_log_recovery\records")
results={}
for p in sorted(B.glob('*/*/trajectory.npz')):
 t=time.perf_counter()
 with np.load(p) as old:wall=old['wall'][-1]
 old_s=time.perf_counter()-t
 t=time.perf_counter();z=load_record(p);load_s=time.perf_counter()-t
 st=Stage(CycleConfig(**json.loads((p.parent/'manifest.json').read_text())['config']));scn=mujoco.MjvScene(st.model,maxgeom=20000)
 samples=[]
 for at in np.linspace(0,z['time_s'][-1],30):
  t=time.perf_counter();i,q=sample_pose(z,at);st.pose(q);scn.ngeom=0;st.draw(scn,z['wall'][i]*1000);st.draw_blade(scn,z['blade'][i]*1000,0);boundaries(st,scn,.03);samples.append(time.perf_counter()-t)
 results[p.parent.parent.name]={'old_wall_read_ms':old_s*1000,'startup_load_s':load_s,'memory_MB':sum(a.nbytes for a in z.values())/1e6,'scene_update_median_ms':float(np.median(samples)*1000),'scene_update_p95_ms':float(np.quantile(samples,.95)*1000),'record_median_interval_ms':float(np.median(np.diff(z['time_s']))*1000)}
 print(p.parent.parent.name,results[p.parent.parent.name],flush=True)
 del z,st,scn
Path(__file__).with_name('benchmark.json').write_text(json.dumps(results,indent=2))

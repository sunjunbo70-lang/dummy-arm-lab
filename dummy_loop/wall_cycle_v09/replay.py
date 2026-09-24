"""Native MuJoCo replay using measured simulation timestamps (L1 only)."""
import argparse,json,time
from pathlib import Path
import numpy as np
import mujoco
import mujoco.viewer
from ..wall_cycle.view import Stage
from ..wall_cycle.config import CycleConfig

def main():
 p=argparse.ArgumentParser();p.add_argument('trajectory',type=Path);a=p.parse_args();z=np.load(a.trajectory);manifest=json.loads((a.trajectory.parent/'manifest.json').read_text(encoding='utf8'));st=Stage(CycleConfig(**manifest['config']));times=z['time_s'];state={'paused':False,'t':float(times[0])}
 def key(k):
  if k==32:state['paused']=not state['paused']
  if k==82:state['t']=float(times[0])
 with mujoco.viewer.launch_passive(st.model,st.data,key_callback=key,show_left_ui=False,show_right_ui=False) as v:
  st.camera(v.cam);last=time.monotonic()
  while v.is_running():
   now=time.monotonic()
   if not state['paused']:state['t']=min(state['t']+now-last,float(times[-1]))
   last=now;i=int(np.clip(np.searchsorted(times,state['t'],side='right')-1,0,len(times)-1))
   with v.lock():
    st.pose(z['q'][i]);v.user_scn.ngeom=0;st.draw(v.user_scn,z['wall'][i]*1000);st.draw_blade(v.user_scn,z['blade'][i]*1000,0)
   v.sync();time.sleep(.01)
if __name__=='__main__':main()

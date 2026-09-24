"""Native MuJoCo timestamp replay (recorded execution, not policy inference)."""
import argparse,json,time
from pathlib import Path
import numpy as np
import mujoco
import mujoco.viewer
from ..wall_cycle.view import Stage
from ..wall_cycle.config import CycleConfig

def boundaries(st,scn,buffer):
 fr=st.ex.frame
 for half,col in [(.1,[1.,.8,.15,1.]),(.1+buffer,[.2,.9,.5,1.])]:
  corners=[(-half,.25-half),(half,.25-half),(half,.25+half),(-half,.25+half)]
  for k in range(4):
   if scn.ngeom>=scn.maxgeom:return
   g=scn.geoms[scn.ngeom];mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_CAPSULE,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array(col,np.float32));mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_CAPSULE,.001,fr.to_world(st.ex._uvn(*corners[k],-.0015)),fr.to_world(st.ex._uvn(*corners[(k+1)%4],-.0015)));scn.ngeom+=1

def main():
 p=argparse.ArgumentParser();p.add_argument('trajectory',type=Path);a=p.parse_args();z=np.load(a.trajectory);manifest=json.loads((a.trajectory.parent/'manifest.json').read_text(encoding='utf8'));st=Stage(CycleConfig(**manifest['config']));times=z['time_s'];state={'paused':False,'t':float(times[0]),'speed':1.}
 def key(k):
  if k==32:state['paused']=not state['paused']
  if k==82:state['t']=float(times[0])
  if k==91:state['speed']=max(.125,state['speed']/2)
  if k==93:state['speed']=min(8,state['speed']*2)
  if k in (262,263):
   i=int(np.searchsorted(times,state['t']));state['paused']=True;state['t']=float(times[np.clip(i+(1 if k==262 else -1),0,len(times)-1)])
 with mujoco.viewer.launch_passive(st.model,st.data,key_callback=key,show_left_ui=False,show_right_ui=False) as v:
  st.camera(v.cam);last=time.monotonic()
  while v.is_running():
   now=time.monotonic()
   if not state['paused']:state['t']=min(state['t']+(now-last)*state['speed'],float(times[-1]))
   last=now;i=int(np.clip(np.searchsorted(times,state['t'],side='right')-1,0,len(times)-1))
   with v.lock():
    st.pose(z['q'][i]);v.user_scn.ngeom=0;st.draw(v.user_scn,z['wall'][i]*1000);st.draw_blade(v.user_scn,z['blade'][i]*1000,0);boundaries(st,v.user_scn,manifest.get('kwargs',{}).get('buffer',.03))
   v.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_150,mujoco.mjtGridPos.mjGRID_TOPLEFT,'Replay / L1\nPhase\nSimulation time\nSpeed\nDomains',f'Recorded MuJoCo execution\n{z["phase"][i]}\n{times[i]:.2f} / {times[-1]:.2f} s\n{state["speed"]}x\nS yellow / B green / D outer'),(mujoco.mjtFontScale.mjFONTSCALE_100,mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,'Space pause | arrows step | [ ] speed | R restart','Material is a 2.5D model; no hardware connection')]);v.sync();time.sleep(.01)
if __name__=='__main__':main()

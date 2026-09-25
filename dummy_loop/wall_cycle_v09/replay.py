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

def load_record(path):
 # NpzFile is lazy and does NOT cache arrays: z['wall'][i] re-inflates all frames.
 # Load each member exactly once before entering the playback clock.
 with np.load(path, allow_pickle=False) as archive:
  frames={key:archive[key] for key in ('time_s','q','wall','blade','phase')}
 t=frames['time_s']
 if len(t)==0 or not np.isfinite(t).all() or np.any(np.diff(t)<0):
  raise ValueError('Replay requires finite, nondecreasing timestamps')
 if any(len(v)!=len(t) for v in frames.values()):raise ValueError('Mismatched frame counts')
 return frames

def sample_pose(frames,t,interpolate=True):
 times=frames['time_s'];i=int(np.clip(np.searchsorted(times,t,side='right')-1,0,len(times)-1))
 q=frames['q'][i]
 if interpolate and i+1<len(times) and times[i+1]>times[i]:
  alpha=float(np.clip((t-times[i])/(times[i+1]-times[i]),0,1))
  q=q+(frames['q'][i+1]-q)*alpha
 return i,q

def main():
 p=argparse.ArgumentParser();p.add_argument('trajectory',type=Path);p.add_argument('--raw-samples',action='store_true',help='Disable display-only joint interpolation');a=p.parse_args()
 print('Loading replay arrays once; long recordings may need several GB of RAM...',flush=True)
 z=load_record(a.trajectory);manifest=json.loads((a.trajectory.parent/'manifest.json').read_text(encoding='utf8'));st=Stage(CycleConfig(**manifest['config']));times=z['time_s'];state={'paused':False,'t':float(times[0]),'speed':1.}
 print(f"Loaded {len(times)} samples / {sum(v.nbytes for v in z.values())/1e6:.0f} MB. No per-frame archive decompression.",flush=True)
 def key(k):
  if k==32:state['paused']=not state['paused']
  if k==82:state['t']=float(times[0])
  if k==91:state['speed']=max(.125,state['speed']/2)
  if k==93:state['speed']=min(8,state['speed']*2)
  if k in (262,263):
   i=int(np.searchsorted(times,state['t'],side='right')-1);state['paused']=True;state['t']=float(times[np.clip(i+(1 if k==262 else -1),0,len(times)-1)])
 with mujoco.viewer.launch_passive(st.model,st.data,key_callback=key,show_left_ui=False,show_right_ui=False) as v:
  st.camera(v.cam);last=time.monotonic();fps=0.
  while v.is_running():
   now=time.monotonic();elapsed=now-last
   if not state['paused']:state['t']=min(state['t']+elapsed*state['speed'],float(times[-1]))
   last=now;i,q=sample_pose(z,state['t'],not a.raw_samples)
   with v.lock():
    st.pose(q);v.user_scn.ngeom=0;st.draw(v.user_scn,z['wall'][i]*1000);st.draw_blade(v.user_scn,z['blade'][i]*1000,0);boundaries(st,v.user_scn,manifest.get('kwargs',{}).get('buffer',.03))
   fps=.9*fps+.1/max(elapsed,1e-6)
   v.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_150,mujoco.mjtGridPos.mjGRID_TOPLEFT,'Replay / L1\nPhase\nSimulation time\nSpeed / updates\nPose display',f"Recorded MuJoCo execution\n{z['phase'][i]}\n{state['t']:.2f} / {times[-1]:.2f} s\n{state['speed']}x / {fps:.1f} Hz\n{'raw samples' if a.raw_samples else 'linear interpolation (display only)'}"),(mujoco.mjtFontScale.mjFONTSCALE_100,mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,'Space pause | arrows step | [ ] speed | R restart','Material is recorded 2.5D state; no hardware connection')]);v.sync()
   time.sleep(max(0,1/60-(time.monotonic()-now)))
if __name__=='__main__':main()

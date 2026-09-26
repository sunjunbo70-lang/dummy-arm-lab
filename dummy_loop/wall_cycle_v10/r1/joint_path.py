"""Pure offline joint-path preflight. No hardware or dynamic execution.
Callbacks: solve(position,rotation,seed)->q or None; fk(q)->(position,rotation);
is_clear(q)->bool. Sampled collision checks do not prove continuous clearance.
"""
from dataclasses import dataclass
import numpy as np
@dataclass(frozen=True)
class JointPath:
 q: np.ndarray
 time_s: np.ndarray
 position_error_m: np.ndarray
 rotation_error_rad: np.ndarray
 collision_samples: int

class PathRejected(RuntimeError):pass

def plan_joint_path(targets,time_s,initial_q,*,solve,fk,is_clear,lo,hi,max_speed,max_jump,collision_step,position_tolerance,rotation_tolerance):
 q0=np.asarray(initial_q,float);lo=np.asarray(lo,float);hi=np.asarray(hi,float);speed=np.asarray(max_speed,float);times=np.asarray(time_s,float)
 if q0.ndim!=1 or any(x.shape!=q0.shape for x in (lo,hi,speed)) or not np.isfinite(np.r_[q0,lo,hi,speed]).all() or (hi<=lo).any() or (speed<=0).any():raise ValueError('Invalid joint limits')
 if not targets or times.shape!=(len(targets),) or not np.isfinite(times).all() or times[0]<0 or (np.diff(times)<=0).any():raise ValueError('Invalid target times')
 if not np.isfinite([max_jump,collision_step,position_tolerance,rotation_tolerance]).all() or min(max_jump,collision_step,position_tolerance,rotation_tolerance)<=0:raise ValueError('Invalid path tolerances')
 def rotation(R):
  R=np.asarray(R,float)
  if R.shape!=(3,3) or not np.isfinite(R).all() or not np.allclose(R.T@R,np.eye(3),atol=1e-7,rtol=0) or abs(np.linalg.det(R)-1)>1e-7:raise ValueError('Invalid rotation')
  return R
 if (q0<lo).any() or (q0>hi).any() or not is_clear(q0.copy()):raise PathRejected('Initial pose invalid or collision')
 path=[];pe=[];re=[];checks=1;previous=q0.copy();previous_time=0.
 for i,(p,R) in enumerate(targets):
  p=np.asarray(p,float);R=rotation(R)
  if p.shape!=(3,) or not np.isfinite(p).all():raise ValueError('Invalid target position')
  q=solve(p.copy(),R.copy(),previous.copy())
  if q is None:raise PathRejected(f'IK failure at {i}')
  q=np.asarray(q,float)
  if q.shape!=q0.shape or not np.isfinite(q).all() or (q<lo).any() or (q>hi).any():raise PathRejected(f'Invalid IK solution at {i}')
  delta=abs(q-previous);dt=times[i]-previous_time
  if delta.max()>max_jump:raise PathRejected(f'Joint jump at {i}')
  if (delta>speed*dt+1e-12).any():raise PathRejected(f'Joint speed limit at {i}')
  actual_p,actual_R=fk(q.copy());actual_p=np.asarray(actual_p,float);actual_R=rotation(actual_R)
  if actual_p.shape!=(3,) or not np.isfinite(actual_p).all():raise PathRejected(f'Invalid FK at {i}')
  ep=float(np.linalg.norm(actual_p-p));er=float(np.arccos(np.clip((np.trace(R.T@actual_R)-1)/2,-1,1)))
  if ep>position_tolerance or er>rotation_tolerance:raise PathRejected(f'Pose residual at {i}: {ep}, {er}')
  n=max(1,int(np.ceil(delta.max()/collision_step)))
  for t in np.linspace(0.,1.,n+1)[1:]:
   checks+=1
   if not is_clear(previous+t*(q-previous)):raise PathRejected(f'Interpolated collision at {i}')
  path.append(q.copy());pe.append(ep);re.append(er);previous=q.copy();previous_time=times[i]
 return JointPath(np.array(path),times.copy(),np.array(pe),np.array(re),checks)

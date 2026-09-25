"""Explicit finite-horizon reward and time-aware GAE, independent of simulator."""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Costs:
    new_material_per_18ml: float = .2
    loading: float = .05
    second: float = .001
    irreversible_per_18ml: float = 1.
    invalid_request: float = 2.

def potential(quality, damaged):
    return -30.*quality-10.*damaged

def transition_reward(before, after, *, duration_s, supplied_ml=0., loss_ml=0.,
                      actual_load=False, invalid=False, terminal=False,
                      success=False, initially_complete=False, costs=Costs()):
    """before/after=(J,P). A rollout boundary must NEVER set terminal=True."""
    values=np.array([*before,*after,duration_s,supplied_ml,loss_ml],float)
    if not np.isfinite(values).all() or min(duration_s,supplied_ml,loss_ml)<0:
        raise ValueError('Nonfinite state or negative incremental cost')
    if actual_load and supplied_ml<=0: raise ValueError('Count only actual positive material transfer')
    cost=dict(new_material=costs.new_material_per_18ml*supplied_ml/18.,
              load_count=costs.loading*bool(actual_load),time=costs.second*duration_s,
              irreversible=costs.irreversible_per_18ml*loss_ml/18.,
              invalid=costs.invalid_request*bool(invalid))
    shaping=(0. if terminal else potential(*after))-potential(*before)
    final=(100.*bool(success and not initially_complete)-10.*after[0]-10.*after[1]) if terminal else 0.
    return float(shaping+final-sum(cost.values())),dict(cost=cost,shaping=shaping,terminal=final)

def time_gae(rewards, values, next_values, terminated, durations, time_constant=600.):
    """Arrays [T,B], gamma=1. Terminal cuts both bootstrap and reverse recursion.

    next_values must be evaluated before reset; last rollout frame bootstraps.
    """
    arrays=[np.asarray(a) for a in (rewards,values,next_values,terminated,durations)]
    r,v,nv,done,dt=arrays
    if any(a.shape!=r.shape for a in arrays) or r.ndim!=2: raise ValueError('Expected matching [T,B] arrays')
    if time_constant<=0 or not np.isfinite(dt).all() or (dt<0).any(): raise ValueError('Invalid time')
    result=np.zeros_like(r,dtype=np.float64);carry=np.zeros(r.shape[1])
    for i in range(len(r)-1,-1,-1):
        alive=~done[i].astype(bool)
        delta=r[i]+alive*nv[i]-v[i]
        carry=delta+alive*np.exp(-dt[i]/time_constant)*carry
        result[i]=carry
    return result,result+v

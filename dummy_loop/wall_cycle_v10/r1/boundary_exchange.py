"""Exponential linear inventory exchange for FROZEN boundary rates/capacities.

An isolated candidate component, not the moving-footprint material solver.
No gap saturation, pressure, changing uncovered area or interface event handling.
Rejects zero-capacity donors and excessive stiffness instead of hiding them with
an area floor, negative-volume clipping or unlimited adaptive retries.
"""
import numpy as np

EXCHANGE_VERSION = 'v10r1.boundary_inventory.frozen_exponential.v1'


def exchange(wall, blade, free_area, blade_area, maps, outside, dt,
             lost=0., max_turnover=1000.):
    """Maps are instantaneous() dictionaries in m2/time, not integrated areas.

    Entry: wall -> blade, exit: blade -> wall/outside. Outside entry is empty.
    Capacities remain fixed for this call. Returned arrays are new objects.
    """
    wall=np.asarray(wall,dtype=float); blade=np.asarray(blade,dtype=float)
    free_area=np.asarray(free_area,dtype=float)
    blade_area=np.broadcast_to(np.asarray(blade_area,dtype=float),blade.shape)
    if wall.ndim!=1 or blade.ndim!=1 or free_area.shape!=wall.shape:
        raise ValueError('Expected flat inventory and matching wall areas')
    vals=np.r_[wall,blade,free_area,blade_area,lost,dt,max_turnover]
    if not np.isfinite(vals).all() or np.any(vals<0) or max_turnover<=0:
        raise ValueError('Expected finite nonnegative inventories/areas/time')
    if len(maps)!=2 or len(outside)!=2:
        raise ValueError('Expected incoming and outgoing maps')
    nw=len(wall); sink=nw+len(blade); edges=[]
    for kind,d in enumerate(maps):
        for (b,w),rate in d.items():
            if not (0<=b<len(blade) and 0<=w<nw) or not np.isfinite(rate) or rate<0:
                raise ValueError('Invalid boundary rate/index')
            if rate>0:
                edges.append((w,nw+b,rate) if kind==0 else (nw+b,w,rate))
    for kind,ext in enumerate(outside):
        ext=np.asarray(ext,dtype=float)
        if ext.shape!=blade.shape or not np.isfinite(ext).all() or np.any(ext<0):
            raise ValueError('Invalid outside rates')
        if kind==1:
            edges.extend((nw+b,sink,float(r)) for b,r in enumerate(ext) if r>0)
    if not edges or dt==0:return wall.copy(),blade.copy(),float(lost)
    active=np.array(sorted({i for src,dst,_ in edges for i in (src,dst)}))
    index={int(v):i for i,v in enumerate(active)}
    capacity=np.r_[free_area,blade_area,1.]
    rows=[];cols=[];data=[]
    outgoing=np.zeros(len(active))
    for src,dst,rate in edges:
        if capacity[src]<=0:
            raise ValueError('Positive flux from zero-capacity donor: split geometry event first')
        k=rate/capacity[src];s=index[src];d=index[dst]
        rows.append(d);cols.append(s);data.append(k)
        outgoing[s]+=k
    if float(outgoing.max())*dt>max_turnover:
        raise ValueError('Frozen exchange stiffness budget exceeded: split geometry event first')
    for i,k in enumerate(outgoing):
        rows.append(i);cols.append(i);data.append(-k)
    # Uniformization: exp(Q dt) is a positive Poisson mixture of Markov
    # steps P=I+Q/lambda. Normalize only the truncated Poisson weights,
    # never the material state. Each step conserves the inventory ledger.
    state=np.r_[wall,blade,lost]
    updated=state[active].copy()
    lam=float(outgoing.max())
    chunks=max(1,int(np.ceil(lam*dt/8.)))
    mu=lam*dt/chunks
    weights=[float(np.exp(-mu))]
    for k in range(1,256):
        weights.append(weights[-1]*mu/k)
        if k>mu:
            next_weight=weights[-1]*mu/(k+1)
            tail_bound=next_weight/(1.-mu/(k+2))
            if tail_bound<2e-16:break
    else:raise FloatingPointError('Poisson expansion budget exhausted')
    weights=np.asarray(weights);weights/=weights.sum()
    off=[(r,c,v/lam) for r,c,v in zip(rows,cols,data) if r!=c]
    dst=np.array([e[0] for e in off],dtype=int)
    src=np.array([e[1] for e in off],dtype=int)
    prob=np.array([e[2] for e in off])
    stay=1.-outgoing/lam
    for _ in range(chunks):
        term=updated.copy();acc=weights[0]*term
        for weight in weights[1:]:
            nxt=stay*term
            np.add.at(nxt,dst,prob*term[src])
            term=nxt;acc+=weight*term
        updated=acc
    if not np.isfinite(updated).all() or np.any(updated<0):
        raise FloatingPointError('Invalid exponential result; no clipping permitted')
    state[active]=updated
    return state[:nw],state[nw:sink],float(state[sink])

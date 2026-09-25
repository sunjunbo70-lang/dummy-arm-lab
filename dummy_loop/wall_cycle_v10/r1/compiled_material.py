"""Compiled CPU candidate. Equivalence to DisplacementMortar is mandatory.
No fastmath: preserve ledger and pressure comparisons. No hardware interfaces.
"""
import numpy as np
from numba import njit

@njit(cache=True)
def support(B,gaps,cell,tau):
    nr,nc=B.shape;total=0.
    for c in range(nc):
        r=0
        while r<nr:
            if B[r,c]<gaps[r]-1e-12:r+=1;continue
            end=r+1
            while end<nr and B[end,c]>=gaps[end]-1e-12:end+=1
            for k in range(r,end):
                dist=(min(k-r+1,end-k)-.5)*cell
                total+=2*tau*dist/gaps[k]
            r=end
    return total*cell*cell

@njit(cache=True)
def gap_solve(B,pitch,force,cell,tau,min_gap,max_gap):
    rise=(np.arange(B.shape[0])+.5)*cell*np.sin(max(pitch,0.))
    if force<=0:return max_gap,0.
    s=support(B,min_gap+rise,cell,tau)
    if s<force:return 0.,s
    s=support(B,max_gap+rise,cell,tau)
    if s>=force:return max_gap,s
    lo=min_gap;hi=max_gap
    for _ in range(20):
        mid=(lo+hi)*.5
        if support(B,mid+rise,cell,tau)>=force:lo=mid
        else:hi=mid
    return lo,support(B,lo+rise,cell,tau)

@njit(cache=True)
def wall_key(u,v,width,cell,nv,nu):
    x=(u+width/2)/cell;y=v/cell
    if abs(x-np.rint(x))<1e-9:x=np.rint(x)
    if abs(y-np.rint(y))<1e-9:y=np.rint(y)
    j=int(np.floor(x));i=int(np.floor(y))
    return i*nu+j if i>=0 and i<nv and j>=0 and j<nu else -1

@njit(cache=True)
def integrate(wall,B,bead,owner,last,gaps_prev,g0_prev,points,phis,pitches,forces,speeds,
              flip,cell,blade_cell,width,tau,couette,h_slump,min_gap,max_gap):
    """In-place wall/B/bead; persistent owner/last allows continuation without lifting.
    owner[k] is last blade flat index for wall cell k, matching CPU reference.
    """
    nr,nc=B.shape;nv,nu=wall.shape;A=blade_cell**2/cell**2
    w=wall.ravel();current=np.full(w.size,-1,np.int64);counts=np.zeros(w.size,np.int64)
    flat=np.full((nr,nc),-1,np.int64);keys=np.empty(nr*nc,np.int64)
    dropped=0.;outside=0.;support_last=0.
    for step in range(len(points)):
        xy=points[step];phi=phis[step];pitch=pitches[step]
        ex=np.array([np.cos(phi),np.sin(phi)]);ew=np.array([-np.sin(phi),np.cos(phi)])
        if flip:ew=-ew
        fresh=np.isnan(last[0]);displacement=0. if fresh else np.sqrt(((xy-last)**2).sum())
        exchange=1. if fresh else min(displacement/blade_cell,1.)
        if not fresh:
            ds=((xy-last)*ew).sum();a=min(max(couette*max(ds,0.)/blade_cell,0.),1.)
            if a>0:
                old=B.copy()
                for r in range(nr):
                    for c in range(nc):
                        B[r,c]=(1-a)*old[r,c]
                        if r<nr-1:B[r,c]+=a*old[r+1,c]
                        if r==0:B[r,c]+=a*old[0,c]
        last=xy.copy();nk=0
        for r in range(nr):
            for c in range(nc):
                loc=xy+(c-(nc-1)/2)*blade_cell*ex+(r-(nr-1)/2)*blade_cell*ew
                k=wall_key(loc[0],loc[1],width,cell,nv,nu);flat[r,c]=k
                if k>=0:
                    if counts[k]==0:keys[nk]=k;nk+=1
                    counts[k]+=1;current[k]=r*nc+c
        # Most loops operate on blade footprint, but leaving ownership scans sparse wall.
        for k in range(w.size):
            old=owner[k]
            if old>=0 and current[k]<0:
                r=old//nc;c=old%nc;exitgap=g0_prev if r==0 else gaps_prev[r]
                amount=min(B[r,c],exitgap)
                w[k]+=amount*A;B[r,c]-=amount;owner[k]=-1
        # Absorb using untouched wall values for duplicate footprint samples.
        for r in range(nr):
            for c in range(nc):
                k=flat[r,c]
                if k>=0 and owner[k]<0:B[r,c]+=w[k]/counts[k]/A
        for j in range(nk):
            k=keys[j]
            if owner[k]<0:w[k]=0.
        for c in range(nc):
            enter=exchange*bead[c];B[nr-1,c]+=enter;bead[c]-=enter
        g0,support_last=gap_solve(B,pitch,forces[step],blade_cell,tau,min_gap,max_gap)
        gaps=g0+(np.arange(nr)+.5)*blade_cell*np.sin(max(pitch,0.))
        gt=max(gaps[0],1e-9);gl=max(gaps[-1],1e-9);ft=gt**3/(gt**3+gl**3)
        trail=np.zeros(nc)
        for c in range(nc):
            eq=B[:,c].copy();excess=0.
            for r in range(nr):
                extra=max(eq[r]-gaps[r],0.);eq[r]-=extra;excess+=extra
            for r in range(nr):
                take=min(max(gaps[r]-eq[r],0.),excess);eq[r]+=take;excess-=take
            for r in range(nr):B[r,c]+=exchange*(eq[r]-B[r,c])
            trail[c]=excess*ft*exchange;bead[c]+=excess*(1-ft)*exchange
        for c in range(nc):
            loc=xy+(c-(nc-1)/2)*blade_cell*ex+(-1-(nr-1)/2)*blade_cell*ew
            k=wall_key(loc[0],loc[1],width,cell,nv,nu)
            if k<0:outside+=trail[c]*blade_cell**2
            elif current[k]>=0:B[0,c]+=trail[c]
            else:w[k]+=trail[c]*A
        for j in range(nk):
            k=keys[j];owner[k]=current[k];current[k]=-1;counts[k]=0
        gaps_prev=gaps;g0_prev=g0
        for k in range(w.size):
            extra=max(w[k]-h_slump,0.)
            w[k]-=extra;dropped+=extra*cell**2
    return last,gaps_prev,g0_prev,dropped,outside,support_last

def integrate_material(m,points,phis,pitches,forces,speeds):
    """Call only inside an existing stroke; air/feed/end retain reference semantics."""
    owner=np.full(m.wall.size,-1,np.int64);nc=m.blade.shape[1]
    for k,(r,c) in m._covered.items():owner[k]=r*nc+c
    B=m._rows();last=np.full(2,np.nan) if m._last_centre is None else m._last_centre.copy()
    gap=np.zeros(B.shape[0]) if m._gaps is None else m._gaps.copy()
    result=integrate(m.wall,B,m._bead,owner,last,gap,m._g0 or 0.,np.ascontiguousarray(points,dtype=float),
        np.asarray(phis,float),np.asarray(pitches,float),np.asarray(forces,float),np.asarray(speeds,float),
        m._flip,m.cfg.cell_m,m.cfg.blade_cell_m,m.cfg.width_m,m.p.tau_y,m.p.couette_fraction,m.h_slump,m.p.min_gap_m,m.p.max_gap_m)
    m._last_centre,m._gaps,m._g0,drop,outside,_=result
    m.dropped_m3+=drop;m.outside_m3+=outside;m._store(B)
    m._covered={int(k):(int(owner[k]//nc),int(owner[k]%nc)) for k in np.flatnonzero(owner>=0)}
    phi=float(phis[-1]);m.e_x=np.array([np.cos(phi),np.sin(phi)]);m.e_w=np.array([-np.sin(phi),np.cos(phi)])*(-1 if m._flip else 1)
    m.last_pitch=float(pitches[-1]);return result

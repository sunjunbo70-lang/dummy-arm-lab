"""Analytic wall-reservoir exchange with linearly changing uncovered area.
Candidate subproblem only. Exit volume is prescribed, not coupled to blade
pressure/gap/inventory. Entry/exit area rates are constant over normalized time.
"""
import numpy as np
VERSION='v10r1.moving_wall_exchange.analytic.v1'

def transfer(volume,area0,entry,exit,deposit,final_area=None):
    """Return (remaining wall volume, picked volume, final uncovered area).

    A(t)=A0+(exit-entry)*t; dV/dt=deposit-entry*V/A(t).
    Areas in m2, volumes in m3. A1 must be nonnegative; callers must split
    geometry events if the linear model would give negative capacity.
    No arbitrary area floor or mass renormalization is applied.
    """
    v,a,e,x,d=np.broadcast_arrays(*[np.asarray(z,float) for z in (volume,area0,entry,exit,deposit)])
    if not all(np.isfinite(z).all() and (z>=0).all() for z in (v,a,e,x,d)):
        raise ValueError('Expected finite nonnegative inventory/areas/deposit')
    change=x-e;b=a+change
    if final_area is not None:
        target=np.broadcast_to(np.asarray(final_area,float),v.shape)
        allowance=32*np.finfo(float).eps*(a+e+x)
        if not np.isfinite(target).all() or (target<0).any() or (abs(target-b)>allowance).any():
            raise ValueError('Final area inconsistent with entry/exit balance')
        b=target;change=b-a
    if (b<0).any():raise ValueError('Negative final area: split geometric event')
    if ((a==0)&(v>0)).any():raise ValueError('Material in zero-area initial reservoir')
    if ((x==0)&(d>0)).any():raise ValueError('Deposit without exiting area')
    if ((a==0)&(b==0)&((e>0)|(x>0))).any():
        raise ValueError('Zero-area throughflow needs interface event treatment')
    f=np.ones(v.shape);source=np.zeros(v.shape)
    positive=(a>0)&(b>0)
    steady=positive&(change==0)
    f[steady]=np.exp(-e[steady]/a[steady])
    source[steady]=-np.expm1(-e[steady]/a[steady])*np.divide(a[steady],e[steady],out=np.zeros_like(a[steady]),where=e[steady]>0)
    source[steady&(e==0)]=1.
    varying=positive&(change!=0)
    z=change[varying]/a[varying]
    logratio=np.log1p(z)
    exponent=-e[varying]/change[varying]*logratio
    f[varying]=np.exp(exponent)
    xv=x[varying];av=a[varying];bv=b[varying]
    # Stable A1-A0*f = change - A0*expm1(exponent).
    numerator=change[varying]-av*np.expm1(exponent)
    source[varying]=np.divide(numerator,xv,out=np.zeros_like(xv),where=xv>0)
    born=(a==0)&(b>0)
    f[born]=0.;source[born]=b[born]/x[born]
    closed=(a>0)&(b==0)
    f[closed]=0.;source[closed]=0.
    remaining=v*f+d*source
    picked=v+d-remaining
    # Only remove subtraction round-off in returned picked amount; use the
    # complementary remaining value so the local ledger stays unchanged.
    tol=32*np.finfo(float).eps*(v+d)
    if (remaining<0).any() or (picked < -tol).any():
        raise FloatingPointError('Nonphysical analytic exchange result')
    tiny=picked<0
    picked=np.where(tiny,0.,picked);remaining=np.where(tiny,v+d,remaining)
    return remaining,picked,b

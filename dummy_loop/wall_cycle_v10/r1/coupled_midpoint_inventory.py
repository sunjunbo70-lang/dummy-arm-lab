"""v7 candidate: jointly midpoint-integrated blade and uncovered-wall exchange.
Finite-volume closure assumes uniform density within each uncovered wall cell.
No change to pressure/flux law. Global convergence and physical validity unproven.
"""
import numpy as np
from .split_inventory import SplitPressure
from .contact_inventory import ContactInventory,turnover

class CoupledMidpointPressure(SplitPressure):
    physics_version='v10r1.area_pressure_candidate.v7_coupled_midpoint_geometry'
    def move(self,center,angle,exit_height):
        if self.pose is None:return ContactInventory.move(self,center,angle,exit_height)
        new_pose=(np.asarray(center,dtype=float),float(angle))
        old,new,leave,enter=turnover(self.pose,new_pose,self.blade.shape,self.bc,self.wall.shape,self.wc,self.origin)
        ob,ow=old[:2];nb,nw=new[:2];nB=self.blade.size;nW=self.wall.size
        height=np.broadcast_to(exit_height,self.blade.shape).ravel()
        if not np.isfinite(height).all() or (height<0).any():raise ValueError('Invalid exit height')
        covered=np.bincount(ow,weights=old[2],minlength=nW)
        free=np.maximum(0.,self.wc**2-covered)
        leave_wall=np.bincount(ow,weights=leave,minlength=nW)
        leave_blade=np.bincount(ob,weights=leave,minlength=nB)
        # Midpoint wall volume / midpoint free area, accounting for half pickup:
        # density=(W0 + D/2)/(free_old + leave_area/2).
        denom=free+.5*leave_wall
        deposited=np.zeros(nW);W=self.wall.ravel();V=self.blade.ravel()
        for _ in range(64):
            density=np.divide(W+.5*deposited,denom,out=np.zeros(nW),where=denom>1e-18)
            take=enter*density[nw]
            picked=np.bincount(nb,weights=take,minlength=nB)
            donor=np.minimum((V+.5*picked)/(self.bc**2+.5*leave_blade),height)
            sent=leave*donor[ob]
            updated=np.bincount(ow,weights=sent,minlength=nW)
            if np.max(abs(updated-deposited))<=1e-20:
                wall=W+updated-np.bincount(nw,weights=take,minlength=nW)
                blade=V+picked-np.bincount(ob,weights=sent,minlength=nB)
                if min(wall.min(),blade.min()) < -1e-18:raise RuntimeError('Negative midpoint inventory')
                self.wall=wall.reshape(self.wall.shape);self.blade=blade.reshape(self.blade.shape)
                self.pose,self.mapping=new_pose,new
                return
            deposited=updated
        raise RuntimeError('Midpoint exchange failed to converge; state unchanged')

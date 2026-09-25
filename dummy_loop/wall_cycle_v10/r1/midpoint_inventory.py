"""Independent v6 candidate: implicit midpoint donor density for geometry exit.

For cell area A, exiting area L and picked-up volume P, midpoint exit E solves
E = L * min((V + P/2 - E/2)/A, gap), hence the expression below.
This removes the old-density exit bias only. Full coupled order is not guaranteed.
"""
import numpy as np
from .split_inventory import SplitPressure
from .contact_inventory import ContactInventory,turnover

def exit_volume(volume,pickup,leaving_area,area,gap):
    density=(volume+.5*pickup)/(area+.5*leaving_area)
    return leaving_area*np.minimum(density,gap)

class MidpointGeometryPressure(SplitPressure):
    physics_version='v10r1.area_pressure_candidate.v6_midpoint_geometry'
    def move(self,center,angle,exit_height):
        if self.pose is None:return ContactInventory.move(self,center,angle,exit_height)
        new_pose=(np.asarray(center,dtype=float),float(angle))
        old,new,leave,enter=turnover(self.pose,new_pose,self.blade.shape,self.bc,self.wall.shape,self.wc,self.origin)
        bi,wi=old[:2]
        height=np.broadcast_to(exit_height,self.blade.shape).ravel()
        if not np.isfinite(height).all() or np.any(height<0):raise ValueError('Invalid exit height')
        before=self.blade.copy()
        covered=np.bincount(wi,weights=old[2],minlength=self.wall.size)
        self._pickup(new,enter,np.maximum(0.,self.wc**2-covered))
        pickup=(self.blade-before).ravel()
        leaving=np.bincount(bi,weights=leave,minlength=self.blade.size)
        total=exit_volume(before.ravel(),pickup,leaving,self.bc**2,height)
        per_area=np.divide(total,leaving,out=np.zeros_like(total),where=leaving>0)
        amount=leave*per_area[bi]
        self.blade-=total.reshape(self.blade.shape)
        self.wall+=np.bincount(wi,weights=amount,minlength=self.wall.size).reshape(self.wall.shape)
        self.pose,self.mapping=new_pose,new

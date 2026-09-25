"""Candidate v2: entry and exit fragments are disjoint within one pose transition.
Sample entering material from OLD uncovered wall before adding exit deposits.
"""
import numpy as np
from .contact_inventory import ContactInventory,turnover
from .pressure_inventory import PressureInventory

class OrderedInventory(ContactInventory):
    def move(self,center,angle,exit_height):
        if self.pose is None:
            return super().move(center,angle,exit_height)
        new_pose=(np.asarray(center,dtype=float),float(angle))
        old,new,leave,enter=turnover(self.pose,new_pose,self.blade.shape,self.bc,self.wall.shape,self.wc,self.origin)
        bi,wi=old[:2]
        height=np.broadcast_to(exit_height,self.blade.shape).ravel()
        if np.any(height<0):raise ValueError('Negative exit height')
        # Exit flow uses pre-transition blade density, not newly picked-up mortar.
        amount=leave*np.minimum(self.blade.ravel()[bi]/self.bc**2,height[bi])
        covered=np.bincount(wi,weights=old[2],minlength=self.wall.size)
        # Entry lies in the previous uncovered region. Exit occupies a different
        # geometric fragment even when both fragments fall in the same wall cell.
        self._pickup(new,enter,np.maximum(0.,self.wc**2-covered))
        self.blade-=np.bincount(bi,weights=amount,minlength=self.blade.size).reshape(self.blade.shape)
        self.wall+=np.bincount(wi,weights=amount,minlength=self.wall.size).reshape(self.wall.shape)
        self.pose,self.mapping=new_pose,new

class OrderedPressure(PressureInventory):
    physics_version='v10r1.area_pressure_candidate.v2_entry_before_exit'
    def move(self,center,angle,exit_height):
        # Explicit dispatch: preserves PressureInventory's constitutive equations.
        if self.pose is None:
            return ContactInventory.move(self,center,angle,exit_height)
        return OrderedInventory.move(self,center,angle,exit_height)

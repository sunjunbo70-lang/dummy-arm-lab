"""Pressure/extrusion coupled to conservative contact inventory: isolated candidate.
No robot/transport-gravity implementation. Not frozen training physics.
"""
import numpy as np
from .contact_inventory import ContactInventory
from .compiled_material import gap_solve
from .overlap import planar_overlap, deposit
from ...wall_cycle.mortar import MortarParams, extrude

class PressureInventory(ContactInventory):
    physics_version='v10r1.area_pressure_candidate.v1'

    def __init__(self,*args,params=None,**kwargs):
        super().__init__(*args,**kwargs)
        self.params=MortarParams() if params is None else params
        self.bead=np.zeros(self.blade.shape[1]) # m^3 per column
        self.gaps=np.zeros(self.blade.shape[0])
        self.g0=0.
        self.dropped=0.

    def total(self):
        return super().total()+self.bead.sum()+self.dropped

    def _edge_deposit(self,volumes,center,angle,side):
        e=np.array([np.cos(angle),np.sin(angle)])
        edge=np.asarray(center)+side*(self.blade.shape[0]+1)*self.bc*.5*e
        m=planar_overlap(edge,angle,(1,self.blade.shape[1]),self.bc,self.wall.shape,self.wc,np.asarray(self.origin))
        wall,lost=deposit(volumes[None,:],m,self.bc,self.wall.shape)
        self.wall += wall
        self.outside += lost

    def contact(self,center,angle,pitch,force):
        """One explicitly substepped pose. Angle is unwrapped row-axis angle.
        Uses previous pressure gap for exiting material, then recomputes pressure.
        Distance relaxation is a candidate, not a calibrated squeeze timescale.
        """
        center=np.asarray(center,dtype=float)
        if not np.isfinite(np.r_[center,angle,pitch,force]).all() or pitch<0 or force<0:
            raise ValueError('Invalid contact input')
        first=self.pose is None
        old=self.pose
        exit_gap=self.gaps.copy();exit_gap[0]=self.g0
        self.move(center,angle,exit_gap[:,None])
        B=self.blade/self.bc**2
        if first:
            exchange=1.
        else:
            # Mean cell-center motion includes pure rotation; no angle wrapping
            # here because the path sampler is required to provide unwrapped phi.
            r,c=np.indices(self.blade.shape)
            local=np.stack(((r+.5-self.blade.shape[0]/2)*self.bc,(c+.5-self.blade.shape[1]/2)*self.bc),axis=-1)
            def positions(pose):
                p,a=pose;e=np.array([np.cos(a),np.sin(a)]);f=np.array([-np.sin(a),np.cos(a)])
                return p+local[...,0,None]*e+local[...,1,None]*f
            travel=float(np.linalg.norm(positions((center,angle))-positions(old),axis=-1).mean())
            exchange=-np.expm1(-travel/self.bc)
            # Retain historical longitudinal Couette assumption; transverse
            # rotational shear is not modeled and must not be claimed as such.
            e=np.array([np.cos(angle),np.sin(angle)])
            ds=max(0.,float((center-old[0])@e))
            a=-np.expm1(-self.params.couette_fraction*ds/self.bc)
            moved=a*B;B-=moved;B[:-1]+=moved[1:];B[0]+=moved[0]
        enter=exchange*self.bead
        B[-1]+=enter/self.bc**2;self.bead-=enter
        p=self.params
        self.g0,support=gap_solve(B,float(pitch),float(force),self.bc,p.tau_y,p.min_gap_m,p.max_gap_m)
        self.gaps=self.g0+(np.arange(B.shape[0])+.5)*self.bc*np.sin(pitch)
        equilibrium,trail,lead=extrude(B,self.gaps)
        B+=exchange*(equilibrium-B)
        self.blade=B*self.bc**2
        self.bead+=exchange*lead*self.bc**2
        self._edge_deposit(exchange*trail*self.bc**2,center,angle,-1)
        # Wall inventory is per uncovered fragment; use its actual area for slump.
        m=self.mapping
        covered=np.bincount(m[1],weights=m[2],minlength=self.wall.size).reshape(self.wall.shape)
        free_area=np.maximum(0.,self.wc**2-covered)
        cap=p.tau_y/(p.rho*9.81)*free_area
        excess=np.maximum(self.wall-cap,0.)
        self.wall-=excess;self.dropped+=excess.sum()
        return dict(gap_m=float(self.g0),support_N=float(support),exchange=float(exchange))

    def lift(self,fraction):
        if self.pose is None:return
        if not 0<=fraction<=1:raise ValueError('Invalid split fraction')
        self._edge_deposit(self.bead,self.pose[0],self.pose[1],1)
        self.bead[:]=0.
        super().lift(fraction)

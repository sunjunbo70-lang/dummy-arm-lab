"""v8.5 candidate: nonautonomous SSPRK2 uses endpoint controls at its stages.
No change to pressure law, flux law, mesh or acceptance criteria.
"""
import numpy as np
from .stationary_boundary_pressure import StationaryBoundaryPressure
class TemporalPressure(StationaryBoundaryPressure):
    physics_version='v10r1.area_pressure_candidate.v8_5_temporal_controls'
    def contact(self,center,angle,pitch,force):
        center=np.asarray(center,dtype=float)
        if self.pose is None:
            info=super().contact(center,angle,pitch,force)
            self.last_controls=(pitch,force)
            return info
        if not np.isfinite(np.r_[center,angle,pitch,force]).all() or pitch<0 or force<0:
            raise ValueError('Invalid contact input')
        old_center,old_angle=self.pose
        mid=(old_center+center)/2;ma=(old_angle+angle)/2
        mp,mf=(np.array(self.last_controls)+[pitch,force])/2
        r,c=np.indices(self.blade.shape)
        local=np.stack(((r+.5-self.blade.shape[0]/2)*self.bc,(c+.5-self.blade.shape[1]/2)*self.bc),axis=-1)
        def coords(p,a):
            return p+local[...,0,None]*[np.cos(a),np.sin(a)]+local[...,1,None]*[-np.sin(a),np.cos(a)]
        travel=float(np.linalg.norm(coords(center,angle)-coords(old_center,old_angle),axis=-1).mean())
        ds=max(0.,float((center-old_center)@np.array([np.cos(ma),np.sin(ma)])))
        exit_gap=self.gaps.copy();exit_gap[0]=self.g0
        self.move(mid,ma,exit_gap[:,None])
        B=self.blade/self.bc**2;bead=self.bead/self.bc**2
        trail=np.zeros_like(bead)
        if travel>0:
            duration=travel/self.bc
            n=max(1,int(np.ceil(duration/.2)));h=duration/n
            couette=self.params.couette_fraction*ds/travel
            for k in range(n):
                p0,f0=np.asarray(self.last_controls)+(np.asarray([pitch,force])-self.last_controls)*(k/n)
                p1,f1=np.asarray(self.last_controls)+(np.asarray([pitch,force])-self.last_controls)*((k+1)/n)
                db,dq,dt=self._flux(B,bead,p0,f0,couette)
                predicted=B+h*db;predicted_bead=bead+h*dq
                db2,dq2,dt2=self._flux(predicted,predicted_bead,p1,f1,couette)
                B=.5*B+.5*(predicted+h*db2)
                bead=.5*bead+.5*(predicted_bead+h*dq2)
                trail+=.5*h*(dt+dt2)
        self.blade=B*self.bc**2;self.bead=bead*self.bc**2
        self._edge_deposit(trail*self.bc**2,mid,ma,-1)
        self.g0,self.gaps,_=self._pressure(B,mp,mf)
        exit_gap=self.gaps.copy();exit_gap[0]=self.g0
        self.move(center,angle,exit_gap[:,None])
        self.g0,self.gaps,support=self._pressure(self.blade/self.bc**2,pitch,force)
        covered=np.bincount(self.mapping[1],weights=self.mapping[2],minlength=self.wall.size).reshape(self.wall.shape)
        cap=self.params.tau_y/(self.params.rho*9.81)*np.maximum(0.,self.wc**2-covered)
        excess=np.maximum(self.wall-cap,0.);self.wall-=excess;self.dropped+=excess.sum()
        self.last_controls=(pitch,force)
        return dict(gap_m=float(self.g0),support_N=float(support))

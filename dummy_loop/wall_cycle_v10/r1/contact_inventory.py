"""Endpoint footprint turnover and conservative contact inventory candidate.
Not a full material model: pressure, extrusion and robot execution are external.
"""
import math
import numpy as np
from numba import njit
from .overlap import planar_overlap, _clip

PHYSICS_VERSION = 'v10r1.swept_inventory_candidate.v1'

@njit(cache=True)
def _halfplane(poly, n, normal, offset):
    out = np.empty((24,2), dtype=np.float64)
    size = 0
    for k in range(n):
        a, b = poly[(k-1)%n], poly[k]
        da, db = (a[0]*normal[0]+a[1]*normal[1])-offset, (b[0]*normal[0]+b[1]*normal[1])-offset
        if (da<=0) != (db<=0):
            out[size] = a + (da/(da-db))*(b-a)
            size += 1
        if db<=0:
            out[size] = b
            size += 1
    return out, size

@njit(cache=True)
def _polygon_area(p,n):
    result=0.
    for k in range(n):
        a,b=p[k]-p[0],p[(k+1)%n]-p[0]
        result += a[0]*b[1]-a[1]*b[0]
    return abs(result)*.5

@njit(cache=True)
def _exclusive_areas(center,angle,other_center,other_angle,bs,bc,ws,wc,origin,bi,wi,areas):
    """Part of each cell/wall overlap outside the other *whole* blade footprint."""
    e=np.array([math.cos(angle),math.sin(angle)])
    f=np.array([-math.sin(angle),math.cos(angle)])
    oe=np.array([math.cos(other_angle),math.sin(other_angle)])
    of=np.array([-math.sin(other_angle),math.cos(other_angle)])
    result=np.empty(areas.size)
    for k in range(areas.size):
        r,c=bi[k]//bs[1],bi[k]%bs[1]
        mid=center+(r+.5-bs[0]/2)*bc*e+(c+.5-bs[1]/2)*bc*f
        poly=np.empty((4,2))
        poly[0]=mid-.5*bc*e-.5*bc*f
        poly[1]=mid+.5*bc*e-.5*bc*f
        poly[2]=mid+.5*bc*e+.5*bc*f
        poly[3]=mid-.5*bc*e+.5*bc*f
        row,col=wi[k]//ws[1],wi[k]%ws[1]
        p,n=_clip(poly,4,0,origin[0]+col*wc,True)
        p,n=_clip(p,n,0,origin[0]+(col+1)*wc,False)
        p,n=_clip(p,n,1,origin[1]+row*wc,True)
        p,n=_clip(p,n,1,origin[1]+(row+1)*wc,False)
        p,n=_halfplane(p,n,oe,(other_center[0]*oe[0]+other_center[1]*oe[1])+bs[0]*bc/2)
        p,n=_halfplane(p,n,-oe,(-other_center[0]*oe[0]-other_center[1]*oe[1])+bs[0]*bc/2)
        p,n=_halfplane(p,n,of,(other_center[0]*of[0]+other_center[1]*of[1])+bs[1]*bc/2)
        p,n=_halfplane(p,n,-of,(-other_center[0]*of[0]-other_center[1]*of[1])+bs[1]*bc/2)
        result[k]=max(0.,areas[k]-_polygon_area(p,n))
    return result


def turnover(old_pose,new_pose,bs=(6,24),bc=.005,ws=(100,100),wc=.005,origin=(-.25,0.)):
    """Endpoint differences, not the entire swept union of a long rotation.
    Caller must substep curves and audit convergence. Returns old/new maps,
    old-cell exit areas and new-cell entry areas, in corresponding sparse order.
    """
    origin=np.asarray(origin,dtype=float)
    a,alpha=np.asarray(old_pose[0],dtype=float),float(old_pose[1])
    b,beta=np.asarray(new_pose[0],dtype=float),float(new_pose[1])
    old=planar_overlap(a,alpha,bs,bc,ws,wc,origin)
    new=planar_overlap(b,beta,bs,bc,ws,wc,origin)
    leaving=_exclusive_areas(a,alpha,b,beta,bs,bc,ws,wc,origin,*old[:3])
    entering=_exclusive_areas(b,beta,a,alpha,bs,bc,ws,wc,origin,*new[:3])
    return old,new,leaving,entering


class ContactInventory:
    """Wall volumes denote material on the currently uncovered area only.
    Blade volumes are tool-attached and shared with the caller's material solver.
    Uniform density within each remaining wall fragment is an explicit approximation.
    No automatic loading, force model, or arbitrary per-callback pickup fraction.
    """
    def __init__(self,wall_volumes,blade_volumes,bc=.005,wc=.005,origin=(-.25,0.)):
        self.wall=np.array(wall_volumes,dtype=float,copy=True)
        self.blade=np.array(blade_volumes,dtype=float,copy=True)
        if np.any(self.wall<0) or np.any(self.blade<0):
            raise ValueError('Volumes must be nonnegative')
        self.bc,self.wc,self.origin=bc,wc,origin
        self.pose=None
        self.mapping=None
        self.outside=0.

    def total(self):
        return self.wall.sum()+self.blade.sum()+self.outside

    def _pickup(self,mapping,areas,available_area):
        bi,wi=mapping[:2]
        density=np.divide(self.wall.ravel(),available_area,out=np.zeros(self.wall.size),where=available_area>1e-18)
        taken=areas*density[wi]
        self.blade += np.bincount(bi,weights=taken,minlength=self.blade.size).reshape(self.blade.shape)
        self.wall -= np.bincount(wi,weights=taken,minlength=self.wall.size).reshape(self.wall.shape)

    def move(self,center,angle,exit_height):
        """Deposit at most min(blade density, caller exit gap) on leaving area.
        exit_height is caller-supplied scalar/array in m, not a smoothing target.
        """
        new_pose=(np.asarray(center,dtype=float),float(angle))
        if self.pose is None:
            m=planar_overlap(new_pose[0],new_pose[1],self.blade.shape,self.bc,self.wall.shape,self.wc,np.asarray(self.origin))
            self._pickup(m,m[2],np.full(self.wall.size,self.wc**2))
        else:
            old,m,leave,enter=turnover(self.pose,new_pose,self.blade.shape,self.bc,self.wall.shape,self.wc,self.origin)
            bi,wi=old[:2]
            height=np.broadcast_to(exit_height,self.blade.shape).ravel()
            if np.any(height<0): raise ValueError('Negative exit height')
            amount=leave*np.minimum(self.blade.ravel()[bi]/self.bc**2,height[bi])
            self.blade -= np.bincount(bi,weights=amount,minlength=self.blade.size).reshape(self.blade.shape)
            self.wall += np.bincount(wi,weights=amount,minlength=self.wall.size).reshape(self.wall.shape)
            # Exit deposits belong to newly uncovered fragments. Mix those with
            # pre-existing uncovered inventory before absorbing new entry area.
            covered=np.bincount(old[1],weights=old[2],minlength=self.wall.size)
            exited=np.bincount(wi,weights=leave,minlength=self.wall.size)
            self._pickup(m,enter,self.wc**2-covered+exited)
        self.pose,self.mapping=new_pose,m

    def lift(self,fraction):
        if not 0<=fraction<=1: raise ValueError('Invalid split fraction')
        if self.pose is None: return
        bi,wi,area,outside=self.mapping
        sent=self.blade.ravel()*fraction
        self.wall += np.bincount(wi,weights=sent[bi]*area/self.bc**2,minlength=self.wall.size).reshape(self.wall.shape)
        self.outside += np.dot(sent,outside/self.bc**2)
        self.blade *= 1-fraction
        self.pose,self.mapping=None,None


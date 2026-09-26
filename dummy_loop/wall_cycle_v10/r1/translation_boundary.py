"""Exact swept boundary maps for constant-orientation linear translation only.
Diagnostic geometry building block, not a replacement for rotating material flow.
Incoming/outgoing strips may include regions absent at BOTH endpoint differences.
"""
import numpy as np
from .overlap import _area_in_box

GEOMETRY_VERSION='v10r1.translation_swept_boundary.v1'

def boundary_map(begin,end,angle,bs=(6,24),bc=.005,ws=(100,100),wc=.005,origin=(-.25,0.),incoming=True):
    begin=np.asarray(begin,dtype=float);end=np.asarray(end,dtype=float)
    if begin.shape!=(2,) or end.shape!=(2,) or not np.isfinite(np.r_[begin,end,angle,bc,wc]).all() or min(bc,wc)<=0:raise ValueError('Invalid geometry')
    e=np.array([np.cos(angle),np.sin(angle)]);f=np.array([-np.sin(angle),np.cos(angle)])
    delta=end-begin;rows,cols=bs;nr,nc=ws;bi=[];wi=[];areas=[];outside=np.zeros(rows*cols)
    for axis,count,tangent_count,normal,tangent in ((0,rows,cols,e,f),(1,cols,rows,f,e)):
        for side in (-1,1):
            signed=float(delta@(side*normal))
            if (signed<=0 if incoming else signed>=0):continue
            for j in range(tangent_count):
                center=begin+side*count*bc/2*normal+(j+.5-tangent_count/2)*bc*tangent
                a=center-.5*bc*tangent;b=center+.5*bc*tangent
                poly=np.array([a,b,b+delta,a+delta]);full=abs(signed)*bc
                index=((0 if side<0 else rows-1)*cols+j) if axis==0 else j*cols+(0 if side<0 else cols-1)
                lo=np.floor((poly.min(0)-origin)/wc).astype(int);hi=np.floor((poly.max(0)-origin)/wc).astype(int);total=0.
                for r in range(max(0,lo[1]),min(nr-1,hi[1])+1):
                    for c in range(max(0,lo[0]),min(nc-1,hi[0])+1):
                        area=float(_area_in_box(poly,origin[0]+c*wc,origin[1]+r*wc,wc))
                        if area>0:bi.append(index);wi.append(r*nc+c);areas.append(area);total+=area
                outside[index]+=max(0.,full-total)
    return np.array(bi,dtype=np.int64),np.array(wi,dtype=np.int64),np.array(areas),outside

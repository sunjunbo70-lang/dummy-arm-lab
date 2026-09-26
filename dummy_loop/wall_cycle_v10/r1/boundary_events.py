"""Bracketed boundary-vertex / wall-grid crossing times for quadrature.
These cuts supplement adaptive checks; tangencies and all other topological
changes are not claimed to be exhaustively detected by this component.
"""
import numpy as np

def crossing_times(begin,end,alpha,beta):
 begin=np.asarray(begin,float);velocity=np.asarray(end,float)-begin;omega=beta-alpha
 vertices=np.array([(r,c) for r in (-.015,.015) for c in np.linspace(-.06,.06,25)]+[(r,c) for c in (-.06,.06) for r in np.linspace(-.015,.015,7)[1:-1]])
 def coords(t):
  a=alpha+t*omega;e=np.array([np.cos(a),np.sin(a)]);f=np.array([-np.sin(a),np.cos(a)])
  return begin+t*velocity+vertices[:,0,None]*e+vertices[:,1,None]*f
 n=max(1,int(np.ceil(abs(omega)/.02)),int(np.ceil(np.linalg.norm(velocity)/.005)))
 if n>4096:raise ValueError('Event bracketing budget exceeded')
 cuts=[0.,1.];origin=(-.25,0.)
 for j in range(n):
  lo=j/n;hi=(j+1)/n;x0=coords(lo);x1=coords(hi)
  for vertex in range(len(vertices)):
   for axis in (0,1):
    low,high=sorted((x0[vertex,axis],x1[vertex,axis]))
    for k in range(max(0,int(np.floor((low-origin[axis])/.005))+1),min(100,int(np.ceil((high-origin[axis])/.005)))+1):
     line=origin[axis]+k*.005;fl=x0[vertex,axis]-line;fh=x1[vertex,axis]-line
     if fl*fh>=0:continue
     left,right=lo,hi
     for _ in range(48):
      mid=(left+right)/2;fm=coords(mid)[vertex,axis]-line
      if fm==0:left=right=mid;break
      if fl*fm>0:left=mid;fl=fm
      else:right=mid
     cuts.append((left+right)/2)
 cuts=sorted(cuts);unique=[cuts[0]]
 for t in cuts[1:]:
  if t-unique[-1]>1e-12:unique.append(t)
 unique[-1]=1.
 return unique

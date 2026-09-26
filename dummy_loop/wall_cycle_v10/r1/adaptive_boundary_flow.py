"""Bounded, support-aware adaptive boundary quadrature candidate.
Compares sparse coarse/fine flux and endpoint geometric closure. This is not
a proof that every zero-net transient event is detected, nor material G0.
"""
import numpy as np
from .boundary_flow import instantaneous
from .overlap import planar_overlap
VERSION='v10r1.boundary_quadrature.adaptive_support.v2_global_rates'

def merge(a,b):
 result=[]
 for x,y in zip(a,b):
  d={}
  for m in (x,y):
   for bi,wi,v in zip(*m[:3]):d[(int(bi),int(wi))]=d.get((int(bi),int(wi)),0.)+v
  keys=sorted(d)
  result.append((np.array([k[0] for k in keys],int),np.array([k[1] for k in keys],int),np.array([d[k] for k in keys]),x[3]+y[3]))
 return result

def adaptive(begin,end,alpha,beta,area_tol=2.5e-13,max_depth=18,max_nodes=4095):
 begin=np.asarray(begin,float);end=np.asarray(end,float)
 if area_tol<=0 or max_depth<0 or max_nodes<1:raise ValueError('Invalid quadrature budget')
 stats=dict(nodes=0,leaves=0,max_depth=0,max_accepted_closure_m2=0.,max_accepted_difference_m2=0.)
 cache={}
 def coverage(t):
  if t not in cache:
   m=planar_overlap(begin+t*(end-begin),alpha+t*(beta-alpha),(6,24),.005,(100,100),.005,np.array([-.25,0.]))
   c=np.bincount(m[1],weights=m[2],minlength=10000)
   cache[t]=np.where(c<2.5e-17,0.,np.where(abs(c-.005**2)<2.5e-17,.005**2,c))
  return cache[t]
 # Keep the root trajectory derivatives fixed. Reconstructing derivatives
 # from two nearly equal subinterval endpoints loses significant digits.
 velocity=end-begin;omega=beta-alpha
 def sample(lo,hi):
  sums=[{},{}];outside=[np.zeros(144),np.zeros(144)]
  for node in (-1/np.sqrt(3),1/np.sqrt(3)):
   t=(lo+hi)/2+(hi-lo)*node/2
   maps,external=instantaneous(begin+t*velocity,alpha+t*omega,velocity,omega)
   for kind in (0,1):
    for key,rate in maps[kind].items():sums[kind][key]=sums[kind].get(key,0.)+rate*(hi-lo)/2
    outside[kind]+=external[kind]*(hi-lo)/2
  answer=[]
  for d,out in zip(sums,outside):
   keys=sorted(d);answer.append((np.array([k[0] for k in keys],int),np.array([k[1] for k in keys],int),np.array([d[k] for k in keys]),out))
  return answer
 def visit(lo,hi,coarse,depth):
  stats['nodes']+=1;stats['max_depth']=max(stats['max_depth'],depth)
  if stats['nodes']>max_nodes:raise RuntimeError('Adaptive boundary node budget exhausted')
  mid=(lo+hi)/2;left=sample(lo,mid);right=sample(mid,hi);fine=merge(left,right)
  difference=0.
  for c,f in zip(coarse,fine):
   d={(int(b),int(w)):float(v) for b,w,v in zip(*c[:3])}
   for b,w,v in zip(*f[:3]):d[(int(b),int(w))]=d.get((int(b),int(w)),0.)-v
   difference=max(difference,max(map(abs,d.values()),default=0.),float(abs(c[3]-f[3]).max(initial=0.)))
  incoming=np.bincount(fine[0][1],weights=fine[0][2],minlength=10000)
  outgoing=np.bincount(fine[1][1],weights=fine[1][2],minlength=10000)
  expected=coverage(hi)-coverage(lo)
  closure=float(abs(incoming-outgoing-expected).max())
  supported=not (((expected>2.5e-17)&(incoming==0)).any() or ((expected<-2.5e-17)&(outgoing==0)).any())
  # Roundoff allowance is geometric, distinct from the integration budget.
  budget=area_tol*(hi-lo)+2.5e-17
  if difference<=budget and closure<=budget and supported:
   stats['leaves']+=1;stats['max_accepted_closure_m2']=max(stats['max_accepted_closure_m2'],closure);stats['max_accepted_difference_m2']=max(stats['max_accepted_difference_m2'],difference)
   return fine
  if depth>=max_depth:raise RuntimeError(f'Adaptive boundary depth budget exhausted at {lo:.17g}:{hi:.17g}: support={supported}, closure={closure}, difference={difference}')
  return merge(visit(lo,mid,left,depth+1),visit(mid,hi,right,depth+1))
 result=visit(0.,1.,sample(0.,1.),0)
 return result,stats

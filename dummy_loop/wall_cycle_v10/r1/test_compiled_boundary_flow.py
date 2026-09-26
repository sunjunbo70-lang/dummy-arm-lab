import unittest
import numpy as np
from .boundary_flow import instantaneous as reference
from .compiled_boundary_flow import instantaneous as compiled
class CompiledBoundaryTests(unittest.TestCase):
 def test_random_rotations_translation_outside_and_zero(self):
  rng=np.random.default_rng(62006)
  for i in range(60):
   p=rng.uniform([-.3,-.03],[.3,.53]);a=rng.uniform(-3,3);v=rng.uniform(-.3,.3,2);w=rng.uniform(-2,2)
   if i==0:v[:]=0.;w=0.
   x,xo=reference(p,a,v,w);y,yo=compiled(p,a,v,w)
   for xd,yd in zip(x,y):
    keys=set(xd)|set(yd)
    self.assertLessEqual(max((abs(xd.get(k,0)-yd.get(k,0)) for k in keys),default=0.),2e-16)
   np.testing.assert_allclose(xo,yo,rtol=1e-12,atol=2e-16)
if __name__=='__main__':unittest.main()

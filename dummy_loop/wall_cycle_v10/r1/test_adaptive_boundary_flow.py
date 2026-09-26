import unittest
import numpy as np
from .adaptive_boundary_flow import adaptive
from .boundary_flow import integrate
class AdaptiveBoundaryTests(unittest.TestCase):
 def test_analytic_translation_area(self):
  maps,stats=adaptive([0.,.25],[.0002,.2501],0.,0.)
  for m in maps:self.assertAlmostEqual(m[2].sum()+m[3].sum(),.12*.0002+.03*.0001,places=14)
 def test_newborn_patch_support_previously_missed(self):
  lo=.31+42*.3/128;hi=.31+43*.3/128
  maps,stats=adaptive([0.,.25],[0.,.25],lo,hi)
  fine=integrate([0.,.25],[0.,.25],lo,hi,intervals=1024)
  for m,f in zip(maps,fine):
   x=np.bincount(m[1],weights=m[2],minlength=10000);y=np.bincount(f[1],weights=f[2],minlength=10000)
   self.assertLess(float(abs(x-y).max()),2.5e-13)
   self.assertFalse(((y>2.5e-17)&(x==0)).any())
 def test_tiny_motion_preserves_analytic_total(self):
  # Normal velocity must not be reconstructed separately at deep nodes.
  dx=1e-10;maps,_=adaptive([0.,.25],[dx,.25],0.,0.,area_tol=1e-20)
  for m in maps:self.assertAlmostEqual((m[2].sum()+m[3].sum())/(.12*dx),1.,places=12)
 def test_budget_exhaustion_is_explicit(self):
  with self.assertRaises(RuntimeError):adaptive([0.,.25],[.02,.27],.31,.61,area_tol=1e-20,max_depth=0)
if __name__=='__main__':unittest.main()

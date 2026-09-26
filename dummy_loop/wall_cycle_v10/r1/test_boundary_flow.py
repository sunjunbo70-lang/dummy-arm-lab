import unittest
import numpy as np
from .boundary_flow import integrate,instantaneous
class BoundaryFlowTests(unittest.TestCase):
 def test_pure_rotation_analytic_area(self):
  for omega in (-.7,.7):
   maps=integrate([0.,.25],[0.,.25],.31,.31+omega,intervals=8)
   expected=abs(omega)*(.03**2+.12**2)/4
   for m in maps:self.assertAlmostEqual(m[2].sum()+m[3].sum(),expected,places=14)
 def test_translation_total_and_zero_motion(self):
  for dx,dy in ((.02,.02),(-.02,.01)):
   for m in integrate([0.,.25],[dx,.25+dy],0.,0.,intervals=8):self.assertAlmostEqual(m[2].sum()+m[3].sum(),.12*abs(dx)+.03*abs(dy),places=14)
  for m in integrate([0.,.25],[0.,.25],.3,.3,intervals=1):self.assertEqual(m[2].sum()+m[3].sum(),0.)
 def test_reversing_path_swaps_entry_exit(self):
  forward=integrate([.23,.25],[.25,.27],.2,.8,intervals=8)
  backward=integrate([.25,.27],[.23,.25],.8,.2,intervals=8)
  for x,y in zip(forward,reversed(backward)):
   np.testing.assert_array_equal(x[0],y[0]);np.testing.assert_array_equal(x[1],y[1]);np.testing.assert_allclose(x[2],y[2],atol=1e-17,rtol=1e-10);np.testing.assert_allclose(x[3],y[3],atol=1e-17,rtol=1e-10)
if __name__=='__main__':unittest.main()

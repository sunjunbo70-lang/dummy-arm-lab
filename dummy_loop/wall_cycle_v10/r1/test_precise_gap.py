import unittest
import numpy as np
from .precise_gap_pressure import precise_gap_solve,PreciseGapPressure
from .compiled_material import gap_solve
class PreciseGapTests(unittest.TestCase):
 def test_smooth_uniform_pressure_root(self):
  B=np.full((6,24),.1);g,s=precise_gap_solve(B,0.,5.,.005,100.,1e-5,.02)
  self.assertAlmostEqual(s,5.,places=8)
  old,_=gap_solve(B,0.,5.,.005,100.,1e-5,.02)
  self.assertLessEqual(abs(g-old),(.02-1e-5)/2**20)
 def test_adaptive_advance_conserves(self):
  x=PreciseGapPressure(np.zeros((100,100)),np.full((6,24),.002*.005**2));v=x.total()
  x.contact([0,.25],.3,.2,2.);x.contact([.00003,.25],.30001,.2,2.);x.lift(.5)
  self.assertAlmostEqual(x.total(),v,places=14);self.assertGreaterEqual(x.wall.min(),0.)
if __name__=='__main__':unittest.main()

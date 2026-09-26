import unittest
import numpy as np
from .quarter_gap_pressure import QuarterGapPressure
from .stationary_boundary_pressure import StationaryBoundaryPressure
class QuarterGapTests(unittest.TestCase):
 def test_constant_controls_match_baseline(self):
  models=[c(np.zeros((100,100)),np.full((6,24),.002*.005**2)) for c in [QuarterGapPressure,StationaryBoundaryPressure]]
  for s in models:
   for i in range(4):s.contact([i*.0001,.25],.3+i*.0001,.2,2.)
   s.lift(.5)
  np.testing.assert_allclose(models[0].wall,models[1].wall,rtol=1e-12,atol=1e-20)
 def test_quarter_controls_and_conservation(self):
  s=QuarterGapPressure(np.zeros((100,100)),np.full((6,24),.002*.005**2));v=s.total();s.contact([0,.25],.3,.2,2.)
  calls=[];f=s._pressure
  def pressure(B,p,F):calls.append((p,F));return f(B,p,F)
  s._pressure=pressure;s.contact([.0001,.25],.3001,.3,4.)
  self.assertTrue(any(np.allclose(x,[.225,2.5]) for x in calls));self.assertTrue(any(np.allclose(x,[.275,3.5]) for x in calls))
  s.lift(.5);self.assertAlmostEqual(v,s.total(),places=14);self.assertGreaterEqual(s.wall.min(),0.)
if __name__=='__main__':unittest.main()

import unittest
import numpy as np
from .stationary_boundary_pressure import StationaryBoundaryPressure
class StationaryTests(unittest.TestCase):
 def test_identical_pose_full_contact_preserves_inventory(self):
  s=StationaryBoundaryPressure(np.full((100,100),.002*.005**2),np.full((6,24),.002*.005**2))
  s.contact([0,.25],.3,.2,2.);w=s.wall.copy();b=s.blade.copy();v=s.total()
  for _ in range(4):s.contact([0,.25],.3,.2,2.)
  np.testing.assert_array_equal(w,s.wall);np.testing.assert_array_equal(b,s.blade);self.assertEqual(v,s.total())
 def test_stationary_force_changes_pressure(self):
  s=StationaryBoundaryPressure(np.zeros((100,100)),np.full((6,24),.003*.005**2))
  s.contact([0,.25],.3,.2,1.);g=s.g0;s.contact([0,.25],.3,.2,15.)
  self.assertLessEqual(s.g0,g)
if __name__=='__main__':unittest.main()

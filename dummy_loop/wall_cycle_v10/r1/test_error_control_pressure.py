import unittest
import numpy as np
from .error_control_pressure import ErrorControlPressure
class ErrorControlTests(unittest.TestCase):
 def test_conservative_nonnegative_advance(self):
  s=ErrorControlPressure(np.zeros((100,100)),np.full((6,24),.002*.005**2));v=s.total()
  s.contact([0,.25],.3,.2,2.);s.contact([.00005,.25],.30001,.2,2.);s.lift(.5)
  self.assertAlmostEqual(v,s.total(),places=14);self.assertGreaterEqual(s.wall.min(),0.);self.assertGreaterEqual(s.boundary_stats['contact_error_nodes'],3)
 def test_failure_does_not_mutate_original(self):
  s=ErrorControlPressure(np.full((100,100),.002*.005**2),np.full((6,24),.002*.005**2));s.contact([0,.25],.3,.2,2.)
  s.max_contact_depth=0;s.error_per_cell_travel_mm=0.;w=s.wall.copy();b=s.blade.copy()
  with self.assertRaises(RuntimeError):s.contact([.0003,.2502],.301,.25,3.)
  np.testing.assert_array_equal(w,s.wall);np.testing.assert_array_equal(b,s.blade)
if __name__=='__main__':unittest.main()

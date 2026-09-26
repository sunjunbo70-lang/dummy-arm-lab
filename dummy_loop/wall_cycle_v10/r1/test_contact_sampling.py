import unittest
import numpy as np
from .contact_sampling import sample_contact,check_continuation
from .actions import decode
from ...wall_cycle_v09.trajectory_action import Contact
class SamplingTests(unittest.TestCase):
 def test_curved_path_swept_bound_and_endpoints(self):
  c=Contact(np.array([[0,.2],[.08,.3],[-.05,.3],[.1,.2]]),np.array([0.,1.]),np.zeros(3),np.ones(3),np.full(3,.05))
  s=sample_contact(c,max_swept_m=.001,tool_radius_m=.062,max_angular_speed_rad_s=1.)
  self.assertLessEqual(s.swept_bound_m.max(),.001);np.testing.assert_array_equal(s.xy[[0,-1]],c.points[[0,-1]])
  self.assertTrue((np.diff(s.time_s)>0).all())
 def test_pure_rotation_has_duration(self):
  c=Contact(np.tile([0,.25],(4,1)),np.array([0.,1.]),np.zeros(3),np.ones(3),np.full(3,.05))
  s=sample_contact(c,max_swept_m=.001,tool_radius_m=.062,max_angular_speed_rad_s=.5)
  self.assertAlmostEqual(s.time_s[-1],2.)
 def test_decoded_continuation_and_discontinuity(self):
  a=decode(np.zeros(19));end=a.at(1.);b=decode(np.zeros(19),current=(end[0],*end[2:]))
  self.assertTrue(check_continuation(a,b));b.forces[0]+=.1;self.assertFalse(check_continuation(a,b))
 def test_depth_failure_explicit(self):
  with self.assertRaises(RuntimeError):sample_contact(decode(np.zeros(19)),max_swept_m=1e-8,tool_radius_m=.062,max_angular_speed_rad_s=1.,max_depth=0)
if __name__=='__main__':unittest.main()

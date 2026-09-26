import unittest
import numpy as np
from .joint_path import plan_joint_path,PathRejected
class JointPathTests(unittest.TestCase):
 def run_path(self,**changes):
  opts=dict(solve=lambda p,R,q:p.copy(),fk=lambda q:(q,np.eye(3)),is_clear=lambda q:True,lo=np.full(3,-2.),hi=np.full(3,2.),max_speed=np.ones(3),max_jump=1.,collision_step=.05,position_tolerance=1e-5,rotation_tolerance=1e-5)
  opts.update(changes)
  return plan_joint_path([(np.array([.2,0,0]),np.eye(3)),(np.array([.4,0,0]),np.eye(3))],[1.,2.],np.zeros(3),**opts)
 def test_path_continuity_and_input_seed_ownership(self):
  def solve(p,R,q):q[:]=p;return q
  result=self.run_path(solve=solve)
  np.testing.assert_allclose(result.q[:,0],[.2,.4]);self.assertGreaterEqual(result.collision_samples,9)
 def test_interior_collision_rejected(self):
  with self.assertRaisesRegex(PathRejected,'Interpolated collision'):self.run_path(is_clear=lambda q:not .08<q[0]<.12)
 def test_speed_rejected(self):
  with self.assertRaisesRegex(PathRejected,'speed'):self.run_path(max_speed=np.full(3,.1))
 def test_rotation_residual_rejected(self):
  with self.assertRaisesRegex(PathRejected,'Pose residual'):self.run_path(fk=lambda q:(q,np.diag([-1.,-1.,1.])))
 def test_ik_jump_rejected(self):
  with self.assertRaisesRegex(PathRejected,'Joint jump'):self.run_path(solve=lambda p,R,q:np.ones(3)*1.5)
if __name__=='__main__':unittest.main()

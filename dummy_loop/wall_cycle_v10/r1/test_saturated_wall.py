import unittest
import numpy as np
from .saturated_wall_exchange import exchange
from .coupled_boundary_exchange import exchange as original
class WallCapTests(unittest.TestCase):
 def args(self):
  entry=(np.array([0]),np.array([0]),np.array([.1]),np.zeros(1))
  exit=(np.array([0]),np.array([0]),np.array([.2]),np.zeros(1))
  return (np.array([.1]),np.array([1.]),np.array([.5]),np.array([.6]),1.,entry,exit,np.array([2.]))
 def test_no_cap_matches_reference(self):
  a=exchange(*self.args());b=original(*self.args())
  for x,y in zip(a[:3],b[:3]):np.testing.assert_array_equal(x,y)
 def test_shed_is_removed_from_blade_and_ledger(self):
  w,b,out,info=exchange(*self.args(),max_wall_height=.3)
  self.assertGreater(info['shed_m3'],0)
  self.assertLessEqual(w[0],.6*.3+1e-15)
  self.assertGreaterEqual(b[0],0)
  self.assertAlmostEqual(w.sum()+b.sum()+out+info['shed_m3'],1.1,places=14)
if __name__=='__main__':unittest.main()

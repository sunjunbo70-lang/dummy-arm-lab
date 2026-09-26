import unittest
import numpy as np
from unittest.mock import patch
from .bounded_edge_pressure import BoundedEdgePressure
class EdgeTests(unittest.TestCase):
 def state(self):return BoundedEdgePressure(np.zeros((100,100)),np.zeros((6,24)))
 def test_real_pressure_path_conserves(self):
  s=self.state();s.wall[:]=.002*s.wc**2;s.blade[:]=.002*s.bc**2;total=s.total()
  for i in range(4):s.contact([i*.0001,.25],.3+i*.001,.2,2.)
  s.lift(.5);self.assertAlmostEqual(s.total(),total,places=14);self.assertGreaterEqual(s.wall.min(),0.)
 def test_roundoff_recipient_excluded_without_losing_donor(self):
  s=self.state();area=s.bc**2;s.mapping=(np.array([0]),np.array([0]),np.array([area]),np.zeros(144))
  bi=np.r_[0,np.arange(24)];wi=np.r_[0,np.arange(1,25)];a=np.r_[1e-18,np.full(24,area)];a[1]-=1e-18
  with patch('dummy_loop.wall_cycle_v10.r1.bounded_edge_pressure.planar_overlap',return_value=(bi,wi,a,np.zeros(24))):s._edge_deposit(np.full(24,1e-8),[0,.25],0,-1)
  self.assertEqual(s.wall.flat[0],0.);self.assertAlmostEqual(s.wall.sum(),24e-8,places=20)
 def test_macroscopic_overlap_rejected(self):
  s=self.state();area=s.bc**2;s.mapping=(np.array([0]),np.array([0]),np.array([area]),np.zeros(144))
  with patch('dummy_loop.wall_cycle_v10.r1.bounded_edge_pressure.planar_overlap',return_value=(np.array([0]),np.array([0]),np.array([1e-8]),np.full(24,area))):
   with self.assertRaises(ValueError):s._edge_deposit(np.full(24,1e-8),[0,.25],0,-1)
if __name__=='__main__':unittest.main()

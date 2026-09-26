import unittest
import numpy as np
from .coupled_boundary_exchange import exchange

def m(b,w,a,n=1,out=None):return np.array(b,int),np.array(w,int),np.array(a,float),np.zeros(n) if out is None else np.array(out,float)
class CoupledBoundaryTests(unittest.TestCase):
 def test_uniform_moving_wall_blade_equilibrium(self):
  # Equal-density boundary reservoirs remain uniform despite changing areas.
  # Blade balanced entry/exit, walls one shrinks and the other expands.
  w,b,l,info=exchange([.002,.002],[.002],[1.,1.],[.8,1.2],1.,m([0],[0],[.2]),m([0],[1],[.2]),.01,tol=1e-16)
  np.testing.assert_allclose(w,[.0016,.0024],atol=1e-15);np.testing.assert_allclose(b,[.002],atol=1e-15)
  self.assertAlmostEqual(w.sum()+b.sum()+l,.006,places=14)
 def test_gap_limits_deposition_and_retains_excess(self):
  w,b,l,_=exchange([0.],[.01],[1.],[1.5],1.,m([],[],[]),m([0],[0],[.5]),.001)
  self.assertAlmostEqual(w[0],.0005);self.assertAlmostEqual(b[0],.0095)
 def test_empty_blade_cannot_deposit_and_external_loss_is_booked(self):
  for stock in (0.,.001):
   w,b,l,_=exchange([0.],[stock],[1.],[1.],1.,m([],[],[]),m([],[],[],out=[.5]),.1)
   self.assertGreaterEqual(b[0],0.);self.assertAlmostEqual(b.sum()+l,stock,places=15)
 def test_multicell_loop_nonnegative_and_conservative(self):
  e=m([0,1,1],[0,0,1],[.1,.1,.1],2);x=m([0,1],[1,0],[.2,.1],2)
  w,b,l,_=exchange([.001,.003],[.002,.0001],[1.,1.],[.9,1.1],1.,e,x,.004)
  self.assertGreaterEqual(min(w.min(),b.min(),l),0.);self.assertAlmostEqual(w.sum()+b.sum()+l,.0061,places=14)
 def test_excess_step_and_inconsistent_geometry_rejected(self):
  with self.assertRaises(ValueError):exchange([0.],[.001],[1.],[3.],1.,m([],[],[]),m([0],[0],[2.]),.1)
  with self.assertRaises(ValueError):exchange([0.],[.001],[1.],[1.],1.,m([],[],[]),m([0],[0],[.2]),.1)
if __name__=='__main__':unittest.main()

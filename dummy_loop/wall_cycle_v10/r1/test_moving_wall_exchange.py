import unittest
import numpy as np
from .moving_wall_exchange import transfer

class MovingWallTests(unittest.TestCase):
 def test_pure_pickup_preserves_remaining_density_and_closes(self):
  remaining,picked,a=transfer([.002,.002],[1.,1.],[.7,1.],[0.,0.],[0.,0.])
  np.testing.assert_allclose(remaining,.002*a,atol=1e-18)
  np.testing.assert_allclose(remaining+picked,.002,atol=1e-18)
 def test_uniform_density_geometric_conservation_including_birth(self):
  a=np.array([1.,1.,0.,1.]);e=np.array([.8,.2,.2,.3]);x=np.array([.2,.8,.8,.3]);h=.002
  w,p,b=transfer(h*a,a,e,x,h*x)
  np.testing.assert_allclose(w,h*b,atol=2e-18)
  np.testing.assert_allclose(p,h*e,atol=2e-18)
 def test_fixed_area_throughflow_exact_exponential(self):
  w,p,a=transfer(2.,1.,.8,.8,.4)
  self.assertAlmostEqual(float(w),.5+1.5*np.exp(-.8),places=14)
  self.assertAlmostEqual(float(w+p),2.4,places=14)
 def test_semigroup_for_linear_area_and_fixed_rates(self):
  rng=np.random.default_rng(62003);a=rng.uniform(.5,1.5,100);e=rng.uniform(0,.4,100);x=rng.uniform(0,.4,100);d=x*rng.uniform(0,2,100);v=a*rng.uniform(0,2,100)
  full=transfer(v,a,e,x,d)
  half=transfer(v,a,e/2,x/2,d/2);end=transfer(half[0],half[2],e/2,x/2,d/2)
  np.testing.assert_allclose(full[0],end[0],atol=2e-14)
  np.testing.assert_allclose(full[1],half[1]+end[1],atol=2e-14)
 def test_final_area_must_match_flow_balance(self):
  with self.assertRaises(ValueError):transfer(1.,1.,.2,.1,0.,final_area=.5)
  w,p,a=transfer(.002,1.,.2,.1,0.,final_area=.9)
  self.assertAlmostEqual(float(w+p),.002)
 def test_invalid_capacity_and_zero_area_throughflow_rejected(self):
  for args in ((1.,0.,0.,1.,0.),(1.,1.,2.,0.,0.),(0.,0.,1.,1.,0.),(1.,1.,.5,0.,1.)):
   with self.assertRaises(ValueError):transfer(*args)

if __name__=='__main__':unittest.main()

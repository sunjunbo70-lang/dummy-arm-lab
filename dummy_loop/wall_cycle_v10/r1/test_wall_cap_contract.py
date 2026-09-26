import unittest
import numpy as np
from .saturated_wall_exchange import exchange
from .fixed_cap_pressure import FixedCapPressure
class WallCapContract(unittest.TestCase):
 def test_random_positive_exchange_ledger_capacity(self):
  rng=np.random.default_rng(260926)
  for _ in range(100):
   a0=rng.uniform(.2,.7);incoming=rng.uniform(0,.1);outgoing=rng.uniform(0,.1);a1=a0-incoming+outgoing
   cap=rng.uniform(.01,.1);w=np.array([rng.uniform(0,cap)*a0]);b=np.array([rng.uniform(0,.2)])
   ent=(np.array([0]),np.array([0]),np.array([incoming]),np.zeros(1))
   ext=(np.array([0]),np.array([0]),np.array([outgoing]),np.array([.01]))
   wall,blade,lost,info=exchange(w,b,np.array([a0]),np.array([a1]),1.,ent,ext,np.array([.2]),max_wall_height=cap)
   self.assertGreaterEqual(wall[0],0);self.assertGreaterEqual(blade[0],0)
   self.assertLessEqual(wall[0],cap*a1+1e-14)
   self.assertAlmostEqual(float(w.sum()+b.sum()),float(wall.sum()+blade.sum()+lost+info['shed_m3']),places=13)
 def test_fixed_cap_contact_lift_conserves(self):
  x=FixedCapPressure(np.zeros((100,100)),np.full((6,24),.002*.005**2));v=x.total()
  x.contact([0,.25],.3,.2,2.);x.contact([.00003,.25],.30001,.2,2.);x.lift(.5)
  self.assertAlmostEqual(x.total(),v,places=14)
  self.assertGreaterEqual(min(x.wall.min(),x.blade.min(),x.bead.min()),0.)
if __name__=='__main__':unittest.main()

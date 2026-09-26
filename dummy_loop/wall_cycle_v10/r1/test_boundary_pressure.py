import unittest
import numpy as np
from .boundary_pressure import BoundaryPressure
class BoundaryPressureTests(unittest.TestCase):
 def test_pressure_motion_and_lift_conserve_full_inventory(self):
  state=BoundaryPressure(np.zeros((100,100)),np.full((6,24),.002*.005**2))
  total=state.total()
  for i in range(3):state.contact([i*.0001,.25],.3+i*.0001,.2,2.)
  state.lift(.5)
  self.assertAlmostEqual(state.total(),total,places=14)
  self.assertGreaterEqual(min(state.wall.min(),state.blade.min(),state.bead.min(),state.dropped,state.outside),-1e-20)
 def test_loaded_wall_initial_pickup_never_overdraws(self):
  state=BoundaryPressure(np.full((100,100),.002*.005**2),np.full((6,24),.002*.005**2))
  total=state.total()
  for i in range(3):state.contact([i*.0001,.25],.3+i*.0001,.2,2.)
  state.lift(.5)
  self.assertGreaterEqual(min(state.wall.min(),state.blade.min()),0.)
  self.assertAlmostEqual(state.total(),total,places=14)
 def test_nonstandard_grid_rejected(self):
  with self.assertRaises(ValueError):BoundaryPressure(np.zeros((10,10)),np.zeros((6,24)))
if __name__=='__main__':unittest.main()

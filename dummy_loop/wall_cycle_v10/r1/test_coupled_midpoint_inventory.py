import unittest
import numpy as np
from .coupled_midpoint_inventory import CoupledMidpointPressure

class CoupledMidpointTests(unittest.TestCase):
    def test_rotating_boundary_inventory_conserved(self):
        rng=np.random.default_rng(983)
        s=CoupledMidpointPressure(rng.uniform(0,1e-7,(100,100)),rng.uniform(0,1e-7,(6,24)));initial=s.total()
        for k in range(40):
            s.contact([.23+.01*np.sin(k/10),.25],k*.01,.2,5.)
            self.assertAlmostEqual(s.total(),initial,delta=1e-16)
            self.assertGreaterEqual(min(s.wall.min(),s.blade.min(),s.bead.min()),-1e-18)
        s.lift(.75);self.assertAlmostEqual(s.total(),initial,delta=1e-16)
    def test_stationary_geometry_keeps_inventory(self):
        rng=np.random.default_rng(211)
        s=CoupledMidpointPressure(rng.uniform(0,1e-7,(100,100)),rng.uniform(0,1e-7,(6,24)))
        s.move([0.,.25],.23,.002)
        wall=s.wall.copy();blade=s.blade.copy()
        for _ in range(20):s.move([0.,.25],.23,.002)
        np.testing.assert_allclose(s.wall,wall,atol=1e-18,rtol=0)
        np.testing.assert_allclose(s.blade,blade,atol=1e-18,rtol=0)
if __name__=='__main__':unittest.main()

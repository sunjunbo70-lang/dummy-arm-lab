import unittest
import numpy as np
from .pressure_inventory import PressureInventory

class PressureTests(unittest.TestCase):
    def make(self):
        rng=np.random.default_rng(2301)
        return PressureInventory(rng.uniform(.001,.004,(100,100))*.005**2,np.full((6,24),.004*.005**2))

    def test_sequence_ledger(self):
        s=self.make();total=s.total()
        for k in range(60):
            s.contact([.01*np.sin(k/20),.25+k*.0002],k*.004,.2,5.)
            self.assertAlmostEqual(s.total(),total,delta=1e-16)
            self.assertGreaterEqual(min(s.wall.min(),s.blade.min(),s.bead.min()),-1e-18)
        s.lift(.75)
        self.assertAlmostEqual(s.total(),total,delta=1e-16)

    def test_stationary_flux_zero(self):
        s=self.make();s.contact([0,.25],.1,.2,5.)
        before=(s.wall.copy(),s.blade.copy(),s.bead.copy())
        for _ in range(20):
            info=s.contact([0,.25],.1,.2,5.)
            self.assertEqual(info['exchange'],0.)
        for a,b in zip(before,(s.wall,s.blade,s.bead)):
            np.testing.assert_allclose(a,b,atol=1e-17,rtol=0)

    def test_pure_rotation_has_flux(self):
        s=self.make();s.contact([0,.25],0,.2,5.)
        self.assertGreater(s.contact([0,.25],.01,.2,5.)['exchange'],0.)

    def test_no_implicit_loading(self):
        s=PressureInventory(np.zeros((100,100)),np.zeros((6,24)))
        for k in range(20):s.contact([k*.001,.25],k*.02,.1,5.)
        s.lift(.75)
        self.assertEqual(s.total(),0.)

if __name__=='__main__':unittest.main()

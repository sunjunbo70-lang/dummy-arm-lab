import unittest
import numpy as np
from .ordered_inventory import OrderedInventory,OrderedPressure
from .contact_inventory import turnover

class OrderedTests(unittest.TestCase):
    def test_exit_material_not_picked_up_in_same_transition(self):
        # Dry wall, wet tool. A turn has entry and exit fragments in the same cell.
        s=OrderedInventory(np.zeros((100,100)),np.full((6,24),1e-7))
        s.move([.001,.251],.3,.001)
        old,new,leave,enter=turnover(s.pose,(np.array([.001,.251]),.35))
        exit_by_wall=np.bincount(old[1],weights=leave,minlength=10000)
        entry_by_wall=np.bincount(new[1],weights=enter,minlength=10000)
        self.assertTrue(np.any((exit_by_wall>1e-12)&(entry_by_wall>1e-12)))
        expected=s.blade.copy()-np.bincount(old[0],weights=leave*.001,minlength=144).reshape(6,24)
        s.move([.001,.251],.35,.001)
        np.testing.assert_allclose(s.blade,expected,atol=1e-19,rtol=0)

    def test_coupled_conservation(self):
        rng=np.random.default_rng(555)
        s=OrderedPressure(rng.uniform(0,1e-7,(100,100)),rng.uniform(0,1e-7,(6,24)))
        initial=s.total()
        for k in range(100):
            s.contact([.23+.01*np.sin(k/10),.25],k*.01,.2,5.)
            self.assertGreaterEqual(min(s.wall.min(),s.blade.min(),s.bead.min()),-1e-18)
            self.assertAlmostEqual(s.total(),initial,delta=1e-16)
        s.lift(.75)
        self.assertAlmostEqual(s.total(),initial,delta=1e-16)

if __name__=='__main__':unittest.main()

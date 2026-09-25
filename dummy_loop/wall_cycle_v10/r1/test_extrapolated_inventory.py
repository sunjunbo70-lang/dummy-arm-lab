import unittest
import numpy as np
from .extrapolated_inventory import ExtrapolatedPressure

class ExtrapolationTests(unittest.TestCase):
    def test_conservative_nonnegative_and_irreversible(self):
        rng=np.random.default_rng(720)
        s=ExtrapolatedPressure(rng.uniform(0,1e-7,(100,100)),rng.uniform(0,1e-7,(6,24)))
        total=s.total();outside=0.;dropped=0.
        for k in range(30):
            s.contact([.24+.01*np.sin(k/10),.25],k*.01,.2,5.)
            self.assertAlmostEqual(s.total(),total,delta=1e-16)
            self.assertGreaterEqual(min(s.wall.min(),s.blade.min(),s.bead.min()),-1e-18)
            self.assertGreaterEqual(s.outside,outside);self.assertGreaterEqual(s.dropped,dropped)
            outside,dropped=s.outside,s.dropped
        self.assertEqual(s.accepted_extrapolations+s.fallback_steps,29)
        s.lift(.75);self.assertAlmostEqual(s.total(),total,delta=1e-16)

    def test_empty_stays_empty(self):
        s=ExtrapolatedPressure(np.zeros((100,100)),np.zeros((6,24)))
        for k in range(10):s.contact([.001*k,.25],.01*k,.2,5.)
        s.lift(.75);self.assertEqual(s.total(),0.)

if __name__=='__main__':unittest.main()


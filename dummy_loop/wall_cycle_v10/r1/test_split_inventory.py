import unittest
import numpy as np
from .split_inventory import SplitPressure

class SplitTests(unittest.TestCase):
    def test_flux_conserves_and_positive_euler_stage(self):
        rng=np.random.default_rng(831)
        s=SplitPressure(np.zeros((100,100)),np.zeros((6,24)))
        for _ in range(100):
            b=rng.uniform(0,.01,(6,24));q=rng.uniform(0,.02,24)
            db,dq,trail=s._flux(b,q,rng.uniform(0,.6),rng.uniform(.5,15),.5)
            self.assertAlmostEqual(db.sum()+dq.sum()+trail.sum(),0.,delta=1e-14)
            self.assertGreaterEqual((b+.2*db).min(),-1e-18)
            self.assertGreaterEqual((q+.2*dq).min(),-1e-18)

    def test_sequence_conservation(self):
        rng=np.random.default_rng(44)
        s=SplitPressure(rng.uniform(0,1e-7,(100,100)),rng.uniform(0,1e-7,(6,24)))
        total=s.total()
        for k in range(50):
            s.contact([.23+.01*np.sin(k/10),.25],k*.01,.2,5.)
            self.assertAlmostEqual(s.total(),total,delta=1e-16)
            self.assertGreaterEqual(min(s.wall.min(),s.blade.min(),s.bead.min()),-1e-18)
        s.lift(.75);self.assertAlmostEqual(s.total(),total,delta=1e-16)

if __name__=='__main__':unittest.main()

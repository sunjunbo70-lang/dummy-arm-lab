import unittest
import numpy as np
from .contact_inventory import turnover,ContactInventory

class InventoryTests(unittest.TestCase):
    def test_partition_identity(self):
        rng=np.random.default_rng(721)
        for _ in range(100):
            p=rng.uniform([-.3,-.02],[.3,.52]);a=rng.uniform(-3,3)
            q=p+rng.uniform(-.012,.012,2);b=a+rng.uniform(-.3,.3)
            old,new,leave,enter=turnover((p,a),(q,b))
            hist=lambda m,w:np.bincount(m[1],weights=w,minlength=10000)
            np.testing.assert_allclose(hist(old,old[2])-hist(old,leave)+hist(new,enter),hist(new,new[2]),atol=2e-16,rtol=0)
            self.assertTrue(np.all(leave<=old[2]+1e-18))
            self.assertTrue(np.all(enter<=new[2]+1e-18))

    def test_stationary_no_repeated_pickup(self):
        s=ContactInventory(np.full((100,100),5e-8),np.zeros((6,24)))
        s.move([.001,.251],.34,.001)
        wall,blade=s.wall.copy(),s.blade.copy()
        for _ in range(50):s.move([.001,.251],.34,.001)
        np.testing.assert_allclose(s.wall,wall,atol=1e-18,rtol=0)
        np.testing.assert_allclose(s.blade,blade,atol=1e-18,rtol=0)

    def test_rotating_sequence_ledger(self):
        rng=np.random.default_rng(802)
        s=ContactInventory(rng.uniform(0,5e-8,(100,100)),rng.uniform(0,1e-7,(6,24)))
        initial=s.total()
        for k in range(100):
            s.move([.23+.02*np.sin(k*.1),.25+.02*np.cos(k*.1)],k*.03,.001)
            self.assertGreaterEqual(s.wall.min(),-1e-18)
            self.assertGreaterEqual(s.blade.min(),-1e-18)
            self.assertAlmostEqual(s.total(),initial,delta=1e-16)
        s.lift(.75)
        self.assertAlmostEqual(s.total(),initial,delta=1e-16)
        self.assertGreater(s.outside,0.)

    def test_straight_pickup_subdivision(self):
        results=[]
        for n in [2,10,100]:
            s=ContactInventory(np.full((100,100),5e-8),np.zeros((6,24)))
            for x in np.linspace(0,.02,n):s.move([x,.25],0,0.)
            results.append(s)
        for s in results[1:]:
            np.testing.assert_allclose(s.wall,results[0].wall,atol=1e-17,rtol=0)
            self.assertAlmostEqual(s.blade.sum(),results[0].blade.sum(),delta=1e-17)

if __name__=='__main__':unittest.main()

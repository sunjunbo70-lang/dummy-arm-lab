import unittest
import numpy as np
from .overlap import planar_overlap, deposit, gather

class OverlapTests(unittest.TestCase):
    def mapping(self, p, angle, bs=(6,24), bc=.005, ws=(100,100), wc=.005):
        return planar_overlap(np.array(p, dtype=float), angle, bs, bc, ws, wc, np.array([-.25,0.]))

    def test_random_conservation_and_no_double_ownership(self):
        rng = np.random.default_rng(9001)
        for _ in range(200):
            p = rng.uniform([-.32,-.08],[.32,.58])
            m = self.mapping(p, rng.uniform(-np.pi,np.pi))
            bi, wi, area, outside = m
            np.testing.assert_allclose(np.bincount(bi, weights=area, minlength=144)+outside, .005**2, atol=1e-16, rtol=0)
            self.assertLessEqual(np.bincount(wi, weights=area, minlength=10000).max(), .005**2 + 1e-16)
            v = rng.uniform(0,1e-7,(6,24))
            wall,lost = deposit(v,m,.005,(100,100))
            self.assertAlmostEqual(wall.sum()+lost,v.sum(),delta=1e-16)
            w = rng.uniform(0,1e-7,(100,100))
            b,left = gather(w,m,.005,(6,24))
            self.assertGreaterEqual(left.min(),-1e-18)
            self.assertAlmostEqual(b.sum()+left.sum(),w.sum(),delta=1e-16)

    def test_aligned_identity(self):
        m = self.mapping([0,.25],0)
        v = np.arange(144,dtype=float).reshape(6,24)*1e-10
        wall,lost = deposit(v,m,.005,(100,100))
        b,left = gather(wall,m,.005,(6,24))
        np.testing.assert_allclose(b,v,atol=1e-20,rtol=1e-10)
        self.assertLess(abs(left.sum()),1e-18)
        self.assertLess(lost,1e-18)

    def test_rotation_continuity_at_grid_alignment(self):
        v=np.ones((6,24))*1e-7
        base=deposit(v,self.mapping([0,.25],0),.005,(100,100))[0]
        diffs=[]
        for angle in [1e-3,1e-4,1e-5]:
            w=deposit(v,self.mapping([0,.25],angle),.005,(100,100))[0]
            diffs.append(np.abs(w-base).sum())
        self.assertLess(diffs[1],diffs[0]*.11)
        self.assertLess(diffs[2],diffs[1]*.11)

    def test_unequal_grid_sizes(self):
        for bc,wc in [(.003,.005),(.011,.005),(.005,.003)]:
            m=self.mapping([.01,.2],.731,bc=bc,wc=wc)
            bi,wi,area,out=m
            np.testing.assert_allclose(np.bincount(bi,weights=area,minlength=144)+out,bc**2,atol=1e-16,rtol=0)
            self.assertLessEqual(np.bincount(wi,weights=area).max(),wc**2+1e-16)

if __name__=='__main__': unittest.main()

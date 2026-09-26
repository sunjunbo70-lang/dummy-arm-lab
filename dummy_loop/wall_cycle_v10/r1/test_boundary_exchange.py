import unittest
import numpy as np
from .boundary_exchange import exchange
from .boundary_flow import instantaneous

class BoundaryExchangeTests(unittest.TestCase):
    def test_analytic_two_reservoir_and_semigroup(self):
        maps=[{(0,0):2.},{(0,0):3.}];out=[np.zeros(1),np.zeros(1)]
        w,b,l=exchange([1.],[0.],[1.],1.,maps,out,.4)
        expected=.6+.4*np.exp(-5*.4)
        self.assertAlmostEqual(w[0],expected,places=14)
        self.assertAlmostEqual(w.sum()+b.sum()+l,1.,places=14)
        w2,b2,l2=exchange([1.],[0.],[1.],1.,maps,out,.2)
        w2,b2,l2=exchange(w2,b2,[1.],1.,maps,out,.2,lost=l2)
        np.testing.assert_allclose(np.r_[w,b,l],np.r_[w2,b2,l2],atol=1e-14)

    def test_outside_sink_and_empty_incoming(self):
        out=[np.array([900.]),np.array([2.])]
        w,b,l=exchange([0.],[1.],[1.],1.,[{},{}],out,.3,lost=.7)
        self.assertAlmostEqual(b[0],np.exp(-.6),places=14)
        self.assertAlmostEqual(w.sum()+b.sum()+l,1.7,places=14)

    def test_real_rotating_flux_nonnegative_conservative(self):
        maps,out=instantaneous([.245,.25],.4,[.1,.04],.6)
        rng=np.random.default_rng(62001)
        wall=rng.uniform(0,1e-7,10000);blade=rng.uniform(0,1e-7,144)
        original=wall.copy();initial=wall.sum()+blade.sum()
        w,b,l=exchange(wall,blade,np.full(10000,2e-5),2.5e-5,maps,out,.002)
        self.assertGreaterEqual(min(w.min(),b.min(),l),0.)
        self.assertAlmostEqual(w.sum()+b.sum()+l,initial,places=15)
        np.testing.assert_array_equal(wall,original)
        self.assertGreater(l,0.)

    def test_zero_capacity_stiffness_and_negative_inputs_rejected(self):
        maps=[{(0,0):1.},{}];out=[np.zeros(1),np.zeros(1)]
        for area in (0.,1e-12):
            with self.assertRaises(ValueError):exchange([1.],[0.],[area],1.,maps,out,1.)
        with self.assertRaises(ValueError):exchange([-1.],[0.],[1.],1.,maps,out,1.)

if __name__=='__main__':unittest.main()

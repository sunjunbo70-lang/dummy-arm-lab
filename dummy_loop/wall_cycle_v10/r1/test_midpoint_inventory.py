import unittest
import numpy as np
from .midpoint_inventory import MidpointGeometryPressure,exit_volume

class MidpointTests(unittest.TestCase):
    def test_midpoint_donor_decay_converges_second_order(self):
        errors=[]
        for n in (10,20,40):
            v=1.
            for _ in range(n):v-=exit_volume(v,0.,1/n,1.,100.)
            errors.append(abs(v-np.exp(-1)))
        self.assertGreater(errors[0]/errors[1],3.9);self.assertGreater(errors[1]/errors[2],3.9)
    def test_random_exchange_nonnegative_and_capped(self):
        rng=np.random.default_rng(54)
        v=rng.random(1000);p=rng.random(1000);leave=rng.random(1000);gap=rng.random(1000)
        e=exit_volume(v,p,leave,1.,gap)
        self.assertTrue((e<=leave*gap+1e-15).all());self.assertTrue((v+p-e>=0).all())
    def test_rotation_and_boundary_conserve(self):
        rng=np.random.default_rng(53)
        s=MidpointGeometryPressure(rng.uniform(0,1e-7,(100,100)),rng.uniform(0,1e-7,(6,24)))
        total=s.total()
        for k in range(40):
            s.contact([.23+.01*np.sin(k/10),.25],k*.01,.2,5.)
            self.assertAlmostEqual(s.total(),total,delta=1e-16)
            self.assertGreaterEqual(min(s.wall.min(),s.blade.min(),s.bead.min()),-1e-18)
        s.lift(.75);self.assertAlmostEqual(s.total(),total,delta=1e-16)
if __name__=='__main__':unittest.main()

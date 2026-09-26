import unittest
import numpy as np
from .translation_boundary import boundary_map
from .contact_inventory import turnover

class BoundaryTests(unittest.TestCase):
    def test_diagonal_area_and_endpoint_missing_sliver(self):
        a=np.array([0.,.25]);d=np.array([.002,.001]);b=a+d
        m=boundary_map(a,b,0.);exact=.12*d[0]+.03*d[1]
        self.assertAlmostEqual(m[2].sum(),exact,places=15)
        _,_,_,enter=turnover((a,0.),(b,0.))
        self.assertAlmostEqual(exact-enter.sum(),d[0]*d[1],places=15)
    def test_partition_invariance_on_wall_and_blade_indices(self):
        a=np.array([0.,.25]);b=a+np.array([.02,.02]);full=boundary_map(a,b,.27)
        def dense(m):return np.bincount(m[0]*10000+m[1],weights=m[2],minlength=1440000)
        pieces=np.zeros(1440000)
        for i in range(8):pieces+=dense(boundary_map(a+(b-a)*i/8,a+(b-a)*(i+1)/8,.27))
        np.testing.assert_allclose(dense(full),pieces,atol=1e-17,rtol=1e-10)
    def test_incoming_outgoing_and_outside_ledger(self):
        for d in ([.002,.003],[-.002,.003],[.002,-.003],[-.002,-.003]):
            a=np.array([.249,.25]);b=a+d;angle=.4;e=np.array([np.cos(angle),np.sin(angle)]);f=np.array([-np.sin(angle),np.cos(angle)])
            exact=.12*abs(np.array(d)@e)+.03*abs(np.array(d)@f)
            for incoming in (True,False):
                m=boundary_map(a,b,angle,incoming=incoming)
                self.assertAlmostEqual(m[2].sum()+m[3].sum(),exact,places=14)
if __name__=='__main__':unittest.main()

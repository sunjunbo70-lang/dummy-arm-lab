import unittest,copy
import numpy as np
from .split_inventory import SplitPressure
from .adaptive_inventory import AdaptivePressure,IntegrationFailure

class AdaptiveTests(unittest.TestCase):
    def make(self):
        return SplitPressure(np.full((100,100),.002*.005**2),np.full((6,24),.001*.005**2))
    def test_latent_bead_error_detected_without_wall_difference(self):
        a=self.make();a.contact([0.,.25],0.,.1,2.);b=copy.deepcopy(a)
        b.bead[5]+=1e-9;b.blade[0,5]-=1e-9
        self.assertEqual(np.max(abs(a.wall-b.wall)),0.)
        self.assertGreater(AdaptivePressure.error_mm(a,b),.01)
    def test_constant_pose_and_small_move_conserve(self):
        m=AdaptivePressure(self.make(),budget_mm=.02);total=m.model.total()
        m.contact([0.,.25],0.,.1,2.)
        m.contact([0.,.25],0.,.1,2.)
        m.contact([.0001,.25],.0001,.1,2.)
        self.assertAlmostEqual(m.model.total(),total,places=16)
        self.assertGreater(m.stats['accepted'],0)
    def test_failed_interval_is_transactional(self):
        m=AdaptivePressure(self.make(),max_trials=0);m.contact([0.,.25],0.,.1,2.)
        before=copy.deepcopy(m.model);stats=copy.deepcopy(m.stats)
        with self.assertRaises(IntegrationFailure):m.contact([.02,.25],.1,.2,3.)
        for k in ('wall','blade','bead'):np.testing.assert_array_equal(getattr(before,k),getattr(m.model,k))
        np.testing.assert_array_equal(before.pose[0],m.model.pose[0]);self.assertEqual(stats,m.stats)
if __name__=='__main__':unittest.main()

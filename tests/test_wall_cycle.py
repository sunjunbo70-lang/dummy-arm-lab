"""L1 tests for the whole-cycle, sensor-observed plastering experiment."""
import unittest
import numpy as np

from dummy_loop.wall_cycle.config import CycleConfig
from dummy_loop.wall_cycle.env import WallCycleEnv, DecodedAction
from dummy_loop.wall_cycle.material import MaterialSystem


class MaterialTests(unittest.TestCase):
    def test_random_load_is_spatial_and_volume_is_conserved(self):
        c=CycleConfig(); m=MaterialSystem(c,4).reset(); m.load_random(18)
        self.assertGreater(np.std(m.blade),0)
        m.stroke('DEPOSIT',(-.05,.0),(.06,.07),.3,.01,8,.06)
        self.assertLess(abs(m.volume_balance()),1e-15)
        self.assertGreater(m.wall.sum(),0)

    def test_load_shape_and_amount_vary(self):
        c=CycleConfig(); rows=[]
        for seed in range(5):
            m=MaterialSystem(c,seed).reset(); x=m.load_random(18)
            rows.append((x['delivered_ml'],m.blade.copy()))
        self.assertGreater(np.std([x[0] for x in rows]),.2)
        self.assertFalse(np.allclose(rows[0][1],rows[1][1]))


class CycleEnvironmentTests(unittest.TestCase):
    def test_policy_observation_does_not_read_true_wall_between_scans(self):
        e=WallCycleEnv(seed=2,initial_mix=False); before=e.reset().copy()
        e.material.wall[:]=.006
        after=e._observe()
        np.testing.assert_allclose(before,after)

    def test_continuous_action_supports_diagonal_curve_and_rotation(self):
        e=WallCycleEnv(seed=1); e.reset()
        d=DecodedAction('REUSE',(-.11,.01),(.09,.065),np.deg2rad(37),.018,9,.073,0)
        got=e.decode(e.encode(d))
        self.assertAlmostEqual(got.blade_angle,np.deg2rad(37),places=6)
        self.assertAlmostEqual(got.bend_m,.018,places=6)
        self.assertNotAlmostEqual(got.start[0],got.end[0])
        self.assertNotAlmostEqual(got.start[1],got.end[1])

    def test_record_contains_full_cycle_and_work_frames(self):
        e=WallCycleEnv(seed=8,initial_mix=False,record=True);e.reset();e.step(e.teacher_action())
        phases={x['phase'] for x in e.events}
        for p in ('LOAD_APPROACH','DISPENSE','TOOL_INSPECT','WALL_APPROACH',
                  'CONTACT_ACQUIRE','WORK_STEP','LIFT','RETREAT','SCAN_RETURN','SCAN'):
            self.assertIn(p,phases)


if __name__=='__main__': unittest.main()

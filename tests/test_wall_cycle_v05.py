"""L1 contracts for wall-cycle v0.5 loading, transport and lab geometry."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from dummy_loop.wall_cycle.config import CycleConfig
from dummy_loop.wall_cycle.env import DecodedAction, WallCycleEnv
from dummy_loop.wall_cycle.mortar import MortarSystem


CFG = CycleConfig(physics='v0.5', tool_profile='lab_20260922',
                  width_m=.12, height_m=.12, lift_wall_fraction=.75,
                  randomize_interface=False)


class V05ActionAndInterface(unittest.TestCase):
    def test_action_carries_explicit_face_up_score(self):
        e = WallCycleEnv(CFG, 1); e.reset()
        self.assertEqual(e.act_dim, 14)
        d = DecodedAction('DEPOSIT', (-.03, .02), (.04, .10), 1.2, 0,
                          2, .05, 12, .2, .1, .8)
        got = e.decode(e.encode(d))
        self.assertAlmostEqual(got.carry_face_up, .8)

    def test_wall_preference_is_nominal_or_randomised_inside_constraint(self):
        e = WallCycleEnv(CFG, 2); e.reset()
        self.assertEqual(e.material.p.lift_wall_fraction, .75)
        r = WallCycleEnv(replace(CFG, randomize_interface=True), 2); r.reset()
        self.assertGreaterEqual(r.material.p.lift_wall_fraction, .60)
        self.assertLessEqual(r.material.p.lift_wall_fraction, .90)


class FeedAndTransport(unittest.TestCase):
    def loaded(self, dt=.02):
        m = MortarSystem(replace(CFG, transport_dt_s=dt), 3).reset()
        result = m.feed(24, 8, .006, .09, .04)
        self.assertGreater(result['retained_ml'], 0)
        self.assertLess(abs(m.volume_balance()), 1e-15)
        return m

    def test_feed_uses_finite_reservoir_and_conserves_volume(self):
        m = self.loaded()
        before = m.feed_board_remaining_m3
        m.feed(24, 8, .006, .09, .04)
        self.assertLess(m.feed_board_remaining_m3, before)
        self.assertLess(abs(m.volume_balance()), 1e-15)

    def test_face_up_transport_retains_more_than_vertical(self):
        up, face_down = self.loaded(), self.loaded()
        up.transport(1.0, 1.5); face_down.transport(-1.0, 1.5)
        self.assertGreater(up.blade_volume_m3, face_down.blade_volume_m3)
        self.assertLess(abs(up.volume_balance()), 1e-15)
        self.assertLess(abs(face_down.volume_balance()), 1e-15)

    def test_transport_converges_with_time_step(self):
        losses = []
        for dt in (.02, .01, .005):
            m = self.loaded(dt); losses.append(m.transport(-1.0, 1.5))
        self.assertGreater(max(losses), 0)
        self.assertLess(max(losses)-min(losses), 0.03*max(losses))




class ViewerPolicyLoading(unittest.TestCase):
    def test_viewer_infers_256_wide_checkpoint(self):
        from dummy_loop.wall.ppo import PPO, PPOConfig
        from dummy_loop.wall_cycle.view import load_policy
        env = SimpleNamespace(obs_dim=5, act_dim=3, cfg=SimpleNamespace(physics='v0.5'))
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'policy.npz'
            src = PPO(5, 3, PPOConfig(hidden=(256, 256), seed=7))
            src.save(path)
            got = load_policy(env, path)
            x = np.arange(5, dtype=float) / 10
            np.testing.assert_allclose(src.act(x, deterministic=True)[0],
                                       got.act(x, deterministic=True)[0])


class EpisodeIsolation(unittest.TestCase):
    def test_reset_clears_cached_executor_ik_seed(self):
        fake = SimpleNamespace(q_scan=np.arange(6, dtype=float),
                               q_last=np.full(6, 99.0), rng=np.random.default_rng(999))
        env = WallCycleEnv(replace(CFG, use_arm=False), 4, executor=fake)
        env.reset()
        np.testing.assert_array_equal(fake.q_last, fake.q_scan)
        self.assertIsNot(fake.q_last, fake.q_scan)
        np.testing.assert_allclose(fake.rng.random(3), np.random.default_rng(4).random(3))


class RewardScaling(unittest.TestCase):
    def test_partial_wall_without_new_feed_has_finite_carry_penalty(self):
        cfg = replace(CFG, use_arm=False)
        env = WallCycleEnv(cfg, 9, initial_mix=False)
        env.reset()
        env.material.blade.fill(.005)
        env.material.initial_m3 = env.material.blade_volume_m3
        env._scan()
        d = DecodedAction('REUSE', (-.02, .02), (.02, .08), np.pi, 0,
                          1, .05, 0, .2, .1, -1)
        _, reward, _, _ = env.step(env.encode(d))
        self.assertTrue(np.isfinite(reward))
        self.assertGreater(reward, -100)


class LabProfile(unittest.TestCase):
    def test_executor_scene_uses_lab_tool(self):
        from dummy_loop.wall_cycle.arm import scene_config
        s = scene_config(CFG)
        self.assertEqual(s.geometry_profile, 'lab_20260922')
        self.assertAlmostEqual(s.trowel_length, .124)
        self.assertTrue(s.lab_geometry['camera_mount']['enabled'])


if __name__ == '__main__':
    unittest.main()

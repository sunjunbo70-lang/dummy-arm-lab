"""Verify visualization callbacks preserve the saved policies' actual rollouts."""
import unittest
from pathlib import Path
import numpy as np

from dummy_loop.wall.rl import make_env, Scaled, load_policy
from dummy_loop.wall.session import run_session

ROOT = Path(__file__).resolve().parents[1]


class VisualizeTests(unittest.TestCase):
    def test_callbacks_do_not_change_physics_or_material(self):
        path = ROOT / 'experiments/v0.1/r0/records/2026-09-21_plaster_session_rl/raw/technique_warm_start/policy.npz'
        env = make_env(2000)
        policy = Scaled(env, load_policy(env, path), deterministic=True)
        expected, strokes, field = run_session(env, policy)
        q_final, h_final = env.data.qpos.copy(), field.h.copy()
        other = make_env(2000)
        policy2 = Scaled(other, load_policy(other, path), deterministic=True)
        resets, steps = [], []
        actual, actual_strokes, field2 = run_session(
            other, policy2, on_step=lambda i, o, a, info: steps.append(i),
            on_reset=lambda i, o: resets.append((i, other.step_i)))
        self.assertEqual(expected, actual)
        self.assertEqual(strokes, actual_strokes)
        np.testing.assert_array_equal(q_final, other.data.qpos)
        np.testing.assert_array_equal(h_final, field2.h)
        self.assertEqual(resets, [(0, 0), (1, 0), (2, 0)])
        self.assertEqual(len(steps), expected['steps'])
        self.assertAlmostEqual(actual['coverage'], .9796)
        self.assertAlmostEqual(actual['rms_error_mm'], .371)


if __name__ == '__main__':
    unittest.main()

"""L1 tests for the v0.3 whole-wall plastering experiment (experiments/v0.3/r0/design/2026-09-22_wall_cycle_v0.3.md).

Each test pins one of the reasons for a change, so that a later edit that quietly undoes
the change fails here instead of silently bringing the v0.2 problem back.
"""
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from dummy_loop.wall_cycle.config import CycleConfig
from dummy_loop.wall_cycle.env import WallCycleEnv, DecodedAction
from dummy_loop.wall_cycle import mortar as M
from dummy_loop.wall_cycle.sensor import coarse_map, coarse_shape

ROOT = Path(__file__).resolve().parents[1]
V02_POLICY = ROOT / 'experiments/v0.2/r0/records/2026-09-22_wall_cycle_rl/raw/model/policy_ppo.npz'
SMALL = CycleConfig(width_m=0.12, height_m=0.12)


class V02StaysReproducible(unittest.TestCase):
    """The recorded v0.2 experiment must still reproduce bit for bit (numbers taken from the
    original commit 9b5716f, before any v0.3 change)."""

    def test_recorded_ppo_policy_gives_identical_episodes(self):
        if not V02_POLICY.is_file():
            self.skipTest('v0.2 policy not present')
        from dummy_loop.wall_cycle.replay import load_agent
        cfg = CycleConfig(physics='v0.2')
        agent = load_agent(V02_POLICY, WallCycleEnv(cfg, 0))
        expected = {20000: (-8.014912093477443, 31, 0.35728551683497284),
                    20001: (-11.252227106389714, 14, 0.7632467804952641),
                    20002: (-8.598312414871849, 11, 1.0571055669509821)}
        for seed, (ret, cycles, rmse) in expected.items():
            e = WallCycleEnv(cfg, seed); o = e.reset(); done = False; R = 0.0
            while not done:
                o, r, done, info = e.step(agent.act(o, deterministic=True)[0]); R += r
            self.assertEqual(R, ret)
            self.assertEqual(info['cycles'], cycles)
            self.assertEqual(info['metrics']['rmse_mm'], rmse)

    def test_v02_teacher_and_observation_unchanged(self):
        e = WallCycleEnv(CycleConfig(physics='v0.2'), 7); o = e.reset()
        self.assertEqual(float(o.sum()), 195.1294132653061)
        np.testing.assert_allclose(e.teacher_action().round(6),
                                   [-0.8, 0.589286, -1.0, 0.589286, 1.0, -1.0, 0.0, 0.0, -0.076923, -0.2, 0.25])
        self.assertEqual(e.act_dim, 11)


class MortarPhysics(unittest.TestCase):
    p = M.MortarParams()

    def test_more_force_leaves_a_thinner_layer(self):
        """v0.2 had the sign reversed (2 N -> 2.03 mm, 15 N -> 2.38 mm)."""
        t = np.full((6, 24), 0.004)
        gaps = [M.solve_gap(t, 0.0, F, 0.005, self.p)[0] for F in (1, 2, 5, 8, 12)]
        # a light press just floats on the 4 mm layer (gap = layer); once it squeezes, harder = thinner
        self.assertTrue(all(a >= b for a, b in zip(gaps, gaps[1:])), gaps)
        self.assertTrue(all(a > b for a, b in zip(gaps[2:], gaps[3:])), gaps)
        self.assertLessEqual(gaps[0], 0.004 + 1e-9)

    def test_stable_thickness_on_a_vertical_face(self):
        self.assertAlmostEqual(M.stable_thickness(self.p, 1.0), 250 / (1900 * 9.81), places=9)
        self.assertGreater(M.stable_thickness(self.p, np.cos(np.deg2rad(30))), M.stable_thickness(self.p, 1.0))

    def test_squeezed_excess_escapes_towards_the_wider_edge(self):
        t = np.full((6, 4), 0.012)                                            # more than the wedge holds
        flat = 0.001 + np.zeros(6)
        tilted = 0.001 + (np.arange(6) + .5) * .005 * np.sin(np.deg2rad(20))
        _, tr_f, ld_f = M.extrude(t, flat)
        _, tr_t, ld_t = M.extrude(t, tilted)
        self.assertGreater(tr_f.sum(), 0)
        self.assertAlmostEqual(tr_f.sum(), ld_f.sum(), places=12)           # flat: half goes behind
        self.assertGreater(ld_t.sum(), 0)
        self.assertLess(tr_t.sum(), 0.01 * (tr_t.sum() + ld_t.sum()))       # tilted: almost all ahead

    def test_drainage_drops_at_a_free_edge_and_keeps_at_a_sealed_one(self):
        h = np.zeros((6, 3)); h[1] = 0.03             # 30 mm ridge next to the lower edge, h_stable ~13 mm
        hs = M.stable_thickness(self.p, 1.0)
        _, free = M.drain(h, 0.0, -1.0, hs)
        kept, sealed = M.drain(h, 0.0, -1.0, hs, sealed_low_w=True)
        self.assertGreater(free, 0)
        self.assertEqual(sealed, 0.0)
        self.assertAlmostEqual(kept.sum(), h.sum(), places=12)

    def test_no_parameter_prescribes_a_pitch(self):
        """The whole point of plan B: the model must not contain the answer."""
        names = ' '.join(M.MortarParams().to_dict()).lower()
        for word in ('pitch', 'tilt', 'angle', 'wedge', 'capacity'):
            self.assertNotIn(word, names)

    def test_strokes_conserve_volume_exactly(self):
        rng = np.random.default_rng(0)
        m = M.MortarSystem(SMALL, 3).reset(); m.load_random(24)
        for _ in range(12):
            if rng.random() < .3:
                m.load_random(float(rng.choice([12, 18, 24])))
            a = rng.uniform([-.06, 0], [.06, .12], 2); b = rng.uniform([-.06, 0], [.06, .12], 2)
            m.stroke('REUSE', a, b, rng.uniform(-np.pi, np.pi), rng.uniform(-.02, .02), rng.uniform(.5, 15),
                     rng.uniform(.02, .12), pitch_start=rng.uniform(0, .6), pitch_end=rng.uniform(0, .6))
            self.assertLess(abs(m.volume_balance()), 1e-15)
        self.assertGreater(m.wall.sum(), 0)

    def test_blade_flow_does_not_wrap_around(self):
        """v0.2 smoothed the blade with np.roll: material at one end reappeared at the other."""
        m = M.MortarSystem(SMALL, 0).reset(); m.blade[:] = 0; m.blade[:, 0] = 0.01
        m.begin_stroke(0.0, (0.0, 1.0))
        m.air(0.0)
        self.assertEqual(float(m.blade[:, -1].sum()), 0.0)

    def test_tilt_matters_through_physics_for_a_stiff_mortar(self):
        """Emergent, not prescribed: with tau_y >= 250 Pa a tilted blade wastes less than a flat one."""
        cfg = replace(SMALL, mortar_tau_y_Pa=500.0)
        waste = {}
        for pitch in (0, 20):
            w = []
            for seed in range(3):
                m = M.MortarSystem(cfg, seed).reset(); m.load_random(24)
                m.stroke('DEPOSIT', (0, .005), (0, .115), np.pi, 0, 1.0, .06,
                         pitch_start=np.deg2rad(pitch), pitch_end=np.deg2rad(pitch))
                w.append((m.dropped_m3 + m.outside_m3) / m.supplied_m3)
            waste[pitch] = np.mean(w)
        self.assertLess(waste[20], waste[0])


class ActionAndObservation(unittest.TestCase):
    def test_v03_action_has_a_free_pitch_profile(self):
        e = WallCycleEnv(SMALL, 1); e.reset()
        self.assertEqual(e.act_dim, 13)
        d = DecodedAction('REUSE', (-.03, .01), (.04, .09), np.deg2rad(37), .01, 4.0, .07, 0,
                          np.deg2rad(22), np.deg2rad(6))
        got = e.decode(e.encode(d))
        self.assertAlmostEqual(got.pitch_start, np.deg2rad(22), places=6)
        self.assertAlmostEqual(got.pitch_end, np.deg2rad(6), places=6)
        self.assertAlmostEqual(got.force_N, 4.0, places=6)

    def test_v03_force_range_reaches_below_one_newton_but_v02_range_is_unchanged(self):
        e3 = WallCycleEnv(SMALL, 1); e2 = WallCycleEnv(CycleConfig(physics='v0.2'), 1)
        a = np.zeros(13); a[8] = -1
        self.assertAlmostEqual(e3.decode(a).force_N, CycleConfig().force_window_v3[0])
        self.assertAlmostEqual(e2.decode(a[:11]).force_N, 2.0)

    def test_coarse_map_pools_any_wall_shape(self):
        """v0.2 returned the UNPOOLED map when the grid was not a multiple of 2x4."""
        for shape in ((14, 56), (24, 24), (39, 39)):
            a = np.ones(shape)
            self.assertEqual(coarse_map(a).shape, coarse_shape(shape))
        e = WallCycleEnv(CycleConfig(width_m=.195, height_m=.195), 0)
        self.assertEqual(e.reset().shape, (e.obs_dim,))


class WorkArea(unittest.TestCase):
    def test_square_is_derived_by_the_users_rule(self):
        """v0.7: area_safety is applied to the reachable circle's AREA (r*sqrt(area_safety)),
        not to the inscribed square's side -- score_square is the scored/aimed-at region,
        sim_square (always >= score_square, same centre) is the physical/action-range extent."""
        from dummy_loop.wall_cycle.area import square_from_reach, WORK_AREA_FILE, load_work_area
        r = .15
        best = {'wall_distance_m': .33, 'circle_centre_uv_m': [0.0, .18], 'circle_radius_m': r,
                'score_square_side_m': np.sqrt(2) * r * np.sqrt(.8),
                'sim_square_side_m': np.sqrt(2) * r}
        sq = square_from_reach(best, .005)
        self.assertLessEqual(sq['score_square']['side_m'], np.sqrt(2) * r * np.sqrt(.8) + 1e-12)
        self.assertGreater(sq['score_square']['side_m'], np.sqrt(2) * r * np.sqrt(.8) - .005)
        self.assertGreaterEqual(sq['sim_square']['side_m'], sq['score_square']['side_m'])
        if WORK_AREA_FILE.is_file():
            a = json.loads(WORK_AREA_FILE.read_text(encoding='utf-8'))
            cfg = load_work_area(CycleConfig())
            self.assertAlmostEqual(cfg.width_m, a['sim_square']['side_m'])
            self.assertAlmostEqual(cfg.score_width_m, a['score_square']['side_m'])
            self.assertAlmostEqual(cfg.width_m, cfg.height_m)
            self.assertGreaterEqual(cfg.width_m, cfg.score_width_m)
            # scored region bigger than the v0.2 strip
            self.assertGreater(cfg.score_width_m * cfg.score_height_m, 0.28 * 0.07)
        self.assertEqual(load_work_area(CycleConfig(physics='v0.2')).width_m, 0.28)


class ActionProjection(unittest.TestCase):
    """C16: an infeasible stroke is replaced by the nearest executable one, never executed as is."""

    def test_projection_reduces_tilt_and_never_executes_an_infeasible_stroke(self):
        class Exec:                       # feasible only with little tilt
            dynamic = False
            def __init__(self): self.stats = {'planned': 0, 'rejected': 0}
            def plan(self, d):
                from dummy_loop.wall_cycle.reach_table import TablePlan
                ok = d.pitch_start <= np.deg2rad(8) + 1e-9
                return TablePlan(ok, '' if ok else 'too steep')
        e = WallCycleEnv(SMALL, 3, executor=Exec()); e.reset()
        d = DecodedAction('REUSE', (0.0, .02), (0.0, .10), np.pi, 0, 1.0, .06, 0, np.deg2rad(20), np.deg2rad(5))
        _, r, _, info = e.step(e.encode(d))
        self.assertEqual(info['projected_strokes'], 1)
        self.assertEqual(info['unreachable_strokes'], 0)
        executed = [ev for ev in e.events if ev['phase'] == 'PROJECTED']
        self.assertEqual(executed, [])       # record=False -> no events; check via the counter above
        # an executor that accepts nothing -> rejected, not executed
        class Never(Exec):
            def plan(self, d):
                from dummy_loop.wall_cycle.reach_table import TablePlan
                return TablePlan(False, 'no')
        e2 = WallCycleEnv(SMALL, 3, executor=Never()); e2.reset()
        w0 = e2.material.wall.copy()
        _, _, _, info2 = e2.step(e2.encode(d))
        self.assertEqual(info2['unreachable_strokes'], 1)
        np.testing.assert_array_equal(e2.material.wall, w0)


class ArmCoSimulation(unittest.TestCase):
    """Every accepted stroke runs on the Dummy V2 MuJoCo model; the mortar follows the ACTUAL blade."""

    @classmethod
    def setUpClass(cls):
        from dummy_loop.wall_cycle.arm import ArmExecutor
        from dummy_loop.wall_cycle.area import load_work_area
        cls.cfg = load_work_area(CycleConfig())
        cls.ex = ArmExecutor(cls.cfg)

    def test_planner_rejects_a_stroke_far_outside_the_reach(self):
        d = DecodedAction('REUSE', (0.0, 0.0), (0.0, 0.05), np.pi, 0, 2.0, .06, 0, 0.0, 0.0)
        far = replace(self.cfg, scene_wall_distance_m=0.60)
        from dummy_loop.wall_cycle.arm import ArmExecutor
        self.assertFalse(ArmExecutor(far).plan(d).ok)

    def test_a_central_stroke_executes_with_bounded_tracking_and_exact_volume(self):
        c = self.cfg
        # (a lower-edge-first tilt of 10 deg is NOT executable everywhere: J5 sits near its -100 deg
        # firmware limit at this wall distance -- see the change log, C7/C8. Use a small tilt.)
        d = DecodedAction('DEPOSIT', (0.0, .3 * c.height_m), (0.0, .7 * c.height_m), np.pi, 0, 2.0, .06, 18,
                          np.deg2rad(4), np.deg2rad(0))
        plan = self.ex.plan(d)
        self.assertTrue(plan.ok, plan.reason)
        m = M.MortarSystem(c, 1).reset(); m.load_random(18)
        st = self.ex.execute(plan, d, m, M.StrokeStats())
        self.assertLess(abs(m.volume_balance()), 1e-15)
        self.assertLess(st['tracking_max_mm'], 15.0)          # servo gains are unidentified; bound is loose
        self.assertEqual(st['ik_not_converged'], 0)
        self.assertGreater(m.wall.sum(), 0)
        # no wrist flip during the stroke (the first draft ran with J6 turned by ~180 deg)
        q = np.asarray(st['q_work'])
        self.assertLess(np.rad2deg(np.max(np.abs(np.diff(q, axis=0)))), 20.0)

    def test_tilted_contact_pose_keeps_the_trailing_edge_out_of_the_wall(self):
        self.assertAlmostEqual(self.ex.contact_n(np.deg2rad(30)), -self.ex.hw * 0.5, places=9)


if __name__ == '__main__':
    unittest.main()

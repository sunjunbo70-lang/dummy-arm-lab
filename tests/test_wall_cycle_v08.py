"""L1 gates for the v0.8 recipe: local teacher, loss ledgers, memory observation, parallel runs."""
import unittest
import numpy as np

from dummy_loop.wall_cycle.env import WallCycleEnv, DecodedAction
from dummy_loop.wall_cycle.reach_table import ReachTableExecutor
from dummy_loop.wall_cycle.recipes import RECIPES, make_config


def short(recipe='v0.8', **kw):
    return make_config(recipe, **{'base_steps': 2500, 'max_steps': 2500, **kw})


class Recipes(unittest.TestCase):
    def test_v07_recipe_is_the_2026_09_23_run(self):
        c = make_config('v0.7')
        self.assertEqual((c.base_steps, c.max_steps, c.max_cycles, c.stall_window), (12000, 24000, 160, 12))
        self.assertEqual((c.teacher_targeting, c.loss_accounting, c.obs_memory), ('global', 'v0.7', False))
        self.assertEqual((c.time_coef, c.finish_fail_penalty, c.feed_transit_physics), (0.0002, 20.0, False))

    def test_v08_recipe_and_score_mask(self):
        c = make_config('v0.8')
        self.assertEqual(RECIPES['v0.8']['outside_coef'], 1.0)
        env = WallCycleEnv(c, 0, executor=ReachTableExecutor(c)); env.reset()
        self.assertEqual(env.material.score_cells, (40, 40))           # explicit, same as v0.7
        self.assertAlmostEqual(env.material.metrics()['score_side_m'], 0.200)
        base = WallCycleEnv(make_config('v0.7'), 0).obs_dim
        self.assertEqual(env.obs_dim, base + 23 * 12 + 2)


class Teacher(unittest.TestCase):
    def _aim(self, targeting):
        c = short(teacher_targeting=targeting)
        env = WallCycleEnv(c, 0, executor=ReachTableExecutor(c)); env.reset()
        # synthetic scan: target thickness everywhere, both side bands of the scored square bare
        env.scan['height'][:] = c.target_m; env.scan['confidence'][:] = 1.0
        env.scan['height'][:, :9] = 0.0; env.scan['height'][:, -9:] = 0.0
        d = env.decode(env.teacher_action())
        return d, (np.asarray(d.start) + d.end) / 2

    def test_global_teacher_aims_at_the_centre_of_a_symmetric_edge_deficit(self):
        _, mid = self._aim('global')
        self.assertLess(abs(mid[0]), 0.03)          # the v0.7 failure mode, kept for comparison

    def test_local_teacher_aims_at_an_edge(self):
        d, mid = self._aim('local')
        self.assertIn(d.mode, ('DEPOSIT', 'REUSE'))
        self.assertGreater(abs(mid[0]), 0.05)


class Ledgers(unittest.TestCase):
    def test_three_ledgers_add_up_and_memory_is_observed(self):
        c = short()
        env = WallCycleEnv(c, 3, executor=ReachTableExecutor(c), initial_mix=False)
        o = env.reset(); done = False
        while not done:
            o, r, done, info = env.step(env.teacher_action())
            self.assertEqual(len(o), env.obs_dim)
        led, m = info['loss_ledger_m3'], env.material
        self.assertAlmostEqual(led['carry_m3'] + led['other_drop_m3'], m.dropped_m3, places=12)
        self.assertAlmostEqual(led['outside_m3'], m.outside_m3, places=12)
        self.assertLess(abs(info['volume_balance_m3']), 1e-12)
        self.assertIn(info['end_reason'], ('finish', 'stall', 'budget', 'cycles', 'force'))

    def test_unsuccessful_finish_costs_the_recipe_penalty(self):
        for recipe, penalty in (('v0.7', 20.0), ('v0.8', 2.0)):
            c = short(recipe)
            env = WallCycleEnv(c, 0, executor=ReachTableExecutor(c)); env.reset()
            a = env.encode(DecodedAction('FINISH', (0, .01), (0, .06), 0, 0, 6, .05, 0))
            _, r, done, _ = env.step(a)
            self.assertTrue(done); self.assertAlmostEqual(r, -.02 - penalty)


class Parallel(unittest.TestCase):
    def test_worker_count_does_not_change_evaluation_episodes(self):
        from dummy_loop.wall_cycle.parallel import Runner
        from dummy_loop.wall_cycle.train import evaluate
        c = short()
        serial = evaluate(c, None, 2, 777)
        runner = Runner(2)
        try:
            parallel = evaluate(c, None, 2, 777, runner=runner, policy_path=None, hidden=(8, 8))
        finally:
            runner.close()
        for k in ('return_mean', 'coverage_mean', 'rmse_mm_mean', 'waste_frac_mean', 'cycles_mean'):
            self.assertEqual(serial[k], parallel[k], k)


class FeedTransit(unittest.TestCase):
    def test_residual_material_meets_transport_physics_on_the_way_to_the_feed(self):
        from dummy_loop.wall_cycle.arm import ArmExecutor
        c = short()
        ex = ArmExecutor(c, seed=0)
        env = WallCycleEnv(c, 0, executor=ex, initial_mix=False, record=True); env.reset()
        _, _, _, info = env.step(env.teacher_action())
        trace = next(e['trajectory'] for e in env.events if e['phase'] == 'ARM_TRACE')
        phases = [x['phase'] for x in trace]
        if 'FEED_ALIGN_UP' in phases:
            self.assertEqual(phases.count('FEED_ALIGN_UP'), 1)
            self.assertGreater(phases.count('FEED_TRANSIT'), 0)
            feed = [x['face_up_score'] for x in trace if x['phase'] == 'FEED_ALIGN_UP']
            self.assertGreaterEqual(min(feed), np.cos(np.deg2rad(c.face_up_hard_deg)))
        self.assertLess(abs(info['volume_balance_m3']), 1e-12)


if __name__ == '__main__': unittest.main()

"""抹涂强化学习闭环的测试（L1，不接硬件）：材料模型、环境、PPO 实现、训练闭环。"""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dummy_loop.wall.material import MaterialConfig, MortarField
from dummy_loop.wall.ppo import PPO, PPOConfig, gae
from dummy_loop.wall.rl import behaviour_clone, evaluate, train
from dummy_loop.wall.session import SessionConfig, area_of, reachable_bands, run_session
from dummy_loop.wall.stroke_env import StrokeEnv, StrokeConfig, rollout


class MaterialTests(unittest.TestCase):
    def field(self):
        return MortarField(MaterialConfig(), (-0.06, 0.06), (-0.05, 0.05))

    def test_volume_is_conserved(self):
        f = self.field()
        v0 = f.volume_balance()
        rng = np.random.default_rng(0)
        v = -0.06
        for _ in range(40):                      # 后缘从下往上扫，间隙随机
            v += 0.004
            f.sweep(v, float(rng.uniform(0, 0.006)), 0.06, 0.0, True)
        self.assertAlmostEqual(f.volume_balance(), v0, places=9)
        self.assertGreater(f.deposited, 0)

    def test_thickness_follows_trailing_edge_gap(self):
        f = self.field(); gap = 0.0025
        f.reset(load=10 * f.initial_load())      # 料管够：这里只看「留下的厚度 = 后缘间隙」
        v = -0.06
        for _ in range(40):
            v += 0.004
            f.sweep(v, gap, 0.06, 0.0, True)
        h = f.h[f.inside]
        self.assertAlmostEqual(float(h.mean()), gap, delta=0.0005)   # 留下的厚度 = 后缘间隙
        self.assertLess(float(h.std()), 0.0005)

    def test_flat_blade_scrapes_material_back_onto_the_tool(self):
        f = self.field()
        f.h[:] = 0.004
        before = f.load
        v = -0.06
        for _ in range(40):
            v += 0.004
            f.sweep(v, 0.001, 0.06, 0.0, True)   # 贴着墙刮
        self.assertGreater(f.load, before)       # 多出来的料回到刀上
        self.assertLess(float(f.h[f.inside].max()), 0.0015)

    def test_far_blade_changes_nothing(self):
        f = self.field(); f.sweep(-0.05, 0.05, 0.06, 0.0, True)
        v0 = f.volume_balance()
        f.sweep(0.05, 0.05, 0.06, 0.0, True)
        self.assertEqual(float(f.h.sum()), 0.0)
        self.assertAlmostEqual(f.volume_balance(), v0, places=12)

    def test_tilting_the_blade_lets_it_hold_more(self):
        """这是「为什么要先立起再放平」在模型里的落点：楔形容量随俯仰增大。"""
        f = self.field()
        flat = f.capacity(0.0, 0.002, 0.12, 0.02)
        tilted = f.capacity(np.deg2rad(25), 0.002, 0.12, 0.02)
        self.assertGreater(tilted, 2 * flat)
        far = f.capacity(np.deg2rad(25), 0.05, 0.12, 0.02)     # 没贴墙：只有粘附层
        self.assertAlmostEqual(far, 0.12 * 0.02 * f.cfg.stick_free, places=9)

    def test_flat_blade_drops_the_load_it_cannot_hold(self):
        f = self.field()
        before, v0 = f.load, f.volume_balance()
        dropped = f.carry(0.0, 0.0005, 0.12, 0.02)             # 贴墙放平：兜不住
        self.assertGreater(dropped, 0)
        self.assertAlmostEqual(f.load, before - dropped, places=12)
        self.assertAlmostEqual(f.volume_balance(), v0, places=12)   # 掉落的料仍记在账上
        f2 = self.field()
        self.assertEqual(f2.carry(np.deg2rad(20), 0.002, 0.30, 0.05), 0.0)   # 楔形够大就不掉

    def test_reload_tops_up_to_one_load_and_counts_supply(self):
        f = self.field()
        one = f.initial_load()
        f.load = 0.3 * one
        f.reload(one)
        self.assertAlmostEqual(f.load, one, places=12)
        self.assertAlmostEqual(f.supplied, one + 0.7 * one, places=12)   # 首份 + 这次补的
        f.reload(one)                                        # 已经满了：不再加
        self.assertAlmostEqual(f.supplied, 1.7 * one, places=12)


class StrokeEnvTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = StrokeEnv(seed=0)

    def test_reset_returns_to_ready_pose_with_a_fresh_load(self):
        obs = self.env.reset()
        self.assertEqual(obs.shape, (self.env.obs_dim,))
        self.assertAlmostEqual(self.env.ctrl.pitch, np.deg2rad(self.env.cfg.start_pitch_deg), places=9)
        self.assertAlmostEqual(self.env.ctrl.target[2], -self.env.cfg.standoff, places=9)
        self.assertGreater(self.env.field.load, 0)
        self.assertEqual(float(self.env.field.h.sum()), 0.0)

    def test_action_validation(self):
        self.env.reset()
        with self.assertRaises(ValueError):
            self.env.step([0, 0])
        with self.assertRaises(ValueError):
            self.env.step([np.nan, 0, 0])

    def test_scripted_stroke_lays_a_layer_and_beats_a_random_policy(self):
        """手写脚本是**预热用的示教**，不是好策略：加入掉料规则后它只能抹到六成多，
        剩下的要靠 PPO 学（见 experiments/2026-09-21_stroke_rl）。这里只钉住「确实抹上去了、比随机强」。"""
        env = StrokeEnv(seed=0)
        ret, steps, m = rollout(env)
        self.assertGreater(m['coverage'], 0.5)
        self.assertLess(m['rms_error_mm'], 2.0)
        self.assertFalse(m['aborted'])
        rng = np.random.default_rng(0)
        scale = env.spec.limits[[1, 2, 4]]
        rnd = rollout(env, lambda o: rng.uniform(-1, 1, 3) * scale)[0]
        self.assertGreater(ret, rnd + 5)

    def test_flat_script_is_worse_than_the_worker_technique(self):
        """对照组：贴墙前就放平 → 兜不住料，掉得多、覆盖差。这是「手法有用」在仿真里的落点。"""
        tech = rollout(StrokeEnv(stroke=StrokeConfig(script_style='technique'), seed=0))[2]
        flat = rollout(StrokeEnv(stroke=StrokeConfig(script_style='flat'), seed=0))[2]
        self.assertGreater(tech['coverage'], flat['coverage'])
        self.assertLess(tech['dropped_frac'], flat['dropped_frac'])

    def test_pitch_puts_the_lower_edge_closer_to_the_wall(self):
        env = StrokeEnv(seed=0); env.reset()
        for _ in range(5):
            env.step([0.0, 0.0, env.cfg.max_dpitch])       # 增大俯仰
        low, high = env.blade_edges()
        self.assertGreater(low[2], high[2])                # 下缘的 n 更大 = 更靠近墙

    def test_material_only_moves_while_the_blade_is_close(self):
        env = StrokeEnv(seed=0); env.reset()
        for _ in range(10):
            env.step([env.cfg.max_dv, -env.cfg.max_dn, 0.0])   # 一边上行一边退出
        self.assertEqual(float(env.field.deposited), 0.0)


class PPOTests(unittest.TestCase):
    def test_ppo_learns_a_toy_reaching_task(self):
        rng = np.random.default_rng(0)
        agent = PPO(2, 2, PPOConfig(steps_per_update=512, lr=1e-3, seed=0))
        x = rng.uniform(-1, 1, 2); t = 0; first = last = None
        for it in range(12):
            O, A, L, R, D, V = [], [], [], [], [], []
            ep, eps = 0.0, []
            for _ in range(agent.cfg.steps_per_update):
                a, raw, lp = agent.act(x); v = agent.value(x)
                x = x + 0.2 * a; t += 1
                r = -float(np.sum(x ** 2)); done = t >= 20
                O.append(x - 0.2 * a); A.append(raw); L.append(lp); R.append(r); D.append(done); V.append(v)
                ep += r
                if done:
                    eps.append(ep); ep = 0.0; t = 0; x = rng.uniform(-1, 1, 2)
            adv, ret = gae(np.array(R), np.array(V), np.array(D), agent.value(x), 0.99, 0.95)
            agent.update(np.array(O), np.array(A), np.array(L), adv, ret)
            m = float(np.mean(eps))
            first = m if first is None else first
            last = m
        self.assertGreater(last, first + 5)

    def test_save_and_load_round_trip(self):
        env = StrokeEnv(seed=0)
        agent = PPO(env.obs_dim, env.act_dim, PPOConfig(seed=0))
        obs = env.reset()
        before = agent.act(obs, deterministic=True)[0]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'p.npz'
            agent.save(path)
            other = PPO(env.obs_dim, env.act_dim, PPOConfig(seed=1)).load(path)
        np.testing.assert_allclose(other.act(obs, deterministic=True)[0], before)


class SessionTests(unittest.TestCase):
    """多刀作业：一刀一条带，整片区域拼起来（dummy_loop/wall/session.py）。"""

    @classmethod
    def setUpClass(cls):
        cls.env = StrokeEnv(seed=3)

    def test_area_is_the_union_of_the_bands(self):
        cfg = SessionConfig(column_centres=(-0.08, 0.0, 0.08))
        half = (self.env.outline['x_tip'] - self.env.outline['x_back']) / 2
        (u0, u1), v = area_of(cfg, half)
        self.assertAlmostEqual(u0, -0.08 - half, places=9)
        self.assertAlmostEqual(u1, 0.08 + half, places=9)
        self.assertEqual(v, cfg.band_v)

    def test_unreachable_bands_are_dropped_not_attempted(self):
        ok, dropped = reachable_bands(self.env, SessionConfig(column_centres=(0.0, 1.5)))
        self.assertEqual(ok, [0.0])
        self.assertEqual(dropped, [1.5])          # 1.5 m 外，机械臂够不到

    def test_three_strokes_cover_more_area_than_one_and_keep_the_books(self):
        env = StrokeEnv(seed=3)
        cfg = SessionConfig(column_centres=(-0.08, 0.0, 0.08))
        m, strokes, field = run_session(env, None, cfg)
        self.assertEqual(len(strokes), 3)
        self.assertEqual(m['strokes'], 3)
        self.assertFalse(m['transit_simulated'])
        self.assertGreater(m['area_cm2'], 150)
        self.assertGreater(m['coverage'], 0.7)
        self.assertFalse(any(s['aborted'] for s in strokes))
        # 体积守恒：墙上 + 刀上 + 掉落 = 累计上料量
        self.assertAlmostEqual(field.volume_balance(), field.supplied, places=9)
        # 三刀各上一份料，总上料量 ≈ 3 份
        self.assertAlmostEqual(field.supplied / env.stroke_load(), 3.0, delta=0.35)


class TrainingLoopTests(unittest.TestCase):
    def test_behaviour_clone_reaches_the_scripted_baseline(self):
        env = StrokeEnv(seed=0)
        agent = PPO(env.obs_dim, env.act_dim, PPOConfig(seed=0))
        mse = behaviour_clone(agent, env, episodes=6, iters=250)
        self.assertLess(mse, 0.01)
        scripted = evaluate(env, None, 2)['return_mean']
        cloned = evaluate(env, agent, 2)['return_mean']
        self.assertGreater(cloned, scripted - 1.0)

    def test_short_training_run_writes_a_record_and_does_not_collapse(self):
        with tempfile.TemporaryDirectory() as d:
            r = train(d, updates=2, cfg=PPOConfig(steps_per_update=256, seed=0, init_log_std=-3.0,
                                                  entropy_coef=0.0, lr=5e-5, epochs=3),
                      eval_every=2, eval_episodes=2, log=lambda s: None)
            self.assertTrue((Path(d) / 'policy.npz').is_file())
            self.assertTrue((Path(d) / 'training.json').is_file())
            self.assertEqual(r['evidence_level'], 'L1')
            self.assertFalse(r['hardware_motion'])
            self.assertIn('reduced-order', r['env']['material_model'])
            self.assertGreater(r['learned_final']['return_mean'], r['random_policy']['return_mean'] + 5)


if __name__ == '__main__':
    unittest.main()

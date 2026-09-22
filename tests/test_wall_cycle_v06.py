"""L1 gates for v0.6 pose-derived loading and monotonic approach."""
from dataclasses import replace
import unittest
import numpy as np

from dummy_loop.wall_cycle.area import load_work_area
from dummy_loop.wall_cycle.arm import ArmExecutor
from dummy_loop.wall_cycle.config import CycleConfig
from dummy_loop.wall_cycle.env import WallCycleEnv


CFG = load_work_area(CycleConfig(
    physics='v0.6', tool_profile='lab_20260922', lift_wall_fraction=.75,
    base_steps=12000, max_steps=24000, extension_steps=2000,
    max_cycles=160, max_reload_cycles=60, stall_limit=1,
    stall_window=12, min_cycles_before_stall=25))


class V06Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ex = ArmExecutor(CFG, seed=0)

    def test_action_removes_declared_face_up_dimension(self):
        env=WallCycleEnv(CFG,0,executor=self.ex);env.reset()
        self.assertEqual(env.act_dim,13)
        self.assertNotIn('carry_face_up',env.action_names)

    def test_teacher_plan_has_face_up_carry_and_monotonic_approach(self):
        env=WallCycleEnv(CFG,0,executor=self.ex);env.teacher_style='technique';env.reset()
        d=env.decode(env.teacher_action());p=self.ex.plan(d)
        self.assertTrue(p.ok,p.reason)
        self.assertGreaterEqual(p.carry_face_up_min,np.cos(np.deg2rad(CFG.face_up_target_deg)))
        self.assertGreater(len(p.q_carry),0);self.assertGreater(len(p.q_rotate),0)
        self.assertTrue(all(a>=b-5e-4 for a,b in zip(p.approach_distances_m,p.approach_distances_m[1:])))

    def test_dynamic_cycle_loads_only_at_face_up_pose(self):
        ex=ArmExecutor(CFG,seed=0)
        env=WallCycleEnv(CFG,0,executor=ex,initial_mix=False,record=True);env.teacher_style='technique';env.reset()
        _,_,_,info=env.step(env.teacher_action())
        trace=next(e['trajectory'] for e in env.events if e['phase']=='ARM_TRACE')
        feed=[x['face_up_score'] for x in trace if x['phase']=='FEED_ALIGN_UP']
        carry=[x['face_up_score'] for x in trace if x['phase']=='CARRY_FACE_UP']
        rotate=[x['face_up_score'] for x in trace if x['phase']=='ROTATE_TO_WALL']
        self.assertGreaterEqual(min(feed+carry),np.cos(np.deg2rad(CFG.face_up_hard_deg)))
        self.assertGreaterEqual(min(rotate),-1e-3)
        self.assertEqual(env.last_stroke['face_down_frames'],0)
        self.assertEqual(env.last_stroke['approach_reversal_count'],0)
        self.assertLess(abs(info['volume_balance_m3']),1e-12)

    def test_long_episode_budget(self):
        self.assertEqual(CFG.base_steps,12000);self.assertEqual(CFG.max_steps,24000)
        self.assertEqual(CFG.max_cycles,160);self.assertEqual(CFG.stall_window,12)


if __name__ == '__main__': unittest.main()

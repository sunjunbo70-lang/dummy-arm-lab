"""墙面抹涂仿真链路的测试（L1，不接硬件）。"""
import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import mujoco

from dummy_loop.episode import EpisodeWriter, load_episode, hardware_fields
from dummy_loop.wall.controller import ActionSpec, EEController, WallFrame
from dummy_loop.wall.errors import Perturbation
from dummy_loop.wall.pipeline import record_episode, replay_episode
from dummy_loop.wall.scene import ARM_MODEL, SceneConfig, build_scene, tool_tilt_matrix, wall_frame, FLANGE_FACE_Y
from dummy_loop.wall.task import WallTask, OBS_SPEC
from dummy_loop.wall.teacher import RasterTeacher


def controller(cfg=None):
    cfg = cfg or SceneConfig(); m, _ = build_scene(cfg); lim = np.deg2rad(cfg.joint_range_deg)
    return EEController(m, WallFrame(*wall_frame(cfg)), [-lim] * 6, [lim] * 6, tool_R=tool_tilt_matrix(cfg))


class SceneTests(unittest.TestCase):
    def test_reference_model_file_not_modified(self):
        before = hashlib.sha256(ARM_MODEL.read_bytes()).hexdigest()
        build_scene(SceneConfig(joint_range_deg=45, spring_k=123))
        self.assertEqual(before, hashlib.sha256(ARM_MODEL.read_bytes()).hexdigest())

    def test_assumptions_are_labelled(self):
        m, _ = build_scene()
        self.assertIn('ASSUMPTIONS_NOT_CALIBRATED', m.names.decode())
        prov = SceneConfig().provenance()
        self.assertIn('ASSUMPTION', prov['joint_range_deg'])
        self.assertIn('PLACEHOLDER', prov['tool_dimensions'])

    def test_tool_length_is_flange_face_to_blade_face(self):
        cfg = SceneConfig(); m, _ = build_scene(cfg); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
        b6 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'link6_1_1')
        tcp = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, 'tcp')
        R = d.xmat[b6].reshape(3, 3); face = d.xpos[b6] + R @ [0, FLANGE_FACE_Y, 0]
        self.assertAlmostEqual(float(np.linalg.norm(d.site_xpos[tcp] - face)), cfg.tool_length, places=6)

    def test_only_blade_and_wall_collide(self):
        m, _ = build_scene()
        colliding = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(m.ngeom)
                     if m.geom_contype[i] or m.geom_conaffinity[i]]
        self.assertEqual(sorted(colliding), ['blade_geom', 'wall_geom'])


class ControllerTests(unittest.TestCase):
    def test_ik_reaches_patch_with_blade_flat(self):
        c = controller()
        for u, v in [(0.018, 0), (-0.03, -0.04), (0.07, 0.04), (0.018, 0.05)]:
            c.reset([u, v, 0.0], 0.0, np.zeros(6))
            p, R = c.ik.fk(c.q_cmd)
            self.assertLess(np.linalg.norm(c.frame.from_world(p) - [u, v, 0]), 2e-3)
            self.assertLess(np.degrees(np.arccos(np.clip(R[:, 1] @ c.frame.R[:, 2], -1, 1))), 1.5)

    def test_unreachable_increment_is_rejected_not_half_applied(self):
        c = controller(); c.reset([0.018, 0, -0.03], 0, np.zeros(6))
        before_t, before_q = c.target.copy(), c.q_cmd.copy()
        c.target = np.array([0.018, 0, -0.5])     # 远在工作空间之外
        q, status, applied = c.step([0, 0, 0.001, 0])
        self.assertEqual(status, 'unreachable'); self.assertTrue(np.all(applied == 0))
        np.testing.assert_allclose(q, before_q)

    def test_action_validation(self):
        spec = ActionSpec()
        with self.assertRaises(ValueError): spec.clip([0, 0, 0])
        with self.assertRaises(ValueError): spec.clip([np.nan, 0, 0, 0])
        np.testing.assert_allclose(spec.clip([1, -1, 1, 1]), [0.01, -0.01, 0.004, 0.05])

    def test_joint_speed_is_limited(self):
        c = controller(); c.reset([0.018, 0, -0.03], 0, np.zeros(6))
        c.target = np.array([0.07, 0.04, -0.03])   # 大跳变
        q0 = c.q_cmd.copy(); q, status, _ = c.step([0, 0, 0, 0])
        self.assertLessEqual(np.abs(q - q0).max(), c.max_joint_step + 1e-12)


class TaskTests(unittest.TestCase):
    def test_nominal_teacher_covers_region_within_force_window(self):
        m, _ = RasterTeacher(WallTask(seed=0)).run()
        self.assertGreaterEqual(m['coverage'], 0.9)
        self.assertEqual(m['bottom_out_steps'], 0); self.assertEqual(m['unreachable_steps'], 0)
        self.assertGreaterEqual(m['in_window_frac_of_contact'], 0.95)

    def test_probe_and_servo_recover_misplaced_wall(self):
        p = Perturbation(wall_dn_m=0.010)        # 墙比名义远 1 cm
        bare, _ = RasterTeacher(WallTask(perturbation=p)).run()
        comp, log = RasterTeacher(WallTask(perturbation=p), probe=True, servo=True).run()
        self.assertGreater(comp['mean_contact_force_N'], bare['mean_contact_force_N'] + 1.0)
        self.assertGreaterEqual(comp['coverage'], bare['coverage'])
        self.assertAlmostEqual(log['wall_correction']['dn_m'], 0.010, delta=0.004)

    def test_latency_does_not_trap_teacher(self):
        m, _ = RasterTeacher(WallTask(perturbation=Perturbation(latency_steps=3))).run()
        self.assertLess(m['steps'], 400)

    def test_privileged_observations_declared(self):
        self.assertEqual(OBS_SPEC['contact_force_N']['availability'], 'privileged')
        self.assertEqual(OBS_SPEC['tcp_true_uvn_m']['availability'], 'privileged')
        self.assertEqual(OBS_SPEC['q_meas_rad']['availability'], 'hardware')


class EpisodeTests(unittest.TestCase):
    def test_record_and_replay_are_identical(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'ep.npz'
            pert = Perturbation.sample(np.random.default_rng(4))
            rec = record_episode(path, pert, seed=4, probe=True, servo=True)
            meta, fr = load_episode(path)
            self.assertTrue(meta['sample_time_known']); self.assertEqual(meta['n_frames'], len(fr['seq']))
            self.assertNotIn('contact_force_N', hardware_fields(meta))
            recorded, replayed = replay_episode(path)
            self.assertEqual(recorded, replayed); self.assertEqual(rec, recorded)

    def test_unknown_sample_time_requires_measured_latency_bound(self):
        meta = {'source': 'hw:test', 'sample_time_known': False, 'fields': {}}
        with self.assertRaises(ValueError): EpisodeWriter('unused.npz', meta)
        EpisodeWriter('unused.npz', {**meta, 'latency_upper_bound_s': 0.03})

    def test_sequence_gap_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            w = EpisodeWriter(Path(d) / 'e.npz', {'source': 'sim:t', 'sample_time_known': True, 'fields': {}})
            for s in (0, 1, 3):
                w.add({'seq': s, 't_sample_s': s, 't_host_s': s})
            with self.assertRaises(ValueError): w.close()

    def test_undeclared_field_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            w = EpisodeWriter(Path(d) / 'e.npz', {'source': 'sim:t', 'sample_time_known': True, 'fields': {}})
            w.add({'seq': 0, 't_sample_s': 0, 't_host_s': 0, 'mystery': 1.0})
            with self.assertRaises(ValueError): w.close()


class SimBackendGuardTests(unittest.TestCase):
    def test_out_of_ctrlrange_target_is_rejected_not_clamped(self):
        from dummy_loop.sim_backend import SimRobot
        r = SimRobot(); r.connect()
        with self.assertRaises(ValueError): r.send_action([1.0, 0, 0, 0, 0, 0])
        r.send_action([0.69, 0, 0, 0, 0, 0])


if __name__ == '__main__':
    unittest.main()

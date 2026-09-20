"""Studio 外观网格缺失时的退回显示：必须缺失可见、转向一致。"""
import re
import unittest
from pathlib import Path

import numpy as np
import mujoco

from dummy_loop.sim_backend import (MODEL, STUDIO_MODEL, STUDIO_TO_REFERENCE_SIGN, SimRobot,
                                    missing_model_assets)


def studio_kinematics_only():
    """去掉网格后加载 Studio 模型：只保留关节层级，不需要不可分发的网格文件。"""
    xml = STUDIO_MODEL.read_text(encoding='utf-8')
    xml = re.sub(r'<mesh [^>]*/>', '', xml)
    xml = re.sub(r'<geom [^>]*mesh="[^"]*"[^>]*/>', '', xml)
    return mujoco.MjModel.from_xml_string(xml)


def joint_frames(m, q):
    d = mujoco.MjData(m)
    ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f'Joint{i}') for i in range(1, 7)]
    d.qpos[m.jnt_qposadr[ids]] = q
    mujoco.mj_forward(m, d)
    return np.array([d.xanchor[j] for j in ids]), np.array([d.xaxis[j] for j in ids])


class StudioFallbackTests(unittest.TestCase):
    def test_sign_map_matches_both_model_files(self):
        _, a_s = joint_frames(studio_kinematics_only(), np.zeros(6))
        _, a_r = joint_frames(mujoco.MjModel.from_xml_path(str(MODEL)), np.zeros(6))
        np.testing.assert_array_equal(np.sign(np.einsum('ij,ij->i', a_s, a_r)), STUDIO_TO_REFERENCE_SIGN)

    def test_fallback_moves_wrist_in_the_same_direction(self):
        S = studio_kinematics_only(); R = mujoco.MjModel.from_xml_path(str(MODEL))
        p0s, _ = joint_frames(S, np.zeros(6)); p0r, _ = joint_frames(R, np.zeros(6))
        rng = np.random.default_rng(0)
        for _ in range(200):
            q = rng.uniform(-np.pi / 3, np.pi / 3, 6)
            ps, _ = joint_frames(S, q); pr, _ = joint_frames(R, STUDIO_TO_REFERENCE_SIGN * q)
            for k in (4, 5):
                ms, mr = ps[k] - p0s[k], pr[k] - p0r[k]
                if np.linalg.norm(ms) > 0.02:
                    cos = ms @ mr / np.linalg.norm(ms) / np.linalg.norm(mr)
                    self.assertGreater(cos, np.cos(np.deg2rad(30)))

    def test_missing_meshes_are_reported_clearly(self):
        if not missing_model_assets(STUDIO_MODEL):
            self.skipTest('Studio meshes present on this machine')
        with self.assertRaises(FileNotFoundError) as ctx:
            SimRobot(STUDIO_MODEL)
        self.assertIn('studio_meshes', str(ctx.exception))

    def test_reference_model_assets_complete(self):
        self.assertEqual(missing_model_assets(MODEL), [])


if __name__ == '__main__':
    unittest.main()

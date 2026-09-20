"""V2 模型核对：MuJoCo 模型的正运动学必须与固件 SolveFK 完全一致。

这是 V2 模型最重要的不变量：上位机把固件角度直接换算成模型关节角
q = deg2rad(固件角 - HOME) 后，画面里的末端位置就是固件认为的末端位置。
"""
import json
import unittest
from pathlib import Path
import numpy as np
import mujoco

from dummy_loop import v2
from dummy_loop.sim_backend import SimRobot, V2_MODEL

ROOT = Path(__file__).resolve().parents[1]


class DummyV2ModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = SimRobot(V2_MODEL).model
        cls.d = mujoco.MjData(cls.m)
        cls.site = cls.m.site('fw_end').id

    def fk_model(self, q_fw):
        self.d.qpos[:6] = v2.firmware_to_model(q_fw)
        mujoco.mj_kinematics(self.m, self.d)
        return self.d.site_xpos[self.site].copy(), self.d.site_xmat[self.site].reshape(3, 3).copy()

    def test_home_matches_firmware_numbers(self):
        p, _ = self.fk_model(v2.HOME_DEG)
        # 固件 V2 参数下 HOME 末端 = (D_BS + L_FA + L_WT, 0, L_BS + L_AM + D_EW)
        c = v2.DH_V2
        np.testing.assert_allclose(p, [c['D_BS'] + c['L_FA'] + c['L_WT'], 0, c['L_BS'] + c['L_AM'] + c['D_EW']], atol=1e-9)

    def test_fk_equals_firmware_solvefk(self):
        rng = np.random.default_rng(7)
        lo, hi = v2.FIRMWARE_LOWER_DEG_V2, v2.FIRMWARE_UPPER_DEG_V2
        for q in [v2.HOME_DEG, v2.FOLD_DEG] + [lo + (hi - lo) * rng.random(6) for _ in range(300)]:
            f = v2.firmware_fk(q)
            p, R = self.fk_model(q)
            np.testing.assert_allclose(p, f['end'], atol=1e-9, err_msg=str(q))
            np.testing.assert_allclose(R, f['R'][6], atol=1e-9, err_msg=str(q))

    def test_limits_and_actuators_follow_v2(self):
        lo, hi = v2.model_limits_rad()
        np.testing.assert_allclose(self.m.jnt_range[:6, 0], lo, atol=1e-6)
        np.testing.assert_allclose(self.m.jnt_range[:6, 1], hi, atol=1e-6)
        np.testing.assert_allclose(self.m.actuator_ctrlrange[:, 0], lo, atol=1e-6)
        peaks = [j['peak_Nm'] for j in v2.joint_params()]
        np.testing.assert_allclose(self.m.actuator_forcerange[:, 1], peaks, atol=1e-3)
        # J1..J3 由 MINI11-50 启停峰值 8.3 限制；J5 由 MINI8-50 的 3.3 限制；J6 直驱只剩电机本身
        self.assertEqual(peaks[:5], [8.3, 8.3, 8.3, 6.0, 3.3])   # J4: 35-28 保持转矩×50 = 6.0 < 8.3，电机先到限
        self.assertLess(peaks[5], 0.2)
        self.assertGreater(self.m.dof_armature[1], 100 * self.m.dof_armature[5])

    def test_end_is_bare_shaft(self):
        params = json.loads((ROOT / 'models' / 'dummy_v2_params.json').read_text(encoding='utf-8'))
        ee = params['end_effector']
        self.assertIn('bare', ee['kind'])
        self.assertTrue(3.0 < ee['diameter_mm'] < 8.0)
        tip = self.m.site('shaft_tip').pos
        self.assertAlmostEqual(tip[0] * 1000, ee['tip_from_wrist_mm'], places=2)
        self.assertEqual(self.m.body('link6').parentid, self.m.body('link5').id)

    def test_masses_are_plausible_and_positive(self):
        mass = self.m.body_mass[1:]
        self.assertTrue(np.all(mass > 0))
        self.assertTrue(2.0 < mass.sum() < 8.0, mass.sum())

    def test_mesh_hashes_match_provenance(self):
        import hashlib
        prov = json.loads((ROOT / 'models' / 'v2_provenance.json').read_text(encoding='utf-8'))
        for rel, h in prov['files'].items():
            if rel == 'dummy_v2.xml':
                continue
            self.assertEqual(hashlib.sha256((ROOT / 'models' / rel).read_bytes()).hexdigest(), h, rel)


if __name__ == '__main__':
    unittest.main()

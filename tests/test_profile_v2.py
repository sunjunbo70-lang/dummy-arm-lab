"""Profile schema v2 的门禁测试。

核心断言：未标定的档案必须加载失败。这不是待修复的缺陷，是设计意图——
参数没标定就等于没获得驱动实机的授权。
"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dummy_loop.core import (CALIBRATED_FIELDS, PROFILE_SCHEMA_VERSION,
                             SCHEMA_PATH, load_profile)

ROOT = Path(__file__).resolve().parents[1]


def calibrated_profile():
    """一份格式上完整的档案，仅用于测试校验逻辑，不代表任何真实设备的标定值。"""
    provenance = {f: {'method': 'unit-test fixture', 'measured_at': '2026-09-18',
                      'evidence_path': 'tests/test_profile_v2.py', 'evidence_level': 'L1'}
                  for f in CALIBRATED_FIELDS}
    return {
        'schema_version': PROFILE_SCHEMA_VERSION,
        'profile_id': 'unit_test_fixture',
        'calibration_verified': True, 'stop_verified': True,
        'physical_stop_verified': True, 'command_mode_verified': True,
        'firmware_identity': 'fixture-fw-0.0', 'calibration_record': 'tests/',
        'firmware_to_canonical_sign': [1, -1, 1, 1, 1, 1],
        'firmware_zero_deg': [0, -75, 180, 0, 0, 0],
        'lower_rad': [-1.0] * 6, 'upper_rad': [1.0] * 6,
        'max_step_rad': [0.05] * 6, 'max_speed_rad_s': [0.5] * 6,
        'firmware_speed_parameter': 5.0, 'command_mode': 2,
        'timing': {'control_period_s': 0.05, 'max_observation_age_s': 0.5,
                   'comm_timeout_s': 1.0, 'per_axis_freshness_available': False,
                   'observation_latency_upper_bound_s': 0.12},
        'gripper': None, 'camera': None,
        'provenance': provenance,
    }


def write(profile):
    handle = tempfile.NamedTemporaryFile('w', suffix='.json', delete=False, encoding='utf-8')
    json.dump(profile, handle, ensure_ascii=False)
    handle.close()
    return Path(handle.name)


class SchemaFileTests(unittest.TestCase):
    def test_schema_file_exists_and_is_valid_json(self):
        schema = json.loads(SCHEMA_PATH.read_text(encoding='utf-8'))
        self.assertEqual(schema['properties']['schema_version']['const'], PROFILE_SCHEMA_VERSION)

    def test_template_is_valid_json_and_declares_v2(self):
        template = json.loads((ROOT / 'configs' / 'hardware.dummy_v2.template.json')
                              .read_text(encoding='utf-8'))
        self.assertEqual(template['schema_version'], PROFILE_SCHEMA_VERSION)

    def test_template_refuses_to_load(self):
        """模板全是 null，必须加载失败——不能因为存在就被当成可用配置。"""
        with self.assertRaises(ValueError):
            load_profile(ROOT / 'configs' / 'hardware.dummy_v2.template.json')

    def test_legacy_unverified_profile_still_refuses(self):
        """v1 遗留档案缺 schema_version，必须以迁移提示失败。"""
        with self.assertRaises(ValueError) as ctx:
            load_profile(ROOT / 'configs' / 'hardware.unverified.json')
        self.assertIn('schema', str(ctx.exception).lower())


class GateTests(unittest.TestCase):
    def test_fixture_loads(self):
        profile, guard, sign, offset = load_profile(write(calibrated_profile()))
        self.assertEqual(profile['schema_version'], PROFILE_SCHEMA_VERSION)
        np.testing.assert_array_equal(sign, [1, -1, 1, 1, 1, 1])
        np.testing.assert_array_equal(offset, [0, -75, 180, 0, 0, 0])
        guard.validate([0.0] * 6, __import__('dummy_loop.core', fromlist=['Observation'])
                       .Observation(np.zeros(6), __import__('time').monotonic(), 'test'), 0.1)

    def test_wrong_schema_version_rejected(self):
        p = calibrated_profile(); p['schema_version'] = 1
        with self.assertRaises(ValueError):
            load_profile(write(p))

    def test_each_verification_flag_blocks(self):
        for flag in ('calibration_verified', 'stop_verified',
                     'physical_stop_verified', 'command_mode_verified'):
            with self.subTest(flag=flag):
                p = calibrated_profile(); p[flag] = False
                with self.assertRaises(ValueError):
                    load_profile(write(p))

    def test_sign_must_be_plus_minus_one(self):
        p = calibrated_profile(); p['firmware_to_canonical_sign'] = [1, 0, 1, 1, 1, 1]
        with self.assertRaises(ValueError):
            load_profile(write(p))


class TimingContractTests(unittest.TestCase):
    def test_missing_timing_block_rejected(self):
        p = calibrated_profile(); del p['timing']
        with self.assertRaises(ValueError):
            load_profile(write(p))

    def test_incomplete_timing_values_rejected(self):
        for key in ('control_period_s', 'max_observation_age_s', 'comm_timeout_s'):
            with self.subTest(key=key):
                p = calibrated_profile(); p['timing'][key] = None
                with self.assertRaises(ValueError):
                    load_profile(write(p))

    def test_unknown_per_axis_freshness_requires_latency_bound(self):
        """拿不到逐轴采样时刻时，必须写下实测延迟上界。

        否则采集的数据无法说明状态与动作的时间关系，训练会安静地学到错位映射。
        """
        p = calibrated_profile()
        p['timing']['per_axis_freshness_available'] = False
        p['timing']['observation_latency_upper_bound_s'] = None
        with self.assertRaises(ValueError) as ctx:
            load_profile(write(p))
        self.assertIn('observation_latency_upper_bound_s', str(ctx.exception))

    def test_per_axis_freshness_available_needs_no_bound(self):
        p = calibrated_profile()
        p['timing']['per_axis_freshness_available'] = True
        p['timing']['observation_latency_upper_bound_s'] = None
        load_profile(write(p))


class ProvenanceTests(unittest.TestCase):
    def test_missing_provenance_for_calibrated_field_rejected(self):
        for field in CALIBRATED_FIELDS:
            with self.subTest(field=field):
                p = calibrated_profile(); del p['provenance'][field]
                with self.assertRaises(ValueError) as ctx:
                    load_profile(write(p))
                self.assertIn(field, str(ctx.exception))

    def test_provenance_entry_needs_method_date_and_evidence(self):
        for key in ('method', 'measured_at', 'evidence_path'):
            with self.subTest(key=key):
                p = calibrated_profile()
                p['provenance']['lower_rad'][key] = '  '
                with self.assertRaises(ValueError):
                    load_profile(write(p))


if __name__ == '__main__':
    unittest.main()

from dataclasses import dataclass
from pathlib import Path
import json
import time
import numpy as np

PROFILE_SCHEMA_VERSION = 2
SCHEMA_PATH = Path(__file__).resolve().parents[1] / 'configs' / 'schema' / 'hardware_profile.v2.json'
# 标定得来的数值字段：置 calibration_verified 时每一项都必须有 provenance 记录。
CALIBRATED_FIELDS = ('firmware_to_canonical_sign', 'firmware_zero_deg', 'lower_rad',
                     'upper_rad', 'max_step_rad', 'max_speed_rad_s', 'firmware_speed_parameter')


def six(value, name='joints'):
    a = np.asarray(value, dtype=float)
    if a.shape != (6,) or not np.all(np.isfinite(a)):
        raise ValueError(f'{name} must contain exactly six finite numbers')
    return a


@dataclass
class Observation:
    q: np.ndarray
    received_at: float
    source: str
    device_freshness_known: bool = False


class Guard:
    def __init__(self, lower, upper, max_step, max_speed):
        self.lower, self.upper = six(lower), six(upper)
        self.max_step, self.max_speed = six(max_step), six(max_speed)
        if np.any(self.lower >= self.upper) or np.any(self.max_step <= 0) or np.any(self.max_speed <= 0):
            raise ValueError('Invalid limits')

    def validate(self, target, state, dt, now=None, max_age=0.5):
        q, target = six(state.q), six(target)
        now = time.monotonic() if now is None else now
        if not np.isfinite(dt) or dt <= 0 or not np.isfinite(state.received_at):
            raise ValueError('Invalid timestamps')
        if not 0 <= now - state.received_at <= max_age:
            raise ValueError('Stale observation')
        if np.any(q < self.lower) or np.any(q > self.upper):
            raise ValueError('Measured state outside calibrated limits')
        if np.any(target < self.lower) or np.any(target > self.upper):
            raise ValueError('Target outside calibrated limits')
        delta = np.abs(target - q)
        if np.any(delta > self.max_step + 1e-9) or np.any(delta / dt > self.max_speed + 1e-9):
            raise ValueError('Requested step/speed exceeds limits')
        return target


def validate_against_schema(profile):
    """装了 jsonschema 就做完整的声明式校验；没装则返回 False，由下面的手写检查兜底。

    schema 说明规则，本函数执行规则。两者必须一致——改了 schema 就要跑测试。
    """
    try:
        import jsonschema
    except ImportError:
        return False
    schema = json.loads(SCHEMA_PATH.read_text(encoding='utf-8'))
    try:
        jsonschema.validate(profile, schema)
    except jsonschema.ValidationError as exc:
        location = '/'.join(str(x) for x in exc.absolute_path) or '(根对象)'
        raise ValueError(f'Profile 不符合 schema v{PROFILE_SCHEMA_VERSION}: {location}: {exc.message}') from exc
    return True


def load_profile(path):
    """读取并校验设备档案。任何一项不合格都拒绝加载，而不是降级放行。

    这个函数同时是配置读取器和准入控制：参数没标定就等于没获得驱动实机的授权，
    在物理上本来就是同一件事。
    """
    p = json.loads(Path(path).read_text(encoding='utf-8'))

    version = p.get('schema_version')
    if version != PROFILE_SCHEMA_VERSION:
        raise ValueError(
            f'Profile schema 版本不符：需要 v{PROFILE_SCHEMA_VERSION}，实际为 {version!r}。'
            ' v1 档案缺少 timing/gripper/camera/provenance，必须先迁移；'
            ' 模板见 configs/hardware.dummy_v2.template.json')

    validate_against_schema(p)

    for key in ('calibration_verified', 'stop_verified', 'physical_stop_verified', 'command_mode_verified'):
        if p.get(key) is not True:
            raise ValueError(f'Hardware motion blocked: {key} is not true')
    for key in ('firmware_identity', 'calibration_record'):
        if not isinstance(p.get(key), str) or not p[key].strip():
            raise ValueError(f'Missing evidence: {key}')

    sign, offset = six(p['firmware_to_canonical_sign']), six(p['firmware_zero_deg'])
    if not np.all(np.isin(sign, [-1, 1])):
        raise ValueError('Mapping sign must be +/-1')
    # All safety limits refer to canonical radians, never motor revolutions.
    guard = Guard(p['lower_rad'], p['upper_rad'], p['max_step_rad'], p['max_speed_rad_s'])
    speed = p.get('firmware_speed_parameter')
    if not isinstance(speed, (int, float)) or not np.isfinite(speed) or not 0 < speed <= 100:
        raise ValueError('Missing verified firmware speed parameter')
    if p.get('command_mode') != 2:
        raise ValueError('This commissioning adapter only supports verified mode 2')

    timing = p.get('timing')
    if not isinstance(timing, dict):
        raise ValueError('Missing timing contract')
    for key in ('control_period_s', 'max_observation_age_s', 'comm_timeout_s'):
        value = timing.get(key)
        if not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f'Timing contract incomplete: {key} must be a positive number')
    if timing.get('per_axis_freshness_available') is not True:
        # 旧协议拿不到逐轴采样时刻。这不阻止运行，但必须写下实测的延迟上界，
        # 否则采集的数据无法说明状态和动作的时间关系。
        bound = timing.get('observation_latency_upper_bound_s')
        if not isinstance(bound, (int, float)) or not np.isfinite(bound) or bound <= 0:
            raise ValueError(
                'per_axis_freshness_available 为 false 时必须提供实测的'
                ' observation_latency_upper_bound_s；不得把主机接收时刻当作传感器采样时刻')

    provenance = p.get('provenance')
    if not isinstance(provenance, dict):
        raise ValueError('Missing provenance block')
    missing = [f for f in CALIBRATED_FIELDS if f not in provenance]
    if missing:
        raise ValueError(f'标定值缺少来源记录 provenance: {", ".join(missing)}')
    for field, record in provenance.items():
        if not isinstance(record, dict):
            raise ValueError(f'provenance[{field}] 必须是对象')
        for key in ('method', 'measured_at', 'evidence_path'):
            if not isinstance(record.get(key), str) or not record[key].strip():
                raise ValueError(f'provenance[{field}] 缺少 {key}')

    gripper = p.get('gripper')
    if isinstance(gripper, dict) and gripper.get('has_position_feedback') is not True:
        # 没有真实反馈时不得把开合指令当作实测开度，调用方需据此拒绝记录假反馈。
        gripper.setdefault('_feedback_is_command_echo', True)

    return p, guard, sign, offset

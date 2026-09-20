"""Episode 录制格式（仿真与实机共用）。

一个 episode 一个 .npz：每个逐帧字段一个数组（第一维 = 帧），外加一段 JSON 元数据 `__meta__`。
设计目标是让 M2 只读录包、M7 采集器、仿真示教三者写出同一种文件，训练代码只认这一种。

时间字段（M2 时效契约，见 docs/plan/M2_只读链路与时效契约.md）：
  seq          帧序号，严格递增、无缺号——断连或丢帧在这里可见
  t_sample_s   传感器采样时刻。仿真里精确已知；实机旧协议下未知 → 写 NaN，不许用主机时刻冒充
  t_host_s     主机收到这一帧的单调时钟时刻
元数据必须声明：
  source               'sim:…' 或 'hw:…'
  sample_time_known    布尔。为 False 时必须给 latency_upper_bound_s（实测上界）
  fields               每个字段的单位、形状、可得性（hardware / hardware_with_sensor / privileged）

privileged 字段只有仿真有。训练脚本应默认拒绝把它们当作策略输入（见 hardware_fields()）。
"""
import json, time
from pathlib import Path
import numpy as np

EPISODE_SCHEMA_VERSION = 1
REQUIRED_FRAME_FIELDS = ('seq', 't_sample_s', 't_host_s')
REQUIRED_META = ('source', 'sample_time_known', 'fields', 'created_unix')


class EpisodeWriter:
    def __init__(self, path, meta: dict):
        self.path = Path(path); self.meta = dict(meta); self.frames = {}
        self.meta.setdefault('created_unix', time.time())
        self.meta['episode_schema_version'] = EPISODE_SCHEMA_VERSION
        _check_meta(self.meta, partial=True)

    def add(self, frame: dict):
        for k in REQUIRED_FRAME_FIELDS:
            if k not in frame:
                raise ValueError(f'frame missing required field {k}')
        for k, v in frame.items():
            if k.startswith('_'):
                continue
            self.frames.setdefault(k, []).append(np.asarray(v))

    def close(self, extra_meta=None):
        if extra_meta:
            self.meta.update(extra_meta)
        arrays = {k: np.stack(v) for k, v in self.frames.items()}
        n = {len(a) for a in arrays.values()}
        if len(n) != 1:
            raise ValueError(f'fields have different frame counts: {n}')
        self.meta['n_frames'] = n.pop()
        _check_meta(self.meta)
        _check_seq(arrays['seq'])
        undeclared = [k for k in arrays if k not in self.meta['fields'] and k not in REQUIRED_FRAME_FIELDS]
        if undeclared:
            raise ValueError(f'fields not declared in meta.fields: {undeclared}')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.path, __meta__=np.array(json.dumps(self.meta, ensure_ascii=False)), **arrays)
        return self.path


def load_episode(path):
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z['__meta__']))
        frames = {k: z[k] for k in z.files if k != '__meta__'}
    if meta.get('episode_schema_version') != EPISODE_SCHEMA_VERSION:
        raise ValueError(f'unsupported episode schema {meta.get("episode_schema_version")}')
    _check_meta(meta); _check_seq(frames['seq'])
    return meta, frames


def hardware_fields(meta, allow_sensor=True):
    """可作为策略输入的字段：排除 privileged；allow_sensor=False 时再排除需加装传感器的字段。"""
    ok = {'hardware'} | ({'hardware_with_sensor'} if allow_sensor else set())
    return [k for k, spec in meta['fields'].items() if spec.get('availability') in ok]


def _check_meta(meta, partial=False):
    for k in REQUIRED_META:
        if k not in meta:
            if partial and k in ('created_unix',):
                continue
            raise ValueError(f'episode meta missing {k}')
    if not isinstance(meta['sample_time_known'], bool):
        raise ValueError('sample_time_known must be a boolean')
    if meta['sample_time_known'] is False:
        b = meta.get('latency_upper_bound_s')
        if not isinstance(b, (int, float)) or not np.isfinite(b) or b <= 0:
            raise ValueError('sample_time_known=False requires a measured latency_upper_bound_s; '
                             'host receive time must not stand in for sensor sample time')
    for k, spec in meta['fields'].items():
        if spec.get('availability') not in ('hardware', 'hardware_with_sensor', 'privileged', 'command'):
            raise ValueError(f'field {k}: availability must be hardware / hardware_with_sensor / privileged / command')


def _check_seq(seq):
    seq = np.asarray(seq)
    if seq.ndim != 1 or len(seq) == 0 or np.any(np.diff(seq) != 1):
        raise ValueError('seq must be a gap-free increasing integer sequence')

"""录制、回放、敏感性分析。"""
from dataclasses import asdict
import json, time
from pathlib import Path
import numpy as np

from ..episode import EpisodeWriter, load_episode
from .controller import WallFrame, ActionSpec
from .errors import Perturbation, single_factor_sweeps
from .scene import SceneConfig
from .task import WallTask, TaskConfig, OBS_SPEC
from .teacher import RasterTeacher

STATUS_CODE = {'ok': 0, 'rate_limited': 1, 'unreachable': 2}
FIELDS = dict(OBS_SPEC)
FIELDS.update({
    'action': {'shape': [4], 'units': ['m', 'm', 'm', 'rad'], 'availability': 'command',
               'note': '墙面系增量 (du, dv, dn, dpsi)，交给控制器前（含补偿修正）'},
    'q_cmd_rad': {'shape': [6], 'units': 'rad', 'availability': 'command', 'note': '控制器发出的关节目标（规范坐标）'},
    'ctrl_status': {'shape': [], 'units': 'enum', 'availability': 'command', 'note': '0 ok, 1 rate_limited, 2 unreachable'},
})


def record_episode(path, perturbation=None, seed=0, probe=False, servo=False, scene=None, task=None):
    env = WallTask(scene, task, perturbation, seed)
    teacher = RasterTeacher(env, probe=probe, servo=servo)
    meta = {'source': f'sim:dummy_wall_trowel:{env.scene.digest()}', 'sample_time_known': True,
            'fields': FIELDS, 'seed': seed, 'teacher': {'kind': 'raster_vertical', 'probe': probe, 'servo': servo},
            'evidence_level': 'L1', 'hardware_motion': False, **env.describe()}
    w = EpisodeWriter(path, meta)
    k = [0]

    def on_step(obs, action, info):
        w.add({'seq': k[0], 't_sample_s': obs['t_sample_s'], 't_host_s': obs['t_host_s'],
               'q_meas_rad': obs['q_meas_rad'], 'tcp_belief_uvn_m': obs['tcp_belief_uvn_m'],
               'target_uvn_m': obs['target_uvn_m'], 'compression_m': obs['compression_m'],
               'contact_force_N': obs['contact_force_N'], 'tcp_true_uvn_m': obs['tcp_true_uvn_m'],
               'action': action, 'q_cmd_rad': info['q_cmd'], 'ctrl_status': STATUS_CODE[info['status']]})
        k[0] += 1
    metrics, log = teacher.run(on_step=on_step)
    w.close({'metrics': metrics, 'teacher_log': log})
    return metrics


def replay_episode(path):
    """用录下的动作在同一场景、同一误差、同一种子下重放，返回 (录制时指标, 重放指标)。

    用于检验链路确定性：录下来的东西足以复现这一条 episode。
    """
    meta, fr = load_episode(path)
    sc = SceneConfig(**{**meta['scene_nominal'], 'wall_size': tuple(meta['scene_nominal']['wall_size']),
                        'camera_pos': tuple(meta['scene_nominal']['camera_pos'])})
    tk = TaskConfig(**{**meta['task'], 'region_u': tuple(meta['task']['region_u']),
                       'region_v': tuple(meta['task']['region_v']), 'force_window': tuple(meta['task']['force_window'])})
    pd = dict(meta['perturbation']); pd['joint_offset_rad'] = tuple(pd['joint_offset_rad'])
    env = WallTask(sc, tk, Perturbation(**pd), meta['seed'])
    env.reset()
    fu = (meta.get('teacher_log') or {}).get('frame_update')
    for i, a in enumerate(fr['action']):
        if fu and i == fu['at_seq']:
            old = env.ctrl.frame; new = WallFrame(np.array(fu['origin']), np.array(fu['R']))
            env.ctrl.target = new.from_world(old.to_world(env.ctrl.target)); env.ctrl.frame = new
        env.step(a)
    return meta['metrics'], env.metrics()


MODES = {'none': (False, False), 'probe': (True, False), 'servo': (False, True), 'probe+servo': (True, True)}


def run_study(out_dir, n_random=20, random_scale=1.0, seed=0):
    """单因素扫描 + 随机误差批量，四种补偿模式对比。"""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); single = {}
    for factor, levels in single_factor_sweeps().items():
        single[factor] = {}
        for value, pert in levels:
            single[factor][str(value)] = {}
            for mode, (pr, sv) in MODES.items():
                env = WallTask(perturbation=pert, seed=seed)
                m, _ = RasterTeacher(env, probe=pr, servo=sv).run()
                single[factor][str(value)][mode] = m
    baseline = {mode: RasterTeacher(WallTask(seed=seed), probe=pr, servo=sv).run()[0] for mode, (pr, sv) in MODES.items()}
    rng = np.random.default_rng(seed); perts = [Perturbation.sample(rng, random_scale) for _ in range(n_random)]
    randomized = {}
    for mode, (pr, sv) in MODES.items():
        ms = []
        for i, p in enumerate(perts):
            try:
                ms.append(RasterTeacher(WallTask(perturbation=p, seed=seed + i), probe=pr, servo=sv).run()[0])
            except RuntimeError as e:
                ms.append({'failed': str(e)})
        ok = [m for m in ms if 'failed' not in m]
        agg = lambda k: [float(np.mean([m[k] for m in ok])), float(np.min([m[k] for m in ok])), float(np.max([m[k] for m in ok]))]
        randomized[mode] = {'n': len(ms), 'failed': len(ms) - len(ok), 'per_episode': ms,
                            'coverage_mean_min_max': agg('coverage'),
                            'in_window_mean_min_max': agg('in_window_frac_of_contact'),
                            'max_force_mean_min_max': agg('max_contact_force_N'),
                            'bottom_out_episodes': int(sum(m['bottom_out_steps'] > 0 for m in ok)),
                            'overforced_episodes': int(sum(m['overforced_area_frac'] > 0 for m in ok))}
    result = {'evidence_level': 'L1', 'hardware_motion': False, 'seed': seed, 'random_scale': random_scale,
              'nominal_scene': SceneConfig().to_dict(), 'task': TaskConfig().to_dict(),
              'baseline_no_error': baseline, 'single_factor': single, 'randomized': randomized,
              'random_perturbations': [p.to_dict() for p in perts], 'elapsed_s': round(time.time() - t0, 1)}
    (out_dir / 'sensitivity.json').write_text(json.dumps(result, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    return result

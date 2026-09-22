"""Cheap gates before full wall-cycle v0.5 PPO training. Simulation only."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dummy_loop.wall_cycle.area import load_work_area
from dummy_loop.wall_cycle.config import CycleConfig
from dummy_loop.wall_cycle.mortar import MortarSystem
from dummy_loop.wall_cycle.train import evaluate


def config():
    return load_work_area(CycleConfig(physics='v0.5', tool_profile='lab_20260922',
                                      lift_wall_fraction=.75))


def reach_summary():
    path = Path(__file__).resolve().parents[2] / 'dummy_loop/wall_cycle/reach_table_lab_v05.npz'
    z = np.load(path, allow_pickle=False); ok = z['ok']
    return {'shape': list(ok.shape), 'feasible_fraction': float(ok.mean()),
            'pitch_deg': z['pitch_deg'].tolist(),
            'feasible_by_pitch': ok.mean(axis=(0,1,2,4)).tolist(),
            'work_points_any_feasible': float(ok.any(axis=(2,3,4)).mean()),
            'work_points_all_directions_at_zero_pitch': float(ok[:,:,:,0,:].all(axis=(2,3)).mean()),
            'meta': json.loads(str(z['meta']))}


def material_sweep(cfg):
    rows = []
    for tool_yield in (50., 80., 120.):
        for face in (-1., 0., .5, 1.):
            losses = []
            for seed in range(10):
                c = CycleConfig(**{**cfg.to_dict(), 'tool_interface_yield_Pa': tool_yield,
                                   'randomize_interface': False})
                m = MortarSystem(c, seed).reset(); m.feed(24, c.feed_normal_force_N,
                    c.feed_scoop_depth_m, c.feed_scoop_distance_m, c.feed_scoop_speed_m_s)
                supplied = m.supplied_m3; loss = m.transport(face, c.carry_duration_s)
                losses.append(loss/max(supplied, 1e-12))
                if abs(m.volume_balance()) > 1e-12:
                    raise RuntimeError('material volume balance failed')
            rows.append({'tool_interface_yield_Pa': tool_yield, 'face_up_score': face,
                         'carry_loss_fraction_mean': float(np.mean(losses)),
                         'carry_loss_fraction_std': float(np.std(losses))})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--episodes', type=int, default=30)
    a = ap.parse_args(); c = config()
    result = {'experiment': 'wall_cycle_v0.5_pretraining_gates', 'evidence_level': 'L1',
              'hardware_motion': False, 'config': c.to_dict(), 'reach': reach_summary(),
              'material_sweep': material_sweep(c),
              'teacher_face_up': evaluate(c, None, a.episodes, 20000, 'technique', 'table'),
              'teacher_legacy_transport': evaluate(c, None, a.episodes, 20000,
                                                   'legacy_transport', 'table')}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'written': str(a.out), 'face_up': result['teacher_face_up'],
                      'legacy': result['teacher_legacy_transport']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

"""Precomputed stroke feasibility for training (fast stand-in for ArmExecutor.plan).

WHY: planning one stroke on the Dummy V2 model (continuation IK for approach, contact,
work path, lift, retreat, with arm-wall and table clearance) costs ~0.1 s. PPO needs tens
of thousands of decisions, so training cannot plan every stroke. Instead this table stores,
for a grid over the work square x blade angle x pitch x motion sense, whether the FULL
planner (arm.py) can execute a short stroke there. During training a stroke is accepted
if its start, middle and end cells are all feasible.

The table is an approximation of the planner (a long stroke can fail between two feasible
cells; nearest-cell lookup can be optimistic at the boundary). Evaluation and the replay
therefore run the full planner plus MuJoCo co-simulation, and the report states how many
strokes the table accepted but the planner rejected ("table/planner disagreement").

    python -m dummy_loop.wall_cycle.reach_table            # rebuild for the current work area
"""
import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import CycleConfig

TABLE_FILE = Path(__file__).with_name('reach_table.npz')
PSI_BINS = 8                       # blade-angle bins over 360 deg
PITCH_BINS_DEG = (0.0, 10.0, 20.0, 30.0)


@dataclass
class TablePlan:
    ok: bool
    reason: str = ''


def _bins(cfg, step):
    nu = int(np.floor(cfg.width_m / step)) + 1
    nv = int(np.floor(cfg.height_m / step)) + 1
    us = np.linspace(-cfg.width_m / 2, cfg.width_m / 2, nu)
    vs = np.linspace(0.0, cfg.height_m, nv)
    return us, vs


def build(cfg: CycleConfig, step=0.02, stroke_len=0.02, log=print):
    """Run the full planner for every cell. ok[iv, iu, psi_bin, pitch_bin, sense]."""
    from .arm import ArmExecutor
    from .env import DecodedAction
    ex = ArmExecutor(cfg)
    us, vs = _bins(cfg, step)
    phis = np.arange(PSI_BINS) * 2 * np.pi / PSI_BINS
    ok = np.zeros((len(vs), len(us), PSI_BINS, len(PITCH_BINS_DEG), 2), bool)
    t0 = time.time()
    for iv, v in enumerate(vs):
        for iu, u in enumerate(us):
            for ip, phi in enumerate(phis):
                e_w = np.array([-np.sin(phi), np.cos(phi)])
                for sense, sgn in enumerate((1.0, -1.0)):
                    d_uv = sgn * e_w * stroke_len
                    start = np.clip(np.array([u, v]) - d_uv / 2, [-cfg.width_m / 2, 0], [cfg.width_m / 2, cfg.height_m])
                    end = np.clip(np.array([u, v]) + d_uv / 2, [-cfg.width_m / 2, 0], [cfg.width_m / 2, cfg.height_m])
                    for ib, pitch in enumerate(PITCH_BINS_DEG):
                        d = DecodedAction('REUSE', tuple(start), tuple(end), phi, 0.0, 2.0, 0.06, 0.0,
                                          np.deg2rad(pitch), np.deg2rad(pitch))
                        ok[iv, iu, ip, ib, sense] = ex.plan(d).ok
        if log:
            log(f'reach table row {iv + 1}/{len(vs)}  {time.time() - t0:.0f}s  feasible so far {ok[:iv + 1].mean():.2f}')
    return {'ok': ok, 'u': us, 'v': vs, 'phi': phis, 'pitch_deg': np.array(PITCH_BINS_DEG),
            'meta': json.dumps({'width_m': cfg.width_m, 'height_m': cfg.height_m,
                                'scene_wall_distance_m': cfg.scene_wall_distance_m,
                                'area_centre_u_m': cfg.area_centre_u_m, 'area_centre_z_m': cfg.area_centre_z_m,
                                'step_m': step, 'stroke_len_m': stroke_len,
                                'built_s': round(time.time() - t0, 1)})}


class ReachTableExecutor:
    """Executor interface used by WallCycleEnv during training (no dynamics)."""
    dynamic = False

    def __init__(self, cfg: CycleConfig, path: Path = TABLE_FILE):
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z['meta']))
        for k in ('width_m', 'height_m', 'scene_wall_distance_m', 'area_centre_u_m', 'area_centre_z_m'):
            if abs(meta[k] - getattr(cfg, k)) > 1e-6:
                raise ValueError(f'reach table was built for {k}={meta[k]}, config has {getattr(cfg, k)}; '
                                 f'rebuild with python -m dummy_loop.wall_cycle.reach_table')
        self.ok, self.u, self.v, self.meta = z['ok'], z['u'], z['v'], meta
        self.pitch = z['pitch_deg']
        self.stats = {'planned': 0, 'rejected': 0}

    def _cell(self, uv, phi, pitch_rad, sense):
        iu = int(np.argmin(np.abs(self.u - uv[0]))); iv = int(np.argmin(np.abs(self.v - uv[1])))
        ip = int(np.round((phi % (2 * np.pi)) / (2 * np.pi / PSI_BINS))) % PSI_BINS
        # conservative: round the pitch UP to the next tabulated bin
        ib = int(np.searchsorted(self.pitch, np.rad2deg(pitch_rad) - 1e-6))
        ib = min(ib, len(self.pitch) - 1)
        return bool(self.ok[iv, iu, ip, ib, sense])

    def plan(self, d):
        self.stats['planned'] += 1
        p0, p2 = np.asarray(d.start, float), np.asarray(d.end, float)
        vec = p2 - p0
        e_w = np.array([-np.sin(d.blade_angle), np.cos(d.blade_angle)])
        sense = 0 if float(vec @ e_w) >= 0 else 1
        for t, pitch in ((0.0, d.pitch_start), (0.5, 0.5 * (d.pitch_start + d.pitch_end)), (1.0, d.pitch_end)):
            if not self._cell(p0 + vec * t, d.blade_angle, pitch, sense):
                self.stats['rejected'] += 1
                return TablePlan(False, f'reach table: infeasible at t={t:.1f}')
        return TablePlan(True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, default=TABLE_FILE)
    ap.add_argument('--step', type=float, default=0.02)
    a = ap.parse_args(argv)
    from .area import load_work_area
    cfg = load_work_area(CycleConfig())
    t = build(cfg, a.step)
    np.savez_compressed(a.out, **t)
    print(json.dumps({'written': str(a.out), 'feasible_fraction': float(t['ok'].mean()), **json.loads(t['meta'])}))


if __name__ == '__main__':
    main()

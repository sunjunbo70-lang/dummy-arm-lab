"""Single-stroke sweep of the v0.3 mortar model: pitch x force x yield stress.

Answers one question before any learning: does blade pitch matter in the physics model,
and if so which way, WITHOUT any rule that mentions pitch? (experiments/v0.3/r0/design/2026-09-22_wall_cycle_v0.3.md, C3)

One upward stroke through the middle of a 12 x 12 cm patch with a random 24 ml load (4 seeds),
constant pitch, nominal path (no arm). Reported per (tau_y, pitch, force): mean layer in the
stroke band, share of the supplied material wasted (fell off / pushed out), share still on
the blade. L1, assumed material parameters.

    python tools/simulation/mortar_sweep.py --out experiments/00_initial_debug/runs/wall_cycle_unclassified/mortar_sweep.json
"""
import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dummy_loop.wall_cycle.config import CycleConfig      # noqa: E402
from dummy_loop.wall_cycle.mortar import MortarSystem     # noqa: E402

TAUS = (100, 250, 500, 1000)
PITCHES = (0, 5, 10, 15, 20, 30)
FORCES = (0.5, 1, 2, 3, 5, 8, 12)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, default=Path('experiments/00_initial_debug/runs/wall_cycle_unclassified/mortar_sweep.json'))
    a = ap.parse_args(argv)
    base = CycleConfig(width_m=0.12, height_m=0.12)
    rows = []
    for tau in TAUS:
        cfg = replace(base, mortar_tau_y_Pa=float(tau))
        for pitch in PITCHES:
            for F in FORCES:
                band, waste, left = [], [], []
                for seed in range(4):
                    m = MortarSystem(cfg, seed).reset(); m.load_random(24)
                    m.stroke('DEPOSIT', (0, .005), (0, .115), np.pi, 0, F, .06,
                             pitch_start=np.deg2rad(pitch), pitch_end=np.deg2rad(pitch))
                    b = m.wall[:, np.abs(m.u) < 0.05]
                    band.append(b.mean() * 1e3)
                    waste.append((m.dropped_m3 + m.outside_m3) / m.supplied_m3)
                    left.append(m.blade_volume_m3 / m.supplied_m3)
                rows.append({'tau_y_Pa': tau, 'pitch_deg': pitch, 'force_N': F, 'band_mm': float(np.mean(band)),
                             'waste_frac': float(np.mean(waste)), 'left_on_blade_frac': float(np.mean(left))})
    summary = {}
    for tau in TAUS:
        rs = [r for r in rows if r['tau_y_Pa'] == tau]
        best = max(rs, key=lambda r: r['band_mm'] - 2 * r['waste_frac'])
        per_pitch = {}
        for p in PITCHES:
            rp = [r for r in rs if r['pitch_deg'] == p]
            bp = max(rp, key=lambda r: r['band_mm'] - 2 * r['waste_frac'])
            per_pitch[p] = {'force_N': bp['force_N'], 'band_mm': round(bp['band_mm'], 3), 'waste_frac': round(bp['waste_frac'], 3)}
        summary[tau] = {'best': best, 'best_per_pitch': per_pitch}
    out = {'evidence_level': 'L1', 'score': 'band_mm - 2 * waste_frac (single stroke)', 'rows': rows, 'summary': summary,
           'setup': '12x12 cm patch, one upward stroke u=0 v 5->115 mm, random 24 ml load, 4 seeds, speed 0.06 m/s'}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1) + '\n', encoding='utf-8')
    for tau, s in summary.items():
        print(tau, 'best pitch', s['best']['pitch_deg'], 'force', s['best']['force_N'],
              {p: (v['band_mm'], v['waste_frac']) for p, v in s['best_per_pitch'].items()})


if __name__ == '__main__':
    main()

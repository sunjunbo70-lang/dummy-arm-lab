"""Work area of the whole-wall experiment, chosen by the user's rule (2026-09-22):

    largest circle inside the arm's real reachable set on the wall
    -> inscribed square -> side x 0.8 (safety margin)

The reachable set comes from tools/simulation/v2_trowel_reach.py (Dummy V2 model +
rigid trowel, both working pitches 0 and 30 deg solvable, no arm part within 5 mm of
the wall, V2 firmware joint limits). The chosen square is stored in work_area.json next
to this file, with the analysis it came from. Joint limits are firmware CANDIDATES
until M3 measures them; re-run the analysis then and this square changes with it.
"""
import json
from dataclasses import replace
from pathlib import Path

WORK_AREA_FILE = Path(__file__).with_name('work_area.json')


def square_from_reach(best: dict, cell_m: float):
    """Round the square side DOWN to whole material cells so it never grows past 80 %."""
    side = best['work_square_side_m']
    n = int(side / cell_m + 1e-9)
    side = n * cell_m
    cu, cz = best['circle_centre_uv_m']
    return {'wall_distance_m': best['wall_distance_m'], 'side_m': round(side, 4),
            'centre_u_m': cu, 'centre_z_m': cz}


def load_work_area(cfg, path: Path = WORK_AREA_FILE):
    """Return cfg with width/height/centre/wall distance set from work_area.json (v0.3 only)."""
    if cfg.physics == 'v0.2' or not Path(path).is_file():
        return cfg
    a = json.loads(Path(path).read_text(encoding='utf-8'))['square']
    return replace(cfg, width_m=a['side_m'], height_m=a['side_m'], scene_wall_distance_m=a['wall_distance_m'],
                   area_centre_u_m=a['centre_u_m'], area_centre_z_m=a['centre_z_m'])

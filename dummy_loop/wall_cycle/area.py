"""Work area of the whole-wall experiment, chosen by the user's rule
(2026-09-22, area rule corrected 2026-09-23 / v0.7):

    largest circle inside the arm's real reachable set on the wall
    -> reduce AREA by area_safety, same centre -> inscribed square = SCORED work area
    -> the un-reduced circle's own inscribed square = SIMULATION/PHYSICAL work area

v0.6 and earlier applied the 0.8 safety factor to the inscribed square's SIDE, not to the
circle's area (tools/simulation/v2_trowel_reach.py's old work_square()); that made the scored
area about 25% smaller than the user's intended "80% of area, then inscribe" rule. v0.7 fixes
the formula and, at the same time, splits the single square into two concentric ones: overtravel
into the (small) margin between the two is allowed and is not counted as waste (see
material.py's score-region crop and env.py:decode()'s clip bound, both keyed off these two
squares). Both share the same centre because they come from the same reachable circle.

The reachable set comes from tools/simulation/v2_trowel_reach.py (Dummy V2 model +
rigid trowel, both working pitches 0 and 30 deg solvable, no arm part within 5 mm of
the wall, V2 firmware joint limits). The chosen squares are stored in work_area.json next
to this file, with the analysis they came from. Joint limits are firmware CANDIDATES
until M3 measures them; re-run the analysis then and these squares change with it.
"""
import json
from dataclasses import replace
from pathlib import Path

WORK_AREA_FILE = Path(__file__).with_name('work_area.json')


def square_from_reach(best: dict, cell_m: float):
    """Round each square's side DOWN to whole material cells so it never grows past its
    safety-derived value. `best` is one row of tools/simulation/v2_trowel_reach.py's output
    (post-v0.7: carries score_square_side_m and sim_square_side_m, not work_square_side_m)."""
    def rounded(key):
        n = int(best[key] / cell_m + 1e-9)
        return round(n * cell_m, 4)
    cu, cz = best['circle_centre_uv_m']
    common = {'wall_distance_m': best['wall_distance_m'], 'centre_u_m': cu, 'centre_z_m': cz}
    return {'score_square': {**common, 'side_m': rounded('score_square_side_m')},
            'sim_square': {**common, 'side_m': rounded('sim_square_side_m')}}


def load_work_area(cfg, path: Path = WORK_AREA_FILE):
    """Return cfg with width/height (= sim/physical square), score_width/score_height (=
    scored square), centre and wall distance set from work_area.json (v0.3+ only). The two
    squares are concentric, so a single centre/wall-distance pair covers both."""
    if cfg.physics == 'v0.2' or not Path(path).is_file():
        return cfg
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    sim, score = data['sim_square'], data['score_square']
    return replace(cfg, width_m=sim['side_m'], height_m=sim['side_m'],
                   score_width_m=score['side_m'], score_height_m=score['side_m'],
                   scene_wall_distance_m=sim['wall_distance_m'],
                   area_centre_u_m=sim['centre_u_m'], area_centre_z_m=sim['centre_z_m'])

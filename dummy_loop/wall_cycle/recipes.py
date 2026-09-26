"""Named experiment recipes for the v0.6-physics whole-wall task.

One place builds the configuration used by training (train.py), replay (view.py) and the
tests, so a replay or a re-run can never silently use different budgets or reward terms from
the run it claims to reproduce.

- 'v0.7': the 2026-09-23 P2 run exactly (reward reshaping, split score/sim squares).
- 'v0.8': 2026-09-24 plan (experiments/v0.8/r0/design/2026-09-24_wall_cycle_v0.8.md): local teacher with
  plateau stop, observed teacher memory, three-ledger loss accounting, outside material at a
  quarter of the waste rate, long budgets with the episode ended by FINISH/stall, feed-transit
  transport physics.

Physics, action space, work area and reach table are identical in both.
"""
from .area import load_work_area
from .config import CycleConfig

RECIPES = {
    'v0.7': dict(base_steps=12000, max_steps=24000, extension_steps=2000, max_cycles=160,
                 max_reload_cycles=60, stall_limit=1, stall_window=12, min_cycles_before_stall=25),
    'v0.8': dict(base_steps=60000, max_steps=100000, extension_steps=5000, max_cycles=2000,
                 max_reload_cycles=60, stall_limit=1, stall_window=20, min_cycles_before_stall=60,
                 teacher_targeting='local', obs_memory=True, loss_accounting='v0.8',
                 outside_coef=1.0, time_coef=0.00005, finish_fail_penalty=2.0,
                 feed_transit_physics=True),
}

# Seeds: v0.7 kept its original sets. For v0.8 the 20000+ set is no longer an independent test
# (its first 8 seeds were used to choose the teacher), so validation/test move to fresh sets and
# 20000+ is kept only as a historical regression set.
SEEDS = {
    'v0.7': {'val': 30000, 'test': 20000, 'legacy': None},
    'v0.8': {'val': 40000, 'test': 50000, 'legacy': 20000},
}


def make_config(recipe='v0.8', use_arm=True, **overrides):
    if recipe not in RECIPES:
        raise ValueError(f'unknown recipe {recipe!r}; choose from {sorted(RECIPES)}')
    kw = dict(use_arm=use_arm, physics='v0.6', tool_profile='lab_20260922', lift_wall_fraction=.75,
              **RECIPES[recipe])
    kw.update(overrides)
    return load_work_area(CycleConfig(**kw))

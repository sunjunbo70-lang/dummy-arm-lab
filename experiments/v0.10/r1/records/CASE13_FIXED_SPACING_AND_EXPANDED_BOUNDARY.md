# Case13 fixed-spacing result and expanded boundary audit

L1 numerical evidence only. Formal RL has not started.

Original case13 at frozen 50/25 micrometre spacings (4836/9672 segments) completes. Pre-lift maximum wall difference 0.0000547704 mm; post-lift 0.000550873 mm; post-lift RMSE difference 0.00000159875 mm; coverage difference zero. Ledger residual at most 2.71e-14 mL; no negative volume. This passes the single-case criteria, not the full material gate.

Evidence: runs/boundary_pressure_case13_gatepair_001/{summary.json,quality_comparison.json,prelift_*.npz,final_*.npz}. Coarse 256/512 result remains preserved and fails accuracy.

Started expanded_boundary_001: 100 single cases plus 50 sequences of 100 actions, unchanged spacings/thresholds/scenes/action recipes from expanded_split_audit, BoundaryPressure v8.1. Manager PID 32992 (launcher 40600), four workers to coexist with old 12-worker v4 audit. No duplicate resource monitor. Existing resource stream continues; new audit records per-case elapsed time. Source snapshot includes current dirty source; do not infer snapshot equals git HEAD alone.

Next: check actual processes and cases/summary in expanded_boundary_001; do not restart. Preserve exceptions and failed cases. Passing case13 does not prove broad equivalence. Full continuous robot executor/GPU environment/trainer and training/evaluation/replay remain outstanding.

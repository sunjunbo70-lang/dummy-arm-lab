# Temporal control candidate rejected; pressure subcycle diagnostic

L1 software. v8.5 case76_temporal_001 finished both resolutions; wall difference0.01633718116mm still exceeds0.01mm and does not improve v8.4. Do not adopt it as a repair. Terminal pressure comparison is saved: both gaps0.0009312020588m, support13.889882N vs requested10.141783N; bead inventory4.889994/4.888454mL. Non-smooth pressure/contact transitions remain possible; this does not by itself prove root cause.

New diagnostic v8.6 SubcyclePressure inherits v8.4 and uses four internal SSPRK2 substeps per previous constitutive interval, retaining midpoint controls, external50/25micrometre spacing, geometry, physical laws and acceptance criteria. This is controlled internal solver refinement, not a gate-spacing adjustment or frozen physics. Component check passed:8flux evaluations instead of2, nonnegative wall and conservation. Evidence runs/case76_subcycle_001/component_check.json.

Case76 v8.6 at3344/6688segments running runs/case76_subcycle_001 PID5688 (launcher32952), saves pre/post lift arrays. Evaluate accuracy AND added runtime; do not claim success before summary. Long v8.4 audit40668 remains alive, first four sequences35/30/37/41 operations at snapshot; no complete100-action sequence yet. Do not duplicate or modify these modules while running. Read-only auxiliary stdin diagnostic session6617 remained pending without output; its intended terminal-pressure computation was successfully saved separately, not needed for adoption.

Formal RL has not begun. Full material G0, robot G1, GPU environment/training and final evaluation/replay remain missing. Previous failures and original user modifications preserved.

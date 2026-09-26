# Matched-time evidence and quarter-control exit-gap candidate

L1 software. case76_divergence_001 completed. Largest incremental bead difference at step2777, t=0.8304425837320574: 3.0283812431494347e-05mm per coarse interval. Both terminal gaps at this point equal 0.0008464262676239013m. Nearby intervals show smooth accumulation, not one large jump. Thus the previous temporal-control/internal-subcycling hypotheses were not supported as sufficient repairs.

Independent v8.7 QuarterGapPressure uses interpolated controls at quarter/three-quarter time when solving exit gaps for the two geometry half-steps. It uses current available inventories, not fully predicted quarter-time inventories; therefore do not claim a proven second-order coupled solver. Physical pressure law/material accounting/external50/25micrometre sampling and acceptance criteria unchanged. Two tests passed: constant-control short path equals v8.4, changing-control stage timing and conservation/nonnegative inventory. No full-G0 claim.

Original case76 at3344/6688segments running runs/case76_quarter_gap_001 actualPID29260 (launcher9672). Read summary and saved pre/postlift arrays before adopting or rejecting; don't duplicate. Previous candidates remain preserved. Long-sequence v8.4 audit40668 continues, first four sequences67/66/75/72 operations at snapshot; no completed100-action sequence then. Single5sec telemetry13564 is alive, no new monitor.

Formal RL not started. Complete material/robot gates, GPU environment/PPO/evaluation/native full-job replay remain outstanding. No hardware action, no user-source modifications.

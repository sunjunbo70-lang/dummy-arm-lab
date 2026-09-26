# Continuous sequence expansion and remaining single76 precision

L1 software only. stationary_sequence_001 completed4operations, passed: maximum wall difference0.00173261699mm, RMSE difference3.3352e-6mm, coverage difference0, ledger1.9652e-13mL. This establishes short continuation execution, not full G0.

Expanded v8.2 finished100singles:99pass; only single76 fails at0.01632131053mm versus0.01mm threshold. Existing long sequences in v8.2 still inherit dtype bug; failures preserved. v8.4 long sequence-only audit started runs/expanded_stationary_sequences_001, manager40668 (launcher25032),4workers,50sequencesx100actions. Do not duplicate. It intentionally reports required_budget_complete=false because singles=0; do not misreport this as full same-version G0. Any later reuse of v8.2 singles requires explicit equivalence rationale/testing.

Remaining single76 split diagnostic runs/case76_pressure_001, manager9672 (launcher31168), original recipe and make_scene(16) (76 modulo60),3344/6688segments at frozen spacing. Saves prelift/final wall/blade/bead. When finished compare pre/post differences and bead contribution to find remaining precision defect; do not shrink spacing solely to pass or change thresholds. Short sequence and single76 physics v8.4.

Component unittest discover still alive actual23116, launcher35204, session85155, with no output. Result remains pending, not a pass. Diagnose that process if it remains stalled; do not create duplicate full-suite runs. Existing old v4 audit33596 remains alive. Formal RL/GPU environment/robot G1 not complete. Preserve user dirty sources and all historical data.



# Fixed-step wall-cap case76 passed; same-version expanded audit started

L1 software. Adaptive v8.10 case76 rejected both resolutions near end:3344 at3199,error7.72072689e-8mm;6688 at6398,error3.55459335e-8mm,depth6. Local adaptive estimator was introduced as a diagnostic, not part of the original frozen G0 thresholds. These failures remain recorded and must not be relabelled successful.

v8.11 FixedCapPressure uses v8.10 wall-cap physical ordering with the original StationaryBoundaryPressure fixed-step contact integrator, so the actual prescribed50/25micrometre endpoint comparison can be measured. No global acceptance threshold changed. Removing adaptive rejection does not establish convergence on its own; the global test remains decisive. Physics version distinct, not frozen.

Original case76 pair completed:3344 steps59.41s,6688 steps84.60s, max final wall difference0.00313508934mm (threshold0.01mm), ledger0.0mL at logged precision, minimum0.0. Earlier v8.4 same case difference0.0163281157mm. Candidate passes this pair, not full G0. Evidence runs/case76_fixed_cap_001/cases.jsonl. Four tests passed including100 random positive exchange cases verifying cap/nonnegativity/total ledger, no-cap identity and contact/lift ledger.

Full same-version audit launched runs/expanded_fixed_cap_001, actualPID22620 launcher24180:100singles+50sequences of100operations,8workers, unchanged50/25micrometre spacings and thresholds. Source snapshot includes new candidate files; commit follows. All old results preserved. Original v8.4 long-sequence manager40668 remains active with4workers and monitor13564; no duplicate monitor. Existing resource log provides machine-wide CPU/GPU utilization but per-process tree is only original manager, not new audit: do not claim per-process monitoring of new workers. Total12 CPU workers across both audits; no GPU benchmark claim.

Next: inspect actual jobs/results, if expanded gate fails locate exact case; if material subsystem passes proceed full environment/G1 and GPU integration. Formal RL has not started; full executor/GPU environment/PPO/evaluation/native full-job replay still missing. No hardware commands/user source changes.

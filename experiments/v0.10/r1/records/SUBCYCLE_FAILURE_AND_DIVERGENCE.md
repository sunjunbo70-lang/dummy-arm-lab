# Pressure subcycling failed; matched-time divergence trace

L1 software. v8.6 case76_subcycle_001 completed both3344/6688 resolutions, maxwall difference0.01635887253mm vs0.01mm, ledger1.36e-14mL. Fourfold internal SSPRK2 refinement did not improve v8.4; not adopted. Preserve this negative result alongside v8.5. No reduced acceptance threshold or removed actions.

New runs/case76_divergence_001 (actualPID2408,launcher28912) advances both v8.4 states at matching path times; logs gap, per-column bead delta, blade/wall error and largest incremental bead changes. This is diagnostic only, not a new physical model. Read summary/top_jumps and trace.jsonl when done before proposing more numerical changes. Existing long sequence manager40668 remains active; do not duplicate or change imported modules.

Resource telemetry audit found no active telemetry process after old monitored audits ended. Restored one5second monitor PID13564 targeting40668 at expanded_stationary_sequences_001/resources_resumed.jsonl. Previous interval is a monitoring gap, cannot claim continuous coverage. Existing raw resource logs preserved; no duplicate current monitor.

Formal RL has not begun; numerical G0 not passed. Full GPU environment, robot continuous execution G1, PPO campaign, evaluations and native full-job replay still missing. Component suite83 tests and separate temporal1test remain latest pass; no test success asserted for new divergence diagnostic. Work only D:/VLA, no hardware.

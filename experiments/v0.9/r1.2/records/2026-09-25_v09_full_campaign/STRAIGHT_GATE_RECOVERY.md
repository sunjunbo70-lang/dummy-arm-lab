# Straight gate recovery (L1)

Original result preserved: campaign_001/training/straight_seed11_0100.
BC executable contacts: 72.3164% of 354 requests, below unchanged 80% gate; improvement criterion passed. No PPO ran in this group.

The straight flag replaces the two internal Bezier control points with collinear points. An unchanged curved-action policy therefore executes a different path and visits different subsequent states. This is a plausible distribution shift, not proof of the sole cause.

Recovery: wait for both original remaining ablations; adopt all completed runs without rerunning. Collect 40 straight-environment DAgger episodes using the failed BC checkpoint, seeds 60720–60759, then 6000 BC updates with existing teacher400 plus these labels. The unchanged gate determines whether PPO100 may proceed. Save all new outputs in campaign_002_straight_recovery. Core environment, policy, reward and physics unchanged.

This adapted straight initialization is a methodological confound. Report the original transfer failure and adaptation cost; do not attribute differences to trajectory curvature alone. No final-test-driven tuning. If gate fails again, retain results and inspect before proceeding.

The continuation uses the same frozen expansion, selection and independent-evaluation rules. Original campaign_001 remains the preserved failed campaign record.

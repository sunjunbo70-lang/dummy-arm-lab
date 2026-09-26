

# Error-control budget failure and pressure precision diagnostic

L1 software. v8.8 case76_error_control_001 failed both resolutions at fixed depth6: 3344 steps failed at2314 (error1.6882930473e-8mm versus tolerance1.5621328488e-8mm); 6688 failed at5463 (7.8121361681e-9 versus7.8111642441e-9mm). No tolerance or depth increase. Failure evidence retained.

v8.9 PreciseGapPressure tests whether the20-iteration pressure bisection precision floor contributes to failed adaptive convergence. Same support law/bracket,40 iterations for subsequent pressure evaluations, same v8.8 local tolerance/depth. Initial-contact inherited solver is unchanged. Support has wet/dry discontinuities, so more iterations do not guarantee a force root or resolve the case. This is a candidate diagnostic, not frozen physics.

Two targeted tests passed (uniform support root precision; conservative nonnegative adaptive short advance/lift). Case76 diagnostic launched with original3344/6688 external segments in runs/case76_precise_gap_001: actualPID17240, launcher33596. Source snapshot includes dirty diff; new files committed separately. Check cases/summary before any expansion.

v8.4 long-sequence manager40668 remains active; four of50 complete and pass, remaining pending. Existing single telemetry13564 retained. Full G0 remains unpassed; full robot/GPU/PPO pipeline and formal RL/evaluation/native full-job replay incomplete. No formal training has started. Old results/user source changes preserved; no hardware actions.

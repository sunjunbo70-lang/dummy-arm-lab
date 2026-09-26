# Case76 lift error attribution and temporal pressure candidate

L1 software. case76_pressure_001 finished: prelift maxwall0.00540715mm, postlift0.01632812mm fails0.01. At worst cell(51,35), prelift wall difference0.00003746mm, bead deposition0.01344008mm, remaining blade lift0.00285058mm. Evidence lift_error_decomposition.json. Error is dominated by carried inventory released at lift; no acceptance threshold changed.

Independent v8.5 TemporalPressure evaluates nonautonomous SSPRK2 flux stages with start/end interpolated pitch/force rather than identical midpoint controls. Same pressure law, geometry and spacings. This is a numerical candidate, not frozen physics or guaranteed improvement. Single stage-control/conservation regression passes. Original case76 at3344/6688 segments running runs/case76_temporal_001 PID32884 (launcher10972). Read summary and pre/post arrays before further adoption; preserve failures.

Original v8.2 full audit complete99/150 (99/100singles, long sequences affected by known dtype bug); v4 complete111/150. Both fail G0, not active anymore. v8.4 long sequence-only audit40668 continues, initial four sequences around15¨C21 operations at snapshot; do not duplicate.

Old no-output unittest PID23116 accumulated2818CPU seconds. Stop-Process failed; verified alive, then taskkill succeeded. Parent35204 ended. Diagnostic rerun in component_tests_diagnostic_001 completed83tests with0errors/0failures; tests.log and result.txt preserved. New temporal test was added after discovery and separately passed1test, so total83+1, not one84-test suite. Diagnostic test run is complete.

Next: evaluate temporal candidate; continue long-sequence audit. Formal RL/GPU environment/full robot executor remain incomplete. All work D:/VLA, no hardware instructions or changes to active solver versions.

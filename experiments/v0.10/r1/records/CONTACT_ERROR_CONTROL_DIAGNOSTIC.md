# Quarter-gap failure and bounded whole-contact error-control diagnostic

L1 software. v8.7 quarter-gap case76 finished with maxwall0.01642326642mm, still above0.01mm. Do not adopt; previous failed candidates retained.

New v8.8 ErrorControlPressure performs full-contact step doubling based on v8.4: compare one step with two half-steps for wall/blade/bead inventories (volume converted to equivalent blade-cell thickness). Accept two-half result only when max difference <=1e-4mm per5mm swept path plus1e-12mm arithmetic floor. Otherwise recursively split; depth6 exhaustion raises explicitly. No extrapolation or inventory clipping, no altered physical laws or gate thresholds. Caller state changes only after successful advance. Local estimator is not a proven global error bound, particularly at nonsmooth contact; full same-version gates remain necessary. This solver changes internal time sampling and costs more; accuracy/runtime must both be reported.

Two targeted tests passed: conservative/nonnegative advance and rejected step leaves original inventories unchanged. Original case76 at external3344/6688segments running runs/case76_error_control_001 actualPID39808 (launcher21204). Check summary/failure and boundary_stats contact_error_nodes/depth/error sum before further expansion, do not merely raise budget if it fails. External gate spacing50/25micrometres unchanged.

Long-sequence v8.4 manager40668 continues. First four traces84/88/94/93lines at snapshot, no completed100-action sequence yet (lines include load/final comparisons, not a training count). Telemetry13564 remains the single monitor. Formal RL has not begun, full material and robot gates, GPU environment/training/evaluation/replay incomplete. Only D:/VLA, no hardware.

# Expanded boundary audit: zero-area inventory diagnostic

L1 software session. Expanded v8.1 audit PID32992 continues unchanged. Snapshot: 32 completed cases, 29 passed. Failures: [(1, "ValueError('Material in zero-area initial reservoir')"), (18, "ValueError('Material in zero-area initial reservoir')"), (21, "ValueError('Material in zero-area initial reservoir')")].

Started independent case21 input capture in runs/zero_area_trace_001, actual PID35368, launcher31772. The diagnostic wraps exchange only in its own process and dumps area, target, wall/blade inventories and incoming/outgoing sparse maps before the original solver rejects. No tolerance change, no clearing material, no mutation to running audit modules. Do not duplicate. Inspect zero_area.json, failure_inputs.npz and summary.json when finished. If a nonzero wall inventory occupies snapped zero area, first quantify its magnitude and upstream source (initial pickup, edge deposition, pressure/slump or exchange); do not assume roundoff without evidence.

Full material G0 has not passed and formal RL has not begun. Original expanded_split_001 PID33596 remains preserved. Followups should read this record and live cases rather than the historical root implementation_state.json.


## Trace completed
Case21 reproduces at the second geometry half-move immediately after internal pressure/trailing-edge deposition (split_inventory.py:67). Captured wall cell5153 contains 1.012734565880737e-24 m3 (1.0127e-18 mL) with snapped zero free area. This establishes a tiny residual at the exchange boundary, not a macroscopic material leak. Exact upstream operation still needs attribution; do not globally loosen nonnegative/capacity checks or erase inventory. Next candidate should consistently handle geometric roundoff at deposition and preserve mass. Original audit remains unchanged; trace PID35368 finished, do not wait for it. Evidence runs/zero_area_trace_001/{zero_area.json,failure_inputs.npz,summary.json}. Formal RL not started.

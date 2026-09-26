# Expanded boundary audit: zero-area inventory diagnostic

L1 software session. Expanded v8.1 audit PID32992 continues unchanged. Snapshot: 32 completed cases, 29 passed. Failures: [(1, "ValueError('Material in zero-area initial reservoir')"), (18, "ValueError('Material in zero-area initial reservoir')"), (21, "ValueError('Material in zero-area initial reservoir')")].

Started independent case21 input capture in runs/zero_area_trace_001, actual PID35368, launcher31772. The diagnostic wraps exchange only in its own process and dumps area, target, wall/blade inventories and incoming/outgoing sparse maps before the original solver rejects. No tolerance change, no clearing material, no mutation to running audit modules. Do not duplicate. Inspect zero_area.json, failure_inputs.npz and summary.json when finished. If a nonzero wall inventory occupies snapped zero area, first quantify its magnitude and upstream source (initial pickup, edge deposition, pressure/slump or exchange); do not assume roundoff without evidence.

Full material G0 has not passed and formal RL has not begun. Original expanded_split_001 PID33596 remains preserved. Followups should read this record and live cases rather than the historical root implementation_state.json.

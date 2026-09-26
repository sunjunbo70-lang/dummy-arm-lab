# Continuous-contact empty-map fix and stationary identity

L1 software session. sequence_type_trace_001/summary.json confirms the original long-sequence failure at boundary_pressure.py:49 (np.divide output integer empty bincount), so v8.3 addresses the observed stack, not just a hypothetical bug.

New independent StationaryBoundaryPressure v8.4 inherits v8.3/v8.2. Exactly equal pose coordinates and angle return from geometry move only; pressure is still recomputed by contact. No approximate-motion tolerance or action removal. This avoids expensive integration of identically zero boundary flux. Three targeted tests passed (stationary inventory unchanged, pressure responds to force, empty-map floating output). Full component suite launched, result pending at recording time (exec session85155); do not claim full-suite pass until completion.

Four-operation sequence0 diagnostic with frozen50/25micrometre spacing is running in runs/stationary_sequence_001, actual PID27260 (launcher37848), using v8.4. This includes load/contact/continuation and does not substitute for100-action sequences. Check summary and step logs; do not duplicate.

Expanded v8.2 remains PID6544, snapshot67/67 singles passing; full audit unfinished. Historical v8.1 ended; prior case76 numerical error must still be checked for v8.2. Old v4 auditPID33596 still alive. No formal training has begun and full G0/G1, GPU environment, PPO campaign and final replays remain outstanding. All original sources/results retained.

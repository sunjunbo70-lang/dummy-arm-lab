# Bounded edge deposition candidate v8.2

L1 software only. Original v8.1 audit PID32992 remains unchanged (83 completed, 78 passed at this snapshot).

Candidate BoundedEdgePressure excludes edge-deposition overlaps into geometrically fully covered wall cells. The existing 2.5e-17 m2 geometric roundoff budget applies per donor column. Larger overlaps raise, and removed tiny overlaps are redistributed over the same column's remaining wall/outside recipients with normalized area; no material is erased and analytic exchange capacity checks are unchanged. Hypothesis: the captured 1e-24 m3 residual arises from edge deposition's independently clipped overlap. This is not yet causal confirmation; original failed-case regression is running.

New physics_version v10r1.area_pressure_candidate.v8_2_edge_support, separate module. Three targeted tests pass: real short pressure/lift conservation, zero-area recipient excluded with preserved donor amount, macroscopic overlap rejected. This does not constitute G0 or hardware validation.

runs/bounded_edge_001 tests original indices1,18,21 at unchanged50/25micrometre spacings. Actual PID 31516; launcher15000. Do not duplicate or modify its imported code. Read cases/summary when finished. If successful, expand to full numerical suite; original v8.1 failures remain preserved. Complete robot executor/GPU environment/training/evaluation/replay still outstanding; no formal RL started.

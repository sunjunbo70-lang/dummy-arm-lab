

# Pressure precision hypothesis and contact rejection trace

L1 software session. v8.9 coarse3344 case76 failed at2314 with error1.68846224217e-8mm versus1.56213284876e-8mm tolerance, almost identical to v8.8. Increasing gap bisection20 to40 has not removed this failure; pressure precision alone is not supported as its cause. Fine6688 remains running PID17240; preserve eventual result. No acceptance tolerance/depth changed.

Added trace_contact_rejection.py, instrumentation of unchanged v8.9 contact. On rejection it records coarse/two-half-step differences separately for wall/blade/bead, maximum-error cell, signed total differences and pressure gaps at each rejected recursion depth; deepest pre-step state is saved for reproducible local diagnosis. This is not another physics candidate. One controlled rejection component test passed: record/pickle produced, original inventory unchanged (runs/contact_rejection_component_001/result.json).

Original case76 coarse reproduction running in runs/contact_rejection_trace_001 PID26144 (launcher32480); check summary/depth JSON/rejected_state.pkl next. Do not restart or blindly increase recursion budget. Long-sequence manager40668 still active, first4/50 completed and passed; sole telemetry13564 retained. Latest resource snapshot four workers roughly99-100percent of one CPU core each, system33.9percent CPU, GPU6percent; this numerical validation is not a GPU training benchmark.

Full G0 unpassed, full execution/GPU/training/evaluation/replay remain missing. Formal RL never started. No hardware, no original results or unrelated user edits overwritten.

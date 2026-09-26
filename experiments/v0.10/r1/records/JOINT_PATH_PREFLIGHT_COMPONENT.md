

# Offline joint-path preflight component while material audit runs

L1 software. v8.11 expanded_fixed_cap_001 actualPID22620 remains active with8workers. Snapshot89 completed single cases, all pass; no completed new-version long sequences yet. Original v8.4 audit40668 continues with4workers and4 completed long sequences. No restarts/duplicate monitoring. Full same-version material gate still incomplete.

Added joint_path.py as an independent offline preflight building block: caller-provided IK/FK/clearance callbacks, all-joint position/rate limits, adjacent joint jump cap, both translational and full SO(3) residual checks, interpolated joint-path collision samples. Solver receives copies of seeds; a rejected path does not return a partially accepted executable plan. Initial pose checked. Rotation orthonormality and finite inputs validated. Caller must provide justified tolerances and limits; no hardware calibration value is invented.

Five synthetic callback tests pass: accepted continuous path with mutating solver callback, interior collision rejection even with clear endpoints, rate-limit rejection, orientation residual rejection, IK branch-jump rejection. Evidence runs/joint_path_component_001/result.json. This is NOT a MuJoCo robot execution test: callbacks still need integration, material tool-frame/pitch conventions need verification, approach/load/lift and dynamic tracking remain missing. Collision sampling is discrete, not a proof of continuous clearance; acceleration/torque constraints also remain outstanding. No G1/full executor claim.

Next: continue same-version100+50x100 material validation and integrate physical robot targets/IK callback into preflight without touching active material audit sources. Formal RL, GPU environment, complete PPO training/evaluation/native full-job replay not yet completed. Hardware untouched; user dirty sources preserved.

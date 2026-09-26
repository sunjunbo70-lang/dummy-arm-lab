"""Configuration for the second-generation whole-cycle plastering experiment."""
from dataclasses import dataclass, asdict


@dataclass
class CycleConfig:
    # 'v0.3' (default): physics-based yield-stress mortar (mortar.py), blade pitch as a free
    #   action (13-dim action), work area from the real reach rule, every stroke executed on
    #   the Dummy V2 MuJoCo model. See experiments/v0.3/r0/design/2026-09-22_wall_cycle_v0.3.md.
    # 'v0.2': the 2026-09-22 experiment exactly as recorded (11-dim action, no pitch).
    #   Kept only so that experiments/v0.2/r0/records/2026-09-22_wall_cycle_rl stays reproducible.
    physics: str = 'v0.3'
    tool_profile: str = 'legacy'

    # Wall and coating. Five-millimetre cells match the previous experiment.
    # v0.2 used 0.28 x 0.07 m, inherited from the 3-band single-stroke session and NOT
    # derived from reach. v0.3 uses the work square from tools/simulation/v2_trowel_reach.py:
    # largest circle inside the real V2+trowel reachable set -> inscribed square -> side x 0.8.
    width_m: float = 0.28
    height_m: float = 0.07
    # v0.7: width_m/height_m is the SIMULATION/PHYSICAL square (wall grid size, action decode
    # range, stroke-endpoint clip bound). score_width_m/score_height_m is the smaller, concentric
    # SCORED square (teacher aiming weight, coverage/RMSE/finish gates only look here). Overtravel
    # into the gap between them is allowed and not counted as waste; both come from
    # work_area.json's sim_square/score_square (area.py:load_work_area), which derives them from
    # the same verified reachable circle with area_safety applied to area, not to the inscribed
    # square's side (see tools/simulation/v2_trowel_reach.py:work_square). Defaults below are the
    # pre-v0.7 (v0.2) fallback and are overridden for v0.3+ configs.
    score_width_m: float = 0.28
    score_height_m: float = 0.07
    # Where that square sits in the scene (world frame: +X forward, +Y left, +Z up).
    scene_wall_distance_m: float = 0.35
    area_centre_u_m: float = -0.03
    area_centre_z_m: float = 0.22
    # Check every stroke on the real Dummy V2 model (dummy_loop/wall_cycle/arm.py).
    use_arm: bool = True
    cell_m: float = 0.005
    target_m: float = 0.002
    acceptable_low_m: float = 0.0015
    acceptable_high_m: float = 0.0025

    # 120 x 27.5 mm pointed trowel, represented as a local material field.
    blade_length_m: float = 0.12
    blade_width_m: float = 0.03
    blade_cell_m: float = 0.005
    blade_capacity_ml: float = 30.0
    load_choices_ml: tuple = (0.0, 12.0, 18.0, 24.0)
    load_scale_range: tuple = (0.8, 1.2)

    # D435 proxy: 848x480@30 is the vendor-recommended D435 operating mode.
    camera_resolution: tuple = (848, 480)
    camera_fps: int = 30
    camera_distance_m: float = 0.45
    scan_frames: int = 45
    scan_warmup_frames: int = 15
    scan_fused_frames: int = 30
    depth_noise_mm: float = 0.6
    # Residuals after fitting the bare-wall reference plane, not raw range error.
    plane_bias_mm: float = 0.20
    invalid_base: float = 0.02
    invalid_edge: float = 0.15
    extrinsic_translation_mm: float = 1.0
    extrinsic_rotation_deg: float = 0.05

    # Long horizon: allow progress, stop repeated non-improving work.
    control_hz: int = 20
    base_steps: int = 6000
    max_steps: int = 12000
    extension_steps: int = 1000
    max_cycles: int = 80
    max_reload_cycles: int = 30
    stall_limit: int = 5
    progress_epsilon: float = 0.0002
    min_cycles_before_stall: int = 0
    stall_window: int = 5

    # Contact and task-space motion.
    max_force_N: float = 40.0
    force_window: tuple = (2.0, 15.0)   # v0.2 action range (unchanged: v0.2 policies decode with it)
    force_window_v3: tuple = (0.5, 15.0)  # v0.3: the mortar carries only a few N on a tilted
                                          # blade (mortar.py), so the low end must be reachable
    wall_clearance_m: float = 0.08
    approach_clearance_m: float = 0.025
    min_stroke_m: float = 0.025
    max_stroke_m: float = 0.16
    speed_range_m_s: tuple = (0.02, 0.12)
    curve_offset_m: float = 0.03
    stroke_samples: int = 28

    # v0.3 mortar physics (dummy_loop/wall_cycle/mortar.py). Physical quantities only --
    # there is deliberately no parameter that says what pitch is good.
    mortar_rho: float = 1900.0          # kg/m^3
    mortar_tau_y_Pa: float = 250.0      # yield stress; Banfill 2003: mortar ~400 Pa, plasters softer
    mortar_mu_p_Pa_s: float = 2.0       # plastic viscosity; Banfill 2003: 1-3 Pa s
    lift_wall_fraction: float = 0.5     # share of a squeezed layer that stays on the wall when the
                                        # blade is pulled straight off (ASSUMED; measure it)
    max_pitch_deg: float = 35.0         # action range for the blade pitch (0 = blade parallel to wall)

    # v0.5 loading and transport. face_up_score = dot(blade material-face normal,
    # world up): +1 face up, 0 vertical, -1 face down. The named 0.75 wall share is
    # an engineering prior; randomisation tests its unknown magnitude.
    interface_wall_share_range: tuple = (0.60, 0.90)
    randomize_interface: bool = True
    tool_interface_yield_Pa: float = 80.0  # provisional smooth-metal slip/peel proxy
    feed_board_volume_ml: float = 600.0
    feed_normal_force_N: float = 8.0
    feed_scoop_depth_m: float = 0.006
    feed_scoop_distance_m: float = 0.09
    feed_scoop_speed_m_s: float = 0.04
    carry_duration_s: float = 1.5
    transport_dt_s: float = 0.02
    transport_relaxation_s: float = 0.35
    face_up_target_deg: float = 10.0
    face_up_hard_deg: float = 15.0
    feed_pose_q_rad: tuple = (0.0297, -1.31, 1.23, 0.0942, -1.48, -0.126)
    rotation_clearance_range_m: tuple = (0.12, 0.10, 0.08)

    # v0.7 reward reshaping (env.py:step()/_project()). Fixes the P2-A reward-hacking failure:
    # a fixed -0.5 projection penalty and a force cliff (-50, episode end) were far cheaper than a
    # genuine stroke's waste/carry-loss cost, so PPO learned to avoid real coverage.
    project_base_penalty: float = 0.5        # base cost of one _project() correction
    project_count_penalty: float = 0.1       # extra cost per projection already used this episode
    coverage_shaping_coef: float = 8.0       # potential-based reward for raw coverage delta
    force_over_coef: float = 8.0             # graded penalty coefficient, scales with (over/max)^2
    force_over_penalty_cap: float = 50.0     # cap, same order as the old cliff penalty
    force_terminate_mult: float = 1.5        # episode only ends above this multiple of max_force_N
    waste_coef: float = 4.0                  # was hardcoded '4' in step()
    carry_loss_coef: float = 4.0             # was hardcoded '4' in step()
    # v0.7: teacher no longer needs to hug 2 cm away from the scored edge now that overtravel
    # into the sim-region margin is allowed and not penalised as waste (was 0.02).
    teacher_edge_margin_m: float = 0.003

    # v0.8 switches (see recipes.py and experiments/v0.8/r0/design/2026-09-24_wall_cycle_v0.8.md). Every
    # default below reproduces v0.7 behaviour; only the 'v0.8' recipe turns them on.
    # 'global': teacher aims at the weighted centroid of the whole deficit map (v0.2-v0.7);
    # 'local' : teacher scores candidate strokes by the local deficit/excess they sweep.
    teacher_targeting: str = 'global'
    teacher_level_weight: float = 2.0        # local teacher: level if 2*excess beats deficit
    teacher_fail_damp: float = 0.7           # local teacher: score x (1 - damp*fail map at centre)
    teacher_plateau_n: int = 25              # local teacher: FINISH after n strokes w/o progress
    teacher_plateau_eps: float = 0.005       #   progress = sensor (coverage - rmse_mm/10) gain
    # Observation memory shared by teacher and student: decayed map of recent strokes that did
    # not improve the wall, plus strokes since the sensor score last improved.
    obs_memory: bool = False
    fail_decay: float = 0.8
    fail_radius_m: float = 0.025
    # 'v0.7': waste = all drops except dropped_m3 change since carry start (that whole change,
    #   including stroke slump, was charged as carry loss over cumulative supply).
    # 'v0.8': three separate ledgers per step -- transport loss (material.carry_loss_m3),
    #   other drops (feed overflow, air drop, wall slump) and material pushed off the grid --
    #   each charged per 18 mL, no cumulative-supply denominator.
    loss_accounting: str = 'v0.7'
    outside_coef: float = None               # None -> same as waste_coef (v0.7)
    time_coef: float = 0.0002                # reward per control step
    finish_fail_penalty: float = 20.0        # FINISH declared below the finish gates
    # v0.8: residual material on the blade is subject to transport physics on the way from the
    # scan pose to the feed pose (v0.6/v0.7 moved there without it).
    feed_transit_physics: bool = False
    edge_band_m: float = 0.03                # reported edge metrics: outer band of the score region

    # Provisional finish gates. Must be revised after D435 and material calibration.
    finish_coverage: float = 0.95
    finish_rmse_mm: float = 0.50
    finish_p95_mm: float = 1.00
    finish_confidence: float = 0.95

    def to_dict(self):
        d = asdict(self)
        for k, v in tuple(d.items()):
            if isinstance(v, tuple):
                d[k] = list(v)
        return d

    @property
    def wall_shape(self):
        return (round(self.height_m / self.cell_m), round(self.width_m / self.cell_m))

    @property
    def blade_shape(self):
        return (round(self.blade_width_m / self.blade_cell_m),
                round(self.blade_length_m / self.blade_cell_m))

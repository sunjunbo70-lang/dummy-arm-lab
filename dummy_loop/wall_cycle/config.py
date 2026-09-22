"""Configuration for the second-generation whole-cycle plastering experiment."""
from dataclasses import dataclass, asdict


@dataclass
class CycleConfig:
    # 'v0.3' (default): physics-based yield-stress mortar (mortar.py), blade pitch as a free
    #   action (13-dim action), work area from the real reach rule, every stroke executed on
    #   the Dummy V2 MuJoCo model. See docs/changes/2026-09-22_wall_cycle_v0.3.md.
    # 'v0.2': the 2026-09-22 experiment exactly as recorded (11-dim action, no pitch).
    #   Kept only so that experiments/2026-09-22_wall_cycle_rl stays reproducible.
    physics: str = 'v0.3'
    tool_profile: str = 'legacy'

    # Wall and coating. Five-millimetre cells match the previous experiment.
    # v0.2 used 0.28 x 0.07 m, inherited from the 3-band single-stroke session and NOT
    # derived from reach. v0.3 uses the work square from tools/simulation/v2_trowel_reach.py:
    # largest circle inside the real V2+trowel reachable set -> inscribed square -> side x 0.8.
    width_m: float = 0.28
    height_m: float = 0.07
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

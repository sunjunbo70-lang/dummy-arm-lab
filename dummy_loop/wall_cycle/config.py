"""Configuration for the second-generation whole-cycle plastering experiment."""
from dataclasses import dataclass, asdict


@dataclass
class CycleConfig:
    # Wall and coating. Five-millimetre cells match the previous experiment.
    width_m: float = 0.28
    height_m: float = 0.07
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
    force_window: tuple = (2.0, 15.0)
    wall_clearance_m: float = 0.08
    approach_clearance_m: float = 0.025
    min_stroke_m: float = 0.025
    max_stroke_m: float = 0.16
    speed_range_m_s: tuple = (0.02, 0.12)
    curve_offset_m: float = 0.03
    stroke_samples: int = 28

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

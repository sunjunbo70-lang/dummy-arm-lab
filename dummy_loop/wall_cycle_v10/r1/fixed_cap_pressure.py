"""Same wall-cap law, original fixed-step integrator; assess actual global gates.
Adaptive local estimator was diagnostic, not the frozen acceptance criterion.
"""
from .saturated_wall_pressure import SaturatedWallPressure
from .stationary_boundary_pressure import StationaryBoundaryPressure
class FixedCapPressure(SaturatedWallPressure):
 physics_version='v10r1.area_pressure_candidate.v8_11_fixed_step_wall_cap'
 contact=StationaryBoundaryPressure.contact

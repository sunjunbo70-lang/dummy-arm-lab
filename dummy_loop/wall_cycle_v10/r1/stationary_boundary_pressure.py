"""Exact stationary-geometry identity; pressure update remains in contact."""
import numpy as np
from .typed_boundary_pressure import TypedBoundaryPressure
class StationaryBoundaryPressure(TypedBoundaryPressure):
 physics_version='v10r1.area_pressure_candidate.v8_4_stationary_identity'
 def move(self,center,angle,exit_height):
  if self.pose is not None and np.array_equal(np.asarray(center,float),self.pose[0]) and float(angle)==self.pose[1]:
   return
  return super().move(center,angle,exit_height)

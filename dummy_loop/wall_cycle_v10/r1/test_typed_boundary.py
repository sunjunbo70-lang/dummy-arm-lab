import unittest
import numpy as np
from unittest.mock import patch
from .typed_boundary_pressure import TypedBoundaryPressure
class TypedTests(unittest.TestCase):
 def test_zero_motion_empty_boundary(self):
  s=TypedBoundaryPressure(np.zeros((100,100)),np.zeros((6,24)));s.move([0,.25],.3,0.)
  m=(np.array([],dtype=int),np.array([],dtype=int),np.array([],dtype=float),np.zeros(144))
  with patch('dummy_loop.wall_cycle_v10.r1.typed_boundary_pressure.adaptive',return_value=([m,m],{'nodes':0})):
   s.move([0,.25],.3,0.)
  self.assertEqual(s.total(),0.)
if __name__=='__main__':unittest.main()

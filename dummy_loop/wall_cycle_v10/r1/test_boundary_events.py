import unittest
import numpy as np
from .boundary_events import crossing_times
class BoundaryEventsTests(unittest.TestCase):
 def test_known_crossing(self):
  cuts=crossing_times([-.06586109001381107,.23409079902027605],[-.064437736628953,.23329740422917122],1.5462576820577338,1.550451733524185)
  self.assertTrue(any(.83592987060546875<t<.83593368530273438 for t in cuts))
 def test_reversed_crossings(self):
  a=crossing_times([0.,.25],[.002,.251],.31,.33)
  b=crossing_times([.002,.251],[0.,.25],.33,.31)
  np.testing.assert_allclose(a,1-np.array(b[::-1]),atol=1e-12)
if __name__=='__main__':unittest.main()

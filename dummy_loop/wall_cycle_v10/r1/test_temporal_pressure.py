import unittest
import numpy as np
from .temporal_pressure import TemporalPressure
class TemporalTests(unittest.TestCase):
 def test_stage_controls_and_conservation(self):
  s=TemporalPressure(np.zeros((100,100)),np.full((6,24),.002*.005**2));v=s.total()
  s.contact([0,.25],.3,.2,2.);calls=[];original=s._flux
  def flux(B,bead,p,f,c):
   calls.append((p,f));return original(B,bead,p,f,c)
  s._flux=flux;s.contact([.0001,.25],.3001,.3,4.)
  np.testing.assert_allclose(calls[0],[.2,2.]);np.testing.assert_allclose(calls[-1],[.3,4.]);s.lift(.5)
  self.assertAlmostEqual(s.total(),v,places=14);self.assertGreaterEqual(s.wall.min(),0.)
if __name__=='__main__':unittest.main()

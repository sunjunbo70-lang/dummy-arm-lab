import tempfile,unittest
from pathlib import Path
import numpy as np
from .replay import load_record,sample_pose
class ReplayTest(unittest.TestCase):
 def test_loaded_arrays_survive_closed_and_removed_archive(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'record.npz'
   np.savez_compressed(p,time_s=[0.,.05],q=np.array([[0.]*6,[1.]*6]),wall=np.zeros((2,2,2)),blade=np.zeros((2,1,1)),phase=['A','B'])
   z=load_record(p);p.unlink()
   i,q=sample_pose(z,.025);self.assertEqual(i,0);np.testing.assert_allclose(q,.5)
   np.testing.assert_allclose(sample_pose(z,.025,False)[1],0.)
   np.testing.assert_allclose(sample_pose(z,.05)[1],1.)
 def test_duplicate_timestamps_and_bounds(self):
  z={'time_s':np.array([0.,.05,.05,.10]),'q':np.array([[0.],[1.],[1.],[2.]])}
  self.assertEqual(sample_pose(z,.05)[0],2)
  np.testing.assert_allclose(sample_pose(z,.075)[1],1.5)
  np.testing.assert_allclose(sample_pose(z,1)[1],2.)
  np.testing.assert_allclose(sample_pose(z,-1)[1],0.)
if __name__=='__main__':unittest.main()

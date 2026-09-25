"""Numerical contracts for new kernels and controlled P0 scenarios."""
import json,unittest
import numpy as np
import torch
from .p0 import make_scene,run_case
from .tensor_kernels import solve_gap,extrude
from ..wall_cycle.mortar import solve_gap as reference_gap,extrude as reference_extrude,MortarParams

class Contracts(unittest.TestCase):
 def test_scene_reproducible_and_no_initial_deficit(self):
  for i in [0,1,23,24,47]:
   a,_=make_scene(i);b,_=make_scene(i);np.testing.assert_array_equal(a.wall,b.wall)
   self.assertTrue((a.wall[a._score_rows,a._score_cols]>=.002).all());self.assertEqual(a.blade_volume_m3,0.)
 def test_load_counterfactual_conservation_and_json(self):
  r=run_case((0,12.,.06,.0025,4.,2.));json.dumps(r);self.assertLess(abs(r['after']['ledger_error_ml']),1e-8);self.assertGreater(r['after']['supplied_ml'],0.)
 def test_tensor_gap_and_extrusion_reference(self):
  rng=np.random.default_rng(99);a=rng.uniform(0,.01,(16,6,24));p=rng.uniform(0,.5,16);f=rng.uniform(0,15,16);tt=torch.tensor(a,dtype=torch.float64);g,s=solve_gap(tt,torch.tensor(p),torch.tensor(f));ref=np.array([reference_gap(a[i],p[i],f[i],.005,MortarParams()) for i in range(16)])
  np.testing.assert_allclose(g.numpy(),ref[:,0],atol=1e-10);np.testing.assert_allclose(s.numpy(),ref[:,1],atol=1e-10)
  gap=np.maximum(g.numpy()[:,None]+(np.arange(6)+.5)*.005*np.sin(p[:,None]),1e-8);out=extrude(tt,torch.tensor(gap));rr=[reference_extrude(a[i],gap[i]) for i in range(16)]
  for k in range(3):np.testing.assert_allclose(out[k].numpy(),np.array([x[k] for x in rr]),atol=1e-12)
if __name__=='__main__':unittest.main()

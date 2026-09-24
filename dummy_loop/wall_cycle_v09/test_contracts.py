"""Executable regression checks for v0.9 preliminary implementation."""
import unittest
import numpy as np
import torch
from .learner import HybridPolicy,ppo_update
from .clean_env import CleanEnv,config

class Contracts(unittest.TestCase):
 def test_hidden_material_not_in_observation(self):
  e=CleanEnv(config(),60000);before=e.reset().copy()
  e.material.wall[:]=.04;e.material.blade[:]=.1;e.material.carry_loss_m3=100;e.last_improvement=88;e.stall_count=999
  np.testing.assert_array_equal(before,e._observe())
 def test_conditional_prob_and_mask(self):
  torch.manual_seed(0);m=HybridPolicy(16);x=torch.randn(32,16);valid=torch.ones(32,4,dtype=torch.bool);valid[:,3]=False
  op,a,raw,lp,v=m.sample(x,valid);self.assertTrue(bool((op!=3).all()))
  lp2,_,_=m.evaluate(x,valid,op,raw);torch.testing.assert_close(lp,lp2)
  changed=raw.clone();inactive=m.parameter_mask[op]==0;changed[inactive]=100
  lp3,_,_=m.evaluate(x,valid,op,changed);torch.testing.assert_close(lp,lp3)
 def test_gpu_ppo_updates_parameters(self):
  if not torch.cuda.is_available():self.skipTest('CUDA unavailable')
  torch.manual_seed(1);m=HybridPolicy(16).cuda();x=torch.randn(256,16,device='cuda');valid=torch.ones(256,4,device='cuda',dtype=torch.bool)
  op,a,raw,lp,v=m.sample(x,valid);before=m.op.weight.detach().clone();opt=torch.optim.Adam(m.parameters(),lr=3e-5)
  rows=ppo_update(m,opt,(x,valid,op,raw,lp,torch.randn_like(v),v+torch.randn_like(v)),epochs=1)
  self.assertFalse(torch.equal(before,m.op.weight));self.assertEqual(len(rows),2)
if __name__=='__main__':unittest.main()

"""A1 contract and numerical tests; never connect hardware."""
import unittest,pickle,copy
import numpy as np
import torch
from .environment import Env
from .trajectory_action import decode
from .config import config
from .reward_quality import reward
from .learner import HybridPolicy,MultiHybridPolicy,bc_loss

class A1Tests(unittest.TestCase):
 def test_mixture_density_and_update(self):
  torch.manual_seed(9);m=MultiHybridPolicy(20);x=torch.randn(32,20);valid=torch.ones(32,4,dtype=torch.bool)
  op,a,raw,old,v=m.sample(x,valid);lp,entropy,value=m.evaluate(x,valid,op,raw)
  torch.testing.assert_close(lp,old);self.assertTrue(torch.isfinite(entropy).all())
  altered=raw.clone();altered[op!=1,1:]+=30
  torch.testing.assert_close(m.evaluate(x,valid,op,altered)[0],lp)
  loss=-lp.mean()-.001*entropy.mean()+value.square().mean()+bc_loss(m,x,valid,op,a)
  loss.backward()
  for par in m.parameters():
   if par.grad is not None:self.assertTrue(torch.isfinite(par.grad).all())
 def test_mixture_cpu_gpu_density(self):
  if not torch.cuda.is_available():self.skipTest('CUDA required')
  torch.manual_seed(7);m=MultiHybridPolicy(20);g=copy.deepcopy(m).cuda();x=torch.randn(16,20);valid=torch.ones(16,4,dtype=torch.bool);op=torch.arange(16)%4;raw=torch.randn(16,19)
  torch.testing.assert_close(m.evaluate(x,valid,op,raw)[0],g.evaluate(x.cuda(),valid.cuda(),op.cuda(),raw.cuda())[0].cpu(),atol=2e-5,rtol=2e-5)
 def test_a0_macro_and_pickle(self):
  from .a0_control import A0Env,A0Policy
  e=A0Env(seed=60013,task='bare');a=np.zeros(19);a[:4]=[-.3,0,.3,0];a[4]=1;a[7]=-.8;a[8]=-.2;a[9]=0;a[10:12]=-.8
  o,r,d,i=e.step((0,a));self.assertEqual(len(e.valid_ops()),5);self.assertEqual(e.decisions,2);self.assertLess(abs(i['metrics']['volume_error_m3']),1e-10);pickle.loads(pickle.dumps(e))
  model=A0Policy(len(o));x=torch.tensor(o)[None];mask=torch.tensor(e.valid_ops())[None];op,params,raw,lp,_=model.sample(x,mask);torch.testing.assert_close(model.evaluate(x,mask,op,raw)[0],lp)
 def test_geometry_continuous_and_bounded(self):
  rng=np.random.default_rng(1);c=config()
  for _ in range(20):
   d=decode(rng.uniform(-1,1,19),c,.05);p=np.array([d.at(t)[0] for t in np.linspace(0,1,200)])
   np.testing.assert_allclose(p[[0,-1]],d.points[[0,-1]])
   self.assertLess(abs(p[:,0]).max()+.062,.25);self.assertLess(abs(p[:,1]-.25).max()+.062,.25)
   self.assertLess(np.linalg.norm(np.diff(p,axis=0),axis=1).max(),.01)
 def test_no_truth_input_and_restore(self):
  e=Env(seed=60022);o=e.observe();e.material.wall[:]=.009;e.material.blade[:]=.02;e.material.carry_loss_m3=5
  np.testing.assert_array_equal(o,e.observe())
  e=Env(seed=60022);f=pickle.loads(pickle.dumps(e));a=e.teacher();x=e.step(a);y=f.step(a)
  np.testing.assert_array_equal(x[0],y[0]);self.assertEqual(x[1:3],y[1:3])
 def test_material_conservation(self):
  e=Env(seed=60025,task='edge_ridge');rng=np.random.default_rng(2)
  for k in range(20):
   a=(0,np.zeros(19)) if k==0 else (1,rng.uniform(-1,1,19));o,r,d,i=e.step(a)
   self.assertLess(abs(i['metrics']['volume_error_m3']),1e-10)
 def test_reward_order(self):
  z=[0,0,0]
  giveup=reward(.8,.8,z,1,terminal=True)
  repair=reward(.8,.2,z,10)+reward(.2,.2,z,1,terminal=True)
  self.assertGreater(repair,giveup)
  self.assertLess(reward(.2,.2,z,1),0)
  self.assertLess(reward(.2,.8,z,1)+reward(.8,.2,z,1),0)
  self.assertGreater(reward(.01,.01,z,1,terminal=True,success=True),reward(.01,.2,z,1,terminal=True))
 def test_cpu_gpu_forward_gradient_adam(self):
  if not torch.cuda.is_available():self.skipTest('CUDA required')
  torch.manual_seed(2);cpu=HybridPolicy(20);gpu=copy.deepcopy(cpu).cuda();x=torch.randn(16,20);valid=torch.ones(16,4,dtype=torch.bool);op=torch.arange(16)%4;raw=torch.randn(16,19)
  opts=[torch.optim.Adam(cpu.parameters(),lr=3e-5),torch.optim.Adam(gpu.parameters(),lr=3e-5)]
  outputs=[]
  for m,opt,device in [(cpu,opts[0],'cpu'),(gpu,opts[1],'cuda')]:
   lp,ent,v=m.evaluate(x.to(device),valid.to(device),op.to(device),raw.to(device));loss=-lp.mean()+v.square().mean();outputs.append((lp.detach().cpu(),v.detach().cpu()));loss.backward();opt.step()
  torch.testing.assert_close(outputs[0][0],outputs[1][0],atol=2e-5,rtol=2e-5)
  for a,b in zip(cpu.parameters(),gpu.parameters()):torch.testing.assert_close(a,b.cpu(),atol=2e-5,rtol=2e-5)
if __name__=='__main__':unittest.main()

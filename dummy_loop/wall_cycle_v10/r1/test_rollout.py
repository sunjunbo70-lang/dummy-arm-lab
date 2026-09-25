import unittest
import torch
import numpy as np
from .rollout import prepare_rollout, optimize_rollout, Schedule
from .returns import time_gae

class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.w=torch.nn.Parameter(torch.tensor(0.)); self.seen=[]
    def evaluate(self,maps,scalars,valid,op,raw,length_bin,entropy=False):
        self.seen.extend(maps[:,0].tolist())
        x=self.w.expand(len(op));z=x*0
        return x,x,z,z

def sample(t=3,b=2,device='cpu'):
    shape=(t,b)
    return dict(maps=torch.arange(t*b,device=device).reshape(t,b,1).float(),
        scalars=torch.zeros(t,b,1,device=device),valid=torch.ones(t,b,7,dtype=torch.bool,device=device),
        op=torch.zeros(shape,dtype=torch.long,device=device),raw=torch.zeros(t,b,19,device=device),
        length_bin=torch.zeros(shape,dtype=torch.long,device=device),old_log_prob=torch.zeros(shape,device=device))

class RolloutTests(unittest.TestCase):
    def test_time_gae_terminal_and_rollout_bootstrap(self):
        r=torch.tensor([[1.,2.],[3.,4.],[5.,6.]])
        v=torch.ones_like(r);nv=torch.full_like(r,2.)
        done=torch.tensor([[False,True],[False,False],[False,True]])
        dt=torch.tensor([[1.,4.],[60.,12.],[2.,9.]])
        out=prepare_rollout(sample(),r,v,nv,done,dt)
        a,ret=time_gae(r.numpy(),v.numpy(),nv.numpy(),done.numpy(),dt.numpy())
        np.testing.assert_allclose(out['returns'].numpy().reshape(3,2),ret,rtol=1e-6)
        self.assertEqual(out['returns'][-2].item(),7.) # nonterminal boundary bootstraps
        self.assertEqual(out['returns'][-1].item(),6.) # budget/finish cuts bootstrap
        self.assertAlmostEqual(out['advantage'].mean().item(),0.,places=6)
        self.assertAlmostEqual(out['advantage'].std(unbiased=False).item(),1.,places=6)
    def batch(self,device='cpu'):
        z=torch.zeros(3,2,device=device)
        return prepare_rollout(sample(device=device),torch.ones_like(z),z,z,z.bool(),z+1)
    def test_shuffle_tail_and_all_epochs(self):
        p=TinyPolicy();opt=torch.optim.SGD(p.parameters(),lr=0.)
        batch=self.batch();before=batch['advantage'].clone()
        result=optimize_rollout(p,opt,batch,schedule=Schedule(epochs=3,minibatch=4))
        self.assertEqual(result['updates'],6);self.assertEqual(result['sample_updates'],18)
        for i in range(3):self.assertEqual(sorted(p.seen[i*6:i*6+6]),list(range(6)))
        torch.testing.assert_close(before,batch['advantage'])
    def test_kl_stops_before_update(self):
        p=TinyPolicy();p.w.data.fill_(2.)
        opt=torch.optim.Adam(p.parameters(),lr=.1)
        result=optimize_rollout(p,opt,self.batch())
        self.assertTrue(result['stopped_for_kl']);self.assertEqual(result['updates'],0)
        self.assertEqual(p.w.item(),2.);self.assertFalse(opt.state)
    def test_bad_mask_and_nonfinite_rejected(self):
        z=torch.zeros(3,2);s=sample();s['valid'][:,:,0]=False
        with self.assertRaises(ValueError):prepare_rollout(s,z,z,z,z.bool(),z)
        with self.assertRaises(ValueError):prepare_rollout(sample(),z,z,z,z.bool(),z,time_constant=float('nan'))
    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda_preparation_matches_cpu(self):
        a=self.batch();b=self.batch('cuda')
        for k in a:torch.testing.assert_close(a[k],b[k].cpu())

if __name__=='__main__':unittest.main()

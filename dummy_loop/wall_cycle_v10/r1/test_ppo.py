import unittest
import copy
import torch
from .ppo import objective,update_minibatch,PPOConfig
from .policy import Policy

class PPOTests(unittest.TestCase):
    def test_clipping_sign_and_detachment(self):
        lp=torch.tensor([1.5,.5]).log().requires_grad_()
        old=torch.zeros(2,requires_grad=True)
        adv=torch.tensor([1.,-1.],requires_grad=True)
        z=torch.zeros(2)
        loss,m=objective(lp,old,z,z,adv,z,z,PPOConfig(clip=.2,operation_entropy=0,parameter_entropy=0))
        self.assertAlmostEqual(m['policy_loss'].item(),-.2,places=6)
        loss.backward()
        torch.testing.assert_close(lp.grad,torch.zeros(2))
        self.assertIsNone(old.grad);self.assertIsNone(adv.grad)

    def test_nonfinite_rejected(self):
        z=torch.zeros(2)
        with self.assertRaises(ValueError):objective(z,z,z,torch.full((2,),float('nan')),z,z,z)

    def test_real_network_update_and_optimizer_restore(self):
        torch.manual_seed(100)
        policy=Policy();optimizer=torch.optim.Adam(policy.parameters(),lr=3e-4)
        maps=torch.randn(4,6,100,100);scalars=torch.randn(4,32)
        valid=torch.zeros(4,7,dtype=torch.bool);valid[:,1]=True
        action=policy.sample(maps,scalars,valid)
        batch=dict(maps=maps,scalars=scalars,valid=valid,op=action['op'],raw=action['raw'],length_bin=action['length_bin'],old_log_prob=action['log_prob'],returns=action['value']+torch.tensor([1.,-1.,.5,-.5]),advantage=torch.tensor([1.,-1.,.5,-.5]))
        before=policy.map_encoder[0].weight.detach().clone()
        metrics=update_minibatch(policy,optimizer,batch)
        self.assertTrue(all(torch.isfinite(v) for v in metrics.values()))
        self.assertFalse(torch.equal(before,policy.map_encoder[0].weight))
        restored=Policy();restored.load_state_dict(policy.state_dict())
        opt2=torch.optim.Adam(restored.parameters(),lr=3e-4);opt2.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        rng=torch.get_rng_state();update_minibatch(policy,optimizer,batch)
        torch.set_rng_state(rng);update_minibatch(restored,opt2,batch)
        for a,b in zip(policy.parameters(),restored.parameters()):torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-7)

if __name__=='__main__':unittest.main()


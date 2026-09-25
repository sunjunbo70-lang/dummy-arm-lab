"""Probability and gradient checks; synthetic inputs are NOT task training."""
import unittest
import torch
from .policy import Policy
from .actions import Op

class PolicyContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def inputs(self):
        torch.manual_seed(19)
        return torch.rand(8,6,100,100),torch.randn(8,32),torch.ones(8,7,dtype=torch.bool)
    def test_sample_and_evaluate_match(self):
        m=Policy();x,s,v=self.inputs();r=m.sample(x,s,v);lp,value,_,_=m.evaluate(x,s,v,r['op'],r['raw'],r['length_bin'])
        torch.testing.assert_close(lp,r['log_prob']);torch.testing.assert_close(value,r['value']);self.assertTrue(torch.isfinite(lp).all())
    def test_continuation_unused_dimensions(self):
        m=Policy();x,s,v=self.inputs();op=torch.full((8,),int(Op.CONTINUE));raw=torch.randn(8,19);bin=torch.zeros(8,dtype=torch.long)
        lp,*_=m.evaluate(x,s,v,op,raw,bin);raw[:,[0,1,8,10,13,16]]+=100
        lp2,*_=m.evaluate(x,s,v,op,raw,bin);torch.testing.assert_close(lp,lp2)
    def test_gradient_reaches_shared_network(self):
        m=Policy();x,s,v=self.inputs();r=m.sample(x,s,v);lp,value,eo,ep=m.evaluate(x,s,v,r['op'],r['raw'],r['length_bin'],entropy=True)
        loss=-(lp*torch.arange(8)).mean()+value.square().mean()-.005*eo.mean()-.001*ep.mean();loss.backward()
        self.assertTrue(torch.isfinite(m.map_encoder[0].weight.grad).all());self.assertGreater(float(m.map_encoder[0].weight.grad.abs().sum()),0)
    def test_cuda_forward_if_available(self):
        if not torch.cuda.is_available():self.skipTest('CUDA absent')
        cpu=Policy();gpu=Policy().cuda();gpu.load_state_dict(cpu.state_dict());x,s,v=self.inputs()
        a=cpu.sample(x,s,v,True);b=gpu.sample(x.cuda(),s.cuda(),v.cuda(),True)
        torch.testing.assert_close(a['log_prob'],b['log_prob'].cpu(),atol=1e-4,rtol=1e-4)

if __name__=='__main__':unittest.main()

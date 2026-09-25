"""Tests for multi-step credit, termination and continuation parameter semantics."""
import unittest
import numpy as np
from .returns import transition_reward,time_gae,potential
from .actions import decode,valid_ops,Op,PARAM_MASK,audit_geometry

class Contracts(unittest.TestCase):
    def total(self,states,durations):
        return sum(transition_reward(a,b,duration_s=t,terminal=i==len(durations)-1)[0]
                   for i,(a,b,t) in enumerate(zip(states,states[1:],durations)))
    def test_temporary_damage_not_permanent_penalty(self):
        self.assertAlmostEqual(self.total([(1.,0.),(1.5,.3),(.5,0.)],[5,5]),
                               self.total([(1.,0.),(.5,0.)],[10]))
    def test_cycle_is_costly(self):
        self.assertLess(self.total([(1.,0.),(2.,.4),(1.,0.)],[5,5]),
                        self.total([(1.,0.),(1.,0.)],[1]))
    def test_terminal_shaping_accounting(self):
        states=[(1.,0.),(1.2,.2),(.7,.05)]
        self.assertAlmostEqual(self.total(states,[3,4]),-potential(*states[0])-10*.7-10*.05-.007)
    def test_gae_terminal_and_rollout_bootstrap(self):
        a,_=time_gae(np.array([[1.],[2.]]),np.zeros((2,1)),np.array([[5.],[100.]]),np.array([[False],[True]]),np.ones((2,1)),600)
        self.assertAlmostEqual(a[1,0],2);self.assertAlmostEqual(a[0,0],6+np.exp(-1/600)*2)
        a,_=time_gae(np.array([[1.]]),np.array([[0.]]),np.array([[5.]]),np.array([[False]]),np.ones((1,1)))
        self.assertEqual(a[0,0],6)
    def test_actual_material_cost(self):
        r,c=transition_reward((1,0),(1,0),duration_s=0,supplied_ml=18,actual_load=True)
        self.assertAlmostEqual(r,-.25)
    def test_continuation_inherits_actual_start(self):
        x=np.zeros(19);state=(np.array([-.05,.25]),.2,.1,3.,.04);d=decode(x,current=state)
        np.testing.assert_array_equal(d.points[0],state[0]);np.testing.assert_allclose(d.at(0)[2:],state[1:]);self.assertFalse(PARAM_MASK[Op.CONTINUE,:2].any())
    def test_contact_state_mask(self):
        m=valid_ops(True,continue_supported=True,segments=4)
        self.assertFalse(m[Op.LOAD]);self.assertFalse(m[Op.CONTINUE]);self.assertTrue(m[Op.LIFT])
    def test_outside_proposal_rejected_not_projected(self):
        x=np.zeros(19);x[0]=1;d=decode(x);self.assertFalse(audit_geometry(d)['accepted']);self.assertGreater(d.points[-1,0],.13)

if __name__=='__main__':unittest.main()

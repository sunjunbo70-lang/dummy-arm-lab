"""Regression: DEPOSIT finishing at LOAD must have serializable acyclic info."""
import json, unittest
import numpy as np
from .a0_control import A0Env
class A0TerminalLogTest(unittest.TestCase):
 def test_load_terminal_info_is_snapshot(self):
  e=A0Env(11);leaf={'executed':True,'metrics':{'J':.7},'end_reason':'budget'}
  e._substep=lambda action:(np.zeros(1),2.0,True,leaf)
  obs,reward,done,info=e.step((0,np.zeros(19)))
  self.assertTrue(done);self.assertEqual(reward,2.0)
  self.assertIsNot(info,info['macro_substeps'][0])
  self.assertNotIn('macro_substeps',info['macro_substeps'][0])
  self.assertEqual(json.loads(json.dumps(info))['macro_substeps'][0]['end_reason'],'budget')
if __name__=='__main__':unittest.main()

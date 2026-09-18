"""Check MuJoCo against transforms + rotation assignments from working Studio.

这些检查依赖从 DummyStudio 提取的数据。上游 peng-zhihui/Dummy-Robot 未声明许可，
因此本仓库不再分发那些提取物，相关用例在数据缺失时自动跳过。

需要运行这些检查时，按 models/README.md 用自己的 DummyStudio 副本重建：

    python tools/modeling/build_studio_model.py

参考仿真（models/dummy_reference.xml，GPL-3.0 上游）不受影响，开箱即用。
"""
import json
from pathlib import Path
import unittest
import numpy as np
import mujoco
from dummy_loop.sim_backend import SimRobot

ROOT=Path(__file__).resolve().parents[1]
TRANSFORMS=ROOT/'models/studio_source/studio_verified_transforms.json'
STUDIO_MESHDIR=ROOT/'models/studio_meshes'
STUDIO_AVAILABLE=TRANSFORMS.is_file() and STUDIO_MESHDIR.is_dir()
STUDIO_REASON=('需要 DummyStudio 提取数据；本仓库不分发，'
               '见 models/README.md 的重建说明')
C=np.array([[0,0,1],[-1,0,0],[0,1,0.]])
IDS=[135,149,136,138,148,133]


def rotation(axis,angle):
    v=np.eye(3)[axis]; x,y,z=v
    k=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    return np.eye(3)+np.sin(angle)*k+(1-np.cos(angle))*(k@k)


def reference(q):
    rows={r['id']:r for r in json.loads(TRANSFORMS.read_text())}
    overrides={i:rotation(a,-np.deg2rad(v)) for i,a,v in zip(IDS,[1,2,2,1,2,1],q)}
    def world(i):
        if i==0: return np.eye(4)
        r=rows[i]; t=np.eye(4); t[:3,3]=np.array(r['pos'])*.001
        if i in overrides: t[:3,:3]=overrides[i]
        else:
            x,y,z,w=r['rot']; out=np.empty(9)
            mujoco.mju_quat2Mat(out,np.array([w,x,y,z])); t[:3,:3]=out.reshape(3,3)
        return world(r['parent'])@t
    return [world(i) for i in IDS]


@unittest.skipUnless(STUDIO_AVAILABLE, STUDIO_REASON)
class ArticulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.robot=SimRobot(ROOT/'models/dummy_studio_visual.xml')

    def test_all_joint_frames_match_studio_on_multiple_poses(self):
        home=reference([0,0,90,0,0,0])
        for q in ([0,0,90,0,0,0],[0,-75,180,0,0,0],[10,-35,125,15,-15,12],[-10,15,95,-10,15,-15]):
            self.robot.reset(np.deg2rad(np.array(q)-[0,0,90,0,0,0]))
            for i,(actual,zero) in enumerate(zip(reference(q),home)):
                body=mujoco.mj_name2id(self.robot.model,mujoco.mjtObj.mjOBJ_BODY,f'link{i+1}')
                np.testing.assert_allclose(self.robot.data.xpos[body],C@actual[:3,3],atol=3e-7)
                np.testing.assert_allclose(self.robot.data.xmat[body].reshape(3,3),C@actual[:3,:3]@zero[:3,:3].T@C.T,atol=1e-6)

    def test_j5_only_moves_wrist_not_forearm_or_stator(self):
        r=self.robot; r.reset(np.zeros(6)); before=r.data.geom_xpos.copy(); before_rot=r.data.geom_xmat.copy()
        r.reset(np.deg2rad([0,0,0,0,15,0]))
        for name in ('dummy_j5_0','dummy_j5_motor_0','dummy_j5_cover_0','dummy_j4_0','dummy_j4_cover_0'):
            g=mujoco.mj_name2id(r.model,mujoco.mjtObj.mjOBJ_GEOM,name)
            np.testing.assert_allclose(r.data.geom_xpos[g],before[g],atol=1e-12)
            np.testing.assert_allclose(r.data.geom_xmat[g],before_rot[g],atol=1e-12)
        wrist=mujoco.mj_name2id(r.model,mujoco.mjtObj.mjOBJ_GEOM,'dummy_j6_0')
        self.assertGreater(np.linalg.norm(r.data.geom_xmat[wrist]-before_rot[wrist]),.1)

    def test_covers_and_motors_stay_on_same_rigid_link(self):
        m=self.robot.model
        for group in [('dummy_j4_0','dummy_j4_cover_0','dummy_j4_motor_0'),('dummy_j5_0','dummy_j5_cover_0','dummy_j5_motor_0'),('dummy_j6_0','dummy_j6_cover_0','dummy_j6_motor_0')]:
            bodies=[m.geom_bodyid[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,name)] for name in group]
            self.assertEqual(len(set(bodies)),1)


if __name__=='__main__': unittest.main()

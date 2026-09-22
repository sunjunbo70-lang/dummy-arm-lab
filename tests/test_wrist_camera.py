"""L1: rear mounting holes, optical transform and old-scene compatibility."""
import copy
import unittest
import mujoco
import numpy as np
from dummy_loop.wall.lab_tool import scene_config
from dummy_loop.wall.scene import build_scene

class WristCameraTests(unittest.TestCase):
    def test_holes_open_and_45mm_apart(self):
        m,_=build_scene(scene_config());d=mujoco.MjData(m);mujoco.mj_forward(m,d)
        p=[d.site_xpos[m.site('d435_mount_hole_'+str(i)).id].copy() for i in range(2)]
        self.assertAlmostEqual(np.linalg.norm(p[1]-p[0]),.045)
        root=m.body('d435_bracket').id
        m.geom_group[:]=5;m.geom_group[m.geom_bodyid==root]=4
        group=np.zeros(6,dtype=np.uint8);group[4]=1
        R=d.xmat[root].reshape(3,3);direction=R@np.array([0.,1.,0.])
        for q in p:
            hit=np.zeros(1,dtype=np.int32)
            distance=mujoco.mj_ray(m,d,q-.020*direction,direction,group,1,-1,hit)
            self.assertEqual(distance,-1) # no convex hull accidentally closes screw bore
        # 4 mm beside a hole must strike the mounting plate.
        hit=np.zeros(1,dtype=np.int32)
        self.assertGreater(mujoco.mj_ray(m,d,p[0]-.020*direction+R@np.array([.004,0,0]),direction,group,1,-1,hit),0)

    def test_camera_follows_joint_and_tcp_unchanged(self):
        c=scene_config();off=copy.deepcopy(c);off.lab_geometry.pop('camera_mount')
        m,_=build_scene(c);old,_=build_scene(off)
        self.assertEqual(old.ncam+1,m.ncam)
        self.assertAlmostEqual(m.body_mass.sum()-old.body_mass.sum(),.100)
        for angle in (0,.5,-.8):
            d=mujoco.MjData(m);o=mujoco.MjData(old);d.qpos[5]=o.qpos[5]=angle
            mujoco.mj_forward(m,d);mujoco.mj_forward(old,o)
            np.testing.assert_allclose(d.site_xpos[m.site('tcp').id],o.site_xpos[old.site('tcp').id])
            R=d.xmat[m.body('trowel').id].reshape(3,3)
            cid=m.camera('d435_wrist').id
            np.testing.assert_allclose(-d.cam_xmat[cid].reshape(3,3)[:,2],R[:,1],atol=1e-12)
            relative=R.T@(d.cam_xpos[cid]-d.xpos[m.body('trowel').id])
            np.testing.assert_allclose(relative,[0,.03205,.070],atol=1e-12)

if __name__=='__main__':unittest.main()

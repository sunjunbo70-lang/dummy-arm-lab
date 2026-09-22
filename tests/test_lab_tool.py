"""L1 mechanical regression tests for photo-supported lab tool."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import mujoco
from dummy_loop.wall.lab_tool import scene_config, blade_polygon
from dummy_loop.wall.scene import SceneConfig, build_scene, export_xml, rigid_tool_length, housing_front_x

class LabToolTests(unittest.TestCase):
    def test_dimensions_and_tcp(self):
        c=scene_config();m,_=build_scene(c);d=mujoco.MjData(m);mujoco.mj_forward(m,d)
        poly=blade_polygon(c)
        self.assertAlmostEqual(np.ptp(poly[:,0]),.124)
        self.assertAlmostEqual(np.ptp(poly[:,1]),.042)
        self.assertAlmostEqual(rigid_tool_length(c),.0705)
        R=d.xmat[m.body('link6').id].reshape(3,3)
        flange=d.xpos[m.body('link6').id]+R@np.array([housing_front_x()+c.j6_reducer_length,0,0])
        np.testing.assert_allclose(d.site_xpos[m.site('tcp').id]-flange,R@np.array([.0705,0,0]),atol=1e-10)

    def test_drawing_datums_and_adapter_stack(self):
        c=scene_config();m,_=build_scene(c);d=mujoco.MjData(m);mujoco.mj_forward(m,d)
        root=m.body('j6_reducer_fixed').id;R=d.xmat[root].reshape(3,3)
        for name,axial in (('reducer_housing_top',.0265),('reducer_output_face',.0275),('output_flange_frame',.0335)):
            local=R.T@(d.site_xpos[m.site(name).id]-d.xpos[root])
            np.testing.assert_allclose(local,[axial,0,0],atol=1e-12)
        points=[d.site_xpos[m.site(f'reducer_output_M3_{i}').id] for i in range(3)]
        centre=np.mean(points,axis=0)
        for point in points:self.assertAlmostEqual(np.linalg.norm(point-centre),.0065)
        with self.assertRaises(ValueError):build_scene(scene_config(j6_reducer_length=.028))

    def test_j6_stationary_housing_rotating_tool(self):
        m,_=build_scene(scene_config());d=mujoco.MjData(m);mujoco.mj_forward(m,d)
        housing=d.geom_xpos[m.geom('j6_reducer_body').id].copy()
        tip=d.site_xpos[m.site('blade_tip').id].copy()
        d.qpos[5]=np.pi/2;mujoco.mj_forward(m,d)
        np.testing.assert_allclose(d.geom_xpos[m.geom('j6_reducer_body').id],housing,atol=1e-12)
        self.assertGreater(np.linalg.norm(d.site_xpos[m.site('blade_tip').id]-tip),.05)

    def test_fixed_mass_is_not_lost_to_explicit_link_inertia(self):
        c=scene_config();m,_=build_scene(c)
        self.assertAlmostEqual(m.body_mass[m.body('j6_reducer_fixed').id],c.j6_reducer_mass*.8)
        # Tool subtree includes flange, existing adapter, proposed clamp and complete trowel.
        root=m.body('tool_mount').id
        ids=[]
        for b in range(m.nbody):
            a=b
            while a and a!=root:a=m.body_parentid[a]
            if a==root:ids.append(b)
        self.assertAlmostEqual(m.body_mass[ids].sum(),c.j6_reducer_mass*.2+c.lab_geometry['adapter_mass_kg']+c.holder_mass+c.tool_mass+sum(c.lab_geometry['camera_mount'][k] for k in ('camera_mass_kg','bracket_mass_kg')),places=7)

    def test_joint_and_actuator_limits_and_dynamics(self):
        m,_=build_scene(scene_config());d=mujoco.MjData(m)
        np.testing.assert_allclose(m.jnt_actfrcrange[5],[-.9,.9])
        np.testing.assert_allclose(m.actuator_forcerange[5],[-.9,.9])
        d.ctrl[5]=.25
        peak=0
        for _ in range(1500):
            mujoco.mj_step(m,d);peak=max(peak,abs(d.qfrc_actuator[5]))
        self.assertTrue(np.isfinite(d.qpos).all())
        self.assertGreater(peak,.07) # historical direct-drive joint clamp must not remain
        self.assertLessEqual(peak,.900001)
        self.assertAlmostEqual(d.qpos[5],.25,delta=.02)

    def test_export_roundtrip(self):
        c=scene_config();m,_=build_scene(c)
        with tempfile.TemporaryDirectory() as td:
            path=export_xml(c,Path(td)/'scene.xml');loaded=mujoco.MjModel.from_xml_path(str(path))
            self.assertEqual(loaded.ngeom,m.ngeom)
            np.testing.assert_allclose(loaded.body_mass,m.body_mass,rtol=2e-5,atol=1e-8)

    def test_legacy_and_invalid_variants(self):
        self.assertEqual(SceneConfig().trowel_length,.120)
        self.assertEqual(SceneConfig().handle_radius,.0075)
        for cfg in (scene_config(tool_mount='spring'),scene_config(blade_tilt_deg=10),SceneConfig(geometry_profile='typo')):
            with self.assertRaises(ValueError):build_scene(cfg)

if __name__=='__main__':unittest.main()

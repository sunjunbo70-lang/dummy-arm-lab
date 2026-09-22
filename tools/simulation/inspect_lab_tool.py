"""Inspect the lab tool in MuJoCo. Simulation only; no hardware imports."""
import argparse, json, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import mujoco
import numpy as np
from dummy_loop.wall.lab_tool import scene_config
from dummy_loop.wall.scene import build_scene, export_xml, rigid_tool_length


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--render',type=Path,help='write overview/close-up PNGs and model audit without opening GUI')
    ap.add_argument('--export',type=Path)
    args=ap.parse_args()
    cfg=scene_config(); model,_=build_scene(cfg); data=mujoco.MjData(model)
    data.qpos[:]=0; data.ctrl[:]=0; mujoco.mj_forward(model,data)
    if args.export: export_xml(cfg,args.export)
    if args.render:
        from PIL import Image
        args.render.mkdir(parents=True,exist_ok=True)
        renderer=mujoco.Renderer(model,height=720,width=960)
        opt=mujoco.MjvOption(); opt.sitegroup[:]=0
        # Wall transparent for inspection (rendering only; no physical change).
        model.geom_rgba[model.geom('wall_geom').id,3]=0
        tcp=data.site_xpos[model.site('tcp').id].copy()
        for name,look,dist,azi,elev in (
            ('overview',[.2,0,.22],.85,135,-20),
            ('tool_side',tcp+[-.045,0,0],.31,90,-12),
            ('tool_front',tcp+[-.02,0,0],.32,10,-28)):
            cam=mujoco.MjvCamera(); cam.lookat[:]=look; cam.distance=dist
            cam.azimuth=azi; cam.elevation=elev
            renderer.update_scene(data,camera=cam,scene_option=opt)
            Image.fromarray(renderer.render()).save(args.render/(name+'.png'))
        renderer.close()
        from dummy_loop.wall.scene import SceneConfig
        old,_=build_scene(SceneConfig()); oldd=mujoco.MjData(old); mujoco.mj_forward(old,oldd)
        audit={'evidence_level':'L1','hardware_motion':False,'scene_digest':cfg.digest(),
               'tcp_home_m':tcp.tolist(),'legacy_tcp_home_m':oldd.site_xpos[old.site('tcp').id].tolist(),
               'tcp_change_mm':((tcp-oldd.site_xpos[old.site('tcp').id])*1000).tolist(),
               'flange_to_contact_mm':rigid_tool_length(cfg)*1000,
               'j6_joint_limit_Nm':model.jnt_actfrcrange[5].tolist(),
               'j6_actuator_limit_Nm':model.actuator_forcerange[5].tolist(),
               'robot_and_tool_mass_kg':float(model.body_mass.sum()-model.body_mass[model.body('wall').id]),
               'fixed_reducer_mass_kg':float(model.body_mass[model.body('j6_reducer_fixed').id]),
               'provenance':cfg.provenance()}
        (args.render/'audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding='utf-8')
        print(json.dumps(audit,ensure_ascii=False,indent=2));return
    from mujoco import viewer as mjviewer
    # Z: home; J: toggle slow J6 +/-30-degree visual sweep; SPACE: pause.
    state={'sweep':False,'pause':False,'reset':False}
    def key(k):
        if k==32: state['pause']=not state['pause']
        elif k in (74,106): state['sweep']=not state['sweep']
        elif k in (90,122): state['reset']=True
    print('SIMULATION ONLY. J: J6 sweep, Z: home, SPACE: pause. Use right-panel Control sliders for six axes.')
    model.geom_rgba[model.geom('wall_geom').id,3]=.12
    with mjviewer.launch_passive(model,data,key_callback=key) as viewer:
        viewer.cam.lookat[:]=[.2,0,.23]; viewer.cam.distance=.8
        viewer.cam.azimuth=135;viewer.cam.elevation=-20
        while viewer.is_running():
            t=time.monotonic()
            if state['reset']:
                mujoco.mj_resetData(model,data);state['reset']=False;state['sweep']=False
            if not state['pause']:
                if state['sweep']: data.ctrl[5]=np.deg2rad(30)*np.sin(data.time*.6)
                for _ in range(10): mujoco.mj_step(model,data)
            viewer.sync();time.sleep(max(0,.02-(time.monotonic()-t)))

if __name__=='__main__':main()

"""D435 rear two-M3 mounting extension; simulation geometry, not validated fabrication."""
import numpy as np
import mujoco
from .lab_tool import _box, _mesh


def add_wrist_camera(spec, cfg):
    c=cfg.lab_geometry.get('camera_mount', {})
    if not c.get('enabled',False): return
    spacing=c['hole_spacing_m']; r=c['clearance_diameter_m']/2
    t=c['plate_thickness_m']; z=c['lateral_offset_m']
    if not (.001 < r < .005 and .060 <= z <= .150 and .003 <= t <= .012):
        raise ValueError('invalid wrist mount dimensions')
    gc=1. if cfg.gravcomp else 0.
    b=spec.body('trowel').add_body(name='d435_bracket',gravcomp=gc)
    blue=[.12,.48,.69,1]; mass=c['bracket_mass_kg']
    # Root overlaps the existing holder plate; stem reaches the mounting crossbar.
    _box(b,'camera_stem',[0,t/2,(.020+z)/2],[.008,t/2,(z-.020)/2],mass*.4,blue)
    _box(b,'camera_rib',[0,-.003,(.020+z-.009)/2],[.004,.003,(z-.029)/2],mass*.1,blue)
    # Plate partition leaves REAL open bores, not a convex mesh spanning each hole.
    half=spacing/2+.006
    _box(b,'camera_pad_top',[0,t/2,z+.0075],[half,t/2,.0015],mass*.1,blue)
    _box(b,'camera_pad_bottom',[0,t/2,z-.0075],[half,t/2,.0015],mass*.1,blue)
    _box(b,'camera_pad_middle',[0,t/2,z],[spacing/2-.006,t/2,.006],mass*.1,blue)
    for i,x in enumerate((-spacing/2,spacing/2)):
        b.add_site(name=f'd435_mount_hole_{i}',pos=[x,t,z],size=[.001,0,0],rgba=[1,.3,0,1])
        for k in range(32):
            vv=[]
            for y in (0,t):
                for outer in (False,True):
                    for a in (k*2*np.pi/32,(k+1)*2*np.pi/32):
                        radius=.006/max(abs(np.cos(a)),abs(np.sin(a))) if outer else r
                        vv.append([x+radius*np.cos(a),y,z+radius*np.sin(a)])
            _mesh(spec,b,f'camera_bore_{i}_{k}',vv,mass*.2/64,blue)
    camera=b.add_body(name='d435_wrist_body',pos=[0,t,z],gravcomp=gc)
    width,depth,height=c['body_size_m']
    _box(camera,'d435_case',[0,depth/2,0],[width/2,depth/2,height/2],c['camera_mass_kg'],[.72,.74,.77,1])
    _box(camera,'d435_front',[0,depth+.0002,0],[width/2-.002,.0002,height/2-.002],0,[.04,.05,.06,1])
    for i,x in enumerate((-.025,0,.025,.038)):
        camera.add_geom(name=f'd435_lens_{i}',type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                        fromto=[x,depth+.0004,0,x,depth+.0008,0],size=[.0035,0,0],
                        mass=0,rgba=[.08,.24,.32,1],contype=0,conaffinity=0)
    # MuJoCo camera looks down -Z; in tool coordinates it looks along +Y, up +Z.
    # Centered optical proxy, NOT the calibrated left-IR optical frame.
    camera.add_camera(name='d435_wrist',pos=[0,depth+.001,0],
                      quat=[2**-.5,2**-.5,0,0],fovy=58)
    camera.add_site(name='d435_optical_proxy',pos=[0,depth+.001,0],size=[.002,0,0])


def export_scad(cfg,path):
    """Additive extension solid in tool-frame mm; fuse root into the existing holder CAD."""
    c=cfg.lab_geometry['camera_mount']
    z=c['lateral_offset_m']*1000;t=c['plate_thickness_m']*1000
    spacing=c['hole_spacing_m']*1000;diam=c['clearance_diameter_m']*1000
    text=f'''// DRAFT ONLY: additive extension; fuse root with existing trowel holder.
// Coordinate: X along handle, Y outward from flange, Z sideways.
// D435 rear M3, 45 mm centers, max screw insertion 3 mm.
// Not a complete flange/clamp CAD. Validate actual D435f and cable clearance.
$fn=64;
z={z}; t={t}; pitch={spacing}; bore={diam};
difference() {{
 union() {{
  translate([-8,0,20]) cube([16,t,z-20]);
  translate([-4,-6,20]) cube([8,6,z-29]);
  translate([-pitch/2-6,0,z-9]) cube([pitch+12,t,18]);
 }}
 for(x=[-pitch/2,pitch/2])
  translate([x,-1,z]) rotate([-90,0,0]) cylinder(h=t+2,d=bore);
}}
'''
    path.write_bytes(text.encode('utf-8'))

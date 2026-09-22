"""Photo-supported tool variant. All lengths SI; unmeasured quantities remain estimates.
Legacy scenes are unchanged. No hardware communication is imported here.
"""
import json
from pathlib import Path
import numpy as np
import mujoco

PROFILE = Path(__file__).resolve().parents[2] / 'models/tools/lab_20260922.json'


def scene_config(**overrides):
    from .scene import SceneConfig
    profile = json.loads(PROFILE.read_text(encoding='utf-8'))
    values = profile['scene'] | overrides
    return SceneConfig(**values)


def provenance(cfg):
    return {
        'profile': cfg.geometry_profile,
        'evidence': 'L1 model using user-provided dimension annotations; not hardware calibration',
        'source_manifest': 'experiments/2026-09-22_lab_tool/source_manifest.json',
        'dimensions': '124 mm blade, 42 mm heel, 35 mm shoulder; handle end sections 24x25.5 and 26.5x28.5 mm',
        'reducer': 'photo engraving 8-30; MINIF08-30 candidate, manufacturer unconfirmed; ratio 30:1 photo interpretation',
        'reducer_geometry': 'MINIF08 drawing + photo-based top adapter; installed length and holes not measured',
        'force_limit': '0.9 Nm continuous catalog candidate; motor torque-speed/efficiency not identified',
        'mount': 'proposed split elliptical clamp, simulated rigid; not a manufacturing-validated design',
        'unmeasured': 'all masses, blade thickness, handle axial length, neck position, adapter holes, stiffness and backlash',
        'gap': '27 mm interpreted as blade back to closest handle surface; accepted provisionally by user; precise gap not independently measured',
        'sensors': 'virtual force and optional wrist pinhole camera; no calibrated physical sensor model',
        'camera_mount': cfg.lab_geometry.get('camera_mount', {'enabled': False}),
        'joint_limits': 'CANDIDATE firmware ranges; arm models/dummy_v2.xml unchanged',
        'legacy_compatibility': 'old reach tables, policies and reported scores apply only to their original scenes',
    }


def blade_polygon(cfg):
    """(distance from tip, transverse coordinate), convex pentagon approximation."""
    L, t = cfg.trowel_length, cfg.trowel_tip_length
    w, shoulder = cfg.trowel_width, cfg.lab_geometry['shoulder_width_m']
    return np.array([[0., 0.], [t, -shoulder/2], [L, -w/2], [L, w/2], [t, shoulder/2]])


def blade_centroid(cfg):
    p = blade_polygon(cfg); q = np.roll(p, -1, axis=0)
    cross = p[:,0]*q[:,1]-q[:,0]*p[:,1]
    return float(np.sum((p[:,0]+q[:,0])*cross)/(3*np.sum(cross)))


def _mesh(spec, body, name, vertices, mass, rgba, contact=False):
    spec.add_mesh(name=name+'_mesh', uservert=np.asarray(vertices).flatten().tolist())
    return body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name+'_mesh',
                         mass=mass, rgba=rgba, contype=2 if contact else 0, conaffinity=0,
                         friction=[0.6, .005, .0001], solref=[.004, 1.], condim=3)


def _box(body, name, pos, size, mass, colour):
    return body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, pos=pos, size=size,
                         mass=mass, rgba=colour, contype=0, conaffinity=0)


def refine_spec(spec, cfg):
    """Replace approximate surfaces with measured outline, tapered handle and split clamp.

    The reducer's fixed mass sits on a child body: adding geoms to link5 would otherwise
    be ignored because link5 has an explicit inertial. Output flange/clamp stay on link6.
    """
    from .scene import housing_front_x, clamp_position
    if cfg.tool_mount != 'rigid':
        raise ValueError('lab tool requires rigid mount')
    g = cfg.lab_geometry
    if not (0 < cfg.trowel_tip_length < cfg.trowel_length and
            0 < g['shoulder_width_m'] <= cfg.trowel_width):
        raise ValueError('invalid measured blade dimensions')
    if cfg.blade_tilt_deg != 0:
        raise ValueError('lab mount is designed at zero fixed tilt; change arm joints for work pitch')
    for name in ('j6_reducer_base', 'j6_reducer_body', 'j6_output_flange', 'holder_geom',
                 'handle_geom', 'neck_geom', 'blade_geom', 'blade_tip_geom'):
        spec.delete(spec.geom(name))
    spec.delete(spec.mesh('trowel_tip'))
    gc = 1. if cfg.gravcomp else 0.
    grey = [.73,.75,.77,1]; dark = [.13,.14,.16,1]; plastic = [.12,.48,.69,1]
    base = spec.body('link5').add_body(name='j6_reducer_fixed', pos=[housing_front_x(),0,0], gravcomp=gc)
    p, L = cfg.j6_reducer_plate_thickness, cfg.j6_reducer_length
    adapter = g['adapter_thickness_m']
    _box(base, 'j6_reducer_base', [p/2,0,0], [p/2,cfg.j6_reducer_plate/2,cfg.j6_reducer_plate/2],
         cfg.j6_reducer_mass*.25, grey)
    base.add_geom(name='j6_reducer_body', type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                  fromto=[p,0,0,L-adapter,0,0], size=[cfg.j6_reducer_radius,0,0],
                  mass=cfg.j6_reducer_mass*.55, rgba=grey, contype=0, conaffinity=0)
    mount = spec.body('tool_mount')
    mount.add_geom(name='j6_output_flange', type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                   fromto=[0,-adapter-.002,0,0,-adapter,0], size=[.013,0,0],
                   mass=cfg.j6_reducer_mass*.20, rgba=dark, contype=0, conaffinity=0)
    a = g['adapter_width_m']/2
    _box(mount, 'existing_adapter', [0,-adapter/2,0], [a,adapter/2,a], g['adapter_mass_kg']*.8, grey)
    for i,(x,z) in enumerate(((-a+.004,-a+.004),(-a+.004,a-.004),(a-.004,-a+.004),(a-.004,a-.004))):
        _box(mount, f'adapter_lug_{i}', [x,.002,z], [.004,.002,.004], g['adapter_mass_kg']*.05, grey)
    holder = spec.body('trowel')
    tr = spec.body('trowel_body')
    c = clamp_position(cfg); h0,h1 = cfg.handle_from_tip
    hh, r = cfg.holder_height, cfg.handle_radius
    yh = hh+r
    dsmall, dlarge = g['handle_small_section_m'], g['handle_large_section_m']
    vertices=[]
    for distance, diam in ((h0,dsmall),(h1,dlarge)):
        for t in np.linspace(0,2*np.pi,48,endpoint=False):
            vertices.append([c-distance,yh+diam[1]/2*np.cos(t),diam[0]/2*np.sin(t)])
    _mesh(spec,tr,'handle_geom',vertices,cfg.tool_mass*.40,[.72,.49,.26,1])
    yn = hh+2*r+cfg.neck_height
    xn = c-cfg.neck_from_tip
    neck_fraction=(cfg.neck_from_tip-h0)/(h1-h0)
    neck_bottom=yh+(dsmall[1]*(1-neck_fraction)+dlarge[1]*neck_fraction)/2-.001
    _box(tr,'neck_geom',[xn,(neck_bottom+yn)/2,0],[.005,(yn-neck_bottom)/2,.003],cfg.tool_mass*.12,dark)
    # Flat flange plate + two longitudinally separated split elliptical rings. Leave the
    # neck unobstructed. Each convex sector is explicit; no convex hull closes the bore.
    _box(holder,'holder_geom',[0,.003,0],[.034,.003,.023],cfg.holder_mass*.30,plastic)
    ring_width=.009; centres=(-.027,.022)
    for ring, xc in enumerate(centres):
        distance=c-xc
        f=(distance-h0)/(h1-h0)
        if not 0 < f < 1:
            raise ValueError('clamp rings must be inside handle ends')
        diam=np.array(dsmall)*(1-f)+np.array(dlarge)*f
        ry,rz=diam[1]/2+g['clamp_clearance_m'],diam[0]/2+g['clamp_clearance_m']
        if abs(xc-xn) < ring_width/2+.006:
            raise ValueError('clamp ring intersects trowel neck; move clamp/neck')
        for k in range(24):
            # two gaps at the split plane, hardware can tighten the proposed cap
            t0=2*np.pi*k/24+(.012 if k in (6,18) else 0)
            t1=2*np.pi*(k+1)/24-(.012 if k in (5,17) else 0)
            vv=[]
            for xx in (xc-ring_width/2,xc+ring_width/2):
                for dr in (0,g['clamp_wall_m']):
                    for t in (t0,t1):
                        vv.append([xx,yh+(ry+dr)*np.cos(t),(rz+dr)*np.sin(t)])
            _mesh(spec,holder,f'clamp_{ring}_{k}',vv,cfg.holder_mass*.50/48,plastic)
        support_top=yh-ry-g['clamp_wall_m']+.001
        _box(holder,f'clamp_support_{ring}',[xc,(.006+support_top)/2,0],
             [ring_width/2,(support_top-.006)/2,.008],cfg.holder_mass*.10,plastic)
        for side in (-1,1):
            z=side*(rz+g['clamp_wall_m']+.003)
            _box(holder,f'clamp_ear_{ring}_{side}',[xc,yh,z],[.006,.004,.004],0,dark)
    # Constant-width material proxy is NOT used here: both collision pieces follow the
    # same measured polygon. Sharp curved nose is approximated as straight shoulders.
    blade=spec.body('blade'); th=cfg.trowel_thickness
    shoulder=g['shoulder_width_m']/2; heel=cfg.trowel_width/2
    for name,poly,share in (
        ('blade_geom',[(c-cfg.trowel_length,-heel),
                       (c-cfg.trowel_tip_length,-shoulder),(c-cfg.trowel_tip_length,shoulder),
                       (c-cfg.trowel_length,heel)],.82),
        ('blade_tip_geom',[(c-cfg.trowel_tip_length,-shoulder),(c,0),
                           (c-cfg.trowel_tip_length,shoulder)],.18)):
        vv=[[x,y,z] for y in (0,th) for x,z in poly]
        _mesh(spec,blade,name,vv,cfg.tool_mass*.48*share,[.55,.58,.61,1],contact=True)
    mount.add_site(name='output_flange_frame',pos=[0,0,0],size=[.002,0,0])
    tr.add_site(name='handle_center',pos=[0,yh,0],size=[.002,0,0])
    blade.add_site(name='blade_tip',pos=[c,th,0],size=[.002,0,0])
    # Both actuator-level AND joint-level clamps must agree. Historical XML limits J6
    # to direct-drive torque, even when the actuator's forcerange is raised.
    tau=min(.07*cfg.j6_reducer_ratio,cfg.j6_reducer_max_Nm)
    spec.joint('Joint6').actfrcrange=[-tau,tau]

    from .wrist_camera import add_wrist_camera
    add_wrist_camera(spec, cfg)

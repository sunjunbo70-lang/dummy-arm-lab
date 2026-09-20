"""由 Dummy V2 装配体 STEP 生成 MuJoCo 模型 models/dummy_v2.xml。

运动学：严格按 V2 固件 DH（dummy_loop/v2.py），模型零位 = 固件 HOME，
关节正方向 = 固件正方向，世界系 = 固件基座系（+X 朝前，+Z 向上）。
外观与质量：来自 V2 装配体 STEP。装配体画在收纳姿态 FOLD=(0,-75,180,0,0,0)，
脚本用减速器中心（关节轴）把 CAD 坐标对齐到固件坐标，再把每个零件放进它所属连杆的坐标系。
末端：J6 是 35-20 电机直驱，输出就是电机裸轴 —— 模型末端是这根轴，没有法兰。

质量：结构件体积 × 等效密度（3D 打印默认 0.85 g/cm³，CNC 铝件用 --structure aluminium），
标准件按材质密度，电机 / 减速器按资料质量（dummy_loop/v2.py）。全部是估计值，称重后替换。

用法（STEP 不随仓库分发，来自 V2 资料包，见 THIRD_PARTY.md）：
  python tools/modeling/build_dummy_v2.py --step "<资料包>/4.Model/Dummy V2 版本 视频款/Dummy-B v2装配体.step"
依赖：gmsh（读零件名与质量属性）、cadquery-ocp（三角化），仅建模时需要，运行仿真不需要。
"""
import argparse, hashlib, json, struct, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dummy_loop import v2  # noqa: E402

LINKS = ['base_link', 'link1', 'link2', 'link3', 'link4', 'link5', 'link6']
GROUP_TO_LINK = {'BASE Moudle-普赢': 0, 'J1': 1, 'J2&J3': 2, 'J4': 3, 'J5': 4, 'J6': 5}
COLORS = ['0.30 0.32 0.36 1', '0.93 0.93 0.95 1', '0.93 0.93 0.95 1', '0.93 0.93 0.95 1',
          '0.93 0.93 0.95 1', '0.93 0.93 0.95 1', '0.75 0.75 0.78 1']
# CAD（mm，装配体坐标）-> 固件基座系（m）。由减速器中心确定：
#   J1 减速器中心 CAD (x=-565.0, y=35.0)；J2 减速器中心 CAD (y=0.0, z=110.0) 对应固件肩部 (0.035, 0, 0.1265)。
#   CAD -Y = 固件 +X（朝前），CAD +X = 固件 +Y，CAD +Z = 固件 +Z。
CAD_J1_X, CAD_J1_Y, CAD_Z_OFFSET = -565.0, 35.0, 16.5
# 采购件中心（CAD mm）：用于把未命名实体归到具体电机 / 减速器，按资料质量分摊。
COMPONENTS = {
    'J1 MINI11-50': ((-565, 35, 52), 'reducer', 'MINI11-50'),
    'J2 MINI11-50': ((-562, 0, 110), 'reducer', 'MINI11-50'),
    'J3 MINI11-50': ((-568, 141, 148), 'reducer', 'MINI11-50'),
    'J4 MINI11-50': ((-566, 138, 201), 'reducer', 'MINI11-50'),
    'J5 MINI8-50': ((-567, 17, 168), 'reducer', 'MINI8-50'),
    'J1 motor 42-34': ((-565, 35, 18), 'motor', '42-34'),
    'J2 motor 42-40': ((-569, 43.5, 121.6), 'motor', '42-40'),
    'J3 motor 42-34': ((-563.5, 97.6, 136.1), 'motor', '42-34'),
    'J4 motor 35-28': ((-565.7, 165, 208), 'motor', '35-28'),
    'J5 motor 35-20': ((-568, 69.3, 182.6), 'motor', '35-20'),
    'J6 motor 35-20': ((-566, -28, 156.5), 'motor', '35-20'),
}
COMPONENT_RADIUS_MM = 34.0
FLOOR_Z = (-24.0 + CAD_Z_OFFSET) / 1000.0
DENSITY = {'steel': 7.85, 'aluminium': 2.70, 'belt': 1.2, 'electronics': 2.0, 'other_purchased': 2.7}
STRUCTURE_DENSITY = {'printed': 0.85, 'aluminium': 2.70}
SHAFT_RADIUS_MAX_MM = 3.0   # J6 电机轴 Φ5：离 J6 轴线 3 mm 以内且在电机前端面之外的实体算输出轴


def cad_to_fw(p_mm):
    p = np.asarray(p_mm, float)
    return np.stack([-(p[..., 1] - CAD_J1_Y), p[..., 0] - CAD_J1_X, p[..., 2] + CAD_Z_OFFSET], -1) / 1000.0


CAD_TO_FW_R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.]])


def material(name):
    n = name or ''
    if any(k in n for k in ('螺丝', 'nut', '垫片', '轴承', 'D字轴')):
        return 'steel'
    if '同步轮' in n:
        return 'aluminium'
    if '同步带' in n:
        return 'belt'
    if 'JACK' in n:
        return 'electronics'
    if 'Dummy-' in n:
        return 'structure'
    return None


def skeleton():
    """模型零位（HOME）下各连杆原点（世界系）与关节轴。连杆坐标系在零位时与世界系同向。"""
    f = v2.firmware_fk(v2.HOME_DEG)
    pts = [np.zeros(3), f['shoulder'], f['elbow'], f['j4_point'], f['wrist'], f['wrist']]
    axes = [f['R'][i][:, 2] for i in range(6)]
    return pts, axes, f


def load_step(step):
    import gmsh
    gmsh.initialize(); gmsh.option.setNumber('General.Terminal', 0)
    gmsh.option.setNumber('Geometry.OCCImportLabels', 1)
    gmsh.model.occ.importShapes(str(step)); gmsh.model.occ.synchronize()
    vols = []
    for _, t in gmsh.model.getEntities(3):
        name = gmsh.model.getEntityName(3, t) or ''
        vols.append(dict(tag=t, name=name, vol=gmsh.model.occ.getMass(3, t),
                         com=np.array(gmsh.model.occ.getCenterOfMass(3, t)),
                         I=np.array(gmsh.model.occ.getMatrixOfInertia(3, t)).reshape(3, 3),
                         bbox=np.array(gmsh.model.getBoundingBox(3, t))))
    return gmsh, vols


def classify(vols):
    lab = [v for v in vols if v['name']]
    for v in lab:
        v['group'] = v['name'].split('/')[2].split(':')[0]
        v['link'] = GROUP_TO_LINK[v['group']]
    for v in vols:
        if v['name']:
            continue
        b = v['bbox']
        best = min(lab, key=lambda w: (np.linalg.norm(np.maximum(0, np.maximum(w['bbox'][:3] - b[3:], b[:3] - w['bbox'][3:]))),
                                       np.linalg.norm(w['com'] - v['com'])))
        v['group'] = 'unlabeled->' + best['group']; v['link'] = best['link']
    # 材质与质量
    comp_members = {k: [] for k in COMPONENTS}
    for v in vols:
        mat = material(v['name'])
        if mat is None and not v['name']:
            dist = {k: np.linalg.norm(v['com'] - np.array(c[0])) for k, c in COMPONENTS.items()}
            k = min(dist, key=dist.get)
            if dist[k] < COMPONENT_RADIUS_MM:
                comp_members[k].append(v); v['component'] = k; mat = 'component'
            else:
                mat = 'other_purchased'
        v['material'] = mat or 'structure'
    return comp_members


def assign_masses(vols, comp_members, structure):
    for v in vols:
        if v['material'] == 'structure':
            v['density'] = STRUCTURE_DENSITY[structure]
        elif v['material'] != 'component':
            v['density'] = DENSITY[v['material']]
    comps = {}
    for k, members in comp_members.items():
        _, kind, spec = COMPONENTS[k]
        mass = (v2.REDUCERS if kind == 'reducer' else v2.MOTORS)[spec]['mass_kg']
        vol = sum(m['vol'] for m in members)
        for m in members:
            m['density'] = mass * 1000.0 / (vol / 1000.0)   # g/cm³，使整组质量等于资料值
        comps[k] = dict(spec=spec, kind=kind, mass_kg=mass, n_solids=len(members),
                        cad_volume_cm3=round(vol / 1000, 2))
    for v in vols:
        v['mass'] = v['density'] * v['vol'] / 1e6    # kg（体积 mm³，密度 g/cm³）
    return comps


def tessellate(step, lin_mm, ang_rad):
    """用 OpenCascade 直接三角化每个实体（比体网格化稳：STEP 里有 gmsh 不能处理的周期曲面）。"""
    from OCP.STEPControl import STEPControl_Reader
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_SOLID, TopAbs_FACE, TopAbs_REVERSED
    from OCP.TopoDS import TopoDS
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.BRep import BRep_Tool
    from OCP.TopLoc import TopLoc_Location
    from OCP.GProp import GProp_GProps
    from OCP.BRepGProp import BRepGProp
    r = STEPControl_Reader(); r.ReadFile(str(step)); r.TransferRoots(); shape = r.OneShape()
    BRepMesh_IncrementalMesh(shape, lin_mm, False, ang_rad, True)
    out = []
    ex = TopExp_Explorer(shape, TopAbs_SOLID)
    while ex.More():
        sol = TopoDS.Solid(ex.Current()); ex.Next()
        g = GProp_GProps(); BRepGProp.VolumeProperties_s(sol, g); c = g.CentreOfMass()
        tris = []
        fe = TopExp_Explorer(sol, TopAbs_FACE)
        while fe.More():
            f = TopoDS.Face(fe.Current()); fe.Next()
            loc = TopLoc_Location(); T = BRep_Tool.Triangulation_s(f, loc)
            if T is None:
                continue
            tr = loc.Transformation()
            P = np.array([[q.X(), q.Y(), q.Z()] for q in (T.Node(i).Transformed(tr) for i in range(1, T.NbNodes() + 1))])
            idx = np.array([T.Triangle(i).Get() for i in range(1, T.NbTriangles() + 1)]) - 1
            if f.Orientation() == TopAbs_REVERSED:
                idx = idx[:, ::-1]
            tris.append(P[idx])
        out.append(dict(vol=g.Mass(), com=np.array([c.X(), c.Y(), c.Z()]),
                        tris=np.concatenate(tris) if tris else np.zeros((0, 3, 3))))
    return out


def attach_triangles(vols, tess):
    """按体积与质心把 OpenCascade 实体对应到 gmsh 实体（两者读的是同一个 STEP）。"""
    used = set()
    for v in vols:
        k = min((i for i in range(len(tess)) if i not in used),
                key=lambda i: abs(tess[i]['vol'] - v['vol']) / max(v['vol'], 1) + np.linalg.norm(tess[i]['com'] - v['com']))
        if abs(tess[k]['vol'] - v['vol']) > 1e-3 * max(v['vol'], 1) or np.linalg.norm(tess[k]['com'] - v['com']) > 1e-3:
            raise RuntimeError(f'无法匹配实体 {v["tag"]} {v["name"]}')
        used.add(k); v['tris'] = tess[k]['tris']


def is_visual(v):
    """外观网格里去掉紧固件（螺纹三角面极多）与极小零件；它们仍计入质量。"""
    return not any(k in v['name'] for k in ('螺丝', 'nut', '垫片')) and v['vol'] > 100.0


def cluster_decimate(tris, cell_mm):
    """顶点聚类减面：把落在同一个 cell_mm 立方格里的顶点合并，丢掉退化与重复三角形。
    只用于外观网格；质量属性来自 CAD 实体本身，不受影响。"""
    if cell_mm <= 0 or len(tris) == 0:
        return tris
    V = tris.reshape(-1, 3)
    key = np.floor(V / cell_mm).astype(np.int64)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    rep = np.zeros((len(uniq), 3)); np.add.at(rep, inv, V); rep /= np.bincount(inv)[:, None]
    F = inv.reshape(-1, 3)
    F = F[(F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])]
    _, keep = np.unique(np.sort(F, axis=1), axis=0, return_index=True)
    return rep[F[np.sort(keep)]]


def write_stl(path, tris):
    tris = np.asarray(tris, np.float32)
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    rec = np.zeros(len(tris), dtype=[('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')])
    rec['n'] = n; rec['v'] = tris
    with open(path, 'wb') as fh:
        fh.write(b'dummy_v2 generated by tools/modeling/build_dummy_v2.py'.ljust(80, b' '))
        fh.write(struct.pack('<I', len(tris))); fh.write(rec.tobytes())


def fold_link_frames():
    """收纳姿态（CAD 姿态）下各连杆坐标系在世界系的位姿：用 MuJoCo 骨架直接算。"""
    import mujoco
    m = mujoco.MjModel.from_xml_string(model_xml(visual=False))
    d = mujoco.MjData(m)
    d.qpos[:6] = v2.firmware_to_model(v2.FOLD_DEG); mujoco.mj_kinematics(m, d)
    out = []
    for name in LINKS:
        b = m.body(name).id
        out.append((d.xpos[b].copy(), d.xmat[b].reshape(3, 3).copy()))
    return out


def _f(a):
    return ' '.join(f'{x:.6g}' for x in np.round(np.asarray(a, float).ravel(), 10) + 0.0)


def model_xml(visual=True, inertials=None, mesh_names=None, shaft=None):
    pts, axes, f = skeleton()
    lo, hi = v2.model_limits_rad()
    jp = v2.joint_params()
    end_local = f['end'] - f['wrist']
    R06 = f['R'][6]
    import mujoco
    q = np.zeros(4); mujoco.mju_mat2Quat(q, R06.flatten())
    L = []
    L.append('<mujoco model="dummy_v2_FIRMWARE_DH_CAD_MASS_ESTIMATES">')
    L.append('  <!-- 由 tools/modeling/build_dummy_v2.py 生成，请勿手改。运动学 = V2 固件 DH；'
             '质量/惯量 = CAD 体积×估计密度 + 资料质量；执行器 = 资料额定值推算。均未经实机辨识。 -->')
    L.append('  <compiler angle="radian" meshdir="meshes_v2" autolimits="true" boundinertia="1e-7"/>')
    L.append('  <option timestep="0.002" integrator="implicitfast"/>')
    L.append('  <visual><global offwidth="960" offheight="720"/></visual>')
    L.append('  <default><geom contype="0" conaffinity="0"/></default>')
    if visual:
        L.append('  <asset>')
        for n in mesh_names:
            L.append(f'    <mesh name="{n}" file="{n}.stl" scale="0.001 0.001 0.001" inertia="shell"/>')
        L.append('  </asset>')
    L.append('  <worldbody>')
    L.append('    <light pos="0.3 -0.6 1.2" dir="-0.2 0.5 -1" diffuse="0.8 0.8 0.8"/>')
    # 底座底面：CAD z = -24 mm -> 固件 z = -7.5 mm
    L.append(f'    <geom name="floor" type="plane" size="1 1 0.01" pos="0 0 {FLOOR_Z:.4f}" rgba="0.18 0.20 0.24 1"/>')
    ind = '    '
    parent = np.zeros(3)
    for i, name in enumerate(LINKS):
        if i == 0:
            L.append(f'{ind}<body name="base_link" gravcomp="1">')
        else:
            p = pts[i - 1] if i <= 6 else None
            L.append(f'{ind}<body name="{name}" pos="{_f(p - parent)}" gravcomp="1">')
            parent = p
            j = jp[i - 1]
            arm = j['reflected_inertia_kgm2']
            L.append(f'{ind}  <joint name="Joint{i}" axis="{_f(np.round(axes[i - 1], 9))}" '
                     f'range="{lo[i - 1]:.6f} {hi[i - 1]:.6f}" armature="{arm:.6g}" damping="0.3" '
                     f'actuatorfrcrange="{-j["peak_Nm"]:.4g} {j["peak_Nm"]:.4g}"/>')
        if inertials and name in inertials:
            ine = inertials[name]
            L.append(f'{ind}  <inertial pos="{_f(ine["com"])}" mass="{ine["mass"]:.6g}" '
                     f'fullinertia="{_f(ine["fullinertia"])}"/>')
        elif not inertials:
            L.append(f'{ind}  <inertial pos="0 0 0" mass="0.1" diaginertia="1e-4 1e-4 1e-4"/>')
        if visual and name in mesh_names:
            L.append(f'{ind}  <geom name="{name}_visual" type="mesh" mesh="{name}" rgba="{COLORS[i]}" '
                     f'density="0" group="1"/>')
        if name == 'link6':
            if shaft:
                L.append(f'{ind}  <!-- J6 裸轴：CAD 量得 Φ{shaft["diameter_mm"]:.1f} mm，'
                         f'轴端距腕心 {shaft["tip_from_wrist_mm"]:.1f} mm（固件 L_WT = {v2.DH_V2["L_WT"] * 1000:.1f} mm） -->')
                L.append(f'{ind}  <site name="shaft_tip" pos="{shaft["tip_from_wrist_mm"] / 1000:.5f} 0 0" '
                         f'size="0.003" rgba="1 0.5 0 1"/>')
            L.append(f'{ind}  <site name="fw_end" pos="{_f(end_local)}" quat="{_f(q)}" size="0.004" rgba="0 0.8 0.2 1"/>')
        ind += '  '
    for i in range(len(LINKS)):
        ind = ind[:-2]; L.append(f'{ind}</body>')
    L.append('  </worldbody>')
    L.append('  <actuator>')
    for i, j in enumerate(jp, 1):
        L.append(f'    <position name="servo{i}" joint="Joint{i}" kp="35" kv="4" '
                 f'ctrlrange="{lo[i - 1]:.6f} {hi[i - 1]:.6f}" forcerange="{-j["peak_Nm"]:.4g} {j["peak_Nm"]:.4g}"/>')
    L.append('  </actuator>')
    L.append('</mujoco>')
    return '\n'.join(L) + '\n'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--step', type=Path, required=True)
    ap.add_argument('--structure', choices=sorted(STRUCTURE_DENSITY), default='printed',
                    help='结构件材质：printed（3D 打印，默认）或 aluminium（CNC 铝件）')
    ap.add_argument('--mesh-tol', type=float, default=0.8, help='三角化弦高误差 mm')
    ap.add_argument('--mesh-angle', type=float, default=0.8, help='三角化角度误差 rad')
    ap.add_argument('--decimate-mm', type=float, default=1.2, help='外观网格顶点聚类格子 mm（0 = 不减面）')
    ap.add_argument('--out', type=Path, default=ROOT / 'models')
    a = ap.parse_args()

    gmsh, vols = load_step(a.step)
    comp_members = classify(vols)
    comps = assign_masses(vols, comp_members, a.structure)
    frames = fold_link_frames()
    pts, axes, f_home = skeleton()

    # J6 输出轴：离 J6 轴线 ≤3 mm、且整体在 J6 电机前端面之外的实体（装配体姿态下量）
    ffold = v2.firmware_fk(v2.FOLD_DEG)
    w, ax6 = ffold['wrist'], ffold['R'][6][:, 2]
    gmsh.finalize()
    attach_triangles(vols, tessellate(a.step, a.mesh_tol, a.mesh_angle))
    # 装配体里 J6 电机轴与电机前端盖是同一个实体：按三角面拆出「离 J6 轴线 ≤3.5 mm、
    # 且在电机前端面之外」的部分作为输出轴（link6）。
    j6m = [v for v in vols if v.get('component') == 'J6 motor 35-20']
    def axial(t):
        P = cad_to_fw(t.reshape(-1, 3)); s_ = (P - w) @ ax6
        r_ = np.linalg.norm((P - w) - np.outer(s_, ax6), axis=1)
        return s_.reshape(-1, 3) * 1000, r_.reshape(-1, 3) * 1000
    motor_face = max(sa[ra.min(1) > 6].max() for sa, ra in (axial(v['tris']) for v in j6m))   # 电机壳体最前端
    shaft_tris, shaft_r, tip, solids = [], 0.0, 0.0, []
    for v in j6m:
        sa, ra = axial(v['tris'])
        sel = (ra.max(1) <= SHAFT_RADIUS_MAX_MM + 0.5) & (sa.min(1) >= motor_face - 1e-6)
        if sel.any():
            shaft_tris.append(v['tris'][sel]); v['tris'] = v['tris'][~sel]; solids.append(v['tag'])
            shaft_r = max(shaft_r, ra[sel].max()); tip = max(tip, sa[sel].max())
    if not shaft_tris:
        raise RuntimeError('没有找到 J6 输出轴')
    shaft_tris = np.concatenate(shaft_tris)
    # 直径按轴自身中心量（CAD 里腕部相对 J1 平面有 0.7 mm 横向偏差，按 J6 名义轴量会偏大）
    Ps = cad_to_fw(shaft_tris.reshape(-1, 3)); Ps = Ps - np.outer((Ps - w) @ ax6, ax6)
    ctr = (Ps.max(0) + Ps.min(0)) / 2
    shaft_r = np.linalg.norm(Ps - ctr, axis=1).max() * 1000
    shaft_axis_offset_mm = float(np.linalg.norm(ctr - (w - ((w - w) @ ax6) * ax6)) * 1000)
    # 外壳（J6 电机座）前端：用户看到的「裸轴」从这里露出
    housing_front = max(axial(v['tris'])[0].max() for v in vols
                        if v['link'] == 5 and v['material'] == 'structure' and len(v['tris']))
    shaft = dict(diameter_mm=round(2 * shaft_r, 2), tip_from_wrist_mm=round(tip, 2),
                 motor_face_from_wrist_mm=round(motor_face, 2), housing_front_from_wrist_mm=round(housing_front, 2),
                 length_beyond_motor_face_mm=round(tip - motor_face, 2),
                 visible_length_beyond_housing_mm=round(tip - housing_front, 2), cad_solids=solids,
                 cad_axis_offset_from_firmware_axis_mm=round(shaft_axis_offset_mm, 2),
                 note='D-cut shaft; the shaft and front cap are one solid in the CAD, split by radius')
    L_sh = (tip - motor_face) / 1000; r_sh = shaft_r / 1000
    m_sh = DENSITY['steel'] * 1000 * np.pi * r_sh ** 2 * L_sh

    # 每个连杆：网格 + 质量属性（连杆坐标系）
    mesh_dir = a.out / 'meshes_v2'; mesh_dir.mkdir(parents=True, exist_ok=True)
    inertials, mesh_names, link_rows, stl_sha = {}, [], {}, {}
    for li, name in enumerate(LINKS):
        pos, R = frames[li]
        if name == 'link6':
            # 输出轴：实心钢圆柱近似（转子质量留在 link5 的电机里，转子惯量进 armature）
            cx = (motor_face + tip) / 2000
            Ixx = 0.5 * m_sh * r_sh ** 2; Iyy = m_sh * (3 * r_sh ** 2 + L_sh ** 2) / 12
            inertials[name] = dict(mass=m_sh, com=np.array([cx, 0, 0]), fullinertia=[Ixx, Iyy, Iyy, 0, 0, 0])
            T = (cad_to_fw(shaft_tris.reshape(-1, 3)) - pos) @ R * 1000.0
            write_stl(mesh_dir / f'{name}.stl', T.reshape(-1, 3, 3))
            mesh_names.append(name); stl_sha[f'meshes_v2/{name}.stl'] = sha(mesh_dir / f'{name}.stl')
            link_rows[name] = dict(mass_kg=round(m_sh, 5), com_m=[round(cx, 5), 0, 0], n_solids=0,
                                   n_triangles=int(len(T) // 3), note='bare J6 shaft, steel cylinder approximation')
            continue
        members = [v for v in vols if v['link'] == li]
        M = sum(v['mass'] for v in members)
        # CAD 实体质心 -> 连杆系
        coms = [R.T @ (cad_to_fw(v['com']) - pos) for v in members]
        c = sum(v['mass'] * p for v, p in zip(members, coms)) / M
        I = np.zeros((3, 3))
        for v, p in zip(members, coms):
            Rc = R.T @ CAD_TO_FW_R
            Ic = v['density'] / 1e6 * (Rc @ v['I'] @ Rc.T) * 1e-6    # mm^5 * g/cm³ -> kg m²
            r = p - c
            I += Ic + v['mass'] * (r @ r * np.eye(3) - np.outer(r, r))
        inertials[name] = dict(mass=M, com=c,
                               fullinertia=[I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]])
        tris = [v['tris'] for v in members if len(v['tris']) and is_visual(v)]
        T = cluster_decimate(np.concatenate(tris), a.decimate_mm).reshape(-1, 3)
        T = (cad_to_fw(T) - pos) @ R * 1000.0          # 连杆系，mm
        write_stl(mesh_dir / f'{name}.stl', T.reshape(-1, 3, 3))
        mesh_names.append(name); stl_sha[f'meshes_v2/{name}.stl'] = sha(mesh_dir / f'{name}.stl')
        by_mat = {}
        for v in members:
            by_mat[v['material']] = by_mat.get(v['material'], 0) + v['mass']
        link_rows[name] = dict(mass_kg=round(M, 4), com_m=np.round(c, 5).tolist(), n_solids=len(members),
                               n_triangles=int(len(T) // 3), n_visual_solids=len(tris),
                               mass_by_material_kg={k: round(x, 4) for k, x in sorted(by_mat.items())},
                               components=sorted({v['component'] for v in members if v.get('component')}))

    xml = model_xml(True, inertials, mesh_names, shaft)
    (a.out / 'dummy_v2.xml').write_text(xml, encoding='utf-8')
    params = {
        'model': 'models/dummy_v2.xml',
        'evidence_level': 'L1 (documents + CAD); nothing identified on the real arm yet',
        'frame': 'world = firmware base frame: +X forward, +Z up, J1 axis at origin',
        'joint_zero': 'model q = deg2rad(firmware_deg - HOME), HOME = [0,0,90,0,0,0]',
        'dh_v2_m': v2.DH_V2,
        'firmware_limits_deg': {'lower': v2.FIRMWARE_LOWER_DEG_V2.tolist(), 'upper': v2.FIRMWARE_UPPER_DEG_V2.tolist()},
        'firmware_reduction': v2.FIRMWARE_REDUCTION_V2.tolist(),
        'drivetrain': v2.joint_params(),
        'actuator_model': ('position servo kp=35 kv=4 (example gains, not identified); forcerange = '
                           'min(reducer start/stop peak, motor holding x ratio); armature = rotor inertia x ratio^2 '
                           '(catalog values); damping 0.3 placeholder'),
        'end_effector': dict(kind='bare J6 motor shaft (direct drive, no flange)', **shaft),
        'structure_material': a.structure, 'structure_density_g_cm3': STRUCTURE_DENSITY[a.structure],
        'other_densities_g_cm3': DENSITY,
        'components': comps,
        'links': link_rows,
        'total_mass_kg': round(sum(r['mass_kg'] for r in link_rows.values()), 3),
        'moving_mass_kg': round(sum(r['mass_kg'] for k, r in link_rows.items() if k != 'base_link'), 3),
        'conflicts': v2.CONFLICTS,
        'cad_alignment': {'cad_pose_firmware_deg': v2.FOLD_DEG.tolist(),
                          'cad_to_firmware': 'x=-(y_cad-35.0), y=x_cad+565.0, z=z_cad+16.5 (mm), from reducer centres'},
    }
    (a.out / 'dummy_v2_params.json').write_text(json.dumps(params, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    prov = {'source_step': str(a.step.name), 'source_step_sha256': sha(a.step),
            'archive': '1.Dummy V2任同学整理.rar (see THIRD_PARTY.md)', 'license': 'GPL-3.0 (archive LICENSE); archive README: 仅用于学习，禁止一切商用行为',
            'generator': 'tools/modeling/build_dummy_v2.py', 'mesh_tolerance_mm': a.mesh_tol, 'mesh_angle_rad': a.mesh_angle, 'decimate_mm': a.decimate_mm,
            'files': {'dummy_v2.xml': sha(a.out / 'dummy_v2.xml'), **stl_sha}}
    (a.out / 'v2_provenance.json').write_text(json.dumps(prov, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({k: params[k] for k in ('total_mass_kg', 'moving_mass_kg', 'end_effector')}, ensure_ascii=False, indent=1))
    for k, r in link_rows.items():
        print(k, r['mass_kg'], r['n_solids'], r['n_triangles'], r.get('components', ''))


if __name__ == '__main__':
    main()

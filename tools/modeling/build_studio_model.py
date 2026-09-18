"""Build visual articulation recovered from the working Studio level0/bridge.

Raw OBJ vertices are in the assembled home pose. Keep every triangle.
Dynamics remain illustrative; this is not a calibrated physical model.
"""
from pathlib import Path
import json
import hashlib
import struct
import xml.etree.ElementTree as ET
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT/'models/studio_source'
OUT = ROOT/'models/studio_meshes'
OUT.mkdir(exist_ok=True)
# Millimetres in assembled coordinates after Unity Y-up -> Z-up conversion.
# Working level0 jointBodys transform IDs: 135,149,136,138,148,133.
# Bridge.Update rotates local axes -Y,-Z,-Z,-Y,-Z,-Y in Unity.
# Unity -> this mesh frame: (x,y,z) -> (z,-x,y), with handedness change.
PIVOTS = np.array([[0,0,109],[0,-35,109],[0,-35,255],
                   [0,-35,307],[0,-150,307],[0,-222,307]])*.001
AXES = ['0 0 1','1 0 0','1 0 0','0 -1 0','1 0 0','0 -1 0']
# Names describe the driven joint, not necessarily that joint's moving body.
# Long forearm + J5 stator belong to J4. Short wrist + J6 stator belong to J5.
PARTS = {113:2,114:4,115:4,116:-1,117:1,118:0,119:3,120:-1,
         121:1,122:4,123:2,124:2,125:1,126:3,127:3,128:-1,129:-1}
MOTORS = {114,124,125,126}
COVERS = {113,121,122,127}

def main():
    m=ET.Element('mujoco',model='DummyStudio_COMPLETE_VISUAL_UNCALIBRATED')
    ET.SubElement(m,'compiler',angle='radian',autolimits='true',meshdir='studio_meshes')
    ET.SubElement(m,'option',timestep='.002',integrator='implicitfast')
    asset=ET.SubElement(m,'asset'); world=ET.SubElement(m,'worldbody')
    base=ET.SubElement(world,'body',name='base')
    bodies=[]; parent=base
    # Reference inertias deliberately remain demo assumptions, separate from visuals.
    masses=[.40,.96,.38,.44,.13,.01]
    for i,pivot in enumerate(PIVOTS):
        offset=pivot-(PIVOTS[i-1] if i else np.zeros(3))
        b=ET.SubElement(parent,'body',name=f'link{i+1}',pos=' '.join(map(str,offset)),gravcomp='1')
        ET.SubElement(b,'inertial',pos='0 0 0',mass=str(masses[i]),diaginertia='.001 .001 .001')
        ET.SubElement(b,'joint',name=f'Joint{i+1}',axis=AXES[i],range='-.7 .7',damping='.3',armature='.02')
        bodies.append(b); parent=b
    manifest=[]
    hierarchy=json.loads((SRC/'robot_hierarchy.json').read_text(encoding='utf-8'))
    labels={r['gameobject_id']:r['parent_name'] for r in hierarchy}
    for path in sorted(SRC.glob('*.obj')):
        part=int(path.name.split('_')[0]); frame=PARTS[part]
        vertices=[]; faces=[]
        for line in path.read_text().splitlines():
            if line.startswith('v '): vertices.append(list(map(float,line.split()[1:4])))
            elif line.startswith('f '):
                ids=[int(t.split('/')[0]) for t in line.split()[1:]]
                ids=[v-1 if v>0 else len(vertices)+v for v in ids]
                for j in range(1,len(ids)-1): faces.append([ids[0],ids[j+1],ids[j]])
        v=np.asarray(vertices)[:,[0,2,1]]*.001
        if frame>=0: v-=PIVOTS[frame]
        triangles=v[np.asarray(faces)]
        normals=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
        lengths=np.linalg.norm(normals,axis=1)
        normals/=np.maximum(lengths,1e-30)[:,None]
        records=np.zeros(len(faces),dtype=[('n','<f4',(3,)),('v','<f4',(3,3)),('a','<u2')])
        records['n']=normals; records['v']=triangles
        color='0.12 0.14 0.17 1' if part in MOTORS else ('0.12 0.35 0.48 1' if part in COVERS else '0.65 0.68 0.72 1')
        if part==116: color='0.10 0.32 0.15 1'
        hashes={}
        # MuJoCo STL decoder caps a single file at 200000 triangles. Split,
        # never drop faces. Interleave so each chunk spans the whole part.
        count=(len(records)+149999)//150000
        for chunk in range(count):
            subset=records[chunk::count]
            filename=f'{part}_{labels[part]}_{chunk}.stl'
            target=OUT/filename
            target.write_bytes(b'DummyStudio visual export; uncalibrated'.ljust(80,b'\0')+struct.pack('<I',len(subset))+subset.tobytes())
            name=f'part{part}_{chunk}'
            ET.SubElement(asset,'mesh',name=name,file=filename)
            ET.SubElement(base if frame<0 else bodies[frame],'geom',name=f'{labels[part]}_{chunk}',type='mesh',mesh=name,rgba=color,contype='0',conaffinity='0',density='0',group='2' if part in COVERS else '0')
            hashes[filename]=hashlib.sha256(target.read_bytes()).hexdigest()
        manifest.append(dict(part=part,label=labels[part],triangles=len(faces),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),stl_sha256=hashes,articulated_body=frame+1))
    actuators=ET.SubElement(m,'actuator')
    for i in range(6): ET.SubElement(actuators,'position',name=f'servo{i+1}',joint=f'Joint{i+1}',kp='35',kv='4',ctrlrange='-.7 .7',forcerange='-5 5')
    ET.SubElement(world,'geom',name='floor',type='plane',size='1 1 .01',pos='0 0 -.021',rgba='.18 .20 .24 1',contype='0',conaffinity='0')
    ET.SubElement(world,'light',pos='0 -1 2',dir='0 0 -1',diffuse='.8 .8 .8',ambient='.3 .3 .3')
    ET.SubElement(world,'light',pos='1 1 1',dir='-1 -1 -1',diffuse='.5 .5 .5',castshadow='false')
    visual=ET.SubElement(m,'visual')
    ET.SubElement(visual,'global',offwidth='960',offheight='720')
    ET.SubElement(visual,'headlight',ambient='.35 .35 .35',diffuse='.7 .7 .7')
    tree=ET.ElementTree(m); ET.indent(tree)
    tree.write(ROOT/'models/dummy_studio_visual.xml',encoding='utf-8',xml_declaration=True)
    (ROOT/'models/studio_provenance.json').write_text(json.dumps({'source':'User-provided DummyStudio extracted OBJ files in dummy_robot_unified_demo/models/dummy/meshes','articulation_source':'Working DummyStudio level0 transforms and DummyRobotBridge.Update; outputs/studio_actual_transforms.json. jointBodys=[135,149,136,138,148,133]. Supersedes cyclic robot_hierarchy export.','warning':'Masses, inertia and servos remain illustrative. NOT calibrated hardware dynamics. No contact or gripper. J6 end_anchor contains no separate source mesh; wrist housing stays on J5. Source licensing not independently established; retain original notices before redistribution.','parts':manifest},indent=2),encoding='utf-8')
    print(json.dumps({'parts':len(manifest),'triangles':sum(r['triangles'] for r in manifest)}))

if __name__=='__main__': main()

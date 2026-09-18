"""Copy and convert a local auk URDF into a clearly labelled MuJoCo reference."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import xml.etree.ElementTree as ET

p=argparse.ArgumentParser()
p.add_argument('--source',type=Path,required=True,help='local dummy-v1-auk repository')
a=p.parse_args()
root=Path(__file__).resolve().parents[2]
out=root/'models'; out.mkdir(exist_ok=True)
src=a.source/'ros2/dummy_ws/src/dummy-ros2_description'
urdf=src/'urdf/dummy-ros2.xacro'
tree=ET.parse(urdf); robot=tree.getroot()
for element in list(robot):
    if element.tag.startswith('{'):
        robot.remove(element)
# CAD export contains zero inertia entries. Regularize for reference simulation only.
for inertia in robot.iter('inertia'):
    for axis in ('ixx','iyy','izz'):
        inertia.set(axis,str(float(inertia.get(axis,'0'))+1e-6))
# Visual and collision geometry are retained but contacts disabled in this MVP.
meshdir=out/'meshes'; meshdir.mkdir(exist_ok=True)
hashes={}
for mesh in robot.iter('mesh'):
    name=Path(mesh.attrib['filename']).name
    origin=src/'meshes'/name
    target=meshdir/name
    shutil.copy2(origin,target)
    hashes[name]=hashlib.sha256(target.read_bytes()).hexdigest()
    mesh.set('filename',name)
extension=ET.SubElement(robot,'mujoco')
ET.SubElement(extension,'compiler',{'discardvisual':'false','fusestatic':'false','balanceinertia':'true','boundinertia':'0.000001'})
temp=out/'reference_import.urdf'
ET.indent(tree); tree.write(temp,encoding='utf-8',xml_declaration=True)
import mujoco
assets={p.name:p.read_bytes() for p in meshdir.glob('*.stl')}
model=mujoco.MjModel.from_xml_string(ET.tostring(robot,encoding='unicode'),assets)
dest=out/'dummy_reference.xml'
with tempfile.TemporaryDirectory(prefix='dummy_mj_') as scratch:
    saved=Path(scratch)/'model.xml'
    mujoco.mj_saveLastXML(str(saved),model)
    shutil.copy2(saved,dest)
doc=ET.parse(dest); m=doc.getroot(); m.set('model','Dummy_auk_REFERENCE_NOT_CALIBRATED')
compiler=m.find('compiler')
compiler.set('meshdir','meshes'); compiler.set('autolimits','true')
for mesh in m.findall('asset/mesh'):
    mesh.set('file',Path(mesh.attrib['file']).name)
for geom in m.iter('geom'):
    geom.set('contype','0'); geom.set('conaffinity','0')
    if geom.get('group')!='1':
        geom.set('rgba','0.25 0.52 0.65 1')
for body in m.iter('body'):
    body.set('gravcomp','1')
for joint in m.iter('joint'):
    joint.set('damping','0.3'); joint.set('armature','0.02')
option=m.find('option')
if option is None: option=ET.SubElement(m,'option')
option.set('timestep','0.002'); option.set('integrator','implicitfast')
act=ET.SubElement(m,'actuator')
for i in range(1,7):
    ET.SubElement(act,'position',{'name':f'servo{i}','joint':f'Joint{i}','kp':'35','kv':'4','ctrlrange':'-0.7 0.7','forcerange':'-5 5'})
world=m.find('worldbody')
ET.SubElement(world,'light',{'pos':'0 -1 2','dir':'0 0 -1'})
ET.SubElement(world,'geom',{'name':'floor','type':'plane','size':'1 1 0.01','pos':'0 0 -0.01','rgba':'0.18 0.20 0.24 1','contype':'0','conaffinity':'0'})
visual=ET.SubElement(m,'visual'); ET.SubElement(visual,'global',{'offwidth':'960','offheight':'720'})
ET.indent(doc); doc.write(dest,encoding='utf-8',xml_declaration=True)
# Do not ship machine-specific absolute paths in the intermediate URDF.
for mesh in robot.iter('mesh'):
    mesh.set('filename','meshes/'+Path(mesh.attrib['filename']).name)
tree.write(temp,encoding='utf-8',xml_declaration=True)
shutil.copy2(a.source/'LICENSE',out/'UPSTREAM_LICENSE')
(out/'provenance.json').write_text(json.dumps({'source':'switchpi/dummy auk',
    'manifest_commit':'3a9d17464308e1532b3d648a80da6db899cc1f01',
    'urdf_sha256':hashlib.sha256(urdf.read_bytes()).hexdigest(), 'mesh_sha256':hashes,
    'warning':'Reference geometry only. Not calibrated to this Dummy V2. Gravity compensation and disabled contacts; servo/inertial adjustments are demo assumptions.'},indent=2),encoding='utf-8')
check=mujoco.MjModel.from_xml_string(dest.read_text(encoding='utf-8'),{'meshes/'+k:v for k,v in assets.items()})
print(json.dumps({'nq':check.nq,'nu':check.nu,'model':str(dest)}))

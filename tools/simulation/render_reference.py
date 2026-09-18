"""Render the reference model without opening a GUI."""
from pathlib import Path
import struct
import sys
import zlib
import argparse
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import mujoco
from dummy_loop.sim_backend import SimRobot

parser=argparse.ArgumentParser()
parser.add_argument('--studio',action='store_true')
parser.add_argument('--motion',action='store_true')
args=parser.parse_args()
r=SimRobot(Path(__file__).resolve().parents[2]/'models/dummy_studio_visual.xml') if args.studio else SimRobot()
r.connect()
if not args.studio or args.motion:
    for _ in range(80): r.send_action([.15,0,0,0,0,0] if args.studio else [.15,-.1,.2,.1,-.1,.05])
camera=mujoco.MjvCamera()
camera.lookat[:]=[0,0,.20]; camera.distance=.95; camera.azimuth=135; camera.elevation=-20
with mujoco.Renderer(r.model,height=720,width=960) as renderer:
    renderer.update_scene(r.data,camera=camera)
    pixels=renderer.render()
def chunk(kind,data):
    return struct.pack('!I',len(data))+kind+data+struct.pack('!I',zlib.crc32(kind+data)&0xffffffff)
height,width,_=pixels.shape
raw=b''.join(b'\x00'+row.tobytes() for row in pixels)
png=b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('!2I5B',width,height,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(raw))+chunk(b'IEND',b'')
output=Path('outputs/studio_motion_preview.png' if args.studio and args.motion else 'outputs/studio_preview.png' if args.studio else 'outputs/reference_preview.png'); output.parent.mkdir(exist_ok=True)
output.write_bytes(png)
print(output)

"""Parametric prototype, millimetres. Run with cadquery 2.8 + matplotlib.
No hardware or simulation configs are modified. See README for datum/assumptions.
"""
import json, math
from pathlib import Path
import cadquery as cq
from cadquery import exporters
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
ROOT=Path(__file__).resolve().parent
P=json.loads((ROOT/'parameters.json').read_text(encoding='utf-8-sig'))
OUT=ROOT/'models'; OUT.mkdir(exist_ok=True)
def box(dx,dy,dz,x=0,y=0,z=0):
    return cq.Workplane('XY').box(dx,dy,dz,centered=(True,True,False)).translate((x,y,z))
def bore(d,h,x,y,z):
    return cq.Workplane('XY').circle(d/2).extrude(h).translate((x,y,z))
def radii(x):
    t=(x+P['handle_length_estimated']/2)/P['handle_length_estimated']
    return tuple((a*(1-t)+b*t)/2+P['clamp_radial_clearance'] for a,b in zip(P['handle_large'],P['handle_small']))
def oval(x,ry,rz,w):
    return cq.Workplane('YZ',origin=(x-w/2,0,P['handle_axis_z'])).ellipse(ry,rz).extrude(w)
base=box(P['base_length'],P['base_width'],P['base_thickness']).edges('|Z').fillet(3)
# Bottom blind insert pockets; the entire contact plane remains at Z=0.
for x in (-P['flange_pitch']/2,P['flange_pitch']/2):
    for y in (-P['flange_pitch']/2,P['flange_pitch']/2):
        base=base.cut(bore(P['insert_pilot'],P['insert_depth'],x,y,0))
        base=base.cut(cq.Workplane('XY').circle((P['insert_pilot']+.5)/2).workplane(offset=.5).circle(P['insert_pilot']/2).loft().translate((x,y,0)))
caps=[]
for x in P['clamp_centres']:
    # Conservative tapered-handle envelope over each short collar, avoiding a
    # too-small bore at the thicker edge. Soft liner absorbs remaining taper.
    ry,rz=radii(x-P['clamp_width']/2)
    outer=oval(x,ry+P['clamp_wall'],rz+P['clamp_wall'],P['clamp_width'])
    cavity=oval(x,ry,rz,P['clamp_width']+2)
    bottom=outer.intersect(box(100,100,P['handle_axis_z']-8,z=8))
    bottom=bottom.union(box(P['clamp_width'],2*(ry+P['clamp_wall']),P['handle_axis_z']-8,x=x,z=8))
    cap=outer.intersect(box(100,100,50,z=P['handle_axis_z']+P['split_gap']))
    for y in (-P['clamp_bolt_y'],P['clamp_bolt_y']):
        bottom=bottom.union(box(P['clamp_width'],13,10,x,y,16))
        cap=cap.union(box(P['clamp_width'],13,5,x,y,P['handle_axis_z']+P['split_gap']))
    bottom=bottom.cut(cavity); cap=cap.cut(cavity)
    for y in (-P['clamp_bolt_y'],P['clamp_bolt_y']):
        bottom=bottom.cut(bore(P['insert_pilot'],P['insert_depth'],x,y,26-P['insert_depth']))
        cap=cap.cut(bore(3.4,10,x,y,26))
    base=base.union(bottom)
    caps.append(cap)
# Integral camera arm, backing plate, and root gusset. Front/lenses face +Z.
base=base.union(box(16,57,6,y=44.5,z=8))
base=base.union(box(57,18,P['camera_plate_thickness'],y=P['camera_offset_y'],z=8))
# Triangular underside rib; lower edge at Z=0, rises to arm underside Z=8.
rib=cq.Workplane('YZ',origin=(-3,0,0)).polyline([(18,0),(58,8),(18,8)]).close().extrude(6)
base=base.union(rib)
for x in (-P['camera_hole_pitch']/2,P['camera_hole_pitch']/2):
    base=base.cut(bore(P['camera_through_diameter'],10,x,P['camera_offset_y'],7))
base=base.clean()
# Fit coupon: four reference hole positions + three insert pilot choices.
coupon=box(38,38,8)
for x in (-13,13):
    for y in (-13,13): coupon=coupon.cut(bore(3.2,10,x,y,-1))
for x,d in zip((-10,0,10),(4.0,4.2,4.4)): coupon=coupon.cut(bore(d,5.5,x,0,2.5))
parts={'01_main_body':base,'02_tail_cap':caps[0],'03_tip_cap':caps[1],'04_fit_coupon':coupon}
audit={'evidence':'L1 CAD only; no physical fit or load validation','parts':{}}
for name,part in parts.items():
    s=part.val(); bb=s.BoundingBox()
    assert s.isValid() and len(part.solids().vals())==1, name
    exporters.export(part,str(OUT/(name+'.step')))
    # Caps print on their axial side (arch lies flat); main body bottom down.
    print_part=part.rotate((0,0,0),(0,1,0),90) if 'cap' in name else part
    print_part=print_part.translate((0,0,-print_part.val().BoundingBox().zmin))
    exporters.export(print_part,str(OUT/(name+'.stl')),tolerance=.05,angularTolerance=.1)
    audit['parts'][name]={'valid':s.isValid(),'solids':len(part.solids().vals()),'volume_mm3':s.Volume(),'bounds_mm':[bb.xlen,bb.ylen,bb.zlen],'stl_on_bed':True}
assembly=cq.Compound.makeCompound([base.val()]+[c.val() for c in caps])
exporters.export(assembly,str(OUT/'adapter_assembly.step'))
# Shape intersections must be empty for separate assembled print parts.
audit['cap_main_interference_mm3']=[base.val().intersect(c.val()).Volume() for c in caps]
assert max(audit['cap_main_interference_mm3'])<1e-6
# Probe flange blind pockets and camera through-bores at their axes.
audit['flange_void_checks']=[not base.val().isInside(cq.Vector(x,y,2),1e-5) for x in (-13,13) for y in (-13,13)]
audit['flange_floor_checks']=[base.val().isInside(cq.Vector(x,y,7),1e-5) for x in (-13,13) for y in (-13,13)]
audit['camera_void_checks']=[not base.val().isInside(cq.Vector(x,70,11),1e-5) for x in (-22.5,22.5)]
assert all(audit['flange_void_checks']+audit['flange_floor_checks']+audit['camera_void_checks'])
# Visual proxies are not manufacturing parts; handle ends based on photos.
handle=cq.Workplane('YZ',origin=(-47.5,0,26)).ellipse(13.25,14.25).workplane(offset=95).ellipse(12,12.75).loft()
blade=cq.Workplane('XY').polyline([(-49.5,-21),(50.5,-17.5),(74.5,0),(50.5,17.5),(-49.5,21)]).close().extrude(1).translate((0,0,67.25))
neck=box(7,7,28,x=-6.5,z=39.25)
camera=box(90,25.05,25,y=70,z=14)
exporters.export(cq.Compound.makeCompound([base.val()]+[c.val() for c in caps]+[handle.val(),blade.val(),neck.val(),camera.val()]),str(OUT/'reference_fit_NOT_FOR_PRINT.step'))
audit['handle_interference_mm3']=[p.val().intersect(handle.val()).Volume() for p in [base]+caps]
audit['camera_main_interference_mm3']=base.val().intersect(camera.val()).Volume()
assert max(audit['handle_interference_mm3'])<1e-5
assert audit['camera_main_interference_mm3']<1e-5
(ROOT/'cad_checks.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
def draw(ax,part,color,alpha=1):
    vv,tt=part.val().tessellate(.2)
    verts=[(v.x,v.y,v.z) for v in vv]
    poly=Poly3DCollection([[verts[i] for i in t] for t in tt],facecolors=color,alpha=alpha,shade=True,lightsource=matplotlib.colors.LightSource(azdeg=300,altdeg=45))
    ax.add_collection3d(poly)
def setup(ax,title,lim=(-55,80,-35,92,0,80)):
    ax.set(xlim=lim[:2],ylim=lim[2:4],zlim=lim[4:],xlabel='X / mm (handle)',ylabel='Y / mm',zlabel='Z / mm',title=title)
    ax.set_box_aspect((lim[1]-lim[0],lim[3]-lim[2],lim[5]-lim[4])); ax.view_init(27,-52)
fig=plt.figure(figsize=(15,7))
ax=fig.add_subplot(121,projection='3d'); draw(ax,base,'#368bad')
for c in caps: draw(ax,c,'#66c4d7')
setup(ax,'V1 adapter: assembled print parts')
ax=fig.add_subplot(122,projection='3d'); draw(ax,base,'#368bad')
for c in caps: draw(ax,c,'#66c4d7')
for p,col in [(handle,'#c59553'),(blade,'#bfc4c8'),(neck,'#696969'),(camera,'#303942')]: draw(ax,p,col)
setup(ax,'Fit concept: trowel + D435 (reference envelopes)')
fig.tight_layout(); fig.savefig(ROOT/'assembly_preview.png',dpi=160); plt.close(fig)
fig=plt.figure(figsize=(13,6))
ax=fig.add_subplot(121,projection='3d'); draw(ax,base,'#368bad'); setup(ax,'Main body: flat underside + 4 blind insert holes'); ax.view_init(-55,-65)
ax=fig.add_subplot(122,projection='3d'); draw(ax,base,'#368bad')
for c in caps: draw(ax,c.translate((0,0,25)),'#66c4d7')
setup(ax,'Exploded: 2 separate caps, 4 clamping screws',(-40,40,-35,85,0,85))
fig.tight_layout();fig.savefig(ROOT/'detail_preview.png',dpi=160);plt.close(fig)
# 2D hole locations for physical measurement; dimensions are design values.
fig,axs=plt.subplots(1,3,figsize=(14,5))
for ax in axs: ax.set_aspect('equal');ax.grid(alpha=.2);ax.set_xlabel('mm');ax.set_ylabel('mm')
a=axs[0]; a.add_patch(plt.Rectangle((-34,-23),68,46,fill=False))
for x in (-13,13):
 for y in (-13,13): a.add_patch(plt.Circle((x,y),2.1,fill=False));a.text(x,y+3,f'{x},{y}',fontsize=8,ha='center')
a.set(xlim=(-40,40),ylim=(-30,30),title='Bottom: 26 x 26 pitch / 4 x dia4.2 depth5.5')
a=axs[1]; a.add_patch(plt.Rectangle((-28.5,-9),57,18,fill=False))
for x in (-22.5,22.5):a.add_patch(plt.Circle((x,0),1.6,fill=False))
a.set(xlim=(-35,35),ylim=(-20,20),title='D435: 45 pitch / 2 x dia3.2 through')
a=axs[2];a.add_patch(plt.Rectangle((-19,-19),38,38,fill=False))
for x,d in zip((-10,0,10),(4.,4.2,4.4)):a.add_patch(plt.Circle((x,0),d/2,fill=False));a.text(x,5,str(d),ha='center')
for x in (-13,13):
 for y in (-13,13):a.add_patch(plt.Circle((x,y),1.6,fill=False))
a.set(xlim=(-24,24),ylim=(-24,24),title='Coupon: pilot diameters + 26 x 26 template')
fig.tight_layout();fig.savefig(ROOT/'dimensions.png',dpi=160);plt.close(fig)
print(json.dumps(audit,indent=2))




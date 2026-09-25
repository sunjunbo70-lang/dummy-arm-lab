"""r1 proposal contract. No IK, no implicit projection or quality-based action mask."""
from enum import IntEnum
import numpy as np
from ...wall_cycle_v09.trajectory_action import Contact

class Op(IntEnum):
    LOAD=0
    CONTACT_NEW=1
    CONTINUE=2
    LIFT=3
    INSPECT_WALL=4
    INSPECT_TOOL=5
    FINISH_REQUEST=6

PARAM_MASK=np.zeros((7,19),bool)
PARAM_MASK[Op.LOAD,0]=True
PARAM_MASK[Op.CONTACT_NEW]=True
PARAM_MASK[Op.CONTINUE]=True
PARAM_MASK[Op.CONTINUE,:2]=False
# First attitude/force/speed is inherited, not an unmodelled projection.
PARAM_MASK[Op.CONTINUE,[8,10,13,16]]=False

def valid_ops(in_contact, *, continue_supported, segments=0, length_m=0., room_ml=30., remaining_ml=1000.):
    m=np.zeros(7,bool)
    if in_contact:
        m[Op.CONTINUE]=continue_supported and segments<4 and length_m<1.04-1e-9
        m[Op.LIFT]=True;m[Op.FINISH_REQUEST]=True
    else:
        m[Op.LOAD]=room_ml>0 and remaining_ml>0
        m[[Op.CONTACT_NEW,Op.INSPECT_WALL,Op.INSPECT_TOOL,Op.FINISH_REQUEST]]=True
    return m

def decode(raw_bounded, *, group='R2', current=None):
    x=np.asarray(raw_bounded,dtype=float)
    if x.shape!=(19,) or not np.isfinite(x).all() or np.max(np.abs(x))>1+1e-7:
        raise ValueError('Action must be finite, bounded, 19-dimensional')
    if group not in ('R0','R1','R2'): raise ValueError(group)
    start=x[:2]*.13+np.array([0.,.25]) if current is None else np.asarray(current[0],float)
    theta=x[2]*np.pi;forward=np.array([np.cos(theta),np.sin(theta)]);side=np.array([-forward[1],forward[0]])
    length=.10 if group=='R0' else .03+(x[3]+1)*.115
    end=start+length*forward
    p1=start+length*forward/3+.03*(x[4]*forward+x[5]*side)
    p2=start+2*length*forward/3+.03*(x[6]*forward+x[7]*side)
    angles=x[8:10]*np.pi;pitches=(x[10:13]+1)*.5*np.deg2rad(35)
    forces=.5+(x[13:16]+1)*7.25;speeds=.02+(x[16:19]+1)*.05
    if current is not None:
        angles[0],pitches[0],forces[0],speeds[0]=current[1:]
    return Contact(np.array([start,p1,p2,end]),angles,pitches,forces,speeds)

def audit_geometry(contact, samples=257):
    """Reject, never clip. Physical swept footprint and IK need separate checks."""
    # Cubic Bezier lies in convex hull; conservative control-hull check is explicit.
    inside=np.all(np.abs(contact.points-np.array([0.,.25]))<=.13+1e-10)
    points=np.array([contact.at(t)[0] for t in np.linspace(0,1,samples)])
    length=float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum())
    return dict(control_hull_inside=bool(inside),actual_arc_length_m=length,
                accepted=bool(inside and .003<=length<=.26+1e-9),physical_verified=False)

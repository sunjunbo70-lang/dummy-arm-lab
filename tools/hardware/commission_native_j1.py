"""Bounded supervised comparison: native move_j result and native angle polling.

High-level path does not write current, speed, calibration or limits.
The old direct-motor path is disabled because its conversion is unverified.
"""
# 一次性监督调试脚本。设备身份改为从 configs/device_identity.json 读取；
# 端口名仍需按目标电脑的实际枚举结果调整。
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))
from dummy_loop.live_control import device_identity as _identity
_VENDOR, _PRODUCT, _SERIAL = _identity()
from pathlib import Path
import argparse
import functools
import json
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'vendor/native_client'))
from dummy_loop.serial_backend import SerialRobot, list_ports


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--direct-motor',action='store_true',help='Disabled: archived conversion does not match observed installed-firmware behavior')
    parser.add_argument('--target-deg',type=int,choices=(3,10),default=10)
    parser.add_argument('--return-to-zero',action='store_true',help='Return from the verified J1=10 degree pose to zero')
    parser.add_argument('--ascii-target',action='store_true',help='Send target via the captured Studio ASCII format; native USB only reads angles')
    args=parser.parse_args()
    if args.direct_motor:
        parser.error('Direct motor control is disabled: installed firmware position units/direction do not match the archived conversion. No device opened.')
    import usb.core
    import usb.backend.libusb1
    import libusb_package
    backend=usb.backend.libusb1.get_backend(find_library=libusb_package.find_library)
    usb.core.find=functools.partial(usb.core.find,backend=backend,idVendor=_VENDOR,idProduct=_PRODUCT)
    import fibre
    end=fibre.Event()
    report={'samples':[],'motion_sent':False,'master_speed_changed':False,'polling':'native_angle_properties','direct_motor':args.direct_motor}
    serial=SerialRobot('COM6',ROOT/'outputs/native_commission_serial.jsonl')
    armed=False
    try:
        matches=[p for p in list_ports() if p['port']=='COM6' and p['vid']==_VENDOR and p['pid']==_PRODUCT and p['serial']==_SERIAL]
        if len(matches)!=1: raise RuntimeError('Unexpected serial device')
        dev=fibre.find_any(path='usb',timeout=12,channel_termination_token=end,logger=fibre.Logger(verbose=False))
        if dev is None or f'{dev.serial_number:012X}'!=_SERIAL: raise RuntimeError('Unexpected native device')
        robot_schema=next(x for x in dev._json_data if x['name']=='robot')
        move_schema=next(x for x in robot_schema['members'] if x['name']=='move_j')
        if [x['type'] for x in move_schema['inputs']]!=['float']*6 or move_schema['outputs'][0]['type']!='bool':
            raise RuntimeError('Unexpected move_j signature')
        if args.direct_motor:
            joint_schema=next(x for x in robot_schema['members'] if x['name']=='joint_1')
            method=next(x for x in joint_schema['members'] if x['name']=='set_position_with_time')
            if [(x['name'],x['type']) for x in method['inputs']]!=[('pos','float'),('time','float')]:
                raise RuntimeError('Unexpected per-motor method schema')
        serial.connect()
        canonical=np.rad2deg(serial.get_state().q)
        raw=lambda:np.array([getattr(dev.robot,f'joint_{i}').angle for i in range(1,7)])
        first_raw=raw()
        offsets=np.rint(canonical-first_raw)
        if np.max(np.abs(canonical-first_raw-offsets))>.03: raise RuntimeError('Native and ASCII angle mapping mismatch')
        sample=lambda:raw()+offsets
        before=sample()
        start_min,start_max=(9.8,10.2) if args.return_to_zero else (-.3,3.2)
        if not np.all(np.isfinite(before)) or not start_min<=before[0]<=start_max or np.max(np.abs(before[1:]-[0,90,0,0,0]))>.15:
            raise RuntimeError('Not the verified near-home pose; no action')
        for _ in range(3):
            time.sleep(.15)
            if np.max(np.abs(sample()-before))>.15: raise RuntimeError('Starting pose not stable')
        target=before.copy(); target[0]=0 if args.return_to_zero else args.target_deg
        if abs(target[0]-before[0])<.3: raise RuntimeError('Diagnostic needs a nonzero J1 displacement')
        report.update(before_deg=before.tolist(),target_deg=target.tolist(),native_offsets=offsets.tolist(),target_transport='studio_ascii' if args.ascii_target else 'native')
        print(json.dumps(report),flush=True)
        if not args.execute: return
        print(f'J1 test begins in 5 seconds; absolute target {target[0]} degrees.',flush=True)
        time.sleep(5)
        if np.max(np.abs(sample()-before))>.15: raise RuntimeError('Pose changed during countdown')
        armed=True
        if args.direct_motor:
            # Archived C++ binds this misleadingly named API to
            # SetPositionWithVelocityLimit(motor revolutions, motor rev/s).
            # J1 inverse=true, reduction=30; no joint limit/current changes.
            report.update(motor_position_rev=-target[0]/12,motor_velocity_limit_rev_s=.2,
                          conversion_basis='archived J1 inverse and 30:1 reduction; feedback envelope retained')
            dev.robot.joint_1.set_position_with_time(-target[0]/12,.2)
            report['motion_sent']=True
        elif args.ascii_target:
            # Handoff requires the previous bounded run's STOP at the stable pose.
            serial._request('!START',lambda s:True if s=='Started ok' else None)
            serial.line_ending='\n'
            serial.commissioning_target(np.deg2rad(target),studio_format=True)
            report.update(motion_sent=True,ascii_queue_ack_received=True)
        else:
            accepted=dev.robot.move_j(*target.tolist())
            report.update(native_target_accepted=accepted,motion_sent=True)
            print(json.dumps({'native_target_accepted':accepted}),flush=True)
            if accepted is not True: raise RuntimeError('Firmware rejected native target')
            # Previous session has an acknowledged STOP. Target is preloaded before enable.
            serial._request('!START',lambda s:True if s=='Started ok' else None)
            accepted=dev.robot.move_j(*target.tolist())
            report['native_target_accepted_after_start']=accepted
            if accepted is not True: raise RuntimeError('Firmware rejected target after enable')
        deadline=time.monotonic()+8
        last_progress=time.monotonic(); last_q=before[0]; settled=0
        while time.monotonic()<deadline:
            q=sample()
            report['samples'].append({'time_s':time.monotonic(),'q_deg':q.tolist()})
            print(json.dumps({'q_deg':np.round(q,3).tolist()}),flush=True)
            if not np.all(np.isfinite(q)) or np.max(np.abs(q[1:]-before[1:]))>.5 or not min(before[0],target[0])-.3<=q[0]<=max(before[0],target[0])+.3:
                raise RuntimeError('Unexpected native feedback; stop')
            if abs(q[0]-last_q)>.1: last_progress=time.monotonic(); last_q=q[0]
            error=np.max(np.abs(q-target))
            settled=settled+1 if error<.15 else 0
            if settled>=4:
                report['result']='native_feedback_reached_target'; break
            if time.monotonic()-last_progress>2: raise RuntimeError('Native feedback stalled for 2 seconds')
            time.sleep(.1)
        else: raise TimeoutError('Native target not reached in 8 seconds')
    except BaseException as exc:
        report['error']=str(exc)
        raise
    finally:
        if armed:
            try:
                if args.direct_motor:
                    hold=float(dev.robot.joint_1.angle)
                    if not np.isfinite(hold) or not -.5<=hold<=10.5: raise RuntimeError('Cannot establish bounded motor hold')
                    dev.robot.joint_1.set_position_with_time(-hold/12,.2)
                    report['motor_hold_rpc_returned']=True
                    report['hold_angle_deg']=hold
                else:
                    serial.stop_commissioning(); report['stop_reply_received']=True
            except Exception as exc: report['stop_error']=str(exc); print('STOP NOT CONFIRMED: use physical stop.',flush=True)
        serial.close(); end.set()
        path=ROOT/f'outputs/native_j1_{time.time_ns()}.json'
        path.write_text(json.dumps(report,indent=2),encoding='utf-8')
        print('REPORT '+str(path),flush=True)


if __name__=='__main__': main()

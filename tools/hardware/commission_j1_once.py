"""Operator-supervised native-firmware J1 bounded test, not sim mapping.

Fixed device and narrow pose envelope for this commissioning session only.
Does not mark the general hardware profile calibrated or enable policy control.
"""
# 一次性监督调试脚本。设备身份改为从 configs/device_identity.json 读取；
# 端口名仍需按目标电脑的实际枚举结果调整。
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))
from dummy_loop.live_control import device_identity as _identity
_VENDOR, _PRODUCT, _SERIAL = _identity()
from pathlib import Path
import argparse
import json
import sys
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dummy_loop.serial_backend import SerialRobot, list_ports


def check_pose(q, baseline=None):
    q = np.asarray(q)
    low = np.array([-170,-73,35,-180,-120,-720])
    high = np.array([170,90,180,180,120,720])
    if q.shape != (6,) or not np.isfinite(q).all() or np.any(q<low) or np.any(q>high):
        raise RuntimeError('Feedback outside archived firmware limits')
    if baseline is not None and np.max(np.abs(q-baseline)) > .25:
        raise RuntimeError('Pose changed during startup; abort this test')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--delta-deg',type=int,choices=(1,10),default=1)
    parser.add_argument('--handoff-home',action='store_true',help='Use verified Studio home pose and do not send START')
    parser.add_argument('--resume-stopped',action='store_true',help='Home handoff only: preload target before START after acknowledged stop')
    parser.add_argument('--studio-format',action='store_true',help='Six angles plus trailing comma and LF, preserve firmware speed')
    parser.add_argument('--current-folded-handoff',action='store_true',help='Session-specific folded pose, preserve other axes; no START')
    parser.add_argument('--speed-parameter',type=int,choices=(1,3),default=1)
    parser.add_argument('--resend-after-start',action='store_true',help='Bounded sequence diagnostic: send the same target once after START')
    parser.add_argument('--refresh-target',action='store_true',help='Bounded Studio-format diagnostic: refresh identical target at most 10 Hz until near arrival')
    args=parser.parse_args()
    if args.refresh_target and not (args.handoff_home and args.studio_format and args.resume_stopped):
        parser.error('--refresh-target requires stopped Studio-format home handoff')
    if args.resend_after_start and not (args.current_folded_handoff and args.resume_stopped):
        parser.error('--resend-after-start requires folded stopped handoff')
    if args.current_folded_handoff and (args.handoff_home or args.studio_format):
        parser.error('Folded handoff requires explicit speed and excludes home/resume options')
    if args.resume_stopped and not (args.handoff_home or args.current_folded_handoff):
        parser.error('--resume-stopped requires a verified session handoff pose')
    device=[p for p in list_ports() if p['port']=='COM6' and p['vid']==_VENDOR and p['pid']==_PRODUCT and p['serial']==_SERIAL]
    if len(device)!=1: raise RuntimeError('Expected robot identity is not on COM6')
    root=Path(__file__).resolve().parents[2]
    report={'scope':f'operator_supervised_native_J1_plus_{args.delta_deg}_deg_only','motion_sent':False,'samples':[]}
    robot=SerialRobot('COM6',root/'outputs/j1_once_serial.jsonl',timeout=1)
    if args.studio_format: robot.line_ending='\n'
    armed=False
    try:
        robot.connect()
        before=np.rad2deg(robot.get_state().q)
        check_pose(before)
        # Require the same folded starting region observed this session.
        reference=np.array([0,0,90,0,0,0]) if args.handoff_home else np.array([-.4,-72.1,178.21,.94,-.02,.24])
        tolerance=2
        if args.resend_after_start:
            # Last recorded stopped pose; this is not a general envelope expansion.
            reference=np.array([1.80,-72.71,178.13,1.56,-1.44,-.01])
            tolerance=.25
        if np.max(np.abs(before-reference))>tolerance:
            raise RuntimeError('Starting pose differs from this session; inspect before moving')
        for _ in range(2):
            time.sleep(.15)
            check_pose(np.rad2deg(robot.get_state().q),before)
        target=before.copy(); target[0]+=args.delta_deg
        if args.resume_stopped and args.handoff_home:
            target[0]=10.0
        check_pose(target)
        report.update(before_deg=before.tolist(),target_deg=target.tolist(),firmware_speed_parameter='unchanged' if args.studio_format else args.speed_parameter,studio_format=args.studio_format)
        print(json.dumps(report),flush=True)
        if not args.execute: return
        if args.resend_after_start or (args.handoff_home and args.studio_format):
            print(f'J1 TEST STARTS IN 5 SECONDS; requested delta {args.delta_deg} degrees.',flush=True)
            time.sleep(5)
            check_pose(np.rad2deg(robot.get_state().q),before)
        armed=True
        if args.handoff_home or args.current_folded_handoff:
            if not (args.handoff_home and args.studio_format):
                robot._request('#CMDMODE 2', lambda s: True if s in ('ok Set command mode to [2]','Set command mode to [2]') else None)
            else:
                report['command_mode']='unchanged_studio_handoff'
            robot.speed=args.speed_parameter
            report['start_sent']=False
        else:
            robot.enable_commissioning(speed=args.speed_parameter,mode=2)
            report['start_sent']=True
        time.sleep(.15)
        check_pose(np.rad2deg(robot.get_state().q),before)
        report['motion_sent']=True
        robot.commissioning_target(np.deg2rad(target),studio_format=args.studio_format)
        if args.resume_stopped:
            # The last acknowledged STOP and subsequent stable readback are
            # recorded in this session. Preload a nonzero, in-range target
            # before enable, rather than restoring an unknown old target.
            time.sleep(.15)
            check_pose(np.rad2deg(robot.get_state().q),before)
            report['start_sent']=True
            robot._request('!START',lambda s: True if s=='Started ok' else None)
            if args.resend_after_start:
                robot.commissioning_target(np.deg2rad(target),studio_format=args.studio_format)
                report['same_target_resent_after_start']=True
        deadline=time.monotonic()+30
        motion_deadline=time.monotonic()+3
        settled=0
        progress_time=time.monotonic()
        progress_q=before[0]
        report['target_refresh_count']=0
        next_refresh=time.monotonic()
        while time.monotonic()<deadline:
            measured=np.rad2deg(robot.get_state().q)
            report['samples'].append({'time_s':time.monotonic(),'q_deg':measured.tolist()})
            check_pose(measured)
            if abs(measured[0]-progress_q)>.1:
                progress_time=time.monotonic(); progress_q=measured[0]
            if time.monotonic()>motion_deadline and max(abs(row['q_deg'][0]-before[0]) for row in report['samples'])<.1:
                raise RuntimeError('No J1 feedback movement within 3 seconds; no automatic enable/retry')
            if np.max(np.abs(measured[1:]-before[1:]))>.5 or not before[0]-.3 <= measured[0] <= target[0]+.3:
                raise RuntimeError('Unexpected motion feedback; stopping')
            err=float(np.max(np.abs(measured-target)))
            if err>.15 and time.monotonic()-progress_time>3:
                raise RuntimeError('Feedback progress stalled for 3 seconds; stopping without retry')
            print(json.dumps({'q_deg':measured.tolist(),'max_error_deg':err}),flush=True)
            settled=settled+1 if err<.15 else 0
            if settled>=5:
                report.update(result='feedback_reached_target',after_deg=measured.tolist(),freshness='per-axis device timestamps unavailable',end_state='last target held; motor disable not sent')
                print(json.dumps({'result':report['result'],'delta_deg':(measured-before).tolist()}),flush=True)
                break
            if args.refresh_target and err>.2 and time.monotonic()>=next_refresh:
                robot.commissioning_target(np.deg2rad(target),studio_format=True)
                report['target_refresh_count']+=1
                next_refresh=time.monotonic()+.1
            time.sleep(.1)
        else: raise TimeoutError('Target not reached in 30 seconds; no retry')
    except BaseException as exc:
        report['error']=str(exc)
        if armed:
            try:
                robot.stop_commissioning()
                report['stop_reply_received']=True
            except Exception as stop_exc:
                report['stop_error']=str(stop_exc)
                print('STOP NOT CONFIRMED: operator must use physical stop.',flush=True)
        raise
    finally:
        robot.close()
        (root/f'outputs/j1_{args.delta_deg}deg_{time.time_ns()}_result.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__': main()

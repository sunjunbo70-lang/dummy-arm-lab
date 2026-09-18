"""One supervised fold using the two stages captured from this robot's Studio.

No homing calibration, driver settings, or speed changes. Default is read-only.
"""
# 一次性监督调试脚本。设备身份改为从 configs/device_identity.json 读取；
# 端口名仍需按目标电脑的实际枚举结果调整。
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))
from dummy_loop.live_control import device_identity as _identity
_VENDOR, _PRODUCT, _SERIAL = _identity()
import argparse
import functools
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'vendor/native_client')]
from dummy_loop.serial_backend import SerialRobot,list_ports


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    import usb.core
    import usb.backend.libusb1
    import libusb_package
    backend=usb.backend.libusb1.get_backend(find_library=libusb_package.find_library)
    usb.core.find=functools.partial(usb.core.find,backend=backend,idVendor=_VENDOR,idProduct=_PRODUCT)
    import fibre
    end=fibre.Event()
    serial=SerialRobot('COM6',ROOT/'outputs/fold_serial.jsonl')
    report={'motion_sent':False,'stages':[],'settings_changed':False}
    armed=False
    try:
        if not any(p['port']=='COM6' and p['serial']==_SERIAL and p['vid']==_VENDOR and p['pid']==_PRODUCT for p in list_ports()):
            raise RuntimeError('Robot serial identity mismatch')
        dev=fibre.find_any(path='usb',timeout=12,channel_termination_token=end,logger=fibre.Logger(verbose=False))
        if dev is None or f'{dev.serial_number:012X}'!=_SERIAL:
            raise RuntimeError('Native robot identity mismatch')
        serial.connect()
        sample=lambda:np.array([getattr(dev.robot,f'joint_{i}').angle for i in range(1,7)])+np.array([0,-75,180,0,0,0])
        before=sample()
        if not np.all(np.isfinite(before)) or np.max(np.abs(before-[0,0,90,0,0,0]))>.15:
            raise RuntimeError('Fold requires the verified home pose; no motion')
        if np.max(np.abs(np.rad2deg(serial.get_state().q)-before))>.03:
            raise RuntimeError('ASCII/native feedback mismatch')
        targets=[[0,-75,90,0,0,0],[0,-75,180,0,0,0]]
        report.update(before_deg=before.tolist(),planned_targets_deg=targets)
        print(json.dumps(report),flush=True)
        if not args.execute: return
        print('Fold begins in 5 seconds: J2 to -75, then J3 to 180. Other axes remain zero.',flush=True)
        time.sleep(5)
        if np.max(np.abs(sample()-before))>.15: raise RuntimeError('Pose changed during countdown')
        # Last session ended with acknowledged STOP at this same verified pose.
        armed=True
        serial._request('!START',lambda s:True if s=='Started ok' else None)
        serial.line_ending='\n'
        for index,values in enumerate(targets):
            target=np.array(values,dtype=float)
            start=sample()
            stage={'target_deg':values,'samples':[]}
            report['stages'].append(stage)
            serial.commissioning_target(np.deg2rad(target),studio_format=True)
            report['motion_sent']=True
            stage['queue_ack_received']=True
            deadline=time.monotonic()+15
            progress_at=time.monotonic(); best=float(np.max(np.abs(start-target))); settled=0
            while time.monotonic()<deadline:
                q=sample(); now=time.monotonic()
                stage['samples'].append({'time_s':now,'q_deg':q.tolist()})
                print(json.dumps({'stage':index+1,'q_deg':np.round(q,3).tolist()}),flush=True)
                if not np.all(np.isfinite(q)) or np.any(q<np.minimum(start,target)-.5) or np.any(q>np.maximum(start,target)+.5):
                    raise RuntimeError('Feedback outside bounded stage path')
                error=float(np.max(np.abs(q-target)))
                if error<best-.1: best=error; progress_at=now
                settled=settled+1 if error<.15 else 0
                if settled>=4:
                    stage['feedback_reached_target']=True
                    break
                if now-progress_at>2: raise RuntimeError('No progress for 2 seconds')
                time.sleep(.1)
            else: raise TimeoutError('Fold stage did not reach target in 15 seconds')
        report['result']='fold_feedback_reached_target'
    except BaseException as exc:
        report['error']=str(exc)
        raise
    finally:
        if armed:
            try:
                serial.stop_commissioning(); report['stop_reply_received']=True
            except Exception as exc:
                report['stop_error']=str(exc)
                print('STOP NOT CONFIRMED: use physical stop.',flush=True)
        serial.close(); end.set()
        path=ROOT/f'outputs/fold_once_{time.time_ns()}.json'
        path.write_text(json.dumps(report,indent=2),encoding='utf-8')
        print('REPORT '+str(path),flush=True)


if __name__=='__main__': main()

"""Discover archived Fibre interface; optional speed restoration only, no motion calls."""
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'vendor/native_client'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--restore-speed', action='store_true')
    parser.add_argument('--refresh-angles',action='store_true',help='Request CAN position updates, then read cached properties; no motion')
    parser.add_argument('--watch-seconds',type=int,default=0,choices=range(0,181),help='Passively record cached native angles while Studio owns COM6')
    args = parser.parse_args()
    import usb.core
    import usb.backend.libusb1
    import libusb_package
    backend = usb.backend.libusb1.get_backend(find_library=libusb_package.find_library)
    if backend is None:
        raise RuntimeError('Bundled libusb backend unavailable')
    usb.core.find = functools.partial(usb.core.find, backend=backend, idVendor=_VENDOR, idProduct=_PRODUCT)
    import fibre
    shutdown = fibre.Event()
    try:
        device = fibre.find_any(path='usb', serial_number=None, timeout=8,
                                channel_termination_token=shutdown,
                                logger=fibre.Logger(verbose=True))
        if device is None:
            raise RuntimeError('Expected native robot not discovered within 8 seconds')
        if f'{device.serial_number:012X}' != _SERIAL:
            raise RuntimeError('Native serial number mismatch; no write performed')
        interface = device.__dict__['_json_data']
        (ROOT / 'outputs/native_robot_interface.json').write_text(
            json.dumps(interface, indent=2), encoding='utf-8')
        robot = device.robot
        result = {'serial': str(device.serial_number), 'operation': 'read_only_discovery',
                  'joint_angle_properties_deg': [getattr(robot, f'joint_{i}').angle for i in range(1, 7)],
                  'motion_command_sent': False}
        if args.watch_seconds:
            path=ROOT/f'outputs/native_passive_watch_{time.time_ns()}.jsonl'
            print('PASSIVE_WATCH '+str(path),flush=True)
            with path.open('w',encoding='utf-8') as stream:
                deadline=time.monotonic()+args.watch_seconds
                while time.monotonic()<deadline:
                    entry={'time_ns':time.time_ns(),'native_angles_deg':[getattr(robot,f'joint_{i}').angle for i in range(1,7)]}
                    stream.write(json.dumps(entry)+'\n'); stream.flush()
                    time.sleep(.2)
            result['passive_watch_file']=str(path)
        if args.refresh_angles:
            result['angle_refresh_samples']=[]
            for _ in range(3):
                robot.joint_all.update_angle()
                time.sleep(.08)
                result['angle_refresh_samples'].append([getattr(robot,f'joint_{i}').angle for i in range(1,7)])
            result['individual_refresh_samples']=[]
            for _ in range(2):
                for i in range(1,7):
                    getattr(robot,f'joint_{i}').update_angle()
                time.sleep(.08)
                result['individual_refresh_samples'].append([getattr(robot,f'joint_{i}').angle for i in range(1,7)])
        if args.restore_speed:
            # This exact method is verified in the returned interface before calling.
            obj = next(x for x in interface if x['name'] == 'robot')
            method = next(x for x in obj['members'] if x['name'] == 'set_joint_speed')
            if method['type'] != 'function' or len(method.get('inputs', [])) != 1 or method['inputs'][0]['name'] != 'speed' or method['inputs'][0]['type'] != 'float':
                raise RuntimeError('Unexpected speed method schema; no write performed')
            robot.set_joint_speed(30.0)
            result.update(operation='restore_archived_default_speed', speed_parameter=30,
                          rpc_returned=True, direct_speed_readback_available=False)
        print('RESULT ' + json.dumps(result), flush=True)
        (ROOT / f'outputs/native_diagnostic_{time.time_ns()}.json').write_text(
            json.dumps(result, indent=2), encoding='utf-8')
    finally:
        shutdown.set()


if __name__ == '__main__':
    main()

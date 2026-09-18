"""Read-only GETJPOS using the exact native serial library shipped with Studio."""
# 一次性监督调试脚本。设备身份改为从 configs/device_identity.json 读取；
# 端口名仍需按目标电脑的实际枚举结果调整。
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))
from dummy_loop.live_control import device_identity as _identity
_VENDOR, _PRODUCT, _SERIAL = _identity()
import ctypes as C
import json
from pathlib import Path
import time


class Config(C.Structure):
    _fields_ = [(name, C.c_int32) for name in
                ('baud', 'parity', 'stop', 'data', 'discard_null', 'ignore_break', 'skip')]


def main():
    root = Path(__file__).resolve().parents[2]
    from serial.tools import list_ports
    matching = [p for p in list_ports.comports() if p.vid == _VENDOR]
    if len(matching) != 1 or matching[0].pid != _PRODUCT or matching[0].serial_number != _SERIAL:
        raise RuntimeError('Unique Studio vendor match must be the expected robot')
    lib = C.CDLL(str(root / '.diagnostics/DummyStudioCapture/DummyStudio_Data/Plugins/x86_64/spap.dll'))
    signatures = {
        'spapOpenUSB': ([C.c_char_p, C.c_char_p, C.c_char_p, Config], C.c_int),
        'spapClose': ([C.c_int], None),
        'spapWrite': ([C.c_int, C.c_void_p, C.c_int], None),
        'spapReadDataAvailable': ([C.c_int], C.c_int),
        'spapReadData': ([C.c_int, C.c_void_p, C.c_int], C.c_int),
        'spapSetDTR': ([C.c_int, C.c_int], None),
        'spapGetDTR': ([C.c_int], C.c_int),
        'spapGetRTS': ([C.c_int], C.c_int),
    }
    for name, (argtypes, restype) in signatures.items():
        getattr(lib, name).argtypes = argtypes
        getattr(lib, name).restype = restype
    handle = lib.spapOpenUSB(b'1209', b'', b'', Config(115200,0,0,8,0,1,0))
    if handle == -1:
        raise RuntimeError('Studio native serial open failed; no command sent')
    records = []
    try:
        lib.spapSetDTR(handle, 1)  # Scene StartEnableDTR=true, AutoRTSCTS=false.
        print(json.dumps({'dtr':lib.spapGetDTR(handle),'rts':lib.spapGetRTS(handle)}),flush=True)
        for _ in range(3):
            payload = C.create_string_buffer(b'#GETJPOS\r\n')
            lib.spapWrite(handle, payload, 10)
            data = bytearray()
            deadline = time.monotonic()+1
            while time.monotonic() < deadline:
                available = lib.spapReadDataAvailable(handle)
                if available > 0:
                    if available > 4096:
                        raise RuntimeError('Unexpected input size')
                    buffer = C.create_string_buffer(available)
                    count = lib.spapReadData(handle, buffer, available)
                    if count > 0:
                        data.extend(buffer.raw[:count])
                        if b'\n' in data:
                            break
                time.sleep(.01)
            record={'tx':'#GETJPOS\r\n','rx':data.decode('ascii',errors='backslashreplace')}
            records.append(record)
            print(json.dumps(record),flush=True)
            time.sleep(.2)
    finally:
        lib.spapClose(handle)
        (root/'outputs/studio_native_readonly.json').write_text(json.dumps(records,indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()

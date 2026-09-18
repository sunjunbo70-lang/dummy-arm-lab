"""Legacy ASCII transport. RX age is NOT the age of each motor's feedback."""
import json
import re
import time
from pathlib import Path
import numpy as np
from .core import Observation, six

NUMBER = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
JOINTS = re.compile(r'^ok\s+(' + NUMBER + r'(?:\s+' + NUMBER + r'){5})$')


def parse_joint_line(line):
    match = JOINTS.fullmatch(line.strip())
    if not match:
        return None
    return six([float(x) for x in match.group(1).split()])


class SerialRobot:
    def __init__(self, port, log_path, timeout=1.0, serial_factory=None):
        if not port:
            raise ValueError('Explicit serial port required')
        self.port, self.timeout = port, timeout
        self.log_path = Path(log_path)
        self.factory = serial_factory
        self.ser = None
        self.log = None
        self.line_ending = '\r\n'

    def connect(self):
        if self.factory is None:
            import serial
            self.factory = serial.Serial
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = self.log_path.open('a', encoding='utf-8')
        try:
            # Opening USB may still toggle board-specific modem signals: verify on site.
            self.ser = self.factory(self.port, 115200, timeout=0.05, write_timeout=0.5)
            self.ser.reset_input_buffer()
        except BaseException:
            self.close()
            raise

    def _record(self, direction, data):
        self.log.write(json.dumps({'monotonic_s':time.monotonic(), 'direction':direction,
                                   'raw':data.decode('ascii', errors='backslashreplace')}) + '\n')
        self.log.flush()

    def _write(self, text):
        if not self.ser:
            raise RuntimeError('Not connected')
        data = (text + self.line_ending).encode('ascii')
        self._record('tx', data)
        count = self.ser.write(data)
        if count != len(data):
            raise IOError('Incomplete serial write')

    def _wait(self, accept):
        deadline = time.monotonic() + self.timeout
        pending = bytearray()
        while time.monotonic() < deadline:
            chunk = self.ser.read(1)
            if not chunk:
                continue
            pending.extend(chunk)
            if len(pending) > 4096:
                raise ValueError('Oversized serial frame')
            if chunk == b'\n':
                raw = bytes(pending)
                self._record('rx', raw)
                pending.clear()
                value = accept(raw.decode('ascii', errors='replace').strip())
                if value is not None:
                    return value
        raise TimeoutError('No expected response; stop this session and diagnose before retrying')

    def _request(self, command, accept):
        # One outstanding request; no concurrent callers. Caller aborts on timeout.
        self._write(command)
        return self._wait(accept)

    def get_state(self):
        degrees = self._request('#GETJPOS', parse_joint_line)
        return Observation(np.deg2rad(degrees), time.monotonic(), 'legacy_firmware_cache', False)

    def enable_commissioning(self, speed, mode):
        self._request('!START', lambda s: True if s == 'Started ok' else None)
        replies = {f'ok Set command mode to [{mode}]', f'Set command mode to [{mode}]'}
        self._request(f'#CMDMODE {mode}', lambda s: True if s in replies else None)
        self.speed = speed

    def send_action(self, q):
        raise RuntimeError('Autonomous legacy USB execution blocked: per-axis freshness/watchdog not verified. Use simulation or read-only shadow mode.')

    def commissioning_target(self, firmware_radians, studio_format=False):
        q = np.rad2deg(six(firmware_radians))
        text = ('&' + ','.join(f'{v:.2f}' for v in q) + ',') if studio_format else ('&' + ','.join(f'{v:.4f}' for v in q) + f',{self.speed:.4f}')
        self._write(text)
        # In this source protocol, first reply is FIFO remaining space, not completion.
        def queue_reply(s):
            # Queue and worker responses can interleave on this firmware:
            # observed raw frame was '15ok\r\n\r\n'. Neither is arrival proof.
            match = re.fullmatch(r'(?:ok)?([0-9]+)(?:ok)?', s)
            if match is None:
                return None
            n = int(match.group(1))
            if n == 255:
                raise RuntimeError('Firmware command queue rejected target')
            if not 0 <= n < 255:
                raise ValueError('Unexpected queue response')
            return n
        self._wait(queue_reply)

    def stop_commissioning(self):
        self._request('!STOP', lambda s: True if s in ('Stopped ok','okStopped ok') else None)

    def close(self):
        try:
            if self.ser:
                self.ser.close()
        finally:
            self.ser = None
            if self.log:
                self.log.close()
                self.log = None


def list_ports():
    from serial.tools import list_ports as lp
    return [{'port':p.device, 'vid':p.vid, 'pid':p.pid, 'serial':p.serial_number,
             'description':p.description} for p in lp.comports()]

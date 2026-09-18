"""Read-only legacy query diagnostic, preserving partial and binary replies."""
import argparse
import json
from pathlib import Path
import time
import serial

parser = argparse.ArgumentParser()
parser.add_argument('--port', required=True)
args = parser.parse_args()
records = []
with serial.Serial(args.port, 115200, timeout=.1, write_timeout=.5) as port:
    time.sleep(1)
    initial = port.read(port.in_waiting)
    records.append(dict(phase='initial', rx_hex=initial.hex()))
    command = b'#GETJPOS\r\n'
    sent = port.write(command)
    port.flush()
    deadline = time.monotonic()+5
    received = bytearray()
    while time.monotonic() < deadline:
        received.extend(port.read(max(1, port.in_waiting)))
    records.append(dict(phase='query', port=args.port, baud=115200,
                        tx_hex=command.hex(), tx_bytes=sent,
                        rx_hex=received.hex(), rx_bytes=len(received),
                        rx_text=received.decode('ascii', errors='backslashreplace')))
out = Path(__file__).resolve().parents[2]/'outputs/com5_raw_probe.json'
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(records, indent=2), encoding='utf-8')
print(json.dumps(records, indent=2))

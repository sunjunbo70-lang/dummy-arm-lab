import json
from pathlib import Path
import tempfile
import time
import unittest
import numpy as np
from dummy_loop.core import Guard, Observation, load_profile
from dummy_loop.serial_backend import SerialRobot, parse_joint_line


class FakeSerial:
    def __init__(self, *args, **kwargs):
        self.writes=[]; self.buffer=bytearray(); self.closed=False
    def reset_input_buffer(self): self.buffer.clear()
    def write(self,data):
        self.writes.append(data)
        if data==b'#GETJPOS\r\n': self.buffer.extend(b'12\r\nok\r\n[sys] boot\r\nok 0 -2 90 4 5 6\r\n')
        return len(data)
    def read(self,n):
        if not self.buffer: return b''
        value=bytes(self.buffer[:n]); del self.buffer[:n]; return value
    def close(self): self.closed=True


class SafetyTests(unittest.TestCase):
    def setUp(self):
        self.guard=Guard([-1]*6,[1]*6,[.1]*6,[.5]*6)
        self.state=Observation(np.zeros(6),time.monotonic(),'test')
    def test_valid(self): self.guard.validate([.01]*6,self.state,.1)
    def test_nan(self):
        with self.assertRaises(ValueError): self.guard.validate([float('nan')]*6,self.state,.1)
    def test_wrong_shape(self):
        with self.assertRaises(ValueError): self.guard.validate([0]*7,self.state,.1)
    def test_limit(self):
        with self.assertRaises(ValueError): self.guard.validate([2]*6,self.state,10)
    def test_speed(self):
        with self.assertRaises(ValueError): self.guard.validate([.1]*6,self.state,.01)
    def test_stale(self):
        self.state.received_at-=2
        with self.assertRaises(ValueError): self.guard.validate([0]*6,self.state,.1)
    def test_unknown_profile(self):
        with self.assertRaises(ValueError): load_profile(Path('configs/hardware.unverified.json'))


class ProtocolTests(unittest.TestCase):
    def test_studio_slider_wire_format(self):
        class Reply(FakeSerial):
            def write(self,data):
                self.writes.append(data); self.buffer.extend(b'15ok\r\n\r\n'); return len(data)
        with tempfile.TemporaryDirectory() as d:
            r=SerialRobot('fake',Path(d)/'log',serial_factory=Reply)
            r.line_ending='\n'; r.connect()
            try:
                r.commissioning_target(np.deg2rad([10,0,90,0,0,-.01]),studio_format=True)
                self.assertEqual(r.ser.writes,[b'&10.00,0.00,90.00,0.00,0.00,-0.01,\n'])
            finally: r.close()
    def test_interleaved_queue_reply(self):
        class Interleaved(FakeSerial):
            def write(self,data):
                self.writes.append(data)
                self.buffer.extend(b'15ok\r\n\r\n')
                return len(data)
        with tempfile.TemporaryDirectory() as d:
            r=SerialRobot('fake',Path(d)/'log',serial_factory=Interleaved)
            r.connect(); r.speed=1
            try:
                r.commissioning_target(np.deg2rad([10,0,90,0,0,0]))
                self.assertEqual(len(r.ser.writes),1)
            finally: r.close()
    def test_usb_and_uart_mode_handshake(self):
        for prefix in ('ok ', ''):
            class Handshake(FakeSerial):
                def write(self, data):
                    self.writes.append(data)
                    if data == b'!START\r\n': self.buffer.extend(b'Started ok\r\n')
                    if data == b'#CMDMODE 2\r\n':
                        self.buffer.extend((prefix+'Set command mode to [2]\r\n').encode())
                    return len(data)
            with tempfile.TemporaryDirectory() as d:
                r=SerialRobot('fake',Path(d)/'log',serial_factory=Handshake)
                r.connect()
                try:
                    r.enable_commissioning(1,2)
                    self.assertEqual(r.ser.writes,[b'!START\r\n',b'#CMDMODE 2\r\n'])
                    self.assertEqual(r.speed,1)
                finally: r.close()
    def test_exact_shape(self):
        np.testing.assert_array_equal(parse_joint_line('ok 0 -1 90 2 3 4'),[0,-1,90,2,3,4])
        for line in ['ok','20','ok 1 2 3 4 5','ok 1 2 3 4 5 6 7','ok nan 1 2 3 4 5','Stopped ok']:
            self.assertIsNone(parse_joint_line(line))
    def test_readonly_and_split_frames(self):
        with tempfile.TemporaryDirectory() as d:
            robot=SerialRobot('fake',Path(d)/'log.jsonl',serial_factory=FakeSerial)
            robot.connect(); port=robot.ser
            self.assertEqual(port.writes,[])
            state=robot.get_state()
            self.assertEqual(port.writes,[b'#GETJPOS\r\n'])
            self.assertFalse(state.device_freshness_known)
            self.assertAlmostEqual(state.q[2],np.pi/2)
            robot.close(); self.assertTrue(port.closed)
    def test_autonomous_motion_blocked(self):
        with self.assertRaises(RuntimeError): SerialRobot('fake','unused').send_action([0]*6)
    def test_timeout(self):
        class Silent(FakeSerial):
            def write(self,data): self.writes.append(data); return len(data)
        with tempfile.TemporaryDirectory() as d:
            r=SerialRobot('fake',Path(d)/'log',timeout=.005,serial_factory=Silent)
            r.connect()
            try:
                with self.assertRaises(TimeoutError): r.get_state()
            finally: r.close()


class SimulationTests(unittest.TestCase):
    def test_action_is_integrated_not_teleported(self):
        from dummy_loop.sim_backend import SimRobot
        r=SimRobot(); r.connect(); target=np.ones(6)*.1
        r.send_action(target)
        first=r.get_state().q
        self.assertGreater(float(np.linalg.norm(first)),0)
        self.assertGreater(float(np.linalg.norm(first-target)),1e-4)
        for _ in range(100): r.send_action(target)
        self.assertLess(float(np.linalg.norm(r.get_state().q-target)),.005)
    def test_train_reload_closed_loop_unseen_goal(self):
        from dummy_loop.policy import collect,train,LinearPolicy
        from dummy_loop.sim_backend import SimRobot
        with tempfile.TemporaryDirectory() as d:
            data=Path(d)/'data.npz'; checkpoint=Path(d)/'policy.npz'
            collect(data,16,4); train(data,checkpoint); policy=LinearPolicy(checkpoint)
            r=SimRobot(); r.connect(); goal=np.array([-.17,.13,.11,-.08,.16,-.19])
            for _ in range(240):
                state=r.get_state()
                r.send_action(state.q+policy.predict(state.q,goal))
            self.assertLess(float(np.linalg.norm(r.get_state().q-goal)),.02)
    def test_insufficient_goal_diversity_rejected(self):
        from dummy_loop.policy import collect,train
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'data.npz'; collect(path,4)
            with self.assertRaises(ValueError): train(path,Path(d)/'model.npz')


if __name__=='__main__': unittest.main()

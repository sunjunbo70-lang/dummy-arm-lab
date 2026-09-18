import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from dummy_loop.live_control import LiveController,NativeFeedback,next_command,validate_target,HOME,SweepPlan,LOWER,UPPER,pose_steps


class FakeRobot:
    def __init__(self): self.q=HOME.copy(); self.writes=[]; self.fail=False; self.closed=False
    def connect(self): pass
    def get_state(self):
        if self.fail: raise TimeoutError('telemetry lost')
        return SimpleNamespace(q=np.deg2rad(self.q))
    def commissioning_target(self,q,studio_format=False): self.writes.append(('target',np.rad2deg(q))); self.q=np.rad2deg(q)
    def _request(self,text,accept): self.writes.append((text,None)); return True
    def stop_commissioning(self): self.writes.append(('stop',None))
    def close(self): self.closed=True


class LiveTests(unittest.TestCase):
    def wait(self,c,predicate,timeout=2):
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            c.heartbeat()
            if predicate(): return
            time.sleep(.02)
        self.fail('condition timed out: '+str(c.snapshot()))

    def make(self):
        r=FakeRobot(); c=LiveController('unused',factory=lambda:r); c.connect()
        self.addCleanup(lambda:(c.close(),c.thread.join(2)))
        self.wait(c,lambda:c.snapshot()['connected']); return c,r

    def test_connect_and_preview_never_write(self):
        c,r=self.make(); c.set_target(HOME+[10,0,0,0,0,0]); time.sleep(.15)
        self.assertEqual(r.writes,[])

    def test_arm_discards_preview_then_streams_and_stops(self):
        c,r=self.make(); c.set_target(HOME+[10,0,0,0,0,0]); c.arm()
        self.wait(c,lambda:c.snapshot()['active'])
        self.assertTrue(all(np.allclose(q,HOME) for kind,q in r.writes if kind=='target'))
        c.set_target(HOME+[10,0,0,0,0,0]); self.wait(c,lambda:r.q[0]>.5)
        c.stop(); self.wait(c,lambda:not c.snapshot()['active'])
        self.assertEqual(r.writes[-1][0],'stop')

    def test_telemetry_loss_stops_and_disconnects(self):
        c,r=self.make(); c.arm(); self.wait(c,lambda:c.snapshot()['active']); r.fail=True
        self.wait(c,lambda:not c.snapshot()['connected'])
        self.assertEqual(r.writes[-1][0],'stop'); self.assertTrue(r.closed)

    def test_ui_heartbeat_loss_stops(self):
        c,r=self.make(); c.arm(); self.wait(c,lambda:c.snapshot()['active']); c.ui_at=time.monotonic()-2
        c.thread.join(1)
        self.assertFalse(c.snapshot()['connected']); self.assertEqual(r.writes[-1][0],'stop')

    def test_rate_and_lead_bounded_under_stalled_feedback(self):
        q=HOME.copy(); sent=q.copy(); target=q+[10,0,0,0,0,0]
        for _ in range(25):
            new=next_command(sent,q,target,1.)
            self.assertLessEqual(np.max(np.abs(new-sent)),.500001)
            self.assertLessEqual(np.max(np.abs(new-q)),2.000001); sent=new

    def test_bad_values_and_unexpected_motion_rejected(self):
        for q in ([float('nan')]*6,HOME+[100,0,0,0,0,0],[1,2]):
            with self.assertRaises(ValueError): validate_target(q)
        with self.assertRaises(RuntimeError): next_command(HOME,HOME+[4,0,0,0,0,0],HOME,.1)

    def test_sweep_covers_full_envelope_and_keeps_running(self):
        center=np.array([19.73,-27.1,137.03,-1.01,-15,.25])
        p=SweepPlan(center); samples=[]
        for _ in range(1200):
            q,phase=p.propose(.1,20); p.commit(q,phase,q); samples.append(q)
        samples=np.array(samples)
        np.testing.assert_allclose(samples.min(0),LOWER,atol=1e-7)
        np.testing.assert_allclose(samples.max(0),UPPER,atol=1e-7)
        self.assertTrue(np.all(p.reversals>10))
        self.assertTrue(np.all(samples>=LOWER-1e-8) and np.all(samples<=UPPER+1e-8))

    def test_linear_interiors_and_speed_change_have_no_jump(self):
        p=SweepPlan(np.array([0,-20,120,0,0,0.]))
        q,phase=p.propose(.1,5); np.testing.assert_allclose(q-p.target,.5)
        p.commit(q,phase,q)
        new,phase=p.propose(.1,10); np.testing.assert_allclose(new-q,1.)

    def test_auto_manual_override_rejected_and_stop_cancels(self):
        c,r=self.make(); c.start_auto()
        self.wait(c,lambda:c.snapshot()['active'])
        with self.assertRaises(RuntimeError): c.set_target(HOME)
        self.wait(c,lambda:r.q[0]>.2)
        self.assertTrue(c.snapshot()['automatic'])
        c.stop(); self.wait(c,lambda:not c.snapshot()['active'])
        self.assertFalse(c.snapshot()['automatic'])
        self.assertEqual(c.last_auto_report['result'],'interrupted')
        count=len(r.writes); time.sleep(.15); self.assertEqual(len(r.writes),count)

    def test_speed_changes_slew_not_lead_envelope(self):
        target=HOME+[10,0,0,0,0,0]
        self.assertAlmostEqual(next_command(HOME,HOME,target,.1,20)[0],2)
        self.assertAlmostEqual(next_command(HOME,HOME,target,.1,1)[0],.1)
        for speed in (0,21,float('nan')):
            with self.assertRaises(ValueError): next_command(HOME,HOME,target,.1,speed)

    def test_pose_sequences_have_expected_order_and_final_angles(self):
        q=np.array([10,-30,140,5,5,5.])
        for kind,final in [('home',HOME),('fold',[0,-75,180,0,0,0])]:
            stages=pose_steps(q,kind)
            np.testing.assert_array_equal(stages[0],[0,-30,140,0,0,0])
            self.assertEqual(stages[1][2],140)
            np.testing.assert_array_equal(stages[-1],final)

    def test_home_replaces_auto_and_stops_when_complete(self):
        c,r=self.make(); c.start_auto(); self.wait(c,lambda:r.q[0]>.3)
        c.request_pose('home')
        self.wait(c,lambda:'直立复位完成' in c.snapshot()['auto_progress'],timeout=5)
        self.wait(c,lambda:not c.snapshot()['active'])
        np.testing.assert_allclose(r.q,HOME,atol=.02)
        self.assertEqual(r.writes[-1][0],'stop')

    def test_stop_cancels_pending_pose(self):
        c,r=self.make()
        # Hold the state lock to keep the worker from consuming the request.
        with c.lock:
            c.pose_request='fold'; c.stop_event.set()
        c.stop(); time.sleep(.15)
        self.assertIsNone(c.pose_request)
        self.assertFalse(c.snapshot()['active'])

    def test_native_frame_deadline_rejects_partial_stale_read(self):
        native=NativeFeedback.__new__(NativeFeedback)
        native.device=SimpleNamespace(robot=SimpleNamespace(**{f'joint_{i}':SimpleNamespace(angle=0.) for i in range(1,7)}))
        with patch('dummy_loop.live_control.time.monotonic',side_effect=[0,.1,.2,.8]):
            with self.assertRaises(TimeoutError): native.read()

    def test_stale_ui_cancels_native_and_pending_pose(self):
        c=LiveController('unused',factory=FakeRobot)
        token=SimpleNamespace(cancelled=False)
        c.native=SimpleNamespace(close=lambda:setattr(token,'cancelled',True))
        c.connected=True; c.rx_at=time.monotonic()-2; c.pose_request='fold'
        c.expire_feedback()
        self.assertTrue(c.stop_event.is_set()); self.assertTrue(token.cancelled)
        self.assertIsNone(c.pose_request)


if __name__=='__main__': unittest.main()

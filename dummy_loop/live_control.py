"""Supervised GUI commissioning, not the autonomous policy execution API.

All serial I/O is owned by one worker. Latest target replaces old targets.
Response age is host receive age, not motor sensor freshness.
"""
import threading
import time
import os
import sys
import functools
import json
from pathlib import Path
import numpy as np
from .serial_backend import SerialRobot, list_ports

# Conservative UI envelope, NOT calibrated mechanical limits.
LOWER=np.array([-20.,-75.,90.,-15.,-15.,-15.])
UPPER=np.array([20.,20.,180.,15.,15.,15.])
HOME=np.array([0.,0.,90.,0.,0.,0.])


def pose_steps(q,kind):
    if kind not in ('home','fold'): raise ValueError('未知姿态')
    q=validate_target(q); targets=[]
    q[[0,3,4,5]]=0; targets.append(q.copy())
    q[1]=0 if kind=='home' else -75; targets.append(q.copy())
    q[2]=90 if kind=='home' else 180; targets.append(q.copy())
    return targets


def device_identity():
    """从 configs/device_identity.json 读取 USB 身份，环境变量可覆盖序列号。

    序列号原先硬编码在本文件中。它是一台具体设备的标识，属于配置而非代码：
    换机械臂不应该需要改源码。
    """
    path=Path(__file__).resolve().parents[1]/'configs'/'device_identity.json'
    cfg=json.loads(path.read_text(encoding='utf-8'))
    serial=os.environ.get('DUMMY_DEVICE_SERIAL') or cfg.get('serial_number')
    return (int(cfg['vendor_id'],16), int(cfg['product_id'],16),
            serial.strip().upper() if isinstance(serial,str) and serial.strip() else None)


class NativeFeedback:
    def __init__(self):
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'vendor/native_client'))
        import usb.core
        import usb.backend.libusb1
        import libusb_package
        import fibre
        # Do not globally patch usb.core.find across reconnects.
        vendor,product,expected_serial=device_identity()
        backend=usb.backend.libusb1.get_backend(find_library=libusb_package.find_library)
        self.end=fibre.Event()
        original_find=usb.core.find
        usb.core.find=functools.partial(original_find,backend=backend,idVendor=vendor,idProduct=product)
        try:
            self.device=fibre.find_any(path='usb',timeout=12,channel_termination_token=self.end,logger=fibre.Logger(verbose=False))
            if self.device is None:
                raise RuntimeError('未找到原生反馈接口设备')
            if expected_serial and f'{self.device.serial_number:012X}'!=expected_serial:
                raise RuntimeError('原生反馈接口身份验证失败：设备序列号与 configs/device_identity.json 不符')
            self.device.__channel__._resend_timeout=.15
            self.device.__channel__._send_attempts=2
        except BaseException:
            self.end.set(); raise
        finally: usb.core.find=original_find

    def read(self):
        started=time.monotonic()
        values=[]
        for i in range(1,7):
            values.append(getattr(self.device.robot,f'joint_{i}').angle)
            if time.monotonic()-started>.6: raise TimeoutError('原生反馈读取超时，不能使用过期姿态')
        return np.array(values)+np.array([0,-75,180,0,0,0])
    def close(self): self.end.set()


def validate_target(q):
    q=np.asarray(q,dtype=float)
    if q.shape!=(6,) or not np.all(np.isfinite(q)) or np.any(q<LOWER) or np.any(q>UPPER):
        raise ValueError('目标超出本次调试范围')
    return q.copy()


def next_command(sent,feedback,target,dt,speed=5.):
    """Adjustable host slew rate; max 2 deg lead; no catch-up after blocking."""
    if not np.isfinite(speed) or not 1<=speed<=20: raise ValueError('跟随速度应为 1～20°/秒')
    target=validate_target(target)
    sent=np.asarray(sent); feedback=np.asarray(feedback)
    if not np.all(np.isfinite(feedback)): raise ValueError('角度反馈无效')
    step=speed*min(max(dt,0.),.1)
    candidate=sent+np.clip(target-sent,-step,step)
    # Never issue a backwards catch-up jump if feedback unexpectedly changes.
    if np.max(np.abs(sent-feedback))>3.: raise RuntimeError('跟随误差超过 3°，已暂停')
    return np.where(np.abs(candidate-feedback)<=2.,candidate,sent)


class SweepPlan:
    """Continuous triangular joint trajectories over the full software envelope."""
    def __init__(self,center):
        self.center=validate_target(center)
        self.width=UPPER-LOWER
        self.phase=self.center-LOWER
        self.target=self.center.copy()
        self.minimum=self.center.copy(); self.maximum=self.center.copy()
        self.reversals=np.zeros(6,dtype=int); self.paused_ticks=0

    def propose(self,dt,speed):
        if not np.isfinite(speed) or not 1<=speed<=20: raise ValueError('无效跟随速度')
        boundary=(np.floor((self.phase+1e-9)/self.width)+1)*self.width
        phase=np.minimum(self.phase+speed*min(max(dt,0.),.1),boundary)
        mod=np.mod(phase,2*self.width)
        return LOWER+np.where(mod<=self.width,mod,2*self.width-mod),phase

    def commit(self,target,phase,q):
        self.reversals+=(np.floor(phase/self.width)-np.floor(self.phase/self.width)).astype(int)
        self.phase=phase; self.target=target.copy()
        self.minimum=np.minimum(self.minimum,q); self.maximum=np.maximum(self.maximum,q)

    def report(self,result):
        return {'result':result,'mode':'continuous_full_range_linear','start_deg':self.center.tolist(),
                'lower_deg':LOWER.tolist(),'upper_deg':UPPER.tolist(),'reversals':self.reversals.tolist(),
                'observed_min_deg':self.minimum.tolist(),'observed_max_deg':self.maximum.tolist(),
                'feedback_pause_ticks':self.paused_ticks,
                'note':'Joint-space linear targets, not Cartesian straight lines or guaranteed physical constant velocity.'}


class LiveController:
    def __init__(self,log_path,factory=None):
        self.lock=threading.Lock(); self.exit=threading.Event(); self.stop_event=threading.Event()
        self.thread=None; self.factory=factory; self.log_path=log_path
        self.q=None; self.target=None; self.sent=None; self.rx_at=0.
        self.connected=False; self.active=False; self.want_arm=False
        self.status='未连接'; self.generation=0
        self.ui_at=time.monotonic()
        self.speed=5.; self.auto_request=None; self.plan=None; self.auto_progress=''
        self.last_auto_report=None
        self.pose_request=None; self.pose_targets=None; self.pose_kind=None
        self.pose_index=0; self.pose_settled=0; self.pose_at=0.
        self.native=None

    def expire_feedback(self):
        with self.lock:
            if not self.connected or time.monotonic()-self.rx_at<=.75: return
            self.pose_request=None; self.stop_event.set()
            native=self.native
            self.status='反馈已过期，正在中止动作；若停止未确认请使用实体停止'
        if native is not None: native.close()

    def request_pose(self,kind):
        with self.lock:
            if not self.connected or self.q is None: raise RuntimeError('请先连接机械臂')
            if np.any(self.q<LOWER-.1) or np.any(self.q>UPPER+.1): raise RuntimeError('当前姿态超出调试范围')
            pose_steps(np.clip(self.q,LOWER,UPPER),kind)
            self.pose_request=kind
            self.stop_event.set()

    def finish_plan(self,result):
        if self.plan is None: return
        self.last_auto_report=self.plan.report(result)
        if self.factory is None:
            path=Path(self.log_path).with_suffix(f'.auto_{time.time_ns()}.json')
            path.write_text(json.dumps(self.last_auto_report,indent=2),encoding='utf-8')

    def heartbeat(self): self.ui_at=time.monotonic()

    def snapshot(self):
        with self.lock:
            return dict(q=None if self.q is None else self.q.copy(),target=None if self.target is None else self.target.copy(),
                        connected=self.connected,active=self.active,status=self.status,rx_at=self.rx_at,generation=self.generation,
                        speed=self.speed,automatic=self.plan is not None or self.auto_request is not None or self.pose_request is not None or self.pose_targets is not None,auto_progress=self.auto_progress)

    def set_speed(self,value):
        value=float(value)
        if not np.isfinite(value) or not 1<=value<=20: raise ValueError('跟随速度应为 1～20°/秒')
        with self.lock: self.speed=value

    def start_auto(self):
        self.arm(auto=True)

    def connect(self):
        if self.thread and self.thread.is_alive(): return
        self.exit.clear(); self.stop_event.clear()
        self.thread=threading.Thread(target=self._run,daemon=True); self.thread.start()

    def set_target(self,q):
        value=validate_target(q)
        with self.lock:
            if not self.connected: raise RuntimeError('请先连接机械臂')
            if self.plan is not None or self.auto_request is not None or self.pose_request is not None or self.pose_targets is not None: raise RuntimeError('请先停止自动运动，再手动调整')
            self.target=value

    def arm(self,auto=None):
        with self.lock:
            if not self.connected or self.q is None: raise RuntimeError('没有可用的实机反馈')
            if self.active or self.want_arm: raise RuntimeError('请先停止当前跟随，再启动新模式')
            validate_target(np.clip(self.q,LOWER,UPPER))
            if np.any(self.q<LOWER-.1) or np.any(self.q>UPPER+.1): raise RuntimeError('当前姿态超出界面调试范围，只允许读取')
            if auto is not None: SweepPlan(np.clip(self.q,LOWER,UPPER))
            self.auto_request=auto; self.auto_progress=''
            # Discard any offline preview; arming itself must not move the robot.
            self.target=self.q.copy(); self.generation+=1; self.want_arm=True

    def stop(self):
        with self.lock: self.pose_request=None; self.stop_event.set()
    def close(self): self.stop(); self.exit.set()

    def _run(self):
        robot=None; native=None; feedback_log=None; enabled=False
        try:
            with self.lock: self.status='正在连接并读取当前姿态…'
            if self.factory is None:
                vendor,product,expected=device_identity()
                ports=[p for p in list_ports() if p['vid']==vendor and p['pid']==product
                       and (expected is None or p['serial']==expected)]
                if len(ports)!=1: raise RuntimeError('未找到指定机械臂 USB，或设备不唯一')
                robot=SerialRobot(ports[0]['port'],self.log_path,timeout=.6)
            else: robot=self.factory()
            robot.connect()
            if self.factory is None:
                native=NativeFeedback()
                self.native=native
                feedback_log=Path(self.log_path).with_suffix('.feedback.jsonl').open('a',encoding='utf-8')
                read=native.read
                q=read()
                if np.max(np.abs(q-np.rad2deg(robot.get_state().q)))>.05:
                    raise RuntimeError('原生与串口角度偏置不一致')
            else:
                read=lambda:np.rad2deg(robot.get_state().q)
                q=read()
            if q.shape!=(6,) or not np.all(np.isfinite(q)): raise RuntimeError('无效初始姿态')
            with self.lock:
                self.q=q.copy(); self.target=q.copy(); self.sent=q.copy(); self.rx_at=time.monotonic()
                self.connected=True; self.active=False; self.generation+=1; self.status='只读同步：开启跟随后再调整目标'
            previous=q.copy(); progress_at=np.full(6,time.monotonic()); last_send=time.monotonic()
            while not self.exit.is_set():
                begin=time.monotonic()
                if enabled and begin-self.ui_at>1.: raise RuntimeError('界面失去响应，跟随已停止')
                if self.stop_event.is_set():
                    self.stop_event.clear()
                    if enabled: robot.stop_commissioning(); enabled=False
                    with self.lock:
                        if self.plan is not None:
                            self.finish_plan('interrupted'); self.auto_progress='自动测试已中止，不自动回程'
                        self.active=False; self.want_arm=False; self.target=self.q.copy(); self.generation+=1; self.status='跟随已停止；继续读取反馈'
                        self.plan=None; self.auto_request=None
                        self.pose_targets=None; self.pose_kind=None
                q=read()
                if not np.all(np.isfinite(q)): raise RuntimeError('角度反馈无效')
                now=time.monotonic()
                with self.lock:
                    self.q=q.copy(); self.rx_at=now
                    arm=self.want_arm; self.want_arm=False; target=self.target.copy()
                    if self.pose_request is not None and not self.stop_event.is_set() and not self.exit.is_set():
                        self.pose_kind=self.pose_request; self.pose_request=None; arm=True
                    speed=self.speed
                if feedback_log:
                    feedback_log.write(json.dumps({'time_s':now,'q_deg':q.tolist(),'target_deg':target.tolist(),'active':enabled,'speed_deg_s':speed,'auto_progress':self.auto_progress})+'\n')
                    feedback_log.flush()
                if arm:
                    if np.any(q<LOWER-.1) or np.any(q>UPPER+.1): raise RuntimeError('姿态超出调试范围')
                    if self.stop_event.is_set() or self.exit.is_set(): continue
                    # STOP discards prior motion, then preload this measured pose.
                    enabled=True
                    robot.stop_commissioning()
                    q=read()
                    if np.any(q<LOWER-.1) or np.any(q>UPPER+.1): raise RuntimeError('姿态超出调试范围')
                    q=np.clip(q,LOWER,UPPER)
                    robot.line_ending='\n'
                    robot.commissioning_target(np.deg2rad(q),studio_format=True)
                    if self.stop_event.is_set() or self.exit.is_set(): continue
                    # Worker target ACK may precede START ACK on the same line.
                    robot._request('!START',lambda s:True if s in ('Started ok','okStarted ok') else None)
                    robot.commissioning_target(np.deg2rad(q),studio_format=True)
                    with self.lock:
                        self.sent=q.copy(); self.target=q.copy(); self.active=True; self.generation+=1
                        self.status='实时跟随已开启：拖动目标滑块即控制实体'
                        if self.auto_request is not None:
                            self.plan=SweepPlan(q); self.auto_request=None
                        if self.pose_kind is not None:
                            self.pose_targets=pose_steps(q,self.pose_kind)
                            self.pose_index=0; self.pose_settled=0; self.pose_at=time.monotonic()
                    target=q.copy(); previous=q.copy(); progress_at[:]=time.monotonic(); last_send=time.monotonic()
                if enabled and not self.stop_event.is_set() and not self.exit.is_set():
                    moved=np.abs(q-previous)>.05
                    progress_at[moved]=now; previous[moved]=q[moved]
                    error=np.abs(self.sent-q)
                    progress_at[error<.35]=now
                    if np.any((error>=.35)&(now-progress_at>2.)):
                        raise RuntimeError('关节反馈停滞超过 2 秒，跟随停止')
                    with self.lock:
                        if self.pose_targets is not None:
                            name='直立复位' if self.pose_kind=='home' else '折叠'
                            target=self.pose_targets[self.pose_index]
                            if now-self.pose_at>120: raise RuntimeError(name+'阶段到位超时')
                            self.pose_settled=self.pose_settled+1 if np.max(np.abs(q-target))<.15 else 0
                            if self.pose_settled>=3:
                                self.pose_index+=1; self.pose_settled=0; self.pose_at=now
                            if self.pose_index==len(self.pose_targets):
                                self.auto_progress=name+'完成：反馈到位'
                                self.pose_targets=None; self.stop_event.set(); continue
                            target=self.pose_targets[self.pose_index]
                            cmd=next_command(self.sent,q,target,now-last_send,speed)
                            self.target=cmd.copy(); self.generation+=1
                            self.auto_progress=f'{name} {self.pose_index+1}/{len(self.pose_targets)} 阶段'
                            self.status=self.auto_progress+'；按停止中止'
                        elif self.plan is not None:
                            target,phase=self.plan.propose(now-last_send,speed)
                            if np.max(np.abs(self.sent-q))>3.: raise RuntimeError('跟随误差超过 3°')
                            if np.max(np.abs(target-q))<=2.:
                                self.plan.commit(target,phase,q); cmd=target
                                self.auto_progress='全范围线性往复 · 持续运行'
                            else:
                                cmd=self.sent.copy(); self.plan.paused_ticks+=1
                                self.auto_progress='反馈跟随偏慢 · 暂停推进轨迹'
                            self.target=cmd.copy(); self.generation+=1
                            self.status=self.auto_progress+'；按停止结束'
                        else:
                            cmd=next_command(self.sent,q,target,now-last_send,speed)
                    if np.max(np.abs(cmd-self.sent))>=.015:
                        robot.commissioning_target(np.deg2rad(cmd),studio_format=True)
                        self.sent=cmd
                    last_send=now
                self.exit.wait(max(0,.1-(time.monotonic()-begin)))
        except Exception as exc:
            with self.lock: self.status='已断开：'+str(exc)
        finally:
            stop_error=None
            if robot:
                if enabled:
                    try: robot.stop_commissioning()
                    except Exception as exc: stop_error=str(exc)
                robot.close()
            if native: native.close()
            self.native=None
            if feedback_log: feedback_log.close()
            with self.lock:
                if self.plan is not None: self.finish_plan('aborted_on_exit_or_fault')
                self.connected=False; self.active=False; self.want_arm=False
                self.plan=None; self.auto_request=None
                self.pose_request=None; self.pose_targets=None; self.pose_kind=None
                if stop_error: self.status='停止未确认，请使用实体停止装置：'+stop_error
                elif self.exit.is_set(): self.status='已断开'

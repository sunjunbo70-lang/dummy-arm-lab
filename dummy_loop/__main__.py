import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
from .core import Guard, Observation, load_profile, six


def emit(value):
    print(json.dumps(value,ensure_ascii=False,indent=2))


def execute_sim(args):
    from .sim_backend import SimRobot
    from .policy import LinearPolicy
    policy=LinearPolicy(args.policy)
    robot=SimRobot(dt=args.dt)
    if abs(policy.meta['dt']-args.dt)>1e-8:
        raise ValueError('Policy/control dt mismatch')
    goal=six(args.goal)
    if np.max(np.abs(goal))>0.3:
        raise ValueError('Reference demonstration goal must be within +/-0.3 rad')
    guard=Guard([-0.7]*6,[0.7]*6,[0.04]*6,[0.8]*6)
    robot.connect()
    viewer=None
    args.log.parent.mkdir(parents=True,exist_ok=True)
    try:
        if args.viewer:
            import mujoco.viewer
            viewer=mujoco.viewer.launch_passive(robot.model,robot.data)
            viewer.cam.lookat[:]=[0,0,0.2]; viewer.cam.distance=1.; viewer.cam.azimuth=130.; viewer.cam.elevation=-20.
        initial=float(np.linalg.norm(robot.get_state().q-goal))
        with args.log.open('w',encoding='utf-8') as log:
            for i in range(args.steps):
                start=time.monotonic()
                state=robot.get_state()
                pred_start=time.perf_counter()
                delta=policy.predict(state.q,goal)
                inference_ms=(time.perf_counter()-pred_start)*1000
                requested=state.q+delta
                # Explicit rate limiter; record both requested and applied target.
                applied=state.q+np.clip(delta,-0.8*args.dt,0.8*args.dt)
                target=guard.validate(applied,state,args.dt)
                if viewer:
                    with viewer.lock(): robot.send_action(target)
                    viewer.sync()
                else:
                    robot.send_action(target)
                measured=robot.get_state().q
                log.write(json.dumps({'step':i,'sim_time_s':float(robot.data.time),'q_rad':state.q.tolist(),
                    'requested_rad':requested.tolist(),'applied_rad':target.tolist(),'next_q_rad':measured.tolist(),
                    'goal_rad':goal.tolist(),'inference_ms':inference_ms,'source':'reference_sim_only'})+'\n')
                if viewer:
                    if not viewer.is_running(): break
                    time.sleep(max(0,args.dt-(time.monotonic()-start)))
        final=float(np.linalg.norm(robot.get_state().q-goal))
        summary={'backend':'mujoco_reference','initial_error_rad':initial,'final_error_rad':final,
                 'steps':i+1,'software_loop_pass':bool(final<0.02),
                 'scope':'joint_goal_software_smoke_test_not_real_robot_or_grasp_validation','log':str(args.log)}
        emit(summary)
        args.log.with_suffix('.summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        if final>=0.02:
            raise RuntimeError('Reference loop did not meet smoke-test goal tolerance')
    finally:
        if viewer: viewer.close()
        robot.close()


def execute_probe(args):
    from .serial_backend import SerialRobot
    robot=SerialRobot(args.port,args.log)
    robot.connect()
    try:
        for i in range(args.samples):
            start=time.monotonic()
            state=robot.get_state()
            emit({'sample':i,'firmware_joint_deg':np.rad2deg(state.q).tolist(),
                  'round_trip_ms':(time.monotonic()-start)*1000,
                  'per_axis_freshness':'unknown','motion_commands_sent':False})
            time.sleep(max(0,1/args.hz-(time.monotonic()-start)))
    finally:
        robot.close()


def execute_jog(args):
    # Commissioning only. This never executes an autonomous model.
    from .serial_backend import SerialRobot
    p,guard,sign,offset=load_profile(args.profile)
    if not args.enable:
        raise ValueError('Explicit --enable required for commissioning motion')
    if not np.isfinite(args.delta_deg) or not 0<abs(args.delta_deg)<=2:
        raise ValueError('Commissioning step must be nonzero and at most 2 degrees; calibrated limits may be smaller')
    robot=SerialRobot(args.port,args.log)
    enabled=False
    robot.connect()
    try:
        # Mapping: canonical = sign * (firmware - offset), radians throughout below.
        raw=robot.get_state()
        q=sign*(raw.q-np.deg2rad(offset))
        state=Observation(q,raw.received_at,raw.source,False)
        target=q.copy(); target[args.joint-1]+=np.deg2rad(args.delta_deg)
        target=guard.validate(target,state,1.0)
        # Enable intent set first so partial handshake attempts still try verified stop.
        enabled=True
        robot.enable_commissioning(p['firmware_speed_parameter'],p['command_mode'])
        # Refresh after enable; do not rely on pre-enable coordinates.
        after=robot.get_state()
        state=Observation(sign*(after.q-np.deg2rad(offset)),after.received_at,after.source,False)
        target=guard.validate(target,state,1.0)
        firmware=sign*target+np.deg2rad(offset)
        robot.commissioning_target(firmware)
        deadline=time.monotonic()+5.
        seen=0
        while time.monotonic()<deadline:
            raw=robot.get_state(); q=sign*(raw.q-np.deg2rad(offset))
            guard.validate(q,Observation(q,raw.received_at,raw.source),1.0)
            err=float(np.max(np.abs(q-target)))
            emit({'measured_rad':q.tolist(),'target_rad':target.tolist(),'max_error_rad':err,
                  'device_freshness':'unknown_legacy_protocol','scope':'operator_supervised_commissioning'})
            seen=seen+1 if err<np.deg2rad(.2) else 0
            if seen>=3: break
            time.sleep(.1)
        else:
            raise TimeoutError('Target not reached within commissioning timeout')
    finally:
        try:
            if enabled: robot.stop_commissioning()
        except Exception as exc:
            print(f'SOFTWARE STOP FAILED: {exc}. Local operator must use the verified physical stop.',file=sys.stderr)
        finally:
            robot.close()


def execute_shadow(args):
    from .policy import LinearPolicy
    from .serial_backend import SerialRobot
    p,guard,sign,offset=load_profile(args.profile)
    policy=LinearPolicy(args.policy)
    robot=SerialRobot(args.port,args.log)
    robot.connect()
    try:
        for _ in range(args.samples):
            raw=robot.get_state(); q=sign*(raw.q-np.deg2rad(offset))
            emit({'q_rad':q.tolist(),'predicted_delta_rad':policy.predict(q,args.goal).tolist(),
                  'applied':False,'warning':'reference_sim_policy_predictions_are_not_validated_for_hardware'})
            time.sleep(.2)
    finally:
        robot.close()


def main():
    parser=argparse.ArgumentParser(description='Dummy minimum experiment loop. Simulation + read-only USB + gated commissioning.')
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('ports')
    probe=sub.add_parser('probe'); probe.add_argument('--port',required=True)
    probe.add_argument('--samples',type=int,default=10); probe.add_argument('--hz',type=float,default=2)
    probe.add_argument('--log',type=Path,default=Path('outputs/usb_probe.jsonl'))
    collect=sub.add_parser('collect-sim'); collect.add_argument('--output',type=Path,default=Path('outputs/teacher.npz'))
    collect.add_argument('--episodes',type=int,default=40); collect.add_argument('--seed',type=int,default=7)
    train=sub.add_parser('train'); train.add_argument('--dataset',type=Path,default=Path('outputs/teacher.npz'))
    train.add_argument('--output',type=Path,default=Path('outputs/policy.npz'))
    run=sub.add_parser('run-sim'); run.add_argument('--policy',type=Path,default=Path('outputs/policy.npz'))
    run.add_argument('--goal',type=float,nargs=6,default=[.15,-.1,.2,.1,-.1,.05])
    run.add_argument('--steps',type=int,default=240); run.add_argument('--dt',type=float,default=.05)
    run.add_argument('--viewer',action='store_true'); run.add_argument('--log',type=Path,default=Path('outputs/rollout.jsonl'))
    jog=sub.add_parser('jog'); jog.add_argument('--port',required=True)
    jog.add_argument('--profile',type=Path,default=Path('configs/hardware.unverified.json'))
    jog.add_argument('--joint',type=int,choices=range(1,7),required=True); jog.add_argument('--delta-deg',type=float,required=True)
    jog.add_argument('--enable',action='store_true'); jog.add_argument('--log',type=Path,default=Path('outputs/jog.jsonl'))
    shadow=sub.add_parser('shadow'); shadow.add_argument('--port',required=True)
    shadow.add_argument('--profile',type=Path,default=Path('configs/hardware.unverified.json'))
    shadow.add_argument('--policy',type=Path,default=Path('outputs/policy.npz'))
    shadow.add_argument('--goal',type=float,nargs=6,required=True); shadow.add_argument('--samples',type=int,default=10)
    shadow.add_argument('--log',type=Path,default=Path('outputs/shadow_serial.jsonl'))
    args=parser.parse_args()
    for name in ('samples','steps'):
        if hasattr(args,name) and getattr(args,name)<=0: parser.error(f'{name} must be positive')
    if hasattr(args,'hz') and (not np.isfinite(args.hz) or not 0<args.hz<=10): parser.error('Probe hz must be in (0,10]')
    if args.command=='ports':
        from .serial_backend import list_ports
        emit(list_ports())
    elif args.command=='probe': execute_probe(args)
    elif args.command=='collect-sim':
        from .policy import collect
        emit(collect(args.output,args.episodes,args.seed))
    elif args.command=='train':
        from .policy import train
        emit(train(args.dataset,args.output))
    elif args.command=='run-sim': execute_sim(args)
    elif args.command=='jog': execute_jog(args)
    elif args.command=='shadow': execute_shadow(args)


if __name__=='__main__':
    try:
        main()
    except (ValueError,RuntimeError,TimeoutError,OSError) as exc:
        print(f'ERROR: {exc}',file=sys.stderr)
        raise SystemExit(2)

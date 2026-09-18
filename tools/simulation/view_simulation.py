"""Interactive reference simulation. Never imports or connects a robot transport."""
from pathlib import Path
import sys
import time
import threading
import argparse

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import mujoco.viewer
from dummy_loop.core import Guard
from dummy_loop.policy import LinearPolicy
from dummy_loop.sim_backend import SimRobot


def cycle_goals():
    """J1 through J6: +10 degrees, -10 degrees, then home."""
    goals = []
    for joint in range(6):
        for angle in (10, -10, 0):
            q = np.zeros(6)
            q[joint] = np.deg2rad(angle)
            goals.append(q)
    return np.array(goals)


def simultaneous_goal(sim_time):
    """All six targets follow the same 8-second, +/-10-degree sine wave."""
    return np.full(6, np.deg2rad(10)*np.sin(2*np.pi*sim_time/8))


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument('--studio', action='store_true', help='Full visual assembly, provisional articulation; initially paused')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--manual', action='store_true', help='Six-axis diagnostic jogging: 1-6 select, brackets jog')
    mode.add_argument('--cycle', action='store_true', help='Repeat J1-J6 diagnostic motion: +10, -10, 0 degrees per axis')
    mode.add_argument('--simultaneous', action='store_true', help='All six axes oscillate together: +/-10 degrees, 8-second period')
    args = parser.parse_args()
    diagnostic = args.manual or args.cycle or args.simultaneous
    policy = None if diagnostic else LinearPolicy(root / 'outputs/policy.npz')
    robot = SimRobot(root/'models/dummy_studio_visual.xml',dt=.05) if args.studio else SimRobot(dt=.05)
    robot.connect()
    guard = Guard([-0.7]*6, [0.7]*6, [0.04]*6, [0.8]*6)
    goals = np.array([
        [.25, -.20, .25, .18, -.20, .15],
        [-.25, .20, -.20, -.18, .20, -.15],
        [.15, .10, -.10, .25, -.15, .25],
        [0., 0., 0., 0., 0., 0.],
    ])
    if args.studio:
        # Extracted hierarchy cannot justify multi-axis motion yet. Keep the
        # full assembly intact while demonstrating only base rotation.
        goals[:,1:]=0
    if args.cycle:
        goals = cycle_goals()
    ticks_per_goal = 60 if args.cycle else 200
    commands = []
    command_lock = threading.Lock()

    def key_callback(key):
        with command_lock:
            commands.append(key)

    print('Reference simulation only. No real robot connection.', flush=True)
    print('SPACE pause/resume; N next target; R reset; close window to exit.', flush=True)
    paused = args.studio and not diagnostic
    selected = 0
    manual_goal = np.zeros(6)
    if args.manual:
        print('SIX-AXIS DIAGNOSTIC: 1-6 select joint; [ / ] decrease/increase 5 degrees; R zero; SPACE pause. Uncalibrated geometry.', flush=True)
    if args.cycle:
        print('SIX-AXIS DIAGNOSTIC LOOP: J1-J6 each +10/-10/0 deg, 3 seconds per target, 54 seconds per loop. Uncalibrated geometry. SPACE pause; N next; R restart.', flush=True)
    if args.simultaneous:
        print('SIX-AXIS SIMULTANEOUS LOOP: all targets +/-10 deg, 8-second period. Uncalibrated geometry. SPACE pause; R restart; C covers.', flush=True)
    if args.studio and not diagnostic:
        print('COMPLETE VISUAL ASSEMBLY; PAUSED. SPACE: base-only demo. C: hide/show covers. Visual transforms checked against Studio; physical calibration pending.', flush=True)
    index, ticks = 0, 0
    with mujoco.viewer.launch_passive(robot.model, robot.data, key_callback=key_callback) as viewer:
        with viewer.lock():
            viewer.cam.lookat[:] = [0, 0, .20]
            viewer.cam.distance = .90
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -20
        print('VIEWER_READY', flush=True)
        while viewer.is_running():
            start = time.monotonic()
            with command_lock:
                keys = commands[:]
                commands.clear()
            with viewer.lock():
                for key in keys:
                    if args.manual and ord('1') <= key <= ord('6'):
                        selected = key-ord('1')
                        print(f'Selected J{selected+1}', flush=True)
                    elif args.manual and key in (ord('['), ord(']'), ord('-'), ord('=')):
                        direction = -1 if key in (ord('['),ord('-')) else 1
                        manual_goal[selected] = np.clip(manual_goal[selected]+direction*np.deg2rad(5), -.7,.7)
                        print(f'J{selected+1} target {np.rad2deg(manual_goal[selected]):.1f} deg', flush=True)
                    elif key == 32:
                        paused = not paused
                    elif key in (ord('C'), ord('c')):
                        viewer.opt.geomgroup[2] = 1-viewer.opt.geomgroup[2]
                    elif key in (ord('N'), ord('n')) and not args.simultaneous:
                        index = (index+1) % len(goals)
                        ticks = 0
                    elif key in (ord('R'), ord('r')):
                        robot.reset(np.zeros(6))
                        manual_goal[:] = 0
                        index = 0
                        ticks = 0
                if not paused:
                    state = robot.get_state()
                    if diagnostic:
                        goal = simultaneous_goal(robot.data.time+robot.dt) if args.simultaneous else manual_goal if args.manual else goals[index]
                        delta = .15*(goal-state.q)
                    else:
                        delta = policy.predict(state.q, goals[index])
                    target = guard.validate(state.q + np.clip(delta, -.04, .04), state, .05)
                    robot.send_action(target)
                    ticks += 1
                    if ticks >= ticks_per_goal:
                        index = (index+1) % len(goals)
                        ticks = 0
            viewer.sync()
            time.sleep(max(0, .05-(time.monotonic()-start)))
    robot.close()


if __name__ == '__main__':
    main()

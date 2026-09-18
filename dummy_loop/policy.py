"""Small supervised policy for pipeline verification, not a grasp/VLA policy."""
import json
from pathlib import Path
import numpy as np
from .core import six


class LinearPolicy:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as d:
            self.w = d['weights'].copy()
            self.meta = json.loads(str(d['metadata']))
        if self.w.shape != (13,6) or not np.all(np.isfinite(self.w)):
            raise ValueError('Invalid policy weights')
        if self.meta.get('observation') != 'q_goal_rad' or self.meta.get('action') != 'delta_q_rad':
            raise ValueError('Incompatible model contract')

    def predict(self, state, goal):
        return np.r_[six(state), six(goal), 1.] @ self.w


def collect(path, episodes=40, seed=7):
    from .sim_backend import SimRobot
    if episodes < 4:
        raise ValueError('Need at least 4 episodes')
    robot = SimRobot()
    rng = np.random.default_rng(seed)
    xs, ys, ids = [], [], []
    for episode in range(episodes):
        robot.reset(rng.uniform(-0.15, 0.15, 6))
        goal = rng.uniform(-0.25, 0.25, 6)
        for _ in range(60):
            q = robot.get_state().q
            delta = 0.06*(goal-q)
            xs.append(np.r_[q,goal,1.]); ys.append(delta); ids.append(episode)
            robot.send_action(q+delta)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, x=np.array(xs), action=np.array(ys), episode=np.array(ids))
    return {'episodes':episodes,'frames':len(xs),'source':'synthetic_joint_goal_teacher_in_reference_sim', 'seed':seed}


def train(dataset, output):
    with np.load(dataset, allow_pickle=False) as d:
        x,y,episode = d['x'],d['action'],d['episode']
    if x.ndim!=2 or x.shape[1]!=13 or y.shape!=(len(x),6) or episode.shape!=(len(x),):
        raise ValueError('Invalid dataset shape')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Invalid dataset values')
    unique=np.unique(episode)
    if len(unique)<4:
        raise ValueError('Need independent train/validation episodes')
    train_ids=unique[:int(len(unique)*0.8)]
    mask=np.isin(episode,train_ids)
    if np.linalg.matrix_rank(x[mask])<13:
        raise ValueError('Training observations are rank deficient; collect more independent start/goal episodes')
    weights=np.linalg.lstsq(x[mask],y[mask],rcond=None)[0]
    metadata={'observation':'q_goal_rad','action':'delta_q_rad','dt':0.05,
              'scope':'reference_sim_only','algorithm':'supervised_linear_behavior_cloning',
              'training_episodes':len(train_ids),'validation_episodes':len(unique)-len(train_ids)}
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,weights=weights,metadata=json.dumps(metadata))
    return {**metadata,'validation_mse':float(np.mean((x[~mask]@weights-y[~mask])**2))}

from pathlib import Path
import time
import numpy as np
import mujoco
from .core import Observation, six

MODEL = Path(__file__).resolve().parents[1] / 'models' / 'dummy_reference.xml'
STUDIO_MODEL = MODEL.parent / 'dummy_studio_visual.xml'
# 用户实机对应的 V2 模型：运动学 = V2 固件 DH，零位 = 固件 HOME，末端 = J6 裸轴。
# 墙面仿真、上位机画面、查看器都用它。MODEL（参考模型）只保留给 BC 冒烟闭环，
# 以保证已冻结的历史数字仍可复现。见 docs/hardware/DUMMY_V2.md。
V2_MODEL = MODEL.parent / 'dummy_v2.xml'
# Studio 外观模型与参考模型的关节正方向关系（由两份模型的关节轴逐一比对得到，见
# tests/test_studio_fallback.py）：J1、J4、J6 轴向相反。Studio 网格缺失时用参考模型
# 代替显示，必须乘这个符号，否则这三个关节会朝反方向转。
STUDIO_TO_REFERENCE_SIGN = np.array([-1.0, 1.0, 1.0, -1.0, 1.0, -1.0])


def missing_model_assets(model):
    """返回模型引用但磁盘上不存在的网格文件列表（空列表 = 齐全）。"""
    import xml.etree.ElementTree as ET
    model = Path(model)
    doc = ET.fromstring(model.read_text(encoding='utf-8'))
    compiler = doc.find('compiler')
    meshdir = compiler.get('meshdir', '') if compiler is not None else ''
    return [str(Path(meshdir) / m.get('file')) for m in doc.findall('asset/mesh')
            if not (model.parent / meshdir / m.get('file')).is_file()]


def studio_meshes_available():
    return STUDIO_MODEL.is_file() and not missing_model_assets(STUDIO_MODEL)


class SimRobot:
    def __init__(self, model=MODEL, dt=0.05):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('dt must be positive')
        model=Path(model)
        missing = missing_model_assets(model)
        if missing:
            raise FileNotFoundError(
                f'{model.name} 引用的 {len(missing)} 个网格文件不存在（例如 {missing[0]}）。'
                ' Studio 外观网格因上游无许可声明不随仓库分发，需要从原开发机的'
                ' models/studio_meshes/ 复制过来，或按 models/README.md 重建；'
                ' 复制后可用 tools/maintenance/verify_studio_meshes.py 校验。')
        import xml.etree.ElementTree as ET
        doc = ET.fromstring(model.read_text(encoding='utf-8'))
        compiler = doc.find('compiler')
        meshdir = compiler.get('meshdir', '') if compiler is not None else ''
        assets = {}
        for mesh in doc.findall('asset/mesh'):
            filename = (Path(meshdir) / mesh.get('file')).as_posix()
            assets[filename] = (model.parent / filename).read_bytes()
        self.model = mujoco.MjModel.from_xml_string(model.read_text(encoding='utf-8'), assets)
        self.data = mujoco.MjData(self.model)
        self.dt = dt
        self.steps = round(dt / self.model.opt.timestep)
        if self.steps < 1 or abs(self.steps*self.model.opt.timestep-dt) > 1e-8:
            raise ValueError('dt must be a multiple of simulation timestep')
        self.jids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f'Joint{i}') for i in range(1,7)]
        if min(self.jids) < 0 or self.model.nu != 6:
            raise ValueError('Expected six named reference joints and six actuators')
        self.qids = self.model.jnt_qposadr[self.jids]
        self.vids = self.model.jnt_dofadr[self.jids]
        # MuJoCo 会把超出 ctrlrange 的目标静默截断。这里显式拒绝，避免「以为发了 1 rad、
        # 实际只执行 0.7 rad」而没有任何报错（参考模型 ctrlrange 为 ±0.7 rad）。
        limited = self.model.actuator_ctrllimited.astype(bool)
        rng = self.model.actuator_ctrlrange
        self.ctrl_lo = np.where(limited, rng[:, 0], -np.inf)
        self.ctrl_hi = np.where(limited, rng[:, 1], np.inf)

    def connect(self):
        self.reset(np.zeros(6))

    def reset(self, q):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qids] = six(q)
        self.data.ctrl[:] = q
        mujoco.mj_forward(self.model, self.data)

    def get_state(self):
        return Observation(self.data.qpos[self.qids].copy(), time.monotonic(), 'mujoco', True)

    def send_action(self, q):
        q = six(q)
        if np.any(q < self.ctrl_lo - 1e-9) or np.any(q > self.ctrl_hi + 1e-9):
            raise ValueError(f'target outside actuator ctrlrange {self.ctrl_lo.tolist()}..{self.ctrl_hi.tolist()}; '
                             'MuJoCo would silently clamp it')
        self.data.ctrl[:] = q
        for _ in range(self.steps):
            mujoco.mj_step(self.model, self.data)
        if not np.all(np.isfinite(self.data.qpos)):
            raise RuntimeError('Non-finite simulation state')

    def close(self):
        pass

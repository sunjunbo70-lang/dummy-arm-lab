"""固定相机（D435 类）离屏渲染：RGB 与深度。深度单位 m。"""
from pathlib import Path
import json
import numpy as np
import mujoco
from .task import WallTask


def render_camera(out_dir, width=640, height=480):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    env = WallTask(); env.reset()
    for _ in range(40):                        # 压到墙上再拍
        env.step([0, 0, 0.002, 0])
    r = mujoco.Renderer(env.model, height, width)
    try:
        r.update_scene(env.data, camera='d435'); rgb = r.render()
        r.enable_depth_rendering(); r.update_scene(env.data, camera='d435'); depth = r.render()
    finally:
        r.close()
    np.save(out_dir / 'rgb.npy', rgb); np.save(out_dir / 'depth_m.npy', depth)
    written = ['rgb.npy', 'depth_m.npy']
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        plt.imsave(out_dir / 'rgb.png', rgb)
        plt.imsave(out_dir / 'depth.png', np.clip(depth, 0, 1.5), cmap='viridis')
        written += ['rgb.png', 'depth.png']
    except ImportError:
        pass
    info = {'camera': 'd435 (placeholder pose)', 'resolution': [width, height],
            'depth_range_m': [float(depth.min()), float(depth.max())], 'written': written,
            'note': 'ideal pinhole render; no D435 noise model, no IR shadowing, no depth holes'}
    (out_dir / 'camera.json').write_text(json.dumps(info, indent=1, ensure_ascii=False), encoding='utf-8')
    return info

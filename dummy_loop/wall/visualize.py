"""Render actual deterministic PPO rollouts, with material state, to an offline player.

No hardware imports. Render-only geoms never enter the physics model.
"""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .rl import make_env, load_policy, Scaled
from .session import SessionConfig, run_session

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / 'experiments/2026-09-21_plaster_session_rl/raw'


def font(size):
    for path in ('C:/Windows/Fonts/msyh.ttc', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


class RolloutRenderer:
    def __init__(self, env):
        self.env = env
        env.model.vis.headlight.ambient[:] = [0.4, 0.4, 0.4]
        env.model.vis.headlight.diffuse[:] = [0.8, 0.8, 0.8]
        self.renderer = mujoco.Renderer(env.model, height=400, width=600, max_geom=10000)
        self.cameras = []
        for look, distance, azimuth, elevation in (
            ([0.21, 0, 0.17], 0.85, -25, -22),
            ([0.40, 0, 0.185], 0.34, -65, -15),
        ):
            cam = mujoco.MjvCamera()
            cam.lookat[:] = look
            cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
            self.cameras.append(cam)

    def close(self):
        self.renderer.close()

    def view(self, camera):
        env, r = self.env, self.renderer
        r.update_scene(env.data, camera=camera)
        field = env.field
        # A physical-scale height field, drawn on the outward face of the wall.
        # These are visual geoms only: no contact forces or policy inputs change.
        for row, col in np.argwhere(field.h > 1e-6):
            h = float(field.h[row, col])
            p = env.true_frame.to_world([field.cu[row, col], field.cv[row, col], -h / 2 - 0.0001])
            g = r.scene.geoms[r.scene.ngeom]
            shade = 0.49 + 0.12 * min(h / 0.004, 1)
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_BOX,
                               np.array([field.cfg.cell / 2, field.cfg.cell / 2, h / 2]),
                               p, env.true_frame.R.flatten(), np.array([shade, shade, shade * 0.96, 1.]))
            r.scene.ngeom += 1
        return Image.fromarray(r.render().copy())

    def frame(self, name, band, step, phase='stroke'):
        env, f = self.env, self.env.field
        self.cameras[1].lookat[:] = env.true_frame.to_world([env.u_centre, 0.04, -0.02])
        canvas = Image.new('RGB', (1200, 760), '#101b2b')
        canvas.paste(self.view(self.cameras[0]), (0, 75))
        canvas.paste(self.view(self.cameras[1]), (600, 75))
        d = ImageDraw.Draw(canvas)
        d.text((24, 12), '墙面抹涂 · PPO 模型实际仿真回放', font=font(26), fill='white')
        d.text((24, 47), f'{name}   |   第 {band + 1} 刀   |   {step * env.spec.dt:.2f} s（本刀）', font=font(17), fill='#87d7e8')
        for x, label in ((16, '机械臂全景'), (616, '抹刀与墙面近景 · 材料厚度按真实比例显示')):
            d.rounded_rectangle((x - 5, 85, x + 400, 116), radius=5, fill='#101b2b')
            d.text((x, 89), label, font=font(16), fill='white')
        d.text((24, 486), '墙面厚度展开图  /  目标 2 mm', font=font(20), fill='white')
        h = np.clip(f.h / 0.004, 0, 1)
        # dark -> turquoise (target) -> amber (too thick), fixed scale across runs.
        stops = np.array([[25, 39, 59], [45, 200, 184], [250, 177, 67]])
        colors = np.stack([np.interp(h, [0, .5, 1], stops[:, k]) for k in range(3)], axis=-1).astype('uint8')
        heat = Image.fromarray(colors[::-1]).resize((600, 176), Image.Resampling.NEAREST)
        canvas.paste(heat, (24, 523))
        # Show the actual lower edge on the unwrapped map.
        low, _ = env.blade_edges()
        x = 24 + (low[0] - f.u0) / (f.u1 - f.u0) * 600
        y = 523 + (f.v1 - low[1]) / (f.v1 - f.v0) * 176
        if 523 <= y <= 699:
            d.line((x - 30, y, x + 30, y), fill='white', width=3)
        d.text((24, 706), '深蓝 0 mm    青绿 2 mm    橙色 ≥4 mm    白线：刀后缘中心', font=font(15), fill='#b4c7da')
        m = f.metrics()
        lines = [f'整片覆盖  {100*m["coverage"]:.1f}%     厚度 RMS  {m["rms_error_mm"]:.3f} mm',
                 f'刀面俯仰  {np.rad2deg(env.blade_pitch()):.1f}°     后缘间隙  {max(-low[2],0)*1000:.2f} mm',
                 f'接触 / 材料合力  {env.last_force:.2f} N     刀上余料  {env.field.load*1e6:.1f} mL',
                 '换条带：重置到下一待命位（过渡运动未仿真）' if phase == 'reset' else '动作来源：已保存 PPO 权重 → 逆解 → MuJoCo 动力学',
                 'L1 仿真 · 砂浆为高度场代理模型，参数未经实测']
        for i, line in enumerate(lines):
            d.text((650, 502 + i * 43), line, font=font(17), fill='#ffd18a' if i == 3 and phase == 'reset' else '#d7e3ef')
        return canvas


def render_run(policy_path, name, out, seed=2000):
    env = make_env(seed)
    policy = Scaled(env, load_policy(env, policy_path), deterministic=True)
    renderer = RolloutRenderer(env)
    frames, rows, qpos, fields, actions, observations, bands = [], [], [], [], [], [], []
    previous_band = -1

    def capture(band, obs, action, info):
        nonlocal previous_band
        phase = 'reset' if band != previous_band else 'stroke'
        picture = renderer.frame(name, band, env.step_i, phase)
        if phase == 'reset':
            # A labelled edit between independent strokes, never fabricated motion.
            frames.extend([picture] * 15)
        frames.append(picture)
        low, _ = env.blade_edges()
        rows.append({'band': band, 'step': env.step_i, 'pitch_deg': float(np.rad2deg(env.blade_pitch())),
                     'gap_mm': float(max(-low[2], 0) * 1000), 'force_N': float(env.last_force),
                     'phase': phase, 'status': info['status']})
        qpos.append(env.data.qpos.copy()); fields.append(env.field.h.copy())
        actions.append(action.copy()); observations.append(obs.copy()); bands.append(band)
        previous_band = band

    try:
        metrics, strokes, field = run_session(
            env, policy, SessionConfig(), on_step=capture,
            on_reset=lambda band, obs: capture(band, obs, np.zeros(3), {'status': 'reset'}))
        frames.extend([renderer.frame(name, len(strokes)-1, env.step_i)] * 30)
    finally:
        renderer.close()
    encoded = []
    for picture in frames:
        buf = io.BytesIO(); picture.save(buf, format='JPEG', quality=82)
        encoded.append(base64.b64encode(buf.getvalue()).decode('ascii'))
    frames[-1].save(out / 'final.png')
    small = [f.resize((840, 532), Image.Resampling.LANCZOS) for f in frames[::2]]
    small[0].save(out / 'preview.gif', save_all=True, append_images=small[1:], duration=100, loop=0)
    np.savez_compressed(out / 'rollout.npz', qpos=qpos, thickness_m=fields, action=actions,
                        post_observation=observations, band=bands, u=field.cu, v=field.cv, inside=field.inside)
    report = {'name': name, 'policy': str(policy_path.relative_to(ROOT)) if policy_path.is_relative_to(ROOT) else str(policy_path),
              'policy_sha256': hashlib.sha256(policy_path.read_bytes()).hexdigest(),
              'seed': seed, 'evidence_level': 'L1', 'hardware_motion': False,
              'deterministic': True, 'dt_s': env.spec.dt, 'area': metrics, 'strokes': strokes,
              'env': env.describe(), 'frames': len(frames), 'rows': rows,
              'volume_balance_error_m3': field.volume_balance() - field.supplied,
              'note': 'Rendered actual physics states. Labelled holds at band cuts; transit not simulated. Material is a proxy.'}
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return {'name': name, 'images': encoded, 'metrics': metrics}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=ROOT / 'outputs/wall/ppo_visualization')
    ap.add_argument('--policy', type=Path, help='Render one custom checkpoint instead of both saved policies')
    args = ap.parse_args()
    out = args.out.resolve()
    # Each render is a new evidence bundle; never overwrite a previous result.
    out.mkdir(parents=True, exist_ok=False)
    cases = [('technique', 'PPO · 工人手法预热', EVIDENCE / 'technique_warm_start/policy.npz'),
             ('flat', 'PPO · 平刀预热', EVIDENCE / 'flat_warm_start/policy.npz')]
    if args.policy:
        cases = [('custom', 'PPO · 自选权重', args.policy.resolve())]
    runs = []
    for key, name, path in cases:
        dest = out / key; dest.mkdir()
        print(f'Rendering {key}: {path}', flush=True)
        runs.append(render_run(path, name, dest))
        print(json.dumps(runs[-1]['metrics'], ensure_ascii=False), flush=True)
    template = Path(__file__).with_name('player.html').read_text(encoding='utf-8')
    (out / 'index.html').write_text(template.replace('__ROLLOUT_DATA__', json.dumps(runs, ensure_ascii=False).replace('</', '<\\/')), encoding='utf-8')
    print(f'Open: {out / "index.html"}', flush=True)


if __name__ == '__main__':
    main()

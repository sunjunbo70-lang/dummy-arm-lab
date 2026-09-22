"""Intel RealSense D435 observation proxy for the wall height field.

The policy never receives the true field. Parameters are provisional domains that
must be replaced by measurements from the actual camera at its installed pose.
"""
import numpy as np

from .config import CycleConfig


class D435Proxy:
    def __init__(self, cfg: CycleConfig, seed=0):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.last = None

    def scan(self, true_height):
        c = self.cfg
        nv, nu = true_height.shape
        y, x = np.mgrid[-1:1:complex(nv), -1:1:complex(nu)]
        # One scan-level bias and tilt cannot be removed by averaging frames.
        bias = self.rng.normal(0, c.plane_bias_mm/1000)
        tilt_u = self.rng.normal(0, np.deg2rad(c.extrinsic_rotation_deg)) * x * c.width_m/2
        tilt_v = self.rng.normal(0, np.deg2rad(c.extrinsic_rotation_deg)) * y * c.height_m/2
        spatial = self.rng.normal(0, c.depth_noise_mm/1000, true_height.shape)
        # Approximate median fusion: independent noise shrinks, correlated bias remains.
        fused = true_height + bias + tilt_u + tilt_v + spatial/np.sqrt(c.scan_fused_frames)
        edge = np.zeros_like(true_height, bool)
        edge[:, 1:] |= np.abs(np.diff(true_height, axis=1)) > .0008
        edge[1:, :] |= np.abs(np.diff(true_height, axis=0)) > .0008
        p_invalid = c.invalid_base + c.invalid_edge*edge
        valid_count = self.rng.binomial(c.scan_fused_frames, 1-np.clip(p_invalid, 0, .95))
        confidence = valid_count / c.scan_fused_frames
        valid = valid_count >= max(1, round(.6*c.scan_fused_frames))
        measured = np.where(valid, np.clip(fused, -.002, .010), 0.0)
        self.last = {'height': measured, 'confidence': confidence, 'valid': valid,
                     'bias_m': float(bias), 'valid_fraction': float(valid.mean())}
        return self.last


def coarse_shape(shape):
    return (-(-shape[0] // 2), -(-shape[1] // 4))


def coarse_map(a):
    """2x4 mean pooling (14x56 -> 7x14 for the v0.2 wall); raw resolution remains in logs.

    Grids that are not a multiple of 2x4 are padded with their edge values first. (The
    v0.2 version returned the unpooled map in that case, which silently changed the
    observation size for any other wall shape.)"""
    pr, pc = (-a.shape[0]) % 2, (-a.shape[1]) % 4
    if pr or pc:
        a = np.pad(a, ((0, pr), (0, pc)), mode='edge')
    return a.reshape(a.shape[0]//2, 2, a.shape[1]//4, 4).mean(axis=(1, 3))

"""
========================================================================
Synthetic CFD dataset for smoke-testing the Adaptive Mesh CFD pipeline.
========================================================================

Contents
--------
make_synthetic_field  - raw numpy field generator (shockwave + vortex + turbulence)
SyntheticCFDDataset   - PyTorch Dataset wrapping make_synthetic_field

Returns samples as:
    {"input": np.ndarray [H, W, C], "target": np.ndarray [H, W, output_dim]}
"""

from typing import Dict

from torch.utils.data import Dataset

import numpy as np


def make_synthetic_field(
    n_samples: int = 9,
    channels:  int = 3,
    height:    int = 128,
    width:     int = 256,
    seed:      int = 42,
) -> np.ndarray:
    """
    Generate a synthetic CFD-like field with structured physical features.
 
    Returns np.ndarray of shape (n_samples, channels, height, width).
 
    Features per sample
    -------------------
    - Smooth sinusoidal background
    - Shockwave-like velocity discontinuity at the vertical midline
    - Localised vortex in the top-left quadrant
    - Random turbulence noise in the bottom-right quadrant
 
    Channel layout
        0 : velocity_x  (shockwave + vortex)
        1 : velocity_y  (vortex + turbulence, if channels >= 2)
        2 : velocity_z  (smooth background only, if channels >= 3)
    """
    np.random.seed(seed)
    data = np.zeros((n_samples, channels, height, width), dtype=np.float32)
    x = np.linspace(0, 2 * np.pi, width)
    y = np.linspace(0, 2 * np.pi, height)
    xx, yy = np.meshgrid(x, y)
 
    for b in range(n_samples):
        # Smooth background
        data[b, 0] = 0.3 * np.sin(xx) # gentle velocity_x wave
        if channels >= 2:
            data[b, 1] = 0.3 * np.cos(yy) # gentle velocity_y wave
        if channels >= 3:
            data[b, 2] = 0.1 * (np.sin(xx) + np.cos(yy))
 
        # Shockwave-like discontinuity at the vertical midline
        shock_col = width // 2
        data[b, 0, :, :shock_col] += 1.0 # high velocity left of front
        data[b, 0, :, shock_col:] -= 0.5 # low  velocity right of front

        # Thin transition zone (3 columns wide) to create strong gradient
        fade = np.linspace(1.0, -0.5, 6)
        for k, col in enumerate(range(shock_col - 3, shock_col + 3)):
            if 0 <= col < width:
                data[b, 0, :, col] += fade[k] * 0.8
 
        # Vortex in the top-left quadrant
        r_vortex, c_vortex = height // 8, width // 8
        ri_arr = np.arange(2 * r_vortex)[:, None]
        ci_arr = np.arange(2 * c_vortex)[None, :]
        angles = (ri_arr * ci_arr) * 0.5
        data[b, 0, :2*r_vortex, :2*c_vortex] += 0.6 * np.sin(angles)
        if channels >= 2:
            data[c_vortex] += 0.6 * np.cos(angles)
 
        # Random turbulence noise in the bottom-right quadrant
        noise_r0, noise_c0 = height // 2, width // 2
        data[b, 0, noise_r0:, noise_c0:] += np.random.randn(height - noise_r0, width - noise_c0) * 0.4
        if channels >= 2:
            data[b, 1, noise_r0:, noise_c0:] += np.random.randn(height - noise_r0, width - noise_c0) * 0.4
 
    return data
 
 
class SyntheticCFDDataset(Dataset):
    """
    PyTorch Dataset wrapping make_synthetic_field.
 
    Inputs  : all 3 channels of the generated field  [H, W, 3]
    Targets : u, v, and p=-0.5*(u²+v²)              [H, W, 3]
 
    Args
    ----
    n_samples : number of samples to generate
    height    : grid height in pixels
    width     : grid width in pixels
    seed      : random seed for reproducibility
    """
 
    def __init__(
            self, 
            n_samples: int = 64, 
            height: int = 128, 
            width: int = 256, 
            seed: int = 42
    ):
        # Generate raw field: (n_samples, 3, H, W)
        raw = make_synthetic_field(n_samples=n_samples, channels=3,
                                   height=height, width=width, seed=seed)
        # Transpose to [N, H, W, C] for consistency with CFDDataset
        raw = raw.transpose(0, 2, 3, 1)  # (N, H, W, 3)

        self.inputs  = raw.astype(np.float32)                     # [N, H, W, 3]  - all channels as input
        # Synthetic targets: u=ch0, v=ch1, p=-0.5*(u²+v²)
        u = raw[..., 0]
        v = raw[..., 1]
        p = -0.5 * (u ** 2 + v ** 2)
        self.targets = np.stack([u, v, p], axis=-1).astype(np.float32)  # [N, H, W, 3]

        self.input_channels = self.inputs.shape[-1]
        self.output_dim     = self.targets.shape[-1]
        self.H, self.W      = height, width
        print(f"SyntheticCFDDataset: {n_samples} samples  |  "
              f"grid {height}x{width}  |  input_channels={self.input_channels}  "
              f"output_dim={self.output_dim}")

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Dict:
        return {"input": self.inputs[idx], "target": self.targets[idx]}
    

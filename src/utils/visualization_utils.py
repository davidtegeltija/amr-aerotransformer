from typing import Optional

from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import numpy as np


def _channel_image(data: np.ndarray, channel_idx: int = 0) -> np.ndarray:
    """
    Extract a 2-D image from a physical field for display.

    data : (C, H, W) or (H, W, C) or (H, W)
    """
    if data.ndim == 2:
        return data.astype(float)

    if data.ndim == 3:
        if data.shape[0] < data.shape[1] and data.shape[0] < data.shape[2]:
            # (C, H, W)
            return data[channel_idx].astype(float)
        else:
            # (H, W, C)
            return data[:, :, channel_idx].astype(float)

    raise ValueError(f"Unsupported data shape {data.shape}")


def _sum_image(data: np.ndarray) -> np.ndarray:
    """Sum all channels into a single 2-D image for background display."""
    if data.ndim == 2:
        return data.astype(float)

    if data.ndim == 3:
        if data.shape[0] < data.shape[1] and data.shape[0] < data.shape[2]:
            return data.sum(axis=0).astype(float)
        else:
            return data.sum(axis=2).astype(float)

    raise ValueError(f"Unsupported data shape {data.shape}")


def _color_map(
    values: np.ndarray,
    cmap_name: str,
    *,
    alpha: float = 1.0,
    dmin: Optional[float] = None,
    dmax: Optional[float] = None,
    n_levels: Optional[int] = None,
):
    norm = Normalize(
        vmin=dmin if dmin is not None else values.min(),
        vmax=dmax if dmax is not None else values.max(),
    )
    cmap = plt.get_cmap(cmap_name, n_levels)
    colors = cmap(norm(values))
    colors[..., 3] = alpha
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    return cmap, norm, colors
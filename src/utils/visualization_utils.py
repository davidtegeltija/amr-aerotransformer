from datetime import datetime
from pathlib import Path
from typing import Optional

from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
import numpy as np


def save_plot(save_path: str | Path, figure: Figure, dpi: int = 150, use_date_subfolder: bool = False) -> None:
    """ Save a matplotlib figure to disk under a date-organised subfolder """
    save_path = Path(save_path)

    # Check for figure type. Default is PNG
    if save_path.suffix == "":
        save_path = save_path.with_suffix(".png")

    # Add a timestamp
    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M")
    save_path = save_path.with_name(f"{save_path.stem}_{timestamp}{save_path.suffix}")

    # Add a current date subfolder for better organization
    if use_date_subfolder:
        subfolder = datetime.now().strftime("%d-%m-%Y")
        save_path = save_path.parent / subfolder / save_path.name

    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, dpi=dpi, bbox_inches="tight")
    print(f"SUCCESS: Plot saved to {save_path}")


def channel_image(data: np.ndarray, channel_idx: int = 0) -> np.ndarray:
    """ Extract a 2-D image of a single channel from a physical field for display """
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


def sum_image(data: np.ndarray) -> np.ndarray:
    """Sum all channels into a single 2-D image for background display."""
    if data.ndim == 2:
        return data.astype(float)

    if data.ndim == 3:
        if data.shape[0] < data.shape[1] and data.shape[0] < data.shape[2]:
            return data.sum(axis=0).astype(float)
        else:
            return data.sum(axis=2).astype(float)

    raise ValueError(f"Unsupported data shape {data.shape}")


def color_map(
    values: np.ndarray,
    cmap_name: str,
    *,
    alpha: float = 1.0,
    dmin: Optional[float] = None,
    dmax: Optional[float] = None,
    n_levels: Optional[int] = None,
):
    """Build a normalized matplotlib colormap and per-value RGBA colors.

    Args:
        values:    array whose values are mapped to colors
        cmap_name: matplotlib colormap name (e.g. "viridis", "plasma")
        alpha:     opacity applied to all returned RGBA colors
        dmin:      lower bound for normalization (defaults to values.min())
        dmax:      upper bound for normalization (defaults to values.max())
        n_levels:  number of discrete colormap levels (None for continuous)

    Returns:
        (cmap, norm, colors) where colors has shape values.shape + (4,).
    """
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
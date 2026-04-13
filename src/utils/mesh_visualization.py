"""
========================================================================
Mesh and quadtree visualization utilities.
========================================================================

Functions
---------
visualize_mesh           : Overlay the adaptive quadtree mesh on a 2D grid channel.
visualize_mesh_by_depth  : Show one subplot per depth level with patches at that depth highlighted.
visualize_metric_heatmap : Show a heatmap of a chosen physics metric on the original grid.
visualize_patch_features : Reconstruct and display the field from averaged patch features.
"""

from datetime import datetime
from typing import List, Optional, Tuple

from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

from src.amr.quadtree_tokenizer import QuadNode


def visualize_mesh(
    sample: np.ndarray,
    mesh: List[QuadNode],
    *,
    channel: int = 0,
    title: str = "Adaptive Mesh",
    show: bool = True,
    save_path: Optional[str] = None,
) -> Figure:

    H, W, C = sample.shape
    if channel > C:
        raise ValueError(f"Channel {channel} out of range for {C}-channel input")

    fig, ax = plt.subplots(figsize=(6, 10))

    channel_data = _channel_image(sample, channel)
    ax.imshow(channel_data, cmap="viridis", origin="upper")

    # Discrete depth colormap — one distinct color band per depth level
    depths = [p.depth for p in mesh]
    min_d = min(depths) if depths else 0
    max_d = max(depths) if depths else 1
    n_levels = max_d - min_d + 1
    depth_cmap = plt.get_cmap("plasma", n_levels)
    depth_norm = Normalize(vmin=min_d, vmax=max(max_d, min_d + 1))

    # Rectangle overlays
    rects = []
    for patch in mesh:
        r0, c0, r1, c1 = patch.bbox[0], patch.bbox[1], patch.bbox[2], patch.bbox[3]
        height = r1 - r0
        width  = c1 - c0

        # imshow places pixel (0,0) centered at coordinate 0.5
        rects.append(patches.Rectangle((c0 - 0.5, r0 - 0.5), width, height))

    # PatchCollection is a real mappable — the colorbar attaches directly to the
    # border rectangles rather than to a dummy object.
    pc = PatchCollection(rects, cmap=depth_cmap, norm=depth_norm, linewidth=0.75, alpha=1)
    pc.set_array(np.array(depths))
    pc.set_facecolor("none")
    ax.add_collection(pc)

    # Discrete colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="10%", pad=0.15)
    cbar = fig.colorbar(pc, cax=cax)
    cbar.set_label("Quadtree depth")
    cbar.set_ticks(range(min_d, max_d + 1))

    n_tokens = len(mesh)
    uniform_tokens = H * W  # if we tokenised every pixel
    reduction = 1.0 - n_tokens / max(uniform_tokens, 1)
    ax.set_title(f"{title}\n{n_tokens} patches (uniform would be {uniform_tokens}  |  reduction ≈ {reduction*100:.0f}%)", fontsize=11)
    ax.set_xlabel("Column (x)")
    ax.set_ylabel("Row (y)")
    # ax.set_xlim(0, W)
    # ax.set_ylim(H, 0)  # image convention: row 0 at top

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[visualization] Saved → {save_path}")

    if show:
        plt.show()

    return fig

def visualize_mesh_by_depth(
    sample: np.ndarray,
    mesh: List[QuadNode],
    *,
    channel: int = 0,
    max_cols: int = 4,
    title: str = "Adaptive Mesh by Depth",
    show: bool = True,
    save_path: Optional[str] = None,
) -> None:

    depths = sorted(set(p.depth for p in mesh))
    n_depths = len(depths)

    cols = min(n_depths, max_cols)
    rows = (n_depths + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * max_cols, rows * max_cols), squeeze=False)
    channel_data = _channel_image(sample, channel)

    depth_to_patches = {d: [] for d in depths}
    for patch in mesh:
        depth_to_patches[patch.depth].append(patch)

    for ax_idx, depth in enumerate(depths):
        row = ax_idx // cols
        col = ax_idx % cols
        ax = axes[row][col]

        ax.imshow(channel_data, cmap="viridis", origin="upper")
        for patch in depth_to_patches[depth]:
            r0, c0, r1, c1 = patch.bbox[0], patch.bbox[1], patch.bbox[2], patch.bbox[3]
            height = r1 - r0
            width  = c1 - c0
            rect = patches.Rectangle(
                (c0, r0), width, height,
                linewidth=0.75,
                edgecolor="white",
                facecolor="none",
            )
            ax.add_patch(rect)

        ax.set_title(f"Depth {depth}  ({len(depth_to_patches[depth])} patches)")
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused axes
    for ax_idx in range(n_depths, rows * cols):
        row = ax_idx // cols
        col = ax_idx % cols
        axes[row][col].set_visible(False)

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[visualization] Saved → {save_path}")

    if show:
        plt.show()


def visualize_metric_heatmap(
    sample: np.ndarray,
    mesh: List[QuadNode],
    *,
    metric_name: str = "velocity_gradient",
    title: Optional[str] = "Metric Heatmap",
    show: bool = True,
    save_path: Optional[str] = None,
) -> None:
    """
    metric_name : str   Defined by the RefinementCriteria used for the mesh construction.
                        Look at the definition of compute_enabled_metrics() to see possible 
                        metric names
    """
    # Determine domain shape
    if sample.ndim == 3:
        if sample.shape[0] < sample.shape[1]:
            H, W = sample.shape[1], sample.shape[2]
        else:
            H, W = sample.shape[0], sample.shape[1]
    else:
        H, W = sample.shape

    metric_img = np.full((H, W), np.nan)

    for patch in mesh:
        r0, c0, r1, c1 = patch.bbox[0], patch.bbox[1], patch.bbox[2], patch.bbox[3]
        val = patch.metrics.get(metric_name, np.nan)
        metric_img[r0:r1, c0:c1] = val

    fig, axes = plt.subplots(1, 2)

    # Left: metric heatmap
    ax = axes[0]
    im = ax.imshow(metric_img, cmap="hot", origin="upper")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    ax.set_title(title or f"Metric: {metric_name}")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    # Right: overlay on background
    ax2 = axes[1]
    bg = _sum_image(sample)
    ax2.imshow(bg, cmap="gray", origin="upper", alpha=0.5)
    im2 = ax2.imshow(metric_img, cmap="hot", origin="upper", alpha=0.6)
    plt.colorbar(im2, ax=ax2, fraction=0.04, pad=0.04)
    ax2.set_title("Metric overlay")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[visualization] Saved → {save_path}")

    if show:
        plt.show()


def visualize_patch_features(
    sample: np.ndarray,
    mesh: List[QuadNode],
    *,
    channel: int = 0,
    title: str = "Patch Feature Reconstruction",
    show: bool = True,
    save_path: Optional[str] = None,
) -> None:
    
    # Determine domain shape
    if sample.ndim == 3:
        if sample.shape[0] < sample.shape[1]:
            H, W = sample.shape[1], sample.shape[2]
        else:
            H, W = sample.shape[0], sample.shape[1]
    else:
        H, W = sample.shape

    # Original
    original = _channel_image(sample, channel)

    # Reconstructed channel values
    reconstructed = np.full((H, W), np.nan)
    for patch in mesh:
        r0, c0, r1, c1 = patch.bbox[0], patch.bbox[1], patch.bbox[2], patch.bbox[3]
        mean_features = patch.features
        if len(mean_features) >= channel:
            reconstructed[r0:r1, c0:c1] = mean_features[channel]

    fig, axes = plt.subplots(1, 3, figsize=(12, 6))

    # Original field
    ax0 = axes[0]
    im0 = ax0.imshow(original, cmap="plasma", origin="upper")
    # plt.colorbar(im0, ax=axes[0], fraction=0.04, pad=0.04)
    ax0.set_title(f"Original field  (ch {channel})")
    fig.colorbar(im0, ax=ax0)

    # Reconstructed from patches
    ax1 = axes[1]
    im1 = ax1.imshow(reconstructed, cmap="plasma", origin="upper")
    # plt.colorbar(im1, ax=axes[1], fraction=0.04, pad=0.04)
    ax1.set_title(f"AMR reconstruction  ({len(mesh)} patches)")
    fig.colorbar(im1, ax=ax1)

    # Difference
    ax2 = axes[2]
    diff = np.abs(original - reconstructed)
    im2 = ax2.imshow(diff, cmap="Reds", origin="upper")
    # plt.colorbar(im2, ax=axes[2], fraction=0.04, pad=0.04)
    ax2.set_title("Absolute error")
    fig.colorbar(im2, ax=ax2)

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[visualization] Saved → {save_path}")

    if show:
        plt.show()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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

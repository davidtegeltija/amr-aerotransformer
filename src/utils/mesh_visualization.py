"""
Mesh and quadtree visualization utilities

Functions
---------
plot_mesh           : Overlay the adaptive quadtree mesh on a 2D grid channel
plot_mesh_by_depth  : Show one subplot per depth level with patches at that depth highlighted
plot_metric_heatmap : Show a heatmap of a chosen physics metric on the original grid
plot_patch_features : Reconstruct and display the field from averaged patch features
plot_score_map      : Render a per-pixel refinement score as a heatmap (optionally over geometry)
animate_mesh_refinement  : Depth-by-depth animated GIF of the quadtree build (requires Pillow)
"""

from typing import List, Optional

from matplotlib.collections import PatchCollection
from matplotlib.figure import Figure
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

from src.amr.quadtree import QuadNode
from src.utils.visualization_utils import save_plot, channel_image, sum_image, color_map


def plot_mesh(
    sample: np.ndarray,
    mesh: List[QuadNode],
    *,
    channel: int = 0,
    title: str = "Adaptive Mesh",
    show: bool = True,
    save_path: Optional[str] = None,
) -> Figure:
    """Overlay the adaptive quadtree mesh on a 2D grid channel."""
    H, W, C = sample.shape
    if channel > C:
        raise ValueError(f"Channel {channel} out of range for {C}-channel input")

    fig, ax = plt.subplots(figsize=(6, 10))

    channel_data = channel_image(sample, channel)
    ax.imshow(channel_data, cmap="viridis", origin="upper")

    depths = [p.depth for p in mesh]
    min_d = min(depths) if depths else 0
    max_d = max(depths) if depths else 1
    cmap, norm, _ = color_map(np.array(depths), "plasma", dmin=min_d, dmax=max(max_d, min_d + 1), n_levels=max_d - min_d + 1)

    # Rectangle overlays
    rects = []
    for patch in mesh:
        r0, c0, r1, c1 = patch.bbox[0], patch.bbox[1], patch.bbox[2], patch.bbox[3]
        height = r1 - r0
        width  = c1 - c0

        # imshow places pixel (0,0) centered at coordinate 0.5
        rects.append(patches.Rectangle((c0 - 0.5, r0 - 0.5), width, height))

    pc = PatchCollection(rects, cmap=cmap, norm=norm, linewidth=0.75, alpha=1)
    pc.set_array(np.array(depths))
    pc.set_facecolor("none")
    ax.add_collection(pc)

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
        save_plot(save_path, fig, use_date_subfolder=True)

    if show:
        plt.show()

    return fig

def plot_mesh_by_depth(
    sample: np.ndarray,
    mesh: List[QuadNode],
    *,
    channel: int = 0,
    max_cols: int = 4,
    title: str = "Adaptive Mesh by Depth",
    show: bool = True,
    save_path: Optional[str] = None,
) -> None:
    """Show one subplot per depth level with patches at that depth highlighted."""
    depths = sorted(set(p.depth for p in mesh))
    n_depths = len(depths)

    cols = min(n_depths, max_cols)
    rows = (n_depths + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * max_cols, rows * max_cols), squeeze=False)
    channel_data = channel_image(sample, channel)

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
        save_plot(save_path, fig, use_date_subfolder=True)

    if show:
        plt.show()


def plot_metric_heatmap(
    sample: np.ndarray,
    mesh: List[QuadNode],
    *,
    metric_name: str = "velocity_gradient",
    title: Optional[str] = "Metric Heatmap",
    show: bool = True,
    save_path: Optional[str] = None,
) -> None:
    """Show a heatmap of a chosen physics metric on the original grid.

    'metric_name' must match one of the metrics produced by the RefinementCriteria
    used for mesh construction (see `compute_enabled_metrics()` for valid names).
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
    
    if np.isnan(metric_img).all():
        print(f"WARNING: The plot '{title}' is empty. The mesh was created without this metric: {metric_name}")
        return

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
    bg = sum_image(sample)
    ax2.imshow(bg, cmap="gray", origin="upper", alpha=0.5)
    im2 = ax2.imshow(metric_img, cmap="hot", origin="upper", alpha=0.6)
    plt.colorbar(im2, ax=ax2, fraction=0.04, pad=0.04)
    ax2.set_title("Metric overlay")

    plt.tight_layout()

    if save_path:
        save_plot(save_path, fig, use_date_subfolder=True)

    if show:
        plt.show()


def plot_patch_features(
    sample: np.ndarray,
    mesh: List[QuadNode],
    *,
    channel: int = 0,
    title: str = "Patch Feature Reconstruction",
    show: bool = True,
    save_path: Optional[str] = None,
) -> None:
    """Reconstruct and display the field from averaged patch features."""
    # Determine domain shape
    if sample.ndim == 3:
        if sample.shape[0] < sample.shape[1]:
            H, W = sample.shape[1], sample.shape[2]
        else:
            H, W = sample.shape[0], sample.shape[1]
    else:
        H, W = sample.shape

    # Original
    original = channel_image(sample, channel)

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
        save_plot(save_path, fig, use_date_subfolder=True)

    if show:
        plt.show()


def plot_score_map(
    score_map: np.ndarray,
    geometry: Optional[np.ndarray] = None,
    *,
    cmap: str = "viridis",
    vmin: float = 0.0,
    vmax: float = 1.0,
    title: str = "Per-Pixel Refinement Score Map",
    show: bool = True,
    save_path: Optional[str] = None,
) -> Figure:
    """Render a per-pixel refinement score as a heatmap, optionally over geometry."""
    if score_map.ndim != 2:
        raise ValueError(f"score_map must be 2-D (H, W); got shape {score_map.shape}")

    H, W = score_map.shape
    fig, ax = plt.subplots(figsize=(6, 6 * H / max(W, 1)))

    heatmap_alpha = 1.0
    if geometry is not None:
        if geometry.ndim != 3 or geometry.shape[-1] < 3:
            raise ValueError(
                f"geometry must be [H, W, 3] with xyz channels; got shape {geometry.shape}"
            )
        bg = geometry[..., 2].astype(float)
        ax.imshow(bg, cmap="gray", origin="upper", alpha=0.3, aspect="auto")
        heatmap_alpha = 0.7

    im = ax.imshow(score_map, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper", alpha=heatmap_alpha, aspect="auto")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.15)
    fig.colorbar(im, cax=cax, label="Score")

    ax.set_title(title)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    plt.tight_layout()

    if save_path:
        save_plot(save_path, fig, use_date_subfolder=True)

    if show:
        plt.show()

    return fig


# ---------------------------------------------------------------------------
# Depth-by-depth refinement animation
# ---------------------------------------------------------------------------

def animate_mesh_refinement(
    grid: np.ndarray,
    token_list: List[QuadNode],
    channel: int = 0,
    fps: int = 2,
    save_path: str = "refinement.gif",
) -> None:
    """Depth-by-depth animated GIF of the quadtree build (requires Pillow)."""
    try:
        from PIL import Image
    except ImportError:
        print("Pillow not installed. Skipping animation. Run: pip install Pillow")
        return

    import io

    H, W, _ = grid.shape
    max_depth = max(t.depth for t in token_list)
    frames = []

    for d in range(max_depth + 1):
        visible = [t for t in token_list if t.depth <= d]
        fig = plot_mesh(
            grid, visible,
            channel=channel,
            title=f"Quadtree refinement  -  depth ≤ {d}",
            show=False,
        )
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).copy())

    frames[0].save(
        save_path,
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=int(1000 / fps),
    )
    print(f"Saved animation to {save_path}")


if __name__ == "__main__":
    import os

    H, W = 256, 128
    rows = np.arange(H).reshape(H, 1)
    cols = np.arange(W).reshape(1, W)
    r0, c0 = H / 2, W / 2
    sigma_r, sigma_c = H / 6, W / 6
    score_map = np.exp(
        -(((rows - r0) ** 2) / (2 * sigma_r ** 2) + ((cols - c0) ** 2) / (2 * sigma_c ** 2))
    )
    score_map = score_map.astype(np.float32)

    save_path = os.path.join("outputs", "phase2_score_map_test.png")
    ax = plot_score_map(score_map, title="synthetic")
    ax.figure.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[visualization] Saved -> {save_path}")
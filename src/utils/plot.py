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


Flow field and token-level visualization utilities.

Functions
---------
plot_flow_comparison  : Side-by-side comparison of ground truth vs predicted flow fields
plot_token_statistics : Histogram of token counts per sample and (optionally) cell size distribution.
plot_3d_prediction    : 3D surface rendering of predicted fields over wing geometry
"""

from datetime import datetime
from pathlib import Path
from typing import List, Optional

from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from src.amr.quadtree import QuadNode


# ---------------------------------------------------------------------------
# Training Visualization
# ---------------------------------------------------------------------------
def plot_loss_curves(
    train_loss_history: List[float],
    val_loss_history: List[float],
    epochs: int,
    show: bool = False,
    save_path: Optional[str | Path] = None
):
    """ Plot the training and validation loss curves for training diagnostics """
    train_steps = np.arange(1, epochs + 1, 1)

    fig = plt.figure(figsize=(10, 4))
    plt.plot(train_steps, train_loss_history, label="train_loss")
    plt.plot(train_steps, val_loss_history, label="val_loss")
    plt.legend()
    plt.title(f"Training Loss Curves for {epochs} Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)

    if save_path:
        save_plot(save_path, fig)

    if show:
        plt.show()


# ---------------------------------------------------------------------------
# Mesh Visualization
# ---------------------------------------------------------------------------
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


# Depth-by-depth refinement animation
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


# ---------------------------------------------------------------------------
# Prediction Visualization
# ---------------------------------------------------------------------------
def plot_flow_comparison(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    channel_names: Optional[List[str]] = None,
    figsize_per_col: float = 4.0,
    title: str = "Ground truth  vs  Prediction",
    show: bool = False,
    save_path: Optional[str] = None,
) -> None:
    """Side-by-side comparison of ground truth vs predicted flow fields."""
    if ground_truth.shape != prediction.shape:
        raise ValueError(f"Shape mismatch: ground_truth={ground_truth.shape} prediction={prediction.shape}")
    
    output_channels = ground_truth.shape[-1]
    names = channel_names or [f"channel {i}" for i in range(output_channels)]

    # 3 cols per output channel: GT | Pred | |Error|
    n_cols = 3
    n_rows = output_channels
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * figsize_per_col, n_rows * figsize_per_col))

    if n_rows == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(title, fontsize=13, y=1.01)

    for r in range(output_channels):
        gt = ground_truth[..., r]
        pred = prediction[..., r]
        err  = np.abs(gt - pred)

        vmin = min(gt.min(), pred.min())
        vmax = max(gt.max(), pred.max())

        for col, (data, label) in enumerate(zip([gt, pred, err], ["Ground truth", "Prediction", "|Error|"])):
            ax = axes[r, col]
            vm = err.max() if col == 2 else vmax
            im = ax.imshow(
                data, origin="upper", cmap="hot" if col == 2 else "RdBu_r",
                vmin=0 if col == 2 else vmin,
                vmax=vm,
                interpolation="bilinear",
            )
            ax.set_title(f"{names[r]}  -  {label}", fontsize=9)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()

    if save_path:
        save_plot(save_path, fig, use_date_subfolder=True)

    if show:
        plt.show()


def plot_token_statistics(
    token_counts: List[int],
    cell_sizes: Optional[List[float]] = None,
    title: str = "Token statistics",
    show: bool = False,
    save_path: Optional[str] = None,
) -> None:
    """Histogram of token counts per sample and, optionally, cell size distribution."""
    n_plots = 2 if cell_sizes else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    axes[0].hist(token_counts, bins=30, color="steelblue", edgecolor="white")
    axes[0].set_xlabel("Tokens per sample")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"{title}\nmean={np.mean(token_counts):.0f}  min={min(token_counts)}  max={max(token_counts)}")

    if cell_sizes:
        axes[1].hist(cell_sizes, bins=40, color="coral", edgecolor="white", log=True)
        axes[1].set_xlabel("Normalised cell size")
        axes[1].set_ylabel("Count (log)")
        axes[1].set_title("Cell size distribution")

    plt.tight_layout()

    if save_path:
        save_plot(save_path, fig, use_date_subfolder=True)

    if show:
        plt.show()


def plot_3d_prediction(
    geom: np.ndarray,
    prediction: np.ndarray,
    *,
    title: str = "Adaptive Mesh",
    show: bool = True,
    save_path: Optional[str] = None,
) -> None:
    """3D surface rendering of predicted fields over wing geometry."""
    fig = plt.figure()
    ax: Axes3D = fig.add_subplot(projection="3d")

    elev = 68; azim =120 

    _, _, colors = color_map(prediction[..., 0], "gist_rainbow", alpha=1, dmin=-1, dmax=1)    # cp
    x = geom[:, :, 0]
    y = geom[:, :, 1]
    z = geom[:, :, 2]
    ax.plot_surface(x, y, z, facecolors=colors, edgecolor="none", rstride=1, cstride=3, shade=True)
    ax.view_init(elev=elev, azim=azim)

    # Remove background planes (panes)
    ax.set_axis_off()
    # ax.grid(False)
    # ax.xaxis.pane.set_visible(False)
    # ax.yaxis.pane.set_visible(False)
    # ax.zaxis.pane.set_visible(False)

    plt.title(title)
    plt.tight_layout()

    if save_path:
        save_plot(save_path, fig, use_date_subfolder=True)

    if show:
        plt.show()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def save_plot(save_path: str | Path, figure: Figure, dpi: int = 150, use_date_subfolder: bool = False) -> None:
    """ Save a matplotlib figure to disk under a date-organised subfolder """
    save_path = Path(save_path)

    # Check for figure type. Default is PNG
    if save_path.suffix == "":
        save_path = save_path.with_suffix(".png")

    # Add a timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    save_path = save_path.with_name(f"{timestamp}_{save_path.stem}{save_path.suffix}")

    # Add a current date subfolder for better organization
    if use_date_subfolder:
        subfolder = datetime.now().strftime("%Y-%m-%d")
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


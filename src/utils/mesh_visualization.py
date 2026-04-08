"""
========================================================================
Mesh and quadtree visualization utilities.
========================================================================

Functions
---------
visualize_mesh           : Display a physical field with the adaptive quadtree overlaid.
visualize_mesh_by_depth  : Show one subplot per depth level.
visualize_quadtree_mesh  : Overlay the adaptive quadtree mesh on a 2D grid channel.
visualize_metric_heatmap : Show a heatmap of a chosen physics metric on the original grid.
visualize_patch_features : Reconstruct and display the field from averaged patch features.
"""

from datetime import datetime
from typing import List, Optional, Tuple

from matplotlib.colors import Normalize
from matplotlib.figure import Figure
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from src.amr.quadtree_tokenizer import QuadNode



def visualize_mesh(
    sample: np.ndarray,
    mesh: List[dict],
    *,
    title: str = "Adaptive Mesh",
    color_by_depth: bool = True,
    background_channel: int = 0,
    background_cmap: str = "viridis",
    edge_cmap: str = "plasma",
    linewidth: float = 0.6,
    alpha: float = 0.85,
    figsize: Tuple[int, int] = (10, 6),
    show: bool = True,
    save_path: Optional[str] = None,
) -> Figure:
    """
    Display a physical field with the adaptive quadtree mesh overlaid.

    Parameters
    ----------
    sample : np.ndarray
        Single physical field.  Accepted shapes: (C, H, W), (H, W, C), (H, W).
    mesh : List[dict]
        Output of build_adaptive_mesh().
    title : str
        Figure title.
    color_by_depth : bool
        If True, colour patch outlines by depth level; otherwise use blue.
    background_channel : int
        Which channel to use as the background image.
    background_cmap : str
        Colormap for the background field.
    edge_cmap : str
        Colormap used when color_by_depth=True.
    linewidth : float
        Thickness of patch outlines.
    alpha : float
        Opacity of patch outlines.
    figsize : tuple
        Matplotlib figure size.
    show : bool
        Call plt.show() at the end.
    save_path : str or None
        If given, save the figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # --- Background image -----------------------------------------------
    bg = _channel_image(sample, background_channel)
    im = ax.imshow(bg, cmap=background_cmap, origin="upper")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)

    # --- Depth colour mapping -------------------------------------------
    depths = [p["depth"] for p in mesh]
    get_color = _depth_colormap(depths, edge_cmap) if color_by_depth else None

    # --- Draw patches ---------------------------------------------------
    for patch in mesh:
        r0, c0, r1, c1 = patch["bbox"]
        h = r1 - r0
        w = c1 - c0
        color = get_color(patch["depth"]) if get_color is not None else "royalblue"

        rect = patches.Rectangle(
            (c0, r0), w, h,       # matplotlib: (x=col, y=row)
            linewidth=linewidth,
            edgecolor=color,
            facecolor="none",
            alpha=alpha,
        )
        ax.add_patch(rect)

    # --- Legend (depth → colour) ----------------------------------------
    if color_by_depth and depths:
        unique_depths = sorted(set(depths))
        get_c = _depth_colormap(unique_depths, edge_cmap)
        legend_handles = [
            patches.Patch(facecolor=get_c(d), label=f"Depth {d}")
            for d in unique_depths
        ]
        ax.legend(
            handles=legend_handles,
            loc="upper right",
            fontsize=7,
            framealpha=0.7,
        )

    ax.set_title(f"{title}  ({len(mesh)} patches)")
    ax.set_xlabel("Column (x)")
    ax.set_ylabel("Row (y)")
    ax.set_aspect("equal")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[visualization] Saved → {save_path}")

    if show:
        plt.show()

    return fig



def visualize_mesh_by_depth(
    sample: np.ndarray,
    mesh: List[dict],
    *,
    title: str = "Adaptive Mesh by Depth",
    background_channel: int = 0,
    background_cmap: str = "viridis",
    linewidth: float = 0.8,
    figsize_per_col: Tuple[int, int] = (4, 4),
    max_cols: int = 4,
    show: bool = True,
    save_path: Optional[str] = None,
) -> Figure:
    """
    Show one subplot per depth level with patches at that depth highlighted.

    Parameters
    ----------
    (see visualize_mesh for shared params)
    max_cols : int
        Maximum subplots per row.
    """
    depths = sorted(set(p["depth"] for p in mesh))
    n_depths = len(depths)

    cols = min(n_depths, max_cols)
    rows = (n_depths + cols - 1) // cols

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(cols * figsize_per_col[0], rows * figsize_per_col[1]),
        squeeze=False,
    )
    bg = _channel_image(sample, background_channel)

    depth_to_patches = {d: [] for d in depths}
    for p in mesh:
        depth_to_patches[p["depth"]].append(p)

    for ax_idx, depth in enumerate(depths):
        row = ax_idx // cols
        col = ax_idx % cols
        ax = axes[row][col]

        ax.imshow(bg, cmap=background_cmap, origin="upper")
        for patch in depth_to_patches[depth]:
            r0, c0, r1, c1 = patch["bbox"]
            rect = patches.Rectangle(
                (c0, r0), c1 - c0, r1 - r0,
                linewidth=linewidth,
                edgecolor="crimson",
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

    return fig


def visualize_quadtree_mesh(
    grid: np.ndarray,
    token_list: List[QuadNode],
    channel: int = 0,
    cmap: str = "viridis",
    edge_color: str = "white",
    edge_alpha: float = 0.6,
    edge_lw: float = 0.5,
    title: str = "Adaptive Quadtree Mesh",
    figsize: Tuple[float, float] = (9, 7),
    save_path: Optional[str] = None,
    show: bool = False,
) -> Figure:
    """
    Overlay the adaptive quadtree cell boundaries on top of a heatmap of
    one input channel.

    Parameters
    ----------
    grid       : [H, W, C] numpy array
    token_list : list of QuadNode objects
    channel    : which input channel to display as background image
    cmap       : matplotlib colourmap for the background
    edge_color : colour of the cell boundary lines
    edge_alpha : opacity of cell boundaries
    edge_lw    : line width of cell boundaries
    title      : figure title
    figsize    : matplotlib figure size
    save_path  : if given, save figure to this path
    show       : if True, call plt.show()

    Returns
    -------
    matplotlib Figure
    """
    H, W, C = grid.shape
    assert channel < C, f"Channel {channel} out of range for {C}-channel input"

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(
        grid[:, :, channel],
        origin='upper',
        cmap=cmap,
        vmin=grid[:, :, channel].min(),
        vmax=grid[:, :, channel].max(),
        interpolation='nearest',
    )

    # Colour cells by depth for visual clarity
    max_depth = max(t.depth for t in token_list) if token_list else 1
    depth_cmap = plt.get_cmap("plasma")

    for tok in token_list:
        r0, r1 = tok.r0, tok.r1
        c0, c1 = tok.c0, tok.c1
        width  = c1 - c0
        height = r1 - r0

        # Coloured transparent fill (depth-coded)
        fill_color = depth_cmap(tok.depth / max(max_depth, 1))
        rect_fill = patches.Rectangle(
            (c0, r0), width, height,
            linewidth=0, facecolor=(*fill_color[:3], 0.08),
        )
        ax.add_patch(rect_fill)

        # Cell boundary
        rect_border = patches.Rectangle(
            (c0, r0), width, height,
            linewidth=edge_lw,
            edgecolor=edge_color,
            facecolor='none',
            alpha=edge_alpha,
        )
        ax.add_patch(rect_border)

    # Colourbar for depth
    sm = plt.cm.ScalarMappable(
        cmap=depth_cmap,
        norm=Normalize(vmin=0, vmax=max_depth)
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Quadtree depth")

    n_tokens = len(token_list)
    uniform_tokens = H * W  # if we tokenised every pixel
    reduction = 1.0 - n_tokens / max(uniform_tokens, 1)
    ax.set_title(
        f"{title}\n"
        f"N = {n_tokens} tokens  (uniform would be {H}×{W} = {H*W}  |  reduction ≈ {reduction*100:.0f}%)",
        fontsize=11
    )
    ax.set_xlabel("col (pixels)")
    ax.set_ylabel("row (pixels)")
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)  # image convention: row 0 at top

    plt.tight_layout()
    if save_path:
        timestamp = datetime.now().strftime("%d_%m_%Y-%H_%M")
        fig.savefig(f"{save_path}/mesh_visualisation-{timestamp}.png", dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    return fig


def visualize_metric_heatmap(
    sample: np.ndarray,
    mesh: List[dict],
    metric_name: str = "velocity_gradient",
    *,
    title: Optional[str] = None,
    cmap: str = "hot",
    show: bool = True,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 5),
) -> Figure:
    """
    Reconstruct a 2-D heatmap of a physics metric value per patch.

    The colour of each patch is determined by its stored physics metric.

    Parameters
    ----------
    sample : np.ndarray  Physical field (for determining H, W).
    mesh   : List[dict]  Output of build_adaptive_mesh().
    metric_name : str    One of: velocity_gradient, vorticity, momentum,
                         kh_shear, variance, entropy.
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
        r0, c0, r1, c1 = patch["bbox"]
        val = patch["metrics"].get(metric_name, np.nan)
        metric_img[r0:r1, c0:c1] = val

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left: metric heatmap
    ax = axes[0]
    im = ax.imshow(metric_img, cmap=cmap, origin="upper")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    ax.set_title(title or f"Metric: {metric_name}")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    # Right: overlay on background
    ax2 = axes[1]
    bg = _sum_image(sample)
    ax2.imshow(bg, cmap="gray", origin="upper", alpha=0.5)
    im2 = ax2.imshow(metric_img, cmap=cmap, origin="upper", alpha=0.6)
    plt.colorbar(im2, ax=ax2, fraction=0.04, pad=0.04)
    ax2.set_title("Metric overlay")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[visualization] Saved → {save_path}")

    if show:
        plt.show()

    return fig


def visualize_patch_features(
    sample: np.ndarray,
    mesh: List[dict],
    channel_idx: int = 0,
    *,
    title: str = "Patch Feature Reconstruction",
    cmap: str = "plasma",
    show: bool = True,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5),
) -> Figure:
    """
    Reconstruct the field from averaged patch features and display alongside
    the original field.

    Each patch's spatial extent is filled with its mean feature value for the
    selected channel.

    Parameters
    ----------
    channel_idx : int  Which channel to reconstruct.
    """
    # Determine domain shape
    if sample.ndim == 3:
        if sample.shape[0] < sample.shape[1]:
            H, W = sample.shape[1], sample.shape[2]
        else:
            H, W = sample.shape[0], sample.shape[1]
    else:
        H, W = sample.shape

    reconstructed = np.full((H, W), np.nan)

    for patch in mesh:
        r0, c0, r1, c1 = patch["bbox"]
        feats = patch.get("mean_features", [])
        if len(feats) > channel_idx:
            reconstructed[r0:r1, c0:c1] = feats[channel_idx]

    # Original
    orig = _channel_image(sample, channel_idx)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Original field
    im0 = axes[0].imshow(orig, cmap=cmap, origin="upper")
    plt.colorbar(im0, ax=axes[0], fraction=0.04, pad=0.04)
    axes[0].set_title(f"Original field  (ch {channel_idx})")

    # Reconstructed from patches
    im1 = axes[1].imshow(reconstructed, cmap=cmap, origin="upper")
    plt.colorbar(im1, ax=axes[1], fraction=0.04, pad=0.04)
    axes[1].set_title(f"AMR reconstruction  ({len(mesh)} patches)")

    # Difference
    diff = np.abs(orig - reconstructed)
    im2 = axes[2].imshow(diff, cmap="Reds", origin="upper")
    plt.colorbar(im2, ax=axes[2], fraction=0.04, pad=0.04)
    axes[2].set_title("Absolute error")

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[visualization] Saved → {save_path}")

    if show:
        plt.show()

    return fig


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


def _depth_colormap(depths: List[int], cmap_name: str = "plasma"):
    """Return a callable that maps depth → RGBA colour."""
    min_d = min(depths) if depths else 0
    max_d = max(depths) if depths else 1
    cmap = plt.get_cmap(cmap_name)
    norm = Normalize(vmin=min_d, vmax=max(max_d, min_d + 1))
    return lambda d: cmap(norm(d))
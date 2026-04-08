"""
========================================================================
Flow field and token-level visualization utilities.
========================================================================

Functions
---------
plot_flow_comparison  : Side-by-side comparison of GT vs predicted flow fields
plot_token_statistics : Histogram of token counts and cell sizes
animate_refinement    : Depth-by-depth animated GIF of the quadtree build (requires Pillow)
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np

from src.amr.quadtree_tokenizer import QuadNode
from src.utils.mesh_visualization import visualize_quadtree_mesh


def plot_flow_comparison(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    channel_names: Optional[List[str]] = None,
    figsize_per_col: float = 4.0,
    cmap: str = "RdBu_r",
    title: str = "Ground truth  vs  Prediction",
    save_path: Optional[str] = None,
    show: bool = False,
) -> Figure:
    """
    Side-by-side comparison of ground truth and predicted flow fields.

    Args:
        ground_truth  : [H, W, output_dim]
        prediction    : [H, W, output_dim]
        channel_names : list of strings labelling each output channel
        figsize_per_col : figure width per output channel (each has 3 columns: GT/pred/error)
        cmap           : colourmap
        save_path      : if given, save figure
        show           : call plt.show() if True

    Returns:
        matplotlib Figure
    """
    assert ground_truth.shape == prediction.shape, \
        f"Shape mismatch: GT={ground_truth.shape} pred={prediction.shape}"
    output_dim = ground_truth.shape[-1]
    names = channel_names or [f"channel {i}" for i in range(output_dim)]

    # 3 cols per output channel: GT | Pred | |Error|
    n_cols = 3
    n_rows = output_dim
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * figsize_per_col, n_rows * figsize_per_col),
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(title, fontsize=13, y=1.01)

    for r in range(output_dim):
        gt   = ground_truth[..., r]
        pred = prediction[..., r]
        err  = np.abs(gt - pred)

        vmin = min(gt.min(), pred.min())
        vmax = max(gt.max(), pred.max())

        for col, (data, label) in enumerate(zip(
            [gt, pred, err],
            ["Ground truth", "Prediction", "|Error|"]
        )):
            ax = axes[r, col]
            vm = err.max() if col == 2 else vmax
            im = ax.imshow(
                data, origin='upper', cmap='hot' if col == 2 else cmap,
                vmin=0 if col == 2 else vmin,
                vmax=vm,
                interpolation='bilinear',
            )
            ax.set_title(f"{names[r]}  –  {label}", fontsize=9)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save_path:
        timestamp = datetime.now().strftime("%d_%m_%Y-%H_%M")
        fig.savefig(f"{save_path}/prediction_comparison-{timestamp}.png", dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# Token count statistics
# ---------------------------------------------------------------------------

def plot_token_statistics(
    token_counts: List[int],
    cell_sizes: Optional[List[float]] = None,
    title: str = "Token statistics",
    save_path: Optional[str] = None,
    show: bool = False,
) -> Figure:
    """
    Plot histograms of token counts per sample and (optionally) cell size distribution.

    Args:
        token_counts : list of N per-sample token counts
        cell_sizes   : optional flat list of all cell sizes
        save_path    : if given, save figure
        show         : call plt.show() if True

    Returns:
        matplotlib Figure
    """
    n_plots = 2 if cell_sizes else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    axes[0].hist(token_counts, bins=30, color='steelblue', edgecolor='white')
    axes[0].set_xlabel("Tokens per sample")
    axes[0].set_ylabel("Count")
    axes[0].set_title(
        f"{title}\nmean={np.mean(token_counts):.0f}  "
        f"min={min(token_counts)}  max={max(token_counts)}"
    )

    if cell_sizes:
        axes[1].hist(cell_sizes, bins=40, color='coral', edgecolor='white', log=True)
        axes[1].set_xlabel("Normalised cell size")
        axes[1].set_ylabel("Count (log)")
        axes[1].set_title("Cell size distribution")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# Depth-by-depth refinement animation
# ---------------------------------------------------------------------------

def animate_refinement(
    grid: np.ndarray,
    token_list: List[QuadNode],
    channel: int = 0,
    save_path: str = "refinement.gif",
    fps: int = 2,
) -> None:
    """
    Create an animated GIF showing the quadtree building depth by depth.

    Requires: matplotlib + PIL  (pip install Pillow)
    """
    try:
        from PIL import Image
    except ImportError:
        print("Pillow not installed; skipping animation. Run: pip install Pillow")
        return

    import io

    H, W, _ = grid.shape
    max_depth = max(t.depth for t in token_list)
    frames = []

    for d in range(max_depth + 1):
        visible = [t for t in token_list if t.depth <= d]
        fig = visualize_quadtree_mesh(
            grid, visible,
            channel=channel,
            title=f"Quadtree refinement  -  depth ≤ {d}",
            show=False,
        )
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
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

"""
Mesh and quadtree visualization utilities

Functions
---------
plot_mesh           : Overlay the adaptive quadtree mesh on a 2D grid channel
plot_mesh_comparison : Several meshes of one sample side by side over the same 2D field
plot_mesh_by_depth  : Show one subplot per depth level with patches at that depth highlighted
plot_mesh_by_depth_cumulative : Show one subplot per depth level, each including all shallower depths
plot_reconstruction_by_depth_cumulative : The same panels painted with what the mesh reconstructs the target as, one row per affine_output order
plot_metric_heatmap : Show a heatmap of a chosen physics metric on the original grid
plot_patch_features : Reconstruct and display the field from averaged patch features
plot_score_map      : Render a per-pixel refinement score as a heatmap (optionally over geometry)
animate_mesh_refinement  : Depth-by-depth animated GIF of the quadtree build (requires Pillow)


Flow field and token-level visualization utilities.

Functions
---------
plot_flow_comparison  : Side-by-side comparison of ground truth vs predicted flow fields
plot_mesh_prediction_comparison : Ground truth once, then the absolute error of every model with its mesh
plot_next_timestep    : 2x3 next-step flow of one test case at frames t, t+n, t+m, ground truth over prediction, optional mesh overlay
plot_channel_sections : Chordwise curves of one solution channel at selected spanwise stations
plot_coefficient_correlation : Predicted vs target integrated coefficient over a set of samples
plot_token_statistics : Histogram of token counts per sample and (optionally) cell size distribution.
plot_error_by_depth   : Box plot of per-token MAE grouped by quadtree depth (patch size)
plot_3d_prediction    : 3D surface rendering of predicted fields over wing geometry


Ablation visualization utilities.

Functions
---------
plot_perturbation_error : Overall error against the mesh perturbation amplitude, with the unperturbed run as a line
plot_perturbation_robustness : Overall error against the test-time mesh perturbation amplitude, one line per training amplitude
plot_cell_shape_floors : Reconstruction floor of the uniform mesh against its token count, one line per cell shape
plot_error_vs_cost : Test error against the token count or another cost, one line per model
plot_polynomial_order_error : Overall error against the token count, one line per input and output order
plot_polynomial_floor : Error of every input and output order against the reconstruction floor of its output order
plot_model_floor : Error of every surface field of every model against its reconstruction floor
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch

from src.amr.quadtree import QuadNode
from src.evaluation.metrics import SOLUTION_CHANNEL_SCALES
from src.inference.predict import predict_single_amr, predict_single_vit, resolve_mesh_source
from src.inference.predict_autoregressive import predict_next_timestep
from src.models.amr_model import AMRTransformer
from src.models.reconstruction import basis_size, cell_basis
from src.training.loss import nmse_loss

# Colorbar inset beside an image panel, in its axes coordinates: as tall as the image, 9 % of its width
COLORBAR_BOX = [1.06, 0, 0.09, 1]

# Axis labels for the coefficients ``calculate_coefficients`` returns, in its order.
COEFFICIENT_LABELS = ("$C_L$", "$C_D$", "$C_{Mz}$")

# Axis labels for the SuperWing solution channels (cp, cf_tau, cf_z), in the
# order they are stored; see SOLUTION_CHANNEL_SCALES in src.evaluation.metrics.
SOLUTION_CHANNEL_LABELS = ("$C_p$", r"$C_{f,\tau}$", "$C_{f,z}$")


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
    ax.imshow(channel_data, cmap="RdBu_r", origin="upper")

    # depths = [p.depth for p in mesh]
    # min_d = min(depths) if depths else 0
    # max_d = max(depths) if depths else 1
    # cmap, norm, _ = color_map(np.array(depths), "plasma", dmin=min_d, dmax=max(max_d, min_d + 1), n_levels=max_d - min_d + 1)

    # Rectangle overlays
    rects = []
    for patch in mesh:
        r0, c0, r1, c1 = patch.bbox[0], patch.bbox[1], patch.bbox[2], patch.bbox[3]
        height = r1 - r0
        width  = c1 - c0

        # imshow places pixel (0,0) centered at coordinate 0.5
        rects.append(patches.Rectangle((c0 - 0.5, r0 - 0.5), width, height))

    # pc = PatchCollection(rects, cmap=cmap, norm=norm, linewidth=0.75, alpha=1)
    # pc.set_array(np.array(depths))
    pc = PatchCollection(rects, edgecolor="black", linewidth=0.75, alpha=1)
    pc.set_facecolor("none")
    ax.add_collection(pc)

    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="10%", pad=0.15)
    # cbar = fig.colorbar(pc, cax=cax)
    # cbar.set_label("Quadtree depth")
    # cbar.set_ticks(range(min_d, max_d + 1))

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


def plot_mesh_comparison(
    field: np.ndarray,
    meshes: Dict[str, List[QuadNode]],
    *,
    label: str = "$C_p$",
    show: bool = True,
    save_path: Optional[str] = None,
) -> Figure:
    """Draw several meshes of one sample side by side over the same 2D field."""
    fig, axes = plt.subplots(1, len(meshes), figsize=(3.2 * len(meshes), 6.2), squeeze=False)
    for ax, (name, mesh) in zip(axes[0], meshes.items()):
        # origin="lower": on the wing grid the lower TE is at the bottom, the LE in the middle
        im = ax.imshow(field, cmap="RdBu_r", origin="lower", vmin=field.min(), vmax=field.max())
        draw_mesh(ax, mesh)
        ax.set_title(name, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
    cbar = fig.colorbar(im, cax=axes[0][-1].inset_axes(COLORBAR_BOX))
    cbar.set_label(label, fontsize=12)

    if save_path:
        save_plot(save_path, fig, dpi=200, use_date_subfolder=True)

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


# ---------------------------------------------------------------------------
# The quadtree as it stood partway through its build
# ---------------------------------------------------------------------------

def cumulative_frontier(
    mesh: List[QuadNode],
    depth: int,
    H: int,
    W: int,
) -> Dict[Tuple[int, int, int, int], int]:
    """The cells tiling the grid after ``depth`` refinement steps of the build.

    ``build_adaptive_mesh`` returns only the final leaves (``collect_leaves``),
    discarding the root, so a leaf that finalized deeper than ``depth`` has no node
    left to represent the coarser cell it had not split out of yet at that step.
    That ancestor is recomputed by replaying the same quadrant split
    ``QuadNode.compute_child_bboxes`` used to build the tree, descending from the
    root toward the leaf's top-left corner. A leaf that already finalized at or
    above ``depth`` keeps its own bbox, since it never splits again.

    Several leaves under the same unsplit ancestor collapse to one entry, so the
    result tiles the whole domain with no gaps or overlaps, and the frontier one
    step deeper shows exactly that cell split into four wherever it refined
    further.

    Args:
        mesh: The final leaf ``QuadNode`` s, which must tile the ``H x W`` grid.
        depth: How many refinement steps to replay (0 = the root cell alone).
        H: Grid height (rows), bounding the root cell.
        W: Grid width (columns), likewise.

    Returns:
        ``{bbox: cell_depth}``, one entry per cell of the frontier.
    """
    root_bbox = (0, 0, H, W)
    frontier: Dict[Tuple[int, int, int, int], int] = {}

    for leaf in mesh:
        if leaf.depth <= depth:
            frontier[leaf.bbox] = leaf.depth
            continue

        bbox = root_bbox
        for _ in range(depth):
            for child_bbox in QuadNode(bbox=bbox).compute_child_bboxes():
                cr0, cc0, cr1, cc1 = child_bbox
                if cr0 <= leaf.r0 < cr1 and cc0 <= leaf.c0 < cc1:
                    bbox = child_bbox
                    break
        frontier[bbox] = depth

    return frontier


def plot_mesh_by_depth_cumulative(
    sample: np.ndarray,
    mesh: List[QuadNode],
    *,
    channel: int = 0,
    title: str = "Adaptive Mesh by Cumulative Depth",
    show: bool = True,
    save_path: Optional[str] = None,
) -> None:
    """Show one subplot per depth level, each holding the quadtree as it stood after that many refinement steps."""
    depths = sorted(set(p.depth for p in mesh))
    n_depths = len(depths)
    min_d, max_d = depths[0], depths[-1]

    channel_data = channel_image(sample, channel)
    H, W = channel_data.shape

    # Size each subplot to the image's own aspect ratio (rather than a fixed
    # square), so imshow's equal-aspect axes hug the data with no left/right
    # padding; a taller panel_height keeps neighboring panels clearly separated.
    panel_height = 6
    panel_width = panel_height * (W / H)
    # Constrained layout: tight_layout cannot place a colorbar shared across axes
    fig, axes = plt.subplots(1, n_depths, figsize=(n_depths * panel_width, panel_height), squeeze=False, layout="constrained")
    axes = axes[0]

    # cmap, norm, _ = color_map(np.array(depths), "plasma", dmin=min_d, dmax=max(max_d, min_d + 1), n_levels=max_d - min_d + 1)

    pc = None
    for ax_idx, depth in enumerate(depths):
        ax = axes[ax_idx]
        ax.imshow(channel_data, cmap="RdBu_r", origin="upper")

        frontier = cumulative_frontier(mesh, depth, H, W)

        rects = []
        # cell_depths = []
        for bbox, cell_depth in frontier.items():
            r0, c0, r1, c1 = bbox
            height = r1 - r0
            width  = c1 - c0
            # imshow places pixel (0,0) centered at coordinate 0.5
            rects.append(patches.Rectangle((c0 - 0.5, r0 - 0.5), width, height))
            # cell_depths.append(cell_depth)

        # pc = PatchCollection(rects, cmap=cmap, norm=norm, linewidth=0.75, alpha=1)
        # pc.set_array(np.array(cell_depths))
        pc = PatchCollection(rects, edgecolor="black", linewidth=0.75, alpha=1)
        pc.set_facecolor("none")
        ax.add_collection(pc)

        ax.set_title(f"Depth = {depth}  ({len(frontier)} cells)", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])

    # if pc is not None:
    #     cbar = fig.colorbar(pc, ax=axes.tolist(), fraction=0.02, pad=0.02)
    #     cbar.set_label("Quadtree depth")
    #     cbar.set_ticks(range(min_d, max_d + 1))

    fig.suptitle(title, fontsize=12)

    if save_path:
        save_plot(save_path, fig, use_date_subfolder=True)

    if show:
        plt.show()


# ---------------------------------------------------------------------------
# What the mesh can represent: the target projected onto the per-cell basis
# ---------------------------------------------------------------------------
# The head emits basis_size(affine_output) coefficients per channel per cell and
# tokens_to_grid_affine_torch paints the cell with their weighted sum of
# cell_basis columns. Those columns are mutually orthogonal over a cell -- the
# property that makes affine_nmse_loss closed-form -- so the best field any head
# of that order can produce on a given mesh is the target's orthogonal projection
# onto them, one coefficient per column:
#
#     c_j = sum(target * phi_j) / sum(phi_j ** 2)
#
# which are the quantities src/data/collate_fn.py:_affine_leaf_stats caches
# (mean_target, sum_target_dx / sum_xx, ...). Painting that projection is
# therefore the picture of what the transformer is trained to produce: the
# target as seen through (mesh, basis), with no checkpoint involved and no
# coefficient error mixed in. Its residual is the reconstruction floor of that
# (mesh, order) pair.


def project_cells(target: np.ndarray, cells, order: int) -> np.ndarray:
    """Paint each cell with the best fit an ``affine_output=order`` head can reach.

    Args:
        target: ``[H, W, C]`` dense field to project (the ground truth the loss
            scores against).
        cells: Iterable of ``(r0, c0, r1, c1)`` bboxes tiling the grid — the leaf
            bboxes, or one of `cumulative_frontier`'s partway-through frontiers.
        order: The head's ``affine_output``: 0 the cell mean, 1 mean + two slopes,
            2 additionally the three curvature terms.

    Returns:
        ``[H, W, C]`` float64 projection. A term the cell is too small to resolve
        has a zero-norm basis column and simply drops out, exactly as it does in
        the loss.
    """
    n_taylor_terms = basis_size(order)
    out = np.zeros(target.shape, dtype=np.float64)

    for r0, c0, r1, c1 in cells:
        h, w = r1 - r0, c1 - c0
        basis = cell_basis(h, w, n_taylor_terms)                   # [h*w, K]
        norms = (basis ** 2).sum(axis=0)[:, None]                  # [K, 1]
        region = target[r0:r1, c0:c1].reshape(h * w, -1)           # [h*w, C]
        coefficients = np.divide(basis.T @ region, norms,
                                 out=np.zeros((n_taylor_terms, region.shape[1])),
                                 where=norms > 0)                  # [K, C]
        out[r0:r1, c0:c1] = (basis @ coefficients).reshape(h, w, -1)

    return out


def plot_reconstruction_by_depth_cumulative(
    target: np.ndarray,
    mesh: List[QuadNode],
    *,
    orders: Sequence[int] = (0, 1, 2),
    channel: int = 0,
    title: str = "Mesh Reconstruction by Cumulative Depth",
    show: bool = True,
    save_path: Optional[str] = None,
) -> Figure:
    """Show what the mesh reconstructs the target as, per depth and per output order."""
    # A copy, not a view: the target often arrives as a read-only slice of a
    # memory-mapped array, which torch.from_numpy below will not take.
    target = np.array(target, dtype=np.float64)
    H, W, _ = target.shape
    depths = sorted(set(p.depth for p in mesh))
    n_cols = len(depths) + 1                       # + the ground-truth column

    # Shared color scale, so a panel's colors mean the same thing in every cell of
    # the figure and a coarse panel cannot flatter itself by rescaling.
    truth = target[:, :, channel]
    vmin, vmax = float(truth.min()), float(truth.max())

    # Panels carry the image's own aspect ratio, so a tall grid gives narrow
    # columns; constrained layout keeps their titles from colliding.
    panel_height = 4
    panel_width = panel_height * (W / H)
    fig, axes = plt.subplots(len(orders), n_cols,
                             figsize=(n_cols * panel_width, len(orders) * panel_height),
                             squeeze=False, layout="constrained")

    # One frontier per column, shared down the rows: within a column the mesh is
    # identical and the basis is the only thing that varies.
    frontiers = [cumulative_frontier(mesh, depth, H, W) for depth in depths]

    for row, order in enumerate(orders):
        ax = axes[row][0]
        im = ax.imshow(truth, cmap="RdBu_r", origin="upper", vmin=vmin, vmax=vmax)
        ax.set_ylabel(f"Order = {order}", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        if row == 0:
            ax.set_title("Ground truth\n(dense)", fontsize=11)

        for col, (depth, frontier) in enumerate(zip(depths, frontiers), start=1):
            ax = axes[row][col]
            reconstruction = project_cells(target, frontier, order)
            ax.imshow(reconstruction[:, :, channel], cmap="RdBu_r", origin="upper", vmin=vmin, vmax=vmax)

            # Cell outlines, so the faceting can be told from the mesh that caused it.
            rects = [patches.Rectangle((c0 - 0.5, r0 - 0.5), c1 - c0, r1 - r0)
                     for r0, c0, r1, c1 in frontier]
            ax.add_collection(PatchCollection(rects, facecolor="none",
                                              edgecolor="black", linewidth=0.3, alpha=0.4))

            nmse = float(nmse_loss(torch.from_numpy(reconstruction), torch.from_numpy(target)))
            ax.text(0.03, 0.03, f"NMSE {nmse:.2e}", transform=ax.transAxes, fontsize=9,
                    va="bottom", ha="left",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.75, linewidth=0))
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"Depth = {depth}\n({len(frontier)} cells)", fontsize=11)

    # cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
    # cbar.set_label(SOLUTION_CHANNEL_LABELS[channel] if channel < len(SOLUTION_CHANNEL_LABELS)
    #                else f"Channel {channel}")

    fig.suptitle(title, fontsize=13)

    if save_path:
        save_plot(save_path, fig, use_date_subfolder=True)

    if show:
        plt.show()

    return fig


def plot_metric_heatmap(
    sample: np.ndarray,
    mesh: List[QuadNode],
    *,
    metric_name: str = "velocity_gradient",
    title: Optional[str] = "Metric Heatmap",
    show: bool = True,
    save_path: Optional[str] = None,
) -> None:
    """Show a heatmap of a chosen physics metric on the original grid (see ''compute_enabled_metrics()'' for valid metric_name)."""
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
    names = channel_names or SOLUTION_CHANNEL_LABELS[:output_channels]

    # 3 cols per output channel: GT | Pred | |Error|
    n_cols = 3
    n_rows = output_channels
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * figsize_per_col, n_rows * figsize_per_col), layout="constrained")

    if n_rows == 1:
        axes = axes[np.newaxis, :]

    suptitle = fig.suptitle(title, fontsize=13)

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

    # Colorbar tick labels make the content lopsided, so centre the title over
    # the drawn panels rather than the canvas; the tight crop then keeps it centred.
    fig.draw_without_rendering()
    content = Bbox.union([a.get_tightbbox() for a in fig.axes])
    suptitle.set_x((content.x0 + content.x1) / 2 / fig.bbox.width)

    if save_path:
        save_plot(save_path, fig, use_date_subfolder=True)

    if show:
        plt.show()


def plot_mesh_prediction_comparison(
    ground_truth: np.ndarray,
    predictions: Dict[str, np.ndarray],
    meshes: Dict[str, Optional[List[QuadNode]]],
    *,
    label: str = "$C_p$",
    title: Optional[str] = None,
    figsize_per_col: float = 4.0,
    show: bool = True,
    save_path: Optional[str] = None,
) -> Figure:
    """The ground truth once, then the absolute error of every model with its mesh drawn on it (one shared color scale)."""
    errors = {name: np.abs(prediction - ground_truth) for name, prediction in predictions.items()}
    error_max = max(float(error.max()) for error in errors.values())

    n_cols = len(predictions) + 1
    # An empty column after the ground truth leaves room for its colorbar, which the layout does not reserve
    fig, axes = plt.subplots(1, n_cols + 1, figsize=(n_cols * figsize_per_col, 2.2 * figsize_per_col), layout="constrained",
                             width_ratios=[1, 0.01] + [1] * len(predictions))
    # Gaps between the panels, so the error maps read as separate plots
    fig.get_layout_engine().set(wspace=0.1)
    axes[1].remove()
    axes = [axes[0], *axes[2:]]
    if title:
        fig.suptitle(title, fontsize=13)

    # origin="lower": on the wing grid the lower TE is at the bottom, the LE in the middle
    im = axes[0].imshow(ground_truth, origin="lower", cmap="RdBu_r", interpolation="bilinear")
    axes[0].set_title(f"Ground truth {label}", fontsize=13)
    # Both colorbars sit in an inset beside their image, so they share its height and one width
    cbar1 = fig.colorbar(im, cax=axes[0].inset_axes(COLORBAR_BOX))
    cbar1.set_label(label=label, fontsize=13)

    for ax, (name, error) in zip(axes[1:], errors.items()):
        im = ax.imshow(error, origin="lower", cmap="Reds", vmin=0, vmax=error_max, interpolation="bilinear")
        ax.set_title(f"{name} (MAE {error.mean():.2e})", fontsize=13)
        if meshes.get(name):
            draw_mesh(ax, meshes[name])
    cbar2 = fig.colorbar(im, cax=axes[-1].inset_axes(COLORBAR_BOX))
    cbar2.set_label(label=r"$|C_{p,\mathrm{pred}} - C_{p,\mathrm{true}}|$", fontsize=13)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    if save_path:
        save_plot(save_path, fig, dpi=200, use_date_subfolder=True)

    if show:
        plt.show()

    return fig


def plot_next_timestep(
    model,
    args: Dict,
    dataset,
    mesh_source: Optional[tuple] = None,
    *,
    show_mesh: bool = True,
    channel: int = 0,
    title: Optional[str] = None,
    show: bool = True,
    save_path: Optional[str] = None,
) -> Figure:
    """2x3 grid of one test case at frames t, t + n and t + m: ground truth on top, prediction below."""

    # The showcased case and frames are picked where u changes most visibly. All are test cases
    # except cavity, whose test cases are all near-static; case 62 is from the validation split.
    dataset_name = args["dataset"]
    if dataset_name == "cavity_dataset":
        case_id, t, n, m = 62, 1, 65, 135       # geo/case0013: the main vortex forms and deepens
    elif dataset_name == "cylinder_dataset":
        case_id, t, n, m = 146, 100, 200, 700   # prop/case0126: symmetric wake -> elongated wake -> shedding
    elif dataset_name == "dam_dataset":
        case_id, t, n, m = 216, 1, 8, 98        # prop/case0166: jet over the barrier and wake form, upstream flow slows
    elif dataset_name == "tube_dataset":
        case_id, t, n, m = 115, 1, 7, 49        # prop/case0090: inflow front -> crossing -> developed profile
    else:
        raise ValueError(f"No showcase example for dataset {dataset_name!r}")
    frames = (t, t + n, t + m)
    is_amr = isinstance(model, AMRTransformer)
    if is_amr and mesh_source is None:
        mesh_source = resolve_mesh_source(args)

    # A case's rows are contiguous and ordered by pair start, one start every FRAME_STRIDE frames
    case_rows = np.flatnonzero(dataset.case_ids() == case_id)
    truths, predictions, meshes = [], [], []
    for frame in frames:
        start = frame - dataset.time_step
        if start < 0 or start % dataset.FRAME_STRIDE or start // dataset.FRAME_STRIDE >= len(case_rows):
            raise ValueError(f"Frame {frame} is not a target frame of case {case_id}")
        sample = dataset[int(case_rows[start // dataset.FRAME_STRIDE])]

        if is_amr:
            refinement_criteria, scorer = mesh_source
            result = predict_single_amr(model, sample, max_depth=args["max_depth"], min_depth=args["min_depth"],
                                        refinement_criteria=refinement_criteria, scorer=scorer,
                                        offset=args.get("offset", 0.0))
        else:
            result = predict_single_vit(model, sample)
        result = predict_next_timestep(result, dataset)
        truths.append(result["next_timestep_true"][..., channel])
        predictions.append(result["next_timestep_pred"][..., channel])
        meshes.append(result.get("mesh") if show_mesh else None)

    H, W = truths[0].shape
    vmin, vmax = min(float(f.min()) for f in truths), max(float(f.max()) for f in truths)

    panel_height = 4
    fig, axes = plt.subplots(2, 3, figsize=(3 * panel_height * W / H, 2 * panel_height)) # squeeze=False, layout="constrained"

    for col in range(3):
        for row, (label, field) in enumerate((("Ground Truth", truths[col]), ("Prediction", predictions[col]))):
            ax = axes[row][col]
            # The grids are stored y-up (the cavity lid is the last row), so row 0 goes at the bottom
            ax.imshow(field, cmap="RdBu_r", origin="lower", vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(label, fontsize=13)

        mesh = meshes[col]
        if mesh:
            ax = axes[1][col]
            rects = [patches.Rectangle((leaf.c0 - 0.5, leaf.r0 - 0.5), leaf.width, leaf.height) for leaf in mesh]
            ax.add_collection(PatchCollection(rects, facecolor="none", edgecolor="black", linewidth=0.3, alpha=0.4))
            ax.text(0.03, 0.03, f"{len(mesh)} tokens", transform=ax.transAxes, fontsize=9, va="bottom", ha="left",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.75, linewidth=0))

    # name = type(dataset).__name__.removesuffix("Dataset").lower()
    # fig.suptitle(title or f"The evolution of the {name} flow in comparison to the ground truth", fontsize=13)

    if save_path:
        save_plot(save_path, fig, use_date_subfolder=True)

    if show:
        plt.show()

    return fig


def plot_channel_sections(
    geometry: np.ndarray,
    predictions: np.ndarray | Dict[str, np.ndarray],
    ground_truth: Optional[np.ndarray] = None,
    *,
    spanwise_position: Optional[List[int]] = None,
    channel: int = 0,
    title: str = "Chordwise section curves",
    show: bool = True,
    save_path: Optional[str] = None,
) -> Figure:
    """Plot one solution channel against the chordwise coordinate at selected spanwise stations (default 25/50/75 % span)."""
    named_predictions = {"Prediction": predictions} if isinstance(predictions, np.ndarray) else predictions

    # Datasets other than SuperWing carry their own channels, so fall back to
    # the bare index rather than mislabelling one of them.
    channel_label = SOLUTION_CHANNEL_LABELS[channel] if channel < len(SOLUTION_CHANNEL_LABELS) else f"channel {channel}"

    # The channels are stored pre-scaled, so divide the factor out to plot the
    # physical coefficient the axis is labelled with. Cp's factor is 1.
    channel_scale = SOLUTION_CHANNEL_SCALES[channel] if channel < len(SOLUTION_CHANNEL_SCALES) else 1.0

    if ground_truth is not None and ground_truth.shape[:2] != geometry.shape[:2]:
        raise ValueError(f"Grid mismatch: ground_truth={ground_truth.shape[:2]} geometry={geometry.shape[:2]}")

    # Grids are stored [chordwise, spanwise, channel]: axis 0 walks around the
    # section, axis 1 runs root to tip. A station is therefore a *column*.
    n_spanwise_position = geometry.shape[1]
    if spanwise_position is None:
        spanwise_position = [int(n_spanwise_position * f) for f in (0.25, 0.5, 0.75)]

    model_colors = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, len(spanwise_position), figsize=(5 * len(spanwise_position), 4.5), squeeze=False, sharey=True)

    for ax, station in zip(axes[0], spanwise_position):
        if not 0 <= station < n_spanwise_position:
            raise ValueError(f"Station {station} out of range for {n_spanwise_position} spanwise columns")

        x = geometry[:, station, 0].astype(float)
        # chord = x.max() - x.min()
        # x_over_c = (x - x.min()) / (chord if chord > 0 else 1.0)

        if ground_truth is not None:
            ax.plot(x, ground_truth[:, station, channel] / channel_scale, color="black", linestyle="--", linewidth=1.2, label="Ground truth")

        for color_idx, (name, prediction) in enumerate(named_predictions.items()):
            ax.plot(x, prediction[:, station, channel] / channel_scale, color=model_colors(color_idx % 10), linewidth=1.4, label=name)

        ax.axhline(0.0, color="grey", linewidth=0.6, alpha=0.6)
        ax.set_title(f"Spanwise Position {station}  (y/b ≈ {station / max(n_spanwise_position - 1, 1):.2f})", fontsize=10)
        ax.set_xlabel("x")
        ax.grid(True, alpha=0.3)

        # Suction (negative Cp) upwards by convention
        if channel == 0:
            ax.invert_yaxis()

    axes[0][0].set_ylabel(channel_label)
    axes[0][0].legend(fontsize=9)

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()

    if save_path:
        save_plot(save_path, fig, use_date_subfolder=True)

    if show:
        plt.show()

    return fig


def plot_coefficient_correlation(
    reference: np.ndarray,
    predictions: np.ndarray | Dict[str, np.ndarray],
    *,
    coefficient: int = 1,
    title: str = "Integrated coefficient  -  prediction vs target",
    show: bool = True,
    save_path: Optional[str] = None,
) -> Figure:
    """Scatter one integrated coefficient (0 = CL, 1 = CD, 2 = CMz) of every sample against its target."""
    named_predictions = {"Prediction": predictions} if isinstance(predictions, np.ndarray) else predictions

    if not 0 <= coefficient < len(COEFFICIENT_LABELS):
        raise ValueError(f"coefficient {coefficient} out of range for {COEFFICIENT_LABELS}")

    label = COEFFICIENT_LABELS[coefficient]
    target = reference[:, coefficient].astype(float)

    model_colors = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(6, 6))

    # The diagonal spans everything drawn, so both sides of it stay visible even
    # when a model sits entirely off the target range.
    values = [target] + [p[:, coefficient].astype(float) for p in named_predictions.values()]
    low = min(v.min() for v in values)
    high = max(v.max() for v in values)
    margin = 0.05 * (high - low if high > low else max(abs(high), 1.0))
    limits = (low - margin, high + margin)

    ax.plot(limits, limits, color="black", linestyle="--", linewidth=1.2, label="Perfect agreement")

    for color_idx, (name, prediction) in enumerate(named_predictions.items()):
        predicted = prediction[:, coefficient].astype(float)
        mae = np.abs(predicted - target).mean()
        ax.scatter(target, predicted, s=22, alpha=0.75, color=model_colors(color_idx % 10), edgecolor="none", label=f"{name}  (MAE {mae:.5f})")

    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_aspect("equal")
    ax.set_xlabel(f"Target {label}")
    ax.set_ylabel(f"Predicted {label}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    fig.suptitle(f"{title}  ({len(target)} samples)", fontsize=13)
    plt.tight_layout()

    if save_path:
        save_plot(save_path, fig, use_date_subfolder=True)

    if show:
        plt.show()

    return fig


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


def plot_error_by_depth(
    meshes: List[List[QuadNode]],
    predictions: List[np.ndarray],
    targets: List[np.ndarray],
    *,
    channel: int = 0,
    title: str = "Token error by patch depth",
    show: bool = True,
    save_path: Optional[str] = None,
) -> Figure:
    """Box plot of each token's mean absolute error, grouped by its quadtree depth."""
    # The channels are stored pre-scaled; divide the factor out to report the physical error
    channel_scale = SOLUTION_CHANNEL_SCALES[channel] if channel < len(SOLUTION_CHANNEL_SCALES) else 1.0
    channel_label = SOLUTION_CHANNEL_LABELS[channel] if channel < len(SOLUTION_CHANNEL_LABELS) else f"channel {channel}"

    errors_by_depth: Dict[int, List[float]] = {}
    for mesh, prediction, target in zip(meshes, predictions, targets):
        error = np.abs(prediction[..., channel] - target[..., channel]) / channel_scale
        for leaf in mesh:
            token_error = error[leaf.r0:leaf.r1, leaf.c0:leaf.c1].mean()
            errors_by_depth.setdefault(leaf.depth, []).append(token_error)

    depths = sorted(errors_by_depth)
    tick_labels = [f"depth {d}\n{len(errors_by_depth[d])} tokens" for d in depths]

    fig, ax = plt.subplots(figsize=(1.5 * len(depths) + 3, 4.5))
    ax.boxplot([errors_by_depth[d] for d in depths], showfliers=False)
    ax.set_xticks(range(1, len(depths) + 1), tick_labels)
    ax.set_xlabel("Quadtree depth (deeper = smaller patch)")
    ax.set_ylabel(f"Token MAE of {channel_label}")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_title(f"{title}  ({len(meshes)} samples)", fontsize=12)
    plt.tight_layout()

    if save_path:
        save_plot(save_path, fig, use_date_subfolder=True)

    if show:
        plt.show()

    return fig


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
# Ablation Visualization
# ---------------------------------------------------------------------------
def plot_perturbation_error(
    amplitudes: Sequence[float],
    errors: Sequence[float],
    reference_error: float,
    *,
    reference_label: str = "Unperturbed",
    ylabel: str = "Relative $L_2$ error of surface fields (%)",
    show: bool = True,
    save_path: Optional[str] = None,
) -> Figure:
    """Plot the test error of each perturbed run against its amplitude, with the unperturbed run as a reference line."""
    order = np.argsort(amplitudes)
    a = np.asarray(amplitudes, dtype=float)[order]
    error = 100 * np.asarray(errors, dtype=float)[order]
    reference = 100 * reference_error

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axhline(reference, color="black", linestyle="--", linewidth=1.2)
    ax.annotate(f"{reference_label} ({reference:.2f} %)", (a[-1], reference), xytext=(0, -4),
                textcoords="offset points", fontsize=13, color="black", va="top", ha="right")
    ax.plot(a, error, color="tab:blue", linewidth=2, marker="o", markersize=7)

    ax.set_xlabel("Perturbation amplitude", fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        save_plot(save_path, fig, use_date_subfolder=True)

    if show:
        plt.show()

    return fig


def plot_perturbation_robustness(
    errors: Dict[float, Dict[float, float]],
    *,
    ylabel: str = "Relative $L_2$ error (%)",
    show: bool = True,
    save_path: Optional[str] = None,
) -> Figure:
    """Plot the test error against the amplitude of the test-time mesh perturbation, one line per training amplitude."""
    colors = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for i, (train_amplitude, by_test) in enumerate(sorted(errors.items())):
        test = sorted(by_test)
        ax.plot(test, [100 * by_test[a] for a in test], color=colors(i), linewidth=2, marker="o", markersize=7,
                label=f"$a = {train_amplitude:g}$")

    ax.set_xlabel("Test perturbation amplitude", fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=13)
    plt.tight_layout()

    if save_path:
        save_plot(save_path, fig, use_date_subfolder=True)

    if show:
        plt.show()

    return fig


def plot_error_vs_cost(
    curves: Dict[str, Tuple[Sequence[float], Sequence[float], Sequence[float]]],
    *,
    xlabel: str = "Number of tokens",
    ylabel: str = "Relative error (%)",
    xticks: Optional[Sequence[float]] = None,
    reference: Optional[Tuple[float, float, str]] = None,
    show: bool = True,
    save_path: Optional[str] = None,
) -> Figure:
    """Plot the test error of every model against its cost, one line per model."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    handles = []
    for i, (model, (cost, error, cost_std)) in enumerate(curves.items()):
        bars = ax.errorbar(cost, 100 * np.asarray(error), xerr=cost_std, fmt="o-", linewidth=2, markersize=7,
                           capsize=3, label=model)
        # The legend shows only the line and its dots, not the error bars
        handles.append(bars.lines[0])

    # if reference is not None:
    #     ref_cost, ref_error, ref_label = reference
    #     ref_error = 100 * ref_error
    #     ax.axhline(ref_error, color="black", linestyle="--", linewidth=1.2)
    #     ax.plot(ref_cost, ref_error, "o", color="black", markersize=7, zorder=3)
    #     x_max = max(max(cost) for cost, _, _ in curves.values())
    #     ax.annotate(f"{ref_label} ({ref_error:.2f} %)", (x_max, ref_error), xytext=(0, -4),
    #                 textcoords="offset points", fontsize=9, color="black", va="top", ha="right")

    ax.set_xscale("log")
    if xticks is not None:
        ax.set_xticks(xticks, [str(t) for t in xticks])
        ax.minorticks_off()
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(handles, list(curves), fontsize=13)
    plt.tight_layout()

    if save_path:
        save_plot(save_path, fig, use_date_subfolder=True)

    if show:
        plt.show()

    return fig


def plot_cell_shape_floors(
    floors: Dict[str, Tuple[Sequence[int], Sequence[float]]],
    *,
    ylabel: str = "Reconstruction floor (NMSE)",
    show: bool = True,
    save_path: Optional[str] = None,
) -> Figure:
    """Plot the reconstruction floor of the uniform mesh against its token count, one line per cell shape."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, (shape, (tokens, floor)) in enumerate(floors.items()):
        ax.plot(tokens, floor, color=f"C{i}", linewidth=2, marker="o", markersize=7, label=shape)

    all_tokens = sorted({n for tokens, _ in floors.values() for n in tokens})
    ax.set_xscale("log", base=2)
    ax.set_xticks(all_tokens, [str(n) for n in all_tokens])
    ax.set_yscale("log")
    ax.set_xlabel("Number of tokens", fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=12, title="Cell shape", title_fontsize=12)
    plt.tight_layout()

    if save_path:
        save_plot(save_path, fig, use_date_subfolder=True)

    if show:
        plt.show()

    return fig


def plot_polynomial_order_error(
    errors: Dict[Tuple[int, int], Dict[int, float]],
    *,
    ylabel: str = "Relative $L_2$ error of surface fields (%)",
    show: bool = True,
    save_path: Optional[str] = None,
) -> Figure:
    """Plot the test error against the token count for every input and output order."""
    colors = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for (input_order, output_order), by_tokens in sorted(errors.items()):
        tokens = sorted(by_tokens)
        color = colors(output_order)
        ax.plot(tokens, [100 * by_tokens[n] for n in tokens], color=color, linestyle="-" if input_order else "--",
                linewidth=2, marker="o", markersize=6, markerfacecolor=color if input_order else "white",
                label=f"{input_order}-{output_order}")

    all_tokens = sorted({n for by_tokens in errors.values() for n in by_tokens})
    ax.set_xscale("log", base=2)
    ax.set_xticks(all_tokens, [str(n) for n in all_tokens])
    ax.minorticks_off()
    ax.set_xlabel("Number of tokens", fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.grid(True, alpha=0.3)
    # Sorted entries fill the columns in turn, so each column holds one input order
    ax.legend(fontsize=11, ncol=2, title="Token order", title_fontsize=11)
    plt.tight_layout()

    if save_path:
        save_plot(save_path, fig, use_date_subfolder=True)

    if show:
        plt.show()

    return fig


def plot_polynomial_floor(
    floors: Dict[int, float],
    errors: Dict[Tuple[int, int], float],
    *,
    ylabel: str = "Relative $L_2$ error of surface fields (%)",
    show: bool = True,
    save_path: Optional[str] = None,
) -> Figure:
    """Bar the test error of every input order against the reconstruction floor, per output order."""
    colors = plt.get_cmap("tab10")
    output_orders = sorted(floors)
    input_orders = sorted({input_order for input_order, _ in errors})
    width = 0.8 / len(input_orders)
    basis_names = {0: "constant", 1: "affine", 2: "quadratic"}

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for x, output_order in enumerate(output_orders):
        for j, input_order in enumerate(input_orders):
            error = 100 * errors[(input_order, output_order)]
            position = x + (j - (len(input_orders) - 1) / 2) * width
            ax.bar(position, error, width, color=colors(input_order))
            ax.text(position, error, f"{error:.1f}", ha="center", va="bottom", fontsize=8)

        floor = 100 * floors[output_order]
        ax.hlines(floor, x - 0.45, x + 0.45, color="black", linewidth=2)
        ax.text(x + 0.45, floor, f"{floor:.1f}", ha="left", va="center", fontsize=8)

    handles = [patches.Patch(color=colors(input_order), label=f"Input order {input_order}") for input_order in input_orders]
    handles.append(Line2D([], [], color="black", linewidth=2, label="Reconstruction floor"))
    ax.set_xticks(range(len(output_orders)), [f"{o} ({basis_names.get(o, o)})" for o in output_orders])
    ax.set_xlabel("Output order", fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(handles=handles, fontsize=12)
    plt.tight_layout()

    if save_path:
        save_plot(save_path, fig, use_date_subfolder=True)

    if show:
        plt.show()

    return fig


def plot_model_floor(
    errors: Dict[str, Sequence[float]],
    floors: Dict[str, Sequence[float]],
    *,
    channel_labels: Sequence[str] = SOLUTION_CHANNEL_LABELS,
    ylabel: str = "Relative $L_2$ error (%)",
    show: bool = True,
    save_path: Optional[str] = None,
) -> Figure:
    """Bar the test error of every surface field of every model against its reconstruction floor."""
    width = 0.8 / len(channel_labels)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for x, model in enumerate(errors):
        for c in range(len(channel_labels)):
            error, floor = 100 * errors[model][c], 100 * floors[model][c]
            position = x + (c - (len(channel_labels) - 1) / 2) * width
            ax.bar(position, error, width, color=f"C{c}")
            ax.text(position, error, f"{error:.1f}", ha="center", va="bottom", fontsize=8)
            ax.hlines(floor, position - width / 2, position + width / 2, color="black", linewidth=2)
            ax.text(position, floor, f"{floor:.1f}", ha="center", va="bottom", fontsize=8, color="white")

    handles = [patches.Patch(color=f"C{c}", label=label) for c, label in enumerate(channel_labels)]
    handles.append(Line2D([], [], color="black", linewidth=2, label="Reconstruction floor"))
    ax.set_xticks(range(len(errors)), list(errors))
    ax.set_ylabel(ylabel, fontsize=13)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(handles=handles, fontsize=11)
    plt.tight_layout()

    if save_path:
        save_plot(save_path, fig, use_date_subfolder=True)

    if show:
        plt.show()

    return fig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def save_plot(save_path: str | Path, figure: Figure, dpi: int = 150, use_date_subfolder: bool = False) -> None:
    """ Save a matplotlib figure to disk under a date-organised subfolder """
    save_path = Path(save_path)

    # Check for figure type. Default is PNG
    if save_path.suffix == "":
        save_path = save_path.with_suffix(".png")

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


def draw_mesh(ax, mesh: List[QuadNode]) -> None:
    """Outline the mesh cells on an imshow axis and label the panel with its token count."""
    rects = [patches.Rectangle((n.c0 - 0.5, n.r0 - 0.5), n.width, n.height) for n in mesh]
    ax.add_collection(PatchCollection(rects, facecolor="none", edgecolor="black", linewidth=0.3, alpha=0.6))
    ax.text(0.04, 0.02, f"{len(mesh)} tokens", transform=ax.transAxes, fontsize=9, va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8, linewidth=0))


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


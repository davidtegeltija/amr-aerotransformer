"""
========================================================================
Functions for mapping token-level predictions back to the original 
HxW high-resolution grid.
========================================================================

Two modes are provided:

1. **Nearest-fill** (fast, exact): each grid cell is filled with the
   prediction of the token that owns it.  This is equivalent to nearest-
   neighbour interpolation on the irregular quadtree grid.

2. **Bilinear interpolation** (quality): predictions are placed at token
   centres and then bicubically interpolated back onto the full grid.
   Useful for smoother output fields.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F

from src.amr.quadtree import QuadNode


# ---------------------------------------------------------------------------
# Differentiable reconstruction (for the dense training loss)
# ---------------------------------------------------------------------------

def tokens_to_grid_torch(
    token_preds: torch.Tensor,
    token_lists: List[List[QuadNode]],
    tokens_per_sample: List[int],
    H: int,
    W: int,
    leaf_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Differentiable nearest-fill reconstruction of a packed prediction batch.

    Builds, per sample, an integer owner map ``owner[H, W] -> packed token
    index`` (plain numpy ints — indices carry no gradient) and gathers the
    dense grid with advanced indexing, which keeps the autograd graph
    connected to ``token_preds``. This is the loss-side counterpart of
    ``tokens_to_grid`` (which detaches and is for visualization only).

    Args:
        token_preds: [total_N, output_channels] packed per-token predictions
            (any device, graph attached).
        token_lists: Length-B list of per-sample QuadNode leaf lists. Leaves
            of one sample must tile the full H x W grid exactly.
        tokens_per_sample: Per-sample token counts; must sum to total_N.
        H, W: Grid dimensions.
        leaf_weights: Optional [total_N] straight-through ancestry weights
            aligned with token_preds (forward value 1, gradient connects the
            dense loss to the scorer's subdivide decisions). Multiplied into
            the painted predictions when given.

    Returns:
        [B, H, W, output_channels] dense predictions on token_preds' device.
    """
    B = len(token_lists)
    assert sum(tokens_per_sample) == token_preds.shape[0], (
        f"tokens_per_sample sums to {sum(tokens_per_sample)}, "
        f"but token_preds has {token_preds.shape[0]} rows"
    )

    owners = np.empty((B, H, W), dtype=np.int64)
    offset = 0
    for b, leaves in enumerate(token_lists):
        for i, leaf in enumerate(leaves):
            owners[b, leaf.r0:leaf.r1, leaf.c0:leaf.c1] = offset + i
        offset += tokens_per_sample[b]

    owner_flat = torch.from_numpy(owners.reshape(-1)).to(token_preds.device)
    gathered = token_preds[owner_flat]                       # [B*H*W, C_out]
    if leaf_weights is not None:
        gathered = leaf_weights[owner_flat].unsqueeze(-1) * gathered
    return gathered.view(B, H, W, -1)


# ---------------------------------------------------------------------------
# Per-cell polynomial reconstruction (the affine_output path)
# ---------------------------------------------------------------------------
# The head emits a few coefficients per channel and a cell's pixels are painted
# with their weighted sum of basis functions. How many is the model's
# ``affine_output``, which is an order rather than a flag:
#
#   0  {1}                                            the plain constant head
#   1  {1, dx, dy}                                    the original affine head
#   2  {1, dx, dy, dx^2, dx*dy, dy^2}                 the quadratic head
#
# Order 2 is the one to train: the target inside a leaf is only ~7-dimensional
# and its three strongest modes are almost exactly the first three of these
# terms, so going up one order lowers the reconstruction floor 2.7x and shrinks
# the visible seam artefact from +54% to +11% at no token cost. The lower orders
# stay reachable so a new run can be compared against the older heads on the same
# code, and so old checkpoints (which stored affine_output as the bool True, i.e.
# order 1) still rebuild into the head they were trained with.
#
# The terms are mutually orthogonal over a cell, which is what keeps the training
# loss closed-form: every cross term drops, so the cell's squared error is one
# independent term per coefficient, cached per leaf by
# src/data/collate_fn.py:_affine_leaf_stats. The affine head already relied on this
# -- sum_xx / sum_yy are the term norms -- and it carries over to order 2 unchanged,
# except that dx^2 and dy^2 are not orthogonal to the constant until they are
# centred on their cell mean. cell_basis is where that centring is defined, and the
# collate reads its columns from here so the two paths cannot disagree.

# One name per basis column, in the order cell_basis builds them. Only used for
# logging, but it is the single place that says what coefficient k means.
OUTPUT_BASIS_TERMS = ("value", "gx", "gy", "gxx", "gxy", "gyy")
OUTPUT_BASIS_SIZE = len(OUTPUT_BASIS_TERMS)          # the full order-2 basis


def basis_size(order: int) -> int:
    """Coefficients per channel emitted by an ``affine_output`` head of this order.

    Order 0 is one term because a degree-0 polynomial is the constant alone, which
    is exactly the single number per channel the plain head predicts.

    Args:
        order: The model's ``affine_output``: 0, 1 or 2

    Returns:
        How many leading columns of ``cell_basis`` the head predicts: 1, 3 or 6.

    Raises:
        ValueError: for any other order.
    """
    sizes = {0: 1, 1: 3, 2: OUTPUT_BASIS_SIZE}
    if order not in sizes:
        raise ValueError(f"affine_output must be 0, 1 or 2, got {order!r}")
    return sizes[order]


@lru_cache(maxsize=None)
def cell_basis(h: int, w: int, n_taylor_terms: int = OUTPUT_BASIS_SIZE) -> np.ndarray:
    """Quadratic basis over an ``h x w`` cell, with mutually orthogonal columns.

    ``dx``/``dy`` are pixel offsets from the cell centre in units of the cell's own
    width/height, matching what the affine head's ``gx``/``gy`` already meant: a
    coefficient is the change across this cell, and reads the same at every
    refinement depth.

    Because the offsets are symmetric about the centre, every pair of columns is
    orthogonal already except ``(1, dx^2)`` and ``(1, dy^2)``. Subtracting each
    squared term's own cell mean fixes both, and makes ``dx^2`` orthogonal to
    ``dy^2`` too, so no orthogonalisation step is needed.

    A cell too small to resolve a term gets an exact zero column -- a 4x2 cell has
    two ``dx`` values, so ``dx^2`` is constant and centring it gives zero -- which
    drops that term out of the reconstruction and the loss on its own.

    A quadtree has one cell shape per depth, so the cache holds a handful of
    matrices for the whole dataset.

    Args:
        h: Cell height in pixels.
        w: Cell width in pixels.
        n_taylor_terms: How many terms of the expansion the head predicts for one
            cell, per channel (3 is exactly the affine head's 1, dx, dy). 
            Defaults to the full basis.

    Returns:
        ``[h*w, n_taylor_terms]`` float64 array, one column per term in
        ``OUTPUT_BASIS_TERMS``. float64 so ``_affine_leaf_stats`` can accumulate its
        target products in double precision; the painting path casts to float32.
    """
    dx = (np.arange(w) + 0.5 - w / 2.0) / max(w, 1)
    dy = (np.arange(h) + 0.5 - h / 2.0) / max(h, 1)
    DX, DY = np.meshgrid(dx, dy)
    DXX = DX ** 2 - (DX ** 2).mean()
    DYY = DY ** 2 - (DY ** 2).mean()
    terms = [np.ones_like(DX), DX, DY, DXX, DX * DY, DYY][:n_taylor_terms]
    return np.stack([t.ravel() for t in terms], axis=1)       # [h*w, n_taylor_terms]


def precompute_affine_geometry(
    token_lists: List[List[QuadNode]],
    tokens_per_sample: List[int],
    H: int,
    W: int,
    n_taylor_terms: int = OUTPUT_BASIS_SIZE,
) -> tuple:
    """Precompute the packed owner map and the per-pixel basis values.

    Pure geometry (no latents) so the result carries no gradient. Indices are into
    the PACKED token axis (offset by tokens_per_sample), matching
    tokens_to_grid_torch's owner convention.

    Args:
        token_lists: Length-B list of per-sample QuadNode leaf lists. Leaves of one
            sample must tile the full H x W grid exactly.
        tokens_per_sample: Per-sample token counts; prefix sums give packed offsets.
        H: Grid height (rows).
        W: Grid width (columns).
        n_taylor_terms: How many terms of the expansion the head predicts for one
            cell, per channel. It must match, or the painting broadcast fails.

    Returns:
        owner: LongTensor [B, H, W] packed token index owning each pixel.
        basis: FloatTensor [B, H, W, n_taylor_terms] the owning cell's basis
            functions evaluated at that pixel.

    Raises:
        ValueError: if the leaves leave any pixel unowned.
    """
    B = len(token_lists)
    owners = np.full((B, H, W), -1, dtype=np.int64)
    basis = np.zeros((B, H, W, n_taylor_terms), dtype=np.float32)

    offset = 0
    for b, leaves in enumerate(token_lists):
        for i, leaf in enumerate(leaves):
            r0, c0, r1, c1 = leaf.r0, leaf.c0, leaf.r1, leaf.c1
            owners[b, r0:r1, c0:c1] = offset + i
            basis[b, r0:r1, c0:c1] = cell_basis(r1 - r0, c1 - c0, n_taylor_terms).reshape(r1 - r0, c1 - c0, -1)
        offset += tokens_per_sample[b]

    if (owners < 0).any():
        raise ValueError("leaves leave pixels unowned; they must tile the grid")

    return torch.from_numpy(owners), torch.from_numpy(basis)


def tokens_to_grid_affine_torch(
    affine_params: torch.Tensor,
    geom: tuple,
    H: int,
    W: int,
    output_channels: int,
) -> torch.Tensor:
    """Differentiable per-cell polynomial reconstruction.

    Paints each leaf's box with the weighted sum of its basis functions. This is
    the loss-side counterpart of tokens_to_grid_torch.

    Args:
        affine_params: [total_N, C, K] packed per-token coefficients, graph attached.
        geom: (owner, basis) from precompute_affine_geometry; owner [B,H,W] long,
            basis [B,H,W,K] float. Its K must match affine_params'.
        H: Grid height (rows).
        W: Grid width (columns).
        output_channels: C.

    Returns:
        [B, H, W, C] dense predictions on affine_params' device.
    """
    owner, basis = geom
    owner = owner.to(affine_params.device)
    basis = basis.to(affine_params.device)
    p = affine_params[owner.reshape(-1)]              # [B*H*W, C, K] (gather, keeps grad)
    dense = (p * basis.reshape(-1, 1, basis.shape[-1])).sum(dim=-1)     # [B*H*W, C]
    return dense.view(-1, H, W, output_channels)


def tokens_to_grid_affine(
    affine_params: torch.Tensor,
    token_list: List[QuadNode],
    H: int,
    W: int,
    output_channels: int,
) -> torch.Tensor:
    """Non-differentiable polynomial reconstruction for a single sample (viz path).

    Thin wrapper over precompute_affine_geometry + tokens_to_grid_affine_torch
    under no_grad, so the example script and prediction_visualization reuse one
    code path for the polynomial head.

    Args:
        affine_params: [N, C, K] per-token coefficients for one sample. K says which
            order produced them, so the basis is built to match.
        token_list: The sample's QuadNode leaves (tile the full H x W grid).
        H: Grid height (rows).
        W: Grid width (columns).
        output_channels: C.

    Returns:
        [H, W, C] float32 tensor on CPU.
    """
    with torch.no_grad():
        geom = precompute_affine_geometry([token_list], [len(token_list)], H, W, affine_params.shape[-1])
        dense = tokens_to_grid_affine_torch(
            affine_params, geom, H, W, output_channels
        )                                            # [1, H, W, C]
    return dense[0].cpu()


# ---------------------------------------------------------------------------
# Core reconstruction function
# ---------------------------------------------------------------------------

def tokens_to_grid(
    predictions: torch.Tensor,
    token_list: List[QuadNode],
    H: int,
    W: int,
    output_channels: int,
    mode: Literal["fill", "interp", "smooth"],
) -> torch.Tensor:
    """
    Reconstruct the full [H, W, output_channels] prediction grid from token predictions.

    Parameters
    ----------
    predictions     : [N, output_channels] - per-token flow predictions (on any device)
    token_list      : list of N QuadNode objects
    H, W            : original grid dimensions
    output_channels : number of output channels (e.g. 3 for u, v, p)
    mode            : "fill"   - fast nearest-fill (default)
                      "interp" - bilinear interpolation from token centres
                      "smooth" - nearest-fill then a Gaussian low-pass blur
                                 (viz-only cosmetic seam smoothing; see
                                 _smooth_reconstruction)

    Returns
    -------
    grid : [H, W, output_channels] float32 tensor (on CPU)
    """
    assert mode in ("fill", "interp", "smooth"), f"Unknown mode: {mode}"
    preds_np = predictions.detach().cpu().numpy()  # [N, output_channels]

    if mode == "fill":
        # Nearest-fill
        return _fill_reconstruction(preds_np, token_list, H, W, output_channels)
    elif mode == "smooth":
        # Nearest-fill then cosmetic Gaussian blur
        return _smooth_reconstruction(preds_np, token_list, H, W, output_channels)
    else:
        # Bilinear interpolation
        return _interp_reconstruction(preds_np, token_list, H, W, output_channels)

def _fill_reconstruction(
    preds: np.ndarray,
    token_list: List[QuadNode],
    H: int,
    W: int,
    output_channels: int,
) -> torch.Tensor:
    """Fill the bounding box of each token with its prediction."""
    grid = np.zeros((H, W, output_channels), dtype=np.float32)
    # Process tokens from coarsest (largest cells) to finest so that finer
    # cells overwrite coarser ones - consistent with AMR multi-scale storage.
    order = sorted(range(len(token_list)), key=lambda i: -(token_list[i].width * token_list[i].height))
    for idx in order:
        t = token_list[idx]
        grid[t.r0:t.r1, t.c0:t.c1] = preds[idx]
    return torch.from_numpy(grid)

def _smooth_reconstruction(
    preds: np.ndarray,
    token_list: List[QuadNode],
    H: int,
    W: int,
    output_channels: int,
) -> torch.Tensor:
    """Nearest-fill, then a cosmetic Gaussian low-pass blur (viz-only).

    A zero-training sanity check (continuous_prediction_borders_2.md, appendix
    Option 1): paints the blocky nearest-fill grid, then smooths the cell-border
    steps with a per-channel Gaussian filter whose bandwidth tracks the smallest
    cell so blocks larger than the finest detail are softened. This is purely
    cosmetic and inconsistent with the (nearest-fill) training loss, so it does
    not improve accuracy — it only reveals how much blockiness is reconstruction
    vs. genuine model error. It also blurs the sharp shear layer, where error is
    already highest.

    Args:
        preds: [N, output_channels] per-token predictions (numpy).
        token_list: The sample's QuadNode leaves.
        H: Grid height (rows).
        W: Grid width (columns).
        output_channels: C.

    Returns:
        [H, W, output_channels] float32 tensor on CPU.
    """
    from scipy.ndimage import gaussian_filter

    grid = _fill_reconstruction(preds, token_list, H, W, output_channels).numpy()
    # Bandwidth ~ smallest leaf side (in pixels); 0 on the channel axis so
    # channels are filtered independently.
    s = min(min(t.height, t.width) for t in token_list)
    smoothed = gaussian_filter(grid, sigma=(s, s, 0), mode="nearest")
    return torch.from_numpy(smoothed)

def _interp_reconstruction(
    preds: np.ndarray,
    token_list: List[QuadNode],
    H: int,
    W: int,
    output_channels: int,
) -> torch.Tensor:
    """
    Scatter token predictions to their centres on a sparse grid, then
    bilinearly upsample to full resolution using PyTorch's grid_sample.

    This gives smoother transitions across cell boundaries.
    """
    # Build a sparse canvas at the resolution of the finest tokens
    # We use the full H×W canvas with scattered values + count for averaging
    sum_grid   = np.zeros((H, W, output_channels), dtype=np.float64)
    count_grid = np.zeros((H, W, 1), dtype=np.float64)

    for idx, t in enumerate(token_list):
        # Place prediction at the centre pixel of the token bounding box
        rc = (t.r0 + t.r1) // 2
        cc = (t.c0 + t.c1) // 2
        rc = max(0, min(H - 1, rc))
        cc = max(0, min(W - 1, cc))
        sum_grid[rc, cc]   += preds[idx]
        count_grid[rc, cc] += 1.0

    # Average at pixels that received multiple contributions
    with np.errstate(invalid='ignore'):
        avg_grid = np.where(count_grid > 0, sum_grid / count_grid, 0.0)

    # Sparse -> dense interpolation via PyTorch
    # [H, W, C] -> [1, C, H, W]
    canvas = torch.from_numpy(avg_grid.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
    # Bicubic upsample back to full resolution (identity if already H×W)
    out = F.interpolate(canvas, size=(H, W), mode='bicubic', align_corners=True)
    return out.squeeze(0).permute(1, 2, 0)  # [H, W, C]


@torch.no_grad()
def _tokens_to_grid_idw(
    predictions: torch.Tensor,
    token_list: List[QuadNode],
    H: int,
    W: int,
    output_channels: int,
    bandwidth: float = 0.75,
    chunk: int = 8192,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Scattered-data reconstruction by scale-aware inverse-distance weighting.

    Each token's prediction is treated as a sample at its centroid; every output
    pixel is a normalised distance-weighted blend of all token values, with a
    per-token Gaussian bandwidth proportional to the cell size. Coarse cells get
    broad, smooth influence (recovering intra-cell gradients); fine cells stay
    local (keeping shocks sharp). The field is smooth everywhere and seam-free.

    Args:
        predictions: [N, output_channels] per-token predictions (any device).
        token_list: N QuadNode leaves tiling the H x W grid.
        H, W: Grid dimensions.
        output_channels: C.
        bandwidth: Gaussian sigma as a multiple of each token's cell size (px).
            Smaller -> sharper / more local; larger -> smoother. ~0.5-0.8 is sane.
        chunk: Pixels per chunk for the [chunk, N] distance matrix.
        eps: Weight-sum floor.

    Returns:
        [H, W, output_channels] float32 tensor on CPU.
    """
    device = predictions.device
    vals = predictions.reshape(len(token_list), output_channels).float()

    cx = torch.tensor([(t.c0 + t.c1) * 0.5 for t in token_list], device=device)
    cy = torch.tensor([(t.r0 + t.r1) * 0.5 for t in token_list], device=device)
    size = torch.tensor([max(t.c1 - t.c0, t.r1 - t.r0) for t in token_list],
                        device=device, dtype=torch.float32)
    inv2h2 = 1.0 / (2.0 * (bandwidth * size).clamp_min(1.0) ** 2)   # [N]

    rows = torch.arange(H, device=device) + 0.5
    cols = torch.arange(W, device=device) + 0.5
    gy, gx = torch.meshgrid(rows, cols, indexing="ij")
    py, px = gy.reshape(-1), gx.reshape(-1)                          # [P]
    P = px.numel()

    out = torch.empty((P, output_channels), device=device, dtype=torch.float32)
    for s in range(0, P, chunk):
        e = min(s + chunk, P)
        dx = px[s:e, None] - cx[None, :]                            # [p, N]
        dy = py[s:e, None] - cy[None, :]
        w = torch.exp(-(dx * dx + dy * dy) * inv2h2[None, :])        # [p, N]
        w = w / (w.sum(1, keepdim=True) + eps)                       # partition of unity
        out[s:e] = w @ vals                                         # [p, C]
    return out.reshape(H, W, output_channels).cpu()


# ---------------------------------------------------------------------------
# Batch reconstruction (list of samples)
# ---------------------------------------------------------------------------

def batch_tokens_to_grid(
    predictions: torch.Tensor,
    token_lists: List[List[QuadNode]],
    tokens_per_sample: List[int],
    H: int,
    W: int,
    output_channels: int,
    mode: str = "fill",
) -> torch.Tensor:
    """
    Reconstruct a batch of grids from a packed prediction tensor.

    Parameters
    ----------
    predictions     : [total_N, output_channels]
    token_lists     : list of B token lists (one per sample)
    tokens_per_sample : list of B token counts (must sum to total_N)
    H, W            : grid dimensions
    output_channels : output channels
    mode            : "fill" or "interp"

    Returns
    -------
    grids : [B, H, W, output_channels]
    """
    B = len(token_lists)
    grids = []
    offset = 0
    for b in range(B):
        L = tokens_per_sample[b]
        preds_b = predictions[offset:offset + L]
        grid_b  = tokens_to_grid(preds_b, token_lists[b], H, W, output_channels, mode=mode)
        grids.append(grid_b)
        offset += L
    return torch.stack(grids, dim=0)  # [B, H, W, output_channels]

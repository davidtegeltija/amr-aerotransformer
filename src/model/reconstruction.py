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

from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from src.amr.quadtree_tokenizer import QuadNode


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
# Core reconstruction function
# ---------------------------------------------------------------------------

def tokens_to_grid(
    predictions: torch.Tensor,
    token_list: List[QuadNode],
    H: int,
    W: int,
    output_channels: int,
    mode: str = "fill",
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

    Returns
    -------
    grid : [H, W, output_channels] float32 tensor (on CPU)
    """
    assert mode in ("fill", "interp"), f"Unknown mode: {mode}"
    preds_np = predictions.detach().cpu().numpy()  # [N, output_channels]

    if mode == "fill":
        # Nearest-fill
        return _fill_reconstruction(preds_np, token_list, H, W, output_channels)
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

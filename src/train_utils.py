"""
========================================================================
Pure helpers used by the two-phase training loops in src/train.py.
========================================================================

Functions
---------
tau_schedule                : exponential decay of the Gumbel-Softmax temperature.
compute_smooth_loss         : total-variation regularizer on the CNN score map.
average_targets_per_token   : per-leaf mean of a full-resolution target grid, ordered to match the transformer's packed-token sequence.
"""

from pathlib import Path
from typing import List

import torch

from src.amr.quadtree import QuadNode

# ---------------------------------------------------------------------------
# Save the model at its current state
# ---------------------------------------------------------------------------
def save_checkpoint(checkpoint_path, checkpoint_name, model=None, optimizer=None, scheduler=None):
    save_path = Path(checkpoint_path) / checkpoint_name
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "model": model.state_dict() if model else None,
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None}, save_path)
    
# ---------------------------------------------------------------------------
# Gumbel-Softmax temperature schedule
# ---------------------------------------------------------------------------

def tau_schedule(epoch: int, tau_start: float, tau_end: float, T: int) -> float:
    """
    Exponential decay of the Gumbel-Softmax temperature.

        tau(epoch) = tau_start * (tau_end / tau_start) ** (epoch / max(1, T - 1))

    Clamped at ``tau_end`` for ``epoch >= T - 1``. Defined for ``epoch >= 0``.
    """
    if T <= 1:
        return tau_end
    progress = min(max(epoch, 0), T - 1) / (T - 1)
    return tau_start * (tau_end / tau_start) ** progress


# ---------------------------------------------------------------------------
# Spatial smoothness loss (total variation)
# ---------------------------------------------------------------------------

def compute_smooth_loss(score_map: torch.Tensor) -> torch.Tensor:
    """
    Total-variation smoothness loss on a score map.

    Args:
        score_map: ``[B, 1, H, W]`` or ``[B, H, W]`` — works with either by
            squeezing the channel axis when present.

    Returns:
        Scalar tensor: ``mean(|∂/∂W|) + mean(|∂/∂H|)``. Unnormalized by image
        size; with a fixed 256x128 grid the scale is absorbed into
        ``lambda_smooth``.
    """
    if score_map.dim() == 4:
        assert score_map.size(1) == 1, \
            f"expected channel dim 1, got {score_map.size(1)}"
        score_map = score_map.squeeze(1)
    assert score_map.dim() == 3, f"expected [B,H,W], got {tuple(score_map.shape)}"

    dh = (score_map[:, 1:, :] - score_map[:, :-1, :]).abs().mean()
    dw = (score_map[:, :, 1:] - score_map[:, :, :-1]).abs().mean()
    return dh + dw


# ---------------------------------------------------------------------------
# Per-token target averaging
# ---------------------------------------------------------------------------

def average_targets_per_token(
    targets: torch.Tensor,
    token_lists: List[List[QuadNode]],
) -> torch.Tensor:
    """
    Compute per-leaf target means from a full-resolution target grid.

    For each leaf ``QuadNode`` with bbox ``(r0, c0, r1, c1)``, average
    ``targets[b, r0:r1, c0:c1, :]`` over the spatial axes to produce one
    row of the output. Leaves are consumed in the same order as
    ``token_lists[b]``, so the output row order matches the transformer's
    packed-token sequence.

    Args:
        targets: ``[B, H, W, output_dim]`` tensor on any device.
        token_lists: length-B list; each element is the per-sample list of
            leaf ``QuadNode``s produced by the score-guided mesh builder.

    Returns:
        ``packed_targets``: ``[total_N, output_dim]`` on the same device as
        ``targets``.
    """
    assert targets.dim() == 4, \
        f"expected [B,H,W,D], got {tuple(targets.shape)}"
    B = targets.size(0)
    assert len(token_lists) == B, \
        f"token_lists len {len(token_lists)} != B {B}"

    rows = []
    for b in range(B):
        for leaf in token_lists[b]:
            r0, c0, r1, c1 = leaf.bbox
            patch = targets[b, r0:r1, c0:c1, :]
            rows.append(patch.mean(dim=(0, 1)))

    return torch.stack(rows, dim=0)


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # tau_schedule
    assert abs(tau_schedule(0, 5.0, 0.5, 10) - 5.0) < 1e-6
    assert abs(tau_schedule(9, 5.0, 0.5, 10) - 0.5) < 1e-6
    mid = tau_schedule(5, 5.0, 0.5, 10)
    assert 0.5 < mid < 5.0
    print(f"tau_schedule OK (mid={mid:.4f})")

    # compute_smooth_loss
    uniform = torch.full((2, 1, 16, 16), 0.5)
    assert compute_smooth_loss(uniform).item() < 1e-6
    checker = torch.zeros(2, 16, 16)
    checker[:, ::2, ::2] = 1.0
    checker[:, 1::2, 1::2] = 1.0
    assert compute_smooth_loss(checker).item() > 0.4
    print("compute_smooth_loss OK")

    # average_targets_per_token
    from src.amr.quadtree import QuadNode
    grid_targets = torch.arange(2 * 8 * 8 * 3, dtype=torch.float32).reshape(2, 8, 8, 3)
    leaf_top    = QuadNode(bbox=(0, 0, 4, 8), depth=1, is_leaf=True)
    leaf_bottom = QuadNode(bbox=(4, 0, 8, 8), depth=1, is_leaf=True)
    token_lists = [[leaf_top, leaf_bottom], [leaf_top, leaf_bottom]]
    packed = average_targets_per_token(grid_targets, token_lists)
    assert packed.shape == (4, 3)
    assert packed[0, 0] < packed[1, 0]
    print("average_targets_per_token OK")

    print("All smoke tests passed.")

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def nmse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
    channel_dim: int = -1,
) -> torch.Tensor:
    """Normalised Mean Squared Error (per-channel, scale-invariant).

    Divides the per-element squared error by the per-channel variance of the
    target, so output quantities with very different magnitudes (e.g. velocity
    vs. pressure) contribute comparably. Without this normalisation a single
    large-magnitude channel dominates the loss purely through its units.

    Args:
        pred: Predicted tensor, any shape, with a channel axis.
        target: Ground-truth tensor, same shape as ``pred``.
        eps: Numerical floor added to each per-channel variance.
        channel_dim: Axis holding the output channels. Defaults to the last
            axis (-1), matching the ``[..., C]`` token / dense-grid layout.
            The ViT path uses channel-first tensors and passes ``channel_dim=1``.

    Returns:
        Scalar tensor: mean over all elements of squared-error / per-channel
        target variance.
    """
    cdim = channel_dim % target.dim()
    reduce_dims = [d for d in range(target.dim()) if d != cdim]
    # Per-channel variance of the target (population variance), broadcast back
    # over the reduced axes.
    var = target.var(dim=reduce_dims, unbiased=False, keepdim=True)
    return ((pred - target) ** 2 / (var + eps)).mean()


def affine_nmse_loss(
    affine_params: torch.Tensor,
    stats: Dict[str, torch.Tensor],
    eps: float = 1e-8,
) -> torch.Tensor:
    """Dense per-pixel NMSE of an affine-per-cell prediction, in closed form.

    Numerically equal, in value and in gradient, to

        nmse_loss(tokens_to_grid_affine_torch(affine_params, geom, H, W, C), targets)

    but evaluated over the ``[total_N, C]`` token axis instead of the
    ``[B, H, W, C]`` pixel grid, so no owner map, no per-pixel gather and no dense
    target are needed. Everything that depends on the target is precomputed once
    per sample by ``src/data/collate_fn.py:_affine_leaf_stats``; see that function
    for the derivation.

    ``nmse_loss`` normalises by the per-channel variance of the batch's dense
    target and averages over ``B*H*W*C``. Both are recovered from the leaf
    statistics: the leaves tile the grid, so ``sum(num_pixels)`` is exactly
    ``B*H*W`` and the target's first two moments are the leaf moments summed.

    Args:
        affine_params: ``[total_N, C, 3]`` packed per-token (value, gx, gy), graph
            attached.
        stats: Batch-concatenated ``_affine_leaf_stats`` output, on the same device
            as ``affine_params`` and with rows in the same packed order.
        eps: Numerical floor added to each per-channel variance, matching
            ``nmse_loss``.

    Returns:
        Scalar loss tensor.
    """
    value, gx, gy = affine_params[..., 0], affine_params[..., 1], affine_params[..., 2]
    num_pixels = stats["num_pixels"].unsqueeze(1)                        # [N, 1]
    mean_target = stats["mean_target"]                                   # [N, C]
    sum_sq_resid = stats["sum_sq_resid"]                                 # [N, C]

    sse = (num_pixels * (value - mean_target) ** 2
           + stats["sum_xx"].unsqueeze(1) * gx * gx - 2.0 * gx * stats["sum_target_dx"]
           + stats["sum_yy"].unsqueeze(1) * gy * gy - 2.0 * gy * stats["sum_target_dy"]
           + sum_sq_resid)                                               # [N, C]

    total_pixels = num_pixels.sum()                                      # B*H*W
    mean_t = (num_pixels * mean_target).sum(dim=0) / total_pixels        # [C]
    sq_t = (sum_sq_resid + num_pixels * mean_target ** 2).sum(dim=0) / total_pixels
    var = sq_t - mean_t ** 2                                             # [C]

    return (sse.sum(dim=0) / (var + eps)).sum() / (total_pixels * affine_params.shape[1])


def smooth_loss(score_map: torch.Tensor) -> torch.Tensor:
    """
    Total-variation regularizer on the CNN score map.

    Penalizes large differences between neighboring cells (horizontal and vertical) in the
    importance scores outputed by the RefinementNet.
    (real aerodynamic importance regions like shocks and leading edges are spatially contiguous).

    Args:
        score_map:  [B, 1, H, W] or [B, H, W] — works with either by
                    squeezing the channel axis when present.

    Returns:
        Scalar tensor:  mean(|∂/∂W|) + mean(|∂/∂H|). Unnormalized by image
                        size; with a fixed 256x128 grid the scale is absorbed into
                        lambda_smooth.
    """
    if score_map.dim() == 4:
        assert score_map.size(1) == 1, f"expected channel dim 1, got {score_map.size(1)}"
        score_map = score_map.squeeze(1)
    assert score_map.dim() == 3, f"expected [B,H,W], got {tuple(score_map.shape)}"

    dh = (score_map[:, 1:, :] - score_map[:, :-1, :]).abs().mean()
    dw = (score_map[:, :, 1:] - score_map[:, :, :-1]).abs().mean()
    return dh + dw


def budget_loss(soft_N: torch.Tensor, n_target: int, n_floor: int = 16) -> torch.Tensor:
    """
    Hinge penalty on exceeding a target token budget.

    Without it, the scorer would learn to always refine everything (maximum
    tokens) since that gives the lowest dense prediction error. Unlike the
    earlier monotone ``(N / n_max) ** 2`` (whose optimum was zero refinement
    and which single-handedly collapsed the mesh), this form is exactly zero
    below the budget and only pushes back once the mesh overshoots it.

    soft_N counts non-forced "subdivide" draws; each subdivide replaces one
    leaf with four, i.e. adds 3 net tokens on top of the min_depth floor, so
    the differentiable token-count surrogate is ``n_floor + 3 * soft_N``.

    Args:
        soft_N: Scalar tensor — differentiable count of non-forced subdivide
            draws from the Gumbel-softmax tree builder.
        n_target: Target token budget (e.g. 256 for the 256x128 grid).
        n_floor: Token count of the fully-collapsed min_depth mesh
            (4 ** min_depth when the grid permits full min_depth subdivision).

    Returns:
        Scalar tensor: ``relu(n_floor + 3 * soft_N - n_target)^2 / n_target^2``.
    """
    n_surrogate = n_floor + 3.0 * soft_N
    return (torch.relu(n_surrogate - n_target) / n_target) ** 2


# ---------------------------------------------------------------------------
# Scorer loss — supervised regression of predicted depth to the oracle depth
# ---------------------------------------------------------------------------
# The scorer emits a depth map ``d_pred``, trained by regression to the
# variance-oracle depth target (``src/amr/oracle_depth.py``), decoupled from the
# transformer. ``scorer_depth_loss`` combines:
#   * regression (default): smooth-L1 of ``d_pred`` vs the integer oracle depth
#     (``beta=1`` == one depth level);
#   * TV (optional, small): keeps hot regions contiguous (non-zero at the oracle);
#   * decision-consistency (optional, off): per-cell max-reduced hinge pushing each
#     depth-``d`` cell to the correct side of ``d`` (smooth max -> hard max as the
#     temperature grows). Note: inference subdivides on the cell *mean* of
#     ``d_pred``; this term is a conservative max-side approximation of it.
# Loss anchor: at ``d_pred == oracle`` the supervised loss is zero (smooth-L1(x,x)=0,
# decision term in the hard-max limit with zero margin, TV weight 0).


def _masked_smooth_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: Optional[torch.Tensor],
    beta: float,
) -> torch.Tensor:
    """Smooth-L1 reduced over valid pixels only."""
    per_elem = F.smooth_l1_loss(pred, target, beta=beta, reduction="none")
    if valid_mask is None:
        return per_elem.mean()
    m = valid_mask.to(per_elem.dtype)
    denom = m.sum().clamp_min(1.0)
    return (per_elem * m).sum() / denom


def _cell_reduce_max(
    x: torch.Tensor,
    kernel: Tuple[int, int],
    temp: Optional[float],
) -> torch.Tensor:
    """Reduce ``x`` [B,1,H,W] over non-overlapping ``kernel`` cells.

    With ``temp is None`` this is a hard max. With a finite ``temp`` it is a
    numerically-stabilised smooth max
    ``(1/T) * logsumexp(T * x)`` over each cell, which approaches the hard max as
    ``T`` grows but gives gradient to every pixel in the cell.
    """
    kh, kw = kernel
    if temp is None:
        return F.max_pool2d(x, kernel_size=(kh, kw), stride=(kh, kw))
    # Stabilised log-sum-exp pooling: subtract the per-cell hard max first.
    hard = F.max_pool2d(x, kernel_size=(kh, kw), stride=(kh, kw))
    hard_up = F.interpolate(hard, scale_factor=(kh, kw), mode="nearest")
    summ = F.avg_pool2d(torch.exp(temp * (x - hard_up)), kernel_size=(kh, kw),
                        stride=(kh, kw)) * (kh * kw)
    return hard + torch.log(summ) / temp


def _decision_term(
    pred: torch.Tensor,
    oracle: torch.Tensor,
    *,
    min_depth: int,
    max_depth: int,
    margin: float,
    temp: Optional[float],
) -> torch.Tensor:
    """Multi-scale decision-consistency hinge over uniform depth-``d`` cells.

    Implemented with uniform ``(H/2^d, W/2^d)`` pooling, which is exact for a
    power-of-two grid (e.g. 256x128); ``H`` and ``W`` must be divisible by
    ``2^d`` at every decision depth.
    """
    B, _, H, W = pred.shape
    total = pred.new_tensor(0.0)
    n_terms = 0
    for d in range(min_depth, max_depth):
        kh, kw = H // (2 ** d), W // (2 ** d)
        if kh < 1 or kw < 1 or H % (2 ** d) != 0 or W % (2 ** d) != 0:
            break
        cell_pred = _cell_reduce_max(pred, (kh, kw), temp)            # [B,1,h_d,w_d]
        cell_oracle_max = F.max_pool2d(oracle, (kh, kw), (kh, kw))
        subdivide = (cell_oracle_max > d).to(pred.dtype)             # 1 where deeper
        # subdivide cells want cell_pred > d; stop cells want cell_pred <= d.
        hinge_sub = F.relu(d + margin - cell_pred)
        hinge_stop = F.relu(cell_pred - d + margin)
        total = total + (subdivide * hinge_sub + (1.0 - subdivide) * hinge_stop).mean()
        n_terms += 1
    return total / max(1, n_terms)


def scorer_depth_loss(
    pred: torch.Tensor,
    oracle: torch.Tensor,
    *,
    valid_mask: Optional[torch.Tensor] = None,
    beta: float = 1.0,
    tv_weight: float = 0.0,
    decision_weight: float = 0.0,
    min_depth: int = 2,
    max_depth: int = 5,
    margin: float = 0.0,
    decision_temp: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Supervised scorer loss: depth regression (+ optional TV / decision terms).

    Args:
        pred: ``[B, 1, H, W]`` predicted depth ``d_pred``.
        oracle: ``[B, 1, H, W]`` oracle depth target (int or float).
        valid_mask: Optional ``[B, 1, H, W]`` bool mask; the regression term is
            reduced over valid pixels only.
        beta: Smooth-L1 transition (1.0 = one depth level).
        tv_weight: Weight of the (optional, small) TV regulariser on ``d_pred``.
        decision_weight: Weight of the optional decision-consistency term
            (0.0 = off, the default).
        min_depth, max_depth: Decision-depth range for the decision term.
        margin: Decision hinge margin.
        decision_temp: Smooth-max temperature for the decision term; ``None``
            uses the hard max (the anchor limit).

    Returns:
        ``(loss, components)`` where ``components`` holds the scalar value of
        each term for logging.
    """
    oracle = oracle.to(pred.dtype)
    reg = _masked_smooth_l1(pred, oracle, valid_mask, beta)
    loss = reg
    components = {"reg": float(reg.detach())}

    if tv_weight > 0.0:
        tv = smooth_loss(pred)
        loss = loss + tv_weight * tv
        components["tv"] = float(tv.detach())

    if decision_weight > 0.0:
        dec = _decision_term(
            pred, oracle,
            min_depth=min_depth, max_depth=max_depth,
            margin=margin, temp=decision_temp,
        )
        loss = loss + decision_weight * dec
        components["decision"] = float(dec.detach())

    components["total"] = float(loss.detach())
    return loss, components

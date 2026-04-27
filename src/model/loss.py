import torch


def nmse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Normalised Mean Squared Error (scale-invariant).

    Divides the per-element squared error by the per-channel variance of the
    target, making the loss scale-invariant across output quantities that
    may have very different magnitudes (e.g. velocity vs. pressure)
    """
    var = target.var(dim=0, keepdim=True).clamp(min=eps)
    return ((pred - target) ** 2 / var).mean()


def smooth_loss(score_map: torch.Tensor) -> torch.Tensor:
    """
    Total-variation smoothness loss on a score map.
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


def budget_loss(soft_N: torch.Tensor, n_max: int) -> torch.Tensor:
    """
    Penalizes using too many tokens. Without it, the scorer would learn to always refine everything (maximum tokens)
    since that gives the lowest prediction error. This term forces the model to be selective — only refine where it actually matters.
    soft_N is the differentiable (soft) estimate of how many tokens are selected via Gumbel-softmax.

    Args:
        soft_N: scalar tensor — differentiable token count estimate from Gumbel-softmax selection.
        n_max:  maximum possible token count (e.g. 8192 for a 256x128 grid).

    Returns:
        Scalar tensor: ``(soft_N / n_max) ** 2``, in [0, 1].
    """
    return (soft_N / n_max) ** 2
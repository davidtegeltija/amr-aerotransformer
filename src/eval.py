"""
========================================================================
Evaluation loop for the Adaptive Mesh CFD model.
========================================================================

Contents
--------
evaluate - runs the model over a DataLoader and returns mean NMSE loss
"""

import torch
from torch.utils.data import DataLoader

from src.model.amr_model import AdaptiveMeshAeroModel


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


@torch.no_grad()
def evaluate(
    model: AdaptiveMeshAeroModel,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    for batch in loader:
        packed_tokens  = batch["packed_tokens"].to(device)
        packed_targets = batch["packed_targets"].to(device)
        seq_lens       = batch["seq_lens"]
        out = model(packed_tokens, seq_lens)
        total_loss += nmse_loss(out["token_preds"], packed_targets).item()
    model.train()
    return total_loss / len(loader)
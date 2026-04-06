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
from src.train import nmse_loss


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
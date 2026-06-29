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

from src.model.loss import nmse_loss
from src.model.amr_model import AdaptiveMeshAeroModel
from src.model.reconstruction import (
    precompute_affine_geometry,
    tokens_to_grid_affine_torch,
)


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
        tokens_per_sample = batch["tokens_per_sample"]
        out = model(packed_tokens, tokens_per_sample)
        if model.affine_output:
            # Mirror the dense affine training loss so train/val are comparable.
            dense_targets = batch["targets"].to(device)
            Hd, Wd = dense_targets.shape[1], dense_targets.shape[2]
            geom = precompute_affine_geometry(
                batch["token_lists"], tokens_per_sample, Hd, Wd)
            dense_pred = tokens_to_grid_affine_torch(
                out["token_preds"], geom, Hd, Wd, model.output_channels)
            total_loss += nmse_loss(dense_pred, dense_targets).item()
        else:
            total_loss += nmse_loss(out["token_preds"], packed_targets).item()
    model.train()
    return total_loss / len(loader)
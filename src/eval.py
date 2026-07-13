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
from src.model.refinement_net import RefinementNet
from src.model.vit import ViT
from src.model.loss import nmse_loss, scorer_depth_loss
from src.model.reconstruction import (
    precompute_affine_geometry,
    tokens_to_grid_affine_torch,
)


@torch.no_grad()
def evaluate_transformer(
    model: AdaptiveMeshAeroModel,
    val_loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    for batch in val_loader:
        packed_tokens  = batch["packed_tokens"].to(device)
        packed_targets = batch["packed_targets"].to(device)
        tokens_per_sample = batch["tokens_per_sample"]
        out = model(packed_tokens, tokens_per_sample)
        if model.affine_output:
            # Mirror the dense affine training loss so train/val are comparable.
            dense_targets = batch["targets"].to(device)
            Hd, Wd = dense_targets.shape[1], dense_targets.shape[2]
            geom = precompute_affine_geometry(batch["token_lists"], tokens_per_sample, Hd, Wd)
            dense_pred = tokens_to_grid_affine_torch(out["token_preds"], geom, Hd, Wd, model.output_channels)
            total_loss += nmse_loss(dense_pred, dense_targets).item()
        else:
            total_loss += nmse_loss(out["token_preds"], packed_targets).item()

    model.train()
    return total_loss / len(val_loader)


@torch.no_grad()
def evaluate_scorer(
    scorer: RefinementNet, 
    val_loader: DataLoader,
    device: torch.device,
    *, 
    max_depth, 
    min_depth, 
    tv_weight,
    decision_weight, 
    decision_margin, 
    decision_temp,
) -> float:
    """Mean supervised scorer loss over the val split."""
    scorer.eval()
    total_loss = 0.0
    for batch in val_loader:
        grids = batch["grids"].to(device)
        oracle = batch["oracle_depth"].to(device)
        d_pred = scorer(grids)
        _, comp = scorer_depth_loss(
            d_pred, oracle,
            tv_weight=tv_weight, decision_weight=decision_weight,
            min_depth=min_depth, max_depth=max_depth,
            margin=decision_margin, decision_temp=decision_temp,
        )
        total_loss += comp["total"]

    scorer.train()
    return total_loss / len(val_loader)


@torch.no_grad()
def evaluate_vit(
    model: ViT, 
    val_loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    for batch in val_loader:
        grids   = batch["grids"].to(device).permute(0, 3, 1, 2).float()
        targets = batch["targets"].to(device).permute(0, 3, 1, 2).float()
        total_loss += nmse_loss(model(grids), targets, channel_dim=1).item()

    model.train()
    return total_loss / len(val_loader)
"""
========================================================================
Training pipeline for the Adaptive Mesh CFD model.
========================================================================

Key design decisions
--------------------
1. **Sequence packing** (APT / NaViT style): instead of padding variable-
   length token sequences with zeros, we concatenate all tokens in a batch
   into a single packed tensor and use a block-diagonal attention mask.
   This is natively handled by FlashAttention-2's varlen API, so there is
   zero memory wasted on padding and zero extra overhead.

2. **Warmup LR schedule** (AMR-Transformer style):
       lr(t) = (1 / sqrt(d_model)) * min(t^{-0.5}, t · warmup^{-1.5})
   This is the "Transformer warmup" from the original Attention is All You
   Need paper, adapted from the AMR-Transformer implementation.

3. **NMSE loss**: normalised MSE = MSE / (variance of ground-truth + ε),
   which makes the loss scale-invariant across different flow quantities.

4. **Tokenization is done in the DataLoader workers** (CPU) so the GPU
   only ever touches float tensors.

Usage
-----
    python train.py                  # toy synthetic dataset
    python train.py --epochs 200     # more epochs
"""

from __future__ import annotations


from datetime import datetime
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.model.loss import nmse_loss, smooth_loss, budget_loss
from src.eval import evaluate
from src.model.amr_model import AdaptiveMeshAeroModel
from src.utils.train_utils import (
    average_targets_per_token,
    save_checkpoint,
    tau_schedule,
)


# ---------------------------------------------------------------------------
# Learning rate schedule (Transformer warmup)
# ---------------------------------------------------------------------------

class WarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    lr(t) = (1/sqrt(d_model)) * min(t^{-0.5}, t * warmup_steps^{-1.5})

    Identical to the schedule used in AMR-Transformer and the original
    Attention is All You Need paper.
    """

    def __init__(self, optimizer, d_model: int, warmup_steps: int = 1000):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super().__init__(optimizer)

    def get_lr(self):
        step = max(1, self._step_count)
        scale = (self.d_model ** -0.5) * min(
            step ** -0.5,
            step * self.warmup_steps ** -1.5
        )
        return [scale for _ in self.base_lrs]


# ---------------------------------------------------------------------------
# Training loop for deterministic mesh (thresholds are set)
# ---------------------------------------------------------------------------

def train_deterministic_mesh(
    model: AdaptiveMeshAeroModel,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    epochs: int,
    device: torch.device,
    d_model: int = 256,
    warmup_steps: int = 4000,
    checkpoint_path: Optional[str] = None,
):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = WarmupScheduler(optimizer, d_model=d_model, warmup_steps=warmup_steps)

    best_val_loss = float('inf')

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_token_total = 0
        epoch_sample_count = 0
        t0 = time.time()

        with tqdm(train_loader, unit=" batch", leave=False, desc=f"Training Epoch {epoch}/{epochs}") as tq_loader:
            for step, batch in enumerate(tq_loader):
                packed_tokens  = batch["packed_tokens"].to(device)
                packed_targets = batch["packed_targets"].to(device)
                seq_lens       = batch["seq_lens"]

                batch_tokens = sum(seq_lens)
                batch_mean_tokens = batch_tokens / len(seq_lens)
                epoch_token_total += batch_tokens
                epoch_sample_count += len(seq_lens)

                out = model(packed_tokens, seq_lens)

                loss = nmse_loss(out["token_preds"], packed_targets)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()

                # Update progress bar postfix every step; log to stdout every log_every steps
                lr = scheduler.get_last_lr()[0]
                tq_loader.set_postfix(loss=f"{loss.item():.6f}", lr=f"{lr:.2e}", mean_N=f"{batch_mean_tokens:.1f}")

        avg_loss = epoch_loss / len(train_loader)
        epoch_mean = epoch_token_total / max(1, epoch_sample_count)
        elapsed = time.time() - t0

        # Validation 
        val_loss = None
        if val_loader is not None:
            val_loss = evaluate(model, val_loader, device)
            print(f"Epoch {epoch:3d}/{epochs}  train_loss={avg_loss:.6f}  val_loss={val_loss:.6f}  mean_N={epoch_mean:.1f}  time={elapsed:.1f}s")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if checkpoint_path:
                    timestamp = datetime.now().strftime("%d-%m-%Y")
                    checkpoint_name = f"best_model_{timestamp}.pt"
                    save_checkpoint(checkpoint_path, checkpoint_name, model)
                    print(f"  OK Saved best model to {checkpoint_name}")
        else:
            print(f"Epoch {epoch:3d}/{epochs}  train_loss={avg_loss:.6f}  mean_N={epoch_mean:.1f}  time={elapsed:.1f}s")

        # Periodic checkpoint
        if checkpoint_path and epoch % 50 == 0:
            checkpoint_name = f"checkpoint_epoch{epoch:04d}.pt"
            save_checkpoint(checkpoint_path, checkpoint_name, model, optimizer, scheduler)

    print(f"\nTraining complete. Best val loss: {best_val_loss:.6f}")





# ---------------------------------------------------------------------------
# Train only RefinementNet (learned adaptive mesh), transformer frozen
# ---------------------------------------------------------------------------

def train_learned_mesh_p1(
    model,
    train_loader,
    val_loader,
    device,
    *,
    phase2_epochs: int,
    lambda_budget: float = 0.01,
    lambda_smooth: float = 0.001,
    tau_start: float = 5.0,
    tau_end: float = 0.5,
    scorer_lr: float = 1e-3,
    weight_decay: float = 1e-4,
    n_max: int = 1024,
    grad_clip: float = 1.0,
    save_path: str = "outputs/phase2_scorer.pt",
):
    """
    Train the RefinementNet scorer with the transformer frozen.

    Loss per step:
        L = nmse_loss(preds, packed_targets)
            + lambda_budget * budget_loss(soft_N, n_max)
            + lambda_smooth * smooth_loss(score_map)

    Writes the best (lowest val_loss) model state_dict to `save_path`.
    """
    if getattr(model, "refinement_mode", "learned") != "learned":
        raise ValueError("train_learned_mesh_p1 requires refinement_mode='learned'.")

    # 1. Freeze transformer
    for p in model.transformer.parameters():
        p.requires_grad = False
    model.transformer.eval()          # BN/dropout stay frozen too

    # 2. Scorer-only optimizer
    optimizer = AdamW(
        [{"params": model.scorer.parameters(), "lr": scorer_lr,
          "weight_decay": weight_decay}],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=phase2_epochs, eta_min=1e-6)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")

    for epoch in range(phase2_epochs):
        # Gumbel temperature for this epoch
        model.tau = tau_schedule(epoch, tau_start, tau_end, phase2_epochs)

        # --- Train ---
        model.scorer.train()           # scorer-only: keep transformer in eval
        epoch_loss = 0.0
        epoch_pred = 0.0
        epoch_budget = 0.0
        epoch_smooth = 0.0
        epoch_n = 0
        n_steps = 0

        for batch in train_loader:
            grids        = batch["grids"].to(device)
            grid_targets = batch["targets"].to(device)

            out = model(grids)
            packed_targets = average_targets_per_token(grid_targets, out["token_lists"])

            L_pred   = nmse_loss(out["token_preds"], packed_targets)
            L_budget = budget_loss(out["soft_N"], n_max)
            L_smooth = smooth_loss(out["score_map"])

            loss = L_pred + lambda_budget * L_budget + lambda_smooth * L_smooth

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.scorer.parameters(), max_norm=grad_clip)
            optimizer.step()

            # Stats
            epoch_loss   += loss.item()
            epoch_pred   += L_pred.item()
            epoch_budget += L_budget.item()
            epoch_smooth += L_smooth.item()
            epoch_n      += sum(out["seq_lens"])
            n_steps      += 1

        scheduler.step()

        # --- Validate ---
        val_loss = _validate_phase2(model, val_loader, device) if val_loader else None

        # --- Log ---
        print(
            f"[phase2] epoch {epoch:03d}/{phase2_epochs}  "
            f"tau={model.tau:.3f}  "
            f"loss={epoch_loss / n_steps:.4f} "
            f"(pred={epoch_pred / n_steps:.4f} "
            f"budget={epoch_budget / n_steps:.4f} "
            f"smooth={epoch_smooth / n_steps:.4f})  "
            f"mean_N={epoch_n / max(1, n_steps) / max(1, train_loader.batch_size):.1f}"
            + (f"  val={val_loss:.4f}" if val_loss is not None else "")
        )

        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val_loss": val_loss}, save_path)



def _validate_phase2(model, val_loader, device) -> float:
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for batch in val_loader:
            grids = batch["grids"].to(device)
            targets = batch["targets"].to(device)
            out = model(grids)
            packed_targets = average_targets_per_token(targets, out["token_lists"])
            total += nmse_loss(out["token_preds"], packed_targets).item()
            n += 1
    # Restore: scorer back to train, transformer stays frozen+eval
    model.scorer.train()
    model.transformer.eval()
    for p in model.transformer.parameters():
        p.requires_grad = False
    return total / max(1, n)


# ---------------------------------------------------------------------------
# Train RefinementNet + the whole model together
# ---------------------------------------------------------------------------

def train_learned_mesh_p2(
    model,
    train_loader,
    val_loader,
    device,
    *,
    phase3_epochs: int,
    lambda_budget: float = 0.01,
    lambda_smooth: float = 0.001,
    tau_start: float = 0.5,
    tau_end: float = 0.1,
    scorer_lr: float = 1e-3,
    transformer_lr: float = 1e-4,
    weight_decay: float = 1e-4,
    n_max: int = 1024,
    grad_clip: float = 1.0,
    save_path: str = "outputs/phase3_joint.pt",
):
    """
    Joint fine-tuning of scorer and transformer.

    - Transformer is unfrozen.
    - Two param groups: scorer_lr (1e-3) and transformer_lr (1e-4).
    - Tau anneals tau_start -> tau_end across phase3_epochs.

    Loss is identical to phase2:
        L = nmse_loss + lambda_budget * budget_loss(soft_N, n_max)
                      + lambda_smooth * smooth_loss(score_map)

    Writes the best (lowest val_loss) model state_dict to `save_path`.
    """
    if getattr(model, "refinement_mode", "learned") != "learned":
        raise ValueError("train_learned_mesh_p2 requires refinement_mode='learned'.")

    # 1. Unfreeze transformer
    for p in model.transformer.parameters():
        p.requires_grad = True

    # 2. Two-param-group optimizer
    optimizer = AdamW([
        {"params": model.scorer.parameters(),
         "lr": scorer_lr,       "weight_decay": weight_decay},
        {"params": model.transformer.parameters(),
         "lr": transformer_lr,  "weight_decay": weight_decay},
    ])
    scheduler = CosineAnnealingLR(optimizer, T_max=phase3_epochs, eta_min=1e-6)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")

    for epoch in range(phase3_epochs):
        model.tau = tau_schedule(epoch, tau_start, tau_end, phase3_epochs)

        model.train()   # scorer + transformer both in train mode
        epoch_loss = 0.0
        epoch_pred = 0.0
        epoch_budget = 0.0
        epoch_smooth = 0.0
        epoch_n = 0
        n_steps = 0

        for batch in train_loader:
            grids        = batch["grids"].to(device)
            grid_targets = batch["targets"].to(device)

            out = model(grids)
            packed_targets = average_targets_per_token(grid_targets, out["token_lists"])

            L_pred   = nmse_loss(out["token_preds"], packed_targets)
            L_budget = budget_loss(out["soft_N"], n_max)
            L_smooth = smooth_loss(out["score_map"])

            loss = L_pred + lambda_budget * L_budget + lambda_smooth * L_smooth

            optimizer.zero_grad()
            loss.backward()
            # Clip both groups — transformer gradients too, as the finer tau
            # can induce sharper updates than in phase2.
            clip_grad_norm_(
                list(model.scorer.parameters()) + list(model.transformer.parameters()),
                max_norm=grad_clip,
            )
            optimizer.step()

            epoch_loss   += loss.item()
            epoch_pred   += L_pred.item()
            epoch_budget += L_budget.item()
            epoch_smooth += L_smooth.item()
            epoch_n      += sum(out["seq_lens"])
            n_steps      += 1

        scheduler.step()

        val_loss = _validate_phase3(model, val_loader, device) if val_loader else None

        print(
            f"[phase3] epoch {epoch:03d}/{phase3_epochs}  "
            f"tau={model.tau:.3f}  "
            f"loss={epoch_loss / n_steps:.4f} "
            f"(pred={epoch_pred / n_steps:.4f} "
            f"budget={epoch_budget / n_steps:.4f} "
            f"smooth={epoch_smooth / n_steps:.4f})  "
            f"mean_N={epoch_n / max(1, n_steps) / max(1, train_loader.batch_size):.1f}"
            + (f"  val={val_loss:.4f}" if val_loss is not None else "")
        )

        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val_loss": val_loss}, save_path)



def _validate_phase3(model, val_loader, device) -> float:
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for batch in val_loader:
            grids = batch["grids"].to(device)
            targets = batch["targets"].to(device)
            out = model(grids)
            packed_targets = average_targets_per_token(targets, out["token_lists"])
            total += nmse_loss(out["token_preds"], packed_targets).item()
            n += 1
    model.train()
    return total / max(1, n)

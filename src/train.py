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


import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.eval import evaluate
from src.model.amr_model import AdaptiveMeshAeroModel


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Learning rate schedule (Transformer warmup)
# ---------------------------------------------------------------------------

class WarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    lr(t) = (1/sqrt(d_model)) * min(t^{-0.5}, t * warmup_steps^{-1.5})

    Identical to the schedule used in AMR-Transformer and the original
    Attention is All You Need paper.
    """

    def __init__(self, optimizer, d_model: int, warmup_steps: int = 4000):
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
# Training loop
# ---------------------------------------------------------------------------

def train(
    model: AdaptiveMeshAeroModel,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    epochs: int,
    device: torch.device,
    d_model: int = 256,
    warmup_steps: int = 4000,
    checkpoint_dir: Optional[str] = None,
    log_every: int = 10,
):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = WarmupScheduler(optimizer, d_model=d_model, warmup_steps=warmup_steps)

    if checkpoint_dir:
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    best_val_loss = float('inf')

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            packed_tokens  = batch["packed_tokens"].to(device)
            packed_targets = batch["packed_targets"].to(device)
            seq_lens       = batch["seq_lens"]

            out = model(packed_tokens, seq_lens)
            loss = nmse_loss(out["token_preds"], packed_targets)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

            if (step + 1) % log_every == 0:
                lr = scheduler.get_last_lr()[0]
                print(f"  [Epoch {epoch:3d} | Step {step+1:4d}] "
                      f"loss={loss.item():.6f}  lr={lr:.2e}")

        avg_loss = epoch_loss / len(train_loader)
        elapsed = time.time() - t0

        # ---- Validation ----
        val_loss = None
        if val_loader is not None:
            val_loss = evaluate(model, val_loader, device)
            print(f"Epoch {epoch:3d}/{epochs}  "
                  f"train_loss={avg_loss:.6f}  val_loss={val_loss:.6f}  "
                  f"time={elapsed:.1f}s")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if checkpoint_dir:
                    path = Path(checkpoint_dir) / "best_model.pt"
                    torch.save(model.state_dict(), path)
                    print(f"  ✓ Saved best model to {path}")
        else:
            print(f"Epoch {epoch:3d}/{epochs}  "
                  f"train_loss={avg_loss:.6f}  time={elapsed:.1f}s")

        # Periodic checkpoint
        if checkpoint_dir and epoch % 50 == 0:
            path = Path(checkpoint_dir) / f"checkpoint_epoch{epoch:04d}.pt"
            torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict()}, path)

    print(f"\nTraining complete. Best val loss: {best_val_loss:.6f}")


# ---------------------------------------------------------------------------
# Validation loop
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

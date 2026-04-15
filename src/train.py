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
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.eval import evaluate, nmse_loss
from src.model.amr_model import AdaptiveMeshAeroModel


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
    checkpoint_path: Optional[str] = None,
):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = WarmupScheduler(optimizer, d_model=d_model, warmup_steps=warmup_steps)

    best_val_loss = float('inf')

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        with tqdm(train_loader, unit=" batch", leave=False, desc=f"Training Epoch {epoch}/{epochs}") as tq_loader:
            for step, batch in enumerate(tq_loader):
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

                # Update progress bar postfix every step; log to stdout every log_every steps
                lr = scheduler.get_last_lr()[0]
                tq_loader.set_postfix(loss=f"{loss.item():.6f}", lr=f"{lr:.2e}")

                # if (step + 1) % log_every == 0:
                #     tqdm.write(f"  [Epoch {epoch:3d} | Step {step+1:4d}] loss={loss.item():.6f}  lr={lr:.2e}")

        avg_loss = epoch_loss / len(train_loader)
        elapsed = time.time() - t0

        # Validation 
        val_loss = None
        if val_loader is not None:
            val_loss = evaluate(model, val_loader, device)
            print(f"Epoch {epoch:3d}/{epochs}  train_loss={avg_loss:.6f}  val_loss={val_loss:.6f}  time={elapsed:.1f}s")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if checkpoint_path:
                    timestamp = datetime.now().strftime("%d-%m-%Y")
                    checkpoint_name = f"best_model_{timestamp}.pt"
                    save_checkpoint(checkpoint_path, checkpoint_name, model)
                    print(f"  ✓ Saved best model to {checkpoint_name}")
        else:
            print(f"Epoch {epoch:3d}/{epochs}  train_loss={avg_loss:.6f}  time={elapsed:.1f}s")

        # Periodic checkpoint
        if checkpoint_path and epoch % 50 == 0:
            checkpoint_name = f"checkpoint_epoch{epoch:04d}.pt"
            save_checkpoint(checkpoint_path, checkpoint_name, model, optimizer, scheduler)

    print(f"\nTraining complete. Best val loss: {best_val_loss:.6f}")


def save_checkpoint(checkpoint_path, checkpoint_name, model=None, optimizer=None, scheduler=None):
    save_path = Path(checkpoint_path) / checkpoint_name
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "model": model.state_dict() if model else None,
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None}, save_path)

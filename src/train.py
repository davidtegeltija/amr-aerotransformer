"""
========================================================================
Training pipeline for the Adaptive Mesh CFD model.
========================================================================

Key design decisions
--------------------
1. **Sequence packing** (APT / NaViT style): instead of padding variable-
   length token sequences with zeros, we concatenate all tokens in a batch
   into a single packed tensor and use a block-diagonal attention mask.

2. **Warmup LR schedule** (AMR-Transformer style):
       lr(t) = (1 / sqrt(d_model)) * min(t^{-0.5}, t * warmup^{-1.5})

3. **NMSE loss**: per-channel normalised MSE, scale-invariant across flow
   quantities (see src.model.loss.nmse_loss).

4. **Tokenization is done in the DataLoader workers** (CPU) so the GPU
   only ever touches float tensors.

Learned-scorer training
-----------------------
The RefinementNet scorer is trained by **supervised regression** of its
predicted depth map against the variance-oracle depth target
(``train_scorer_supervised``), fully decoupled from the transformer.
"""

from __future__ import annotations


from datetime import datetime
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model.refinement_net import RefinementNet
from src.amr.oracle_depth import max_reachable_depth
from src.model.loss import nmse_loss, scorer_depth_loss
from src.eval import evaluate
from src.model.amr_model import AdaptiveMeshAeroModel
from src.utils.train_utils import average_targets_per_token, save_checkpoint


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

def train_on_deterministic_mesh(
    model: AdaptiveMeshAeroModel,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: torch.device,
    *,
    epochs: int,
    d_model: int = 256,
    warmup_steps: int = 4000,
    save_path: Optional[str] = None,
    writer: Optional[SummaryWriter] = None,
) -> Tuple[List[float], List[Optional[float]]]:
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = WarmupScheduler(optimizer, d_model=d_model, warmup_steps=warmup_steps)

    best_val_loss = float('inf')
    interactive = sys.stderr.isatty()
    global_step = 0

    # Track loss history to see how the network behaves during training
    train_loss_history = []
    val_loss_history = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_token_total = 0
        epoch_sample_count = 0
        t0 = time.time()

        with tqdm(train_loader, unit=" batch", leave=False, desc=f"Training Epoch {epoch}/{epochs}", disable=not interactive) as tq_loader:
            for step, batch in enumerate(tq_loader):
                packed_tokens  = batch["packed_tokens"].to(device)
                packed_targets = batch["packed_targets"].to(device)
                tokens_per_sample = batch["tokens_per_sample"]

                batch_tokens = sum(tokens_per_sample)
                batch_mean_tokens = batch_tokens / len(tokens_per_sample)
                epoch_token_total += batch_tokens
                epoch_sample_count += len(tokens_per_sample)

                out = model(packed_tokens, tokens_per_sample)

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

                # Per-step scalars recover the signal tqdm drops under nohup.
                if writer is not None:
                    writer.add_scalar("Loss/train_step", loss.item(), global_step)
                    writer.add_scalar("LR", lr, global_step)
                    writer.add_scalar("Tokens/mean_N_step", batch_mean_tokens, global_step)
                global_step += 1

        avg_loss = epoch_loss / len(train_loader)
        train_loss_history.append(avg_loss)
        epoch_mean = epoch_token_total / max(1, epoch_sample_count)
        elapsed = time.time() - t0

        if writer is not None:
            writer.add_scalar("Loss/train", avg_loss, epoch)
            writer.add_scalar("Tokens/mean_N", epoch_mean, epoch)
            writer.add_scalar("Time/epoch_s", elapsed, epoch)

        # Validation
        if val_loader is not None:
            val_loss = evaluate(model, val_loader, device)
            val_loss_history.append(val_loss)
            print(f"Epoch {epoch:3d}/{epochs}  train_loss={avg_loss:.6f}  val_loss={val_loss:.6f}  mean_N={epoch_mean:.1f}  time={elapsed:.1f}s")
            if writer is not None:
                writer.add_scalar("Loss/val", val_loss, epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if save_path:
                    timestamp = datetime.now().strftime("%Y-%m-%d")
                    checkpoint_name = f"{timestamp}_deterministic.pt"
                    save_checkpoint(save_path, checkpoint_name, model)
                    print(f"  OK Saved best model to {checkpoint_name}")
        else:
            print(f"Epoch {epoch:3d}/{epochs}  train_loss={avg_loss:.6f}  mean_N={epoch_mean:.1f}  time={elapsed:.1f}s")

        # Periodic checkpoint
        if save_path and epoch % 50 == 0:
            checkpoint_name = f"checkpoint_epoch{epoch:04d}.pt"
            save_checkpoint(save_path, checkpoint_name, model, optimizer, scheduler)

    print(f"\nTraining complete. Best val loss: {best_val_loss:.6f}")

    return train_loss_history, val_loss_history


# ---------------------------------------------------------------------------
# Supervised scorer training (variance-oracle target, transformer decoupled)
# ---------------------------------------------------------------------------

def train_scorer_supervised(
    scorer: RefinementNet,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: torch.device,
    *,
    epochs: int,
    min_depth: int,
    max_depth: int,
    min_cell_size: int,
    tv_weight: float = 0.0,
    decision_weight: float = 0.0,
    decision_margin: float = 0.0,
    decision_temp: Optional[float] = None,
    save_path: Optional[str] = None,
    writer: Optional[SummaryWriter] = None,
) -> Tuple[List[float], List[Optional[float]]]:
    """Train the RefinementNet scorer by supervised regression to the oracle.

    Forward is ``grid -> scorer -> d_pred``; the loss is ``scorer_depth_loss`` against
    the precomputed oracle depth (``batch["oracle_depth"]``). This loop is fully
    decoupled — it depends only on the scorer module; the transformer, token
    packing and the quadtree build never appear here.

    Args:
        scorer: The ``RefinementNet`` to train (e.g. ``model.scorer``). It takes
            channel-last grids ``[B, H, W, C]`` and returns ``d_pred [B, 1, H, W]``.
        train_loader / val_loader: Loaders built with ``ScorerCollateFn`` (yield
            ``grids``, ``targets`` and ``oracle_depth``).
        epochs: Number of epochs.
        min_depth, max_depth, min_cell_size: Quadtree geometry; must match the
            oracle and the mesh builder. The reachable depth is derived from
            ``min_cell_size`` and used as the loss's ``max_depth``.
        tv_weight: Small TV regulariser weight on ``d_pred`` (0 = off).
        decision_weight: Decision-consistency term weight (0 = off, default).
        decision_margin, decision_temp: Decision-term margin / smooth-max temp.
        save_path: Directory to write the best-val checkpoint into (a
            timestamped ``*_scorer_supervised.pt`` file). ``None`` disables
            checkpointing.

    Returns:
        ``(train_loss_history, val_loss_history)``.
    """
    scorer = scorer.to(device)

    optimizer = AdamW(scorer.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_loss = float("inf")
    interactive = sys.stderr.isatty()
    train_loss_history: List[float] = []
    val_loss_history: List[Optional[float]] = []

    reachable: Optional[int] = None

    for epoch in range(epochs):
        scorer.train()
        epoch_loss = 0.0
        n_steps = 0
        t0 = time.time()

        with tqdm(train_loader, unit=" batch", leave=False, desc=f"Scorer Epoch {epoch}/{epochs}", disable=not interactive) as tq_loader:
            for batch in tq_loader:
                grids = batch["grids"].to(device)
                oracle = batch["oracle_depth"].to(device)

                if reachable is None:
                    H, W = grids.shape[1], grids.shape[2]
                    reachable = max_reachable_depth(H, W, min_cell_size, max_depth)

                d_pred = scorer(grids)                           # [B, 1, H, W]
                loss, comp = scorer_depth_loss(
                    d_pred, oracle,
                    tv_weight=tv_weight,
                    decision_weight=decision_weight,
                    min_depth=min_depth,
                    max_depth=reachable,
                    margin=decision_margin,
                    decision_temp=decision_temp,
                )

                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(scorer.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += comp["total"]
                n_steps += 1
                tq_loader.set_postfix(loss=f"{comp['total']:.4f}", reg=f"{comp['reg']:.4f}")

        scheduler.step()
        train_loss_history.append(epoch_loss / max(1, n_steps))

        # --- Supervised validation (cheap model selection) ---
        val_loss = _validate_scorer_supervised(
            scorer, val_loader, device,
            reachable=reachable, min_depth=min_depth, tv_weight=tv_weight,
            decision_weight=decision_weight, decision_margin=decision_margin,
            decision_temp=decision_temp,
        ) if val_loader else None
        val_loss_history.append(val_loss)

        if writer is not None:
            writer.add_scalar("Loss/train", train_loss_history[-1], epoch)
            writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)
            if val_loss is not None:
                writer.add_scalar("Loss/val", val_loss, epoch)

        elapsed = time.time() - t0
        print(f"[scorer] epoch {epoch:03d}/{epochs}  "
              f"train={train_loss_history[-1]:.4f}"
              + (f"  val={val_loss:.4f}" if val_loss is not None else "")
              + f"  time={elapsed:.1f}s")

        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            if save_path:
                timestamp = datetime.now().strftime("%Y-%m-%d")
                checkpoint_name = f"{timestamp}_scorer_supervised.pt"
                save_checkpoint(save_path, checkpoint_name, scorer, epoch=epoch, val_loss=val_loss, prefix="scorer")
                print(f"  OK Saved best scorer to {checkpoint_name}")

    return train_loss_history, val_loss_history


@torch.no_grad()
def _validate_scorer_supervised(
    scorer, val_loader, device, *, reachable, min_depth, tv_weight,
    decision_weight, decision_margin, decision_temp,
) -> float:
    """Mean supervised scorer loss over the val split."""
    scorer.eval()
    total, n = 0.0, 0
    for batch in val_loader:
        grids = batch["grids"].to(device)
        oracle = batch["oracle_depth"].to(device)
        d_pred = scorer(grids)
        _, comp = scorer_depth_loss(
            d_pred, oracle,
            tv_weight=tv_weight, decision_weight=decision_weight,
            min_depth=min_depth, max_depth=reachable,
            margin=decision_margin, decision_temp=decision_temp,
        )
        total += comp["total"]
        n += 1
    scorer.train()
    return total / max(1, n)


# ---------------------------------------------------------------------------
# Train the transformer on meshes from an already-trained (frozen) RefinementNet
# ---------------------------------------------------------------------------

def train_on_learned_mesh(
    model,
    train_loader,
    val_loader,
    device,
    *,
    epochs: int,
    d_model: int = 256,
    warmup_steps: int = 1000,
    save_path: str = "outputs/transformer_on_learned_mesh.pt",
    writer: Optional[SummaryWriter] = None,
) -> Tuple[List[float], List[Optional[float]]]:
    """Train the transformer on meshes produced by an already-trained scorer.

    The RefinementNet scorer is **frozen** (eval mode, ``requires_grad=False``):
    it builds the adaptive mesh each step but receives no gradient. Only the
    transformer is optimised, against the per-token NMSE of its predictions vs.
    the cell-averaged targets. The mesh build inside ``_forward_learned`` is
    already detached, so no gradient could reach the scorer regardless; freezing
    it makes that explicit and keeps its weights fixed.

    Args:
        model: ``AdaptiveMeshAeroModel`` in ``refinement_mode='learned'`` whose
            ``scorer`` has been loaded with trained weights.
        train_loader / val_loader: Loaders built with ``LearnedCollateFn`` (yield
            ``grids`` and dense ``targets``).
        device: Torch device.
        epochs: Number of epochs.
        d_model: Model width used to scale the inverse-sqrt warmup learning rate.
        warmup_steps: Number of warmup steps for the learning-rate schedule.
        save_path: Path to write the best-val full-model checkpoint.
        writer: Optional TensorBoard writer.

    Returns:
        ``(train_loss_history, val_loss_history)``. val entries are ``None`` when
        ``val_loader`` is ``None``.
    """
    if getattr(model, "refinement_mode", "learned") != "learned":
        raise ValueError("train_on_learned_mesh requires refinement_mode='learned'.")
    if model.scorer is None:
        raise ValueError("train_on_learned_mesh requires a scorer; none is present on the model.")

    # Freeze the scorer: it only builds meshes here, it is not trained.
    model.scorer.requires_grad_(False)
    for p in model.transformer.parameters():
        p.requires_grad = True

    optimizer = AdamW(model.transformer.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = WarmupScheduler(optimizer, d_model=d_model, warmup_steps=warmup_steps)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    interactive = sys.stderr.isatty()

    train_loss_history: List[float] = []
    val_loss_history: List[Optional[float]] = []

    for epoch in range(epochs):
        model.train()
        model.scorer.eval()   # keep the frozen scorer in eval mode
        epoch_pred = 0.0
        epoch_n = 0
        n_steps = 0

        with tqdm(train_loader, unit=" batch", leave=False, desc=f"Training Epoch {epoch}/{epochs}", disable=not interactive) as tq_loader:
            for step, batch in enumerate(tq_loader):
                grids        = batch["grids"].to(device)
                grid_targets = batch["targets"].to(device)

                out = model(grids)
                packed_targets = average_targets_per_token(grid_targets, out["token_lists"])

                loss = nmse_loss(out["token_preds"], packed_targets)

                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(model.transformer.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_pred += loss.item()
                epoch_n    += sum(out["tokens_per_sample"])
                n_steps    += 1

        scheduler.step()

        train_loss_history.append(epoch_pred / max(1, n_steps))

        # --- Validate ---
        val_loss = _validate_on_learned_mesh(model, val_loader, device) if val_loader else None
        val_loss_history.append(val_loss)

        # --- Log ---
        mean_N = epoch_n / max(1, n_steps) / max(1, train_loader.batch_size)
        print(
            f"[learned-mesh] epoch {epoch:03d}/{epochs}  "
            f"train={train_loss_history[-1]:.4f}  "
            f"mean_N={mean_N:.1f}"
            + (f"  val={val_loss:.4f}" if val_loss is not None else "")
        )

        if writer is not None:
            writer.add_scalar("Loss/train", train_loss_history[-1], epoch)
            writer.add_scalar("Tokens/mean_N", mean_N, epoch)
            writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)
            if val_loss is not None:
                writer.add_scalar("Loss/val", val_loss, epoch)

        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model": model.state_dict(),
                        "epoch": epoch,
                        "val_loss": val_loss}, save_path)

    return train_loss_history, val_loss_history



def _validate_on_learned_mesh(model, val_loader, device) -> float:
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


# ---------------------------------------------------------------------------
# ViT baseline training (dense prediction, NMSE only)
# ---------------------------------------------------------------------------
if TYPE_CHECKING:
    from src.model.vit import ViT

def train_vit(
    model: "ViT",
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: torch.device,
    *,
    epochs: int,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    save_path: str = "outputs/vit/best.pt",
    writer: Optional[SummaryWriter] = None,
) -> Tuple[List[float], List[Optional[float]]]:
    """Train a ViT on dense [B, H, W, C] grids with NMSE loss only.

    Args:
        model: ViT instance (src.model.vit.ViT).
        train_loader: DataLoader yielding {"grids": [B,H,W,C], "targets": [B,H,W,OC]}.
        val_loader: Optional validation DataLoader (same batch dict shape).
        device: torch device.
        epochs: Number of training epochs.
        lr: AdamW learning rate.
        weight_decay: AdamW weight decay.
        grad_clip: Max grad-norm for clipping.
        save_path: Path to write best-val checkpoint.

    Returns:
        (train_loss_history, val_loss_history). val entries are None when
        val_loader is None.
    """
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    train_hist: List[float] = []
    val_hist: List[Optional[float]] = []
    interactive = sys.stderr.isatty()

    for epoch in range(epochs):
        model.train()
        epoch_loss, n_steps = 0.0, 0
        with tqdm(train_loader, leave=False,
                  desc=f"ViT epoch {epoch}/{epochs}",
                  disable=not interactive) as tq:
            for batch in tq:
                grids   = batch["grids"].to(device).permute(0, 3, 1, 2).float()
                targets = batch["targets"].to(device).permute(0, 3, 1, 2).float()

                preds = model(grids)
                loss  = nmse_loss(preds, targets, channel_dim=1)

                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()

                epoch_loss += loss.item()
                n_steps    += 1
        scheduler.step()

        train_hist.append(epoch_loss / max(1, n_steps))

        val_loss = _validate_vit(model, val_loader, device) if val_loader else None
        val_hist.append(val_loss)

        print(f"[vit] epoch {epoch:03d}/{epochs}  "
              f"train={train_hist[-1]:.4f}"
              + (f"  val={val_loss:.4f}" if val_loss is not None else ""))

        if writer is not None:
            writer.add_scalar("Loss/train", train_hist[-1], epoch)
            writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)
            if val_loss is not None:
                writer.add_scalar("Loss/val", val_loss, epoch)

        if val_loss is not None and val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(),
                        "epoch": epoch, "val_loss": val_loss}, save_path)

    return train_hist, val_hist


def _validate_vit(model, val_loader, device) -> float:
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            grids   = batch["grids"].to(device).permute(0, 3, 1, 2).float()
            targets = batch["targets"].to(device).permute(0, 3, 1, 2).float()
            total += nmse_loss(model(grids), targets, channel_dim=1).item()
            n += 1
    model.train()
    return total / max(1, n)

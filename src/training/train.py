"""
========================================================================
Training pipeline for the Adaptive Mesh CFD model.
========================================================================

Key design decisions
--------------------
1. **Packed batch boundary, padded attention**: the collate concatenates all
   tokens in a batch into a single packed tensor; inside the transformer the
   per-sample embedding sequences are padded to the per-batch max length and attended
   with a boolean key-padding mask (cost B*N_max^2 instead of the dense
   block-diagonal mask's (sum N)^2), then unpadded back to the packed layout.

2. **AdamW + warmup-cosine LR schedule**: linear warmup to a peak learning rate
   that does not depend on ``d_model``, then cosine decay to 1e-6 at the last
   step, with decoupled weight decay on the weight matrices only. The Noam
   schedule of the AMR-Transformer paper,
       lr(t) = (1 / sqrt(d_model)) * min(t^{-0.5}, t * warmup^{-1.5}),
   is what every run before 2026-08-22 used and is still reachable with
   ``schedule="noam"``; it is no longer the default because its peak scales as
   ``d_model ** -0.5`` (so a width sweep changes two things at once) and its
   inverse-sqrt tail never anneals.

3. **NMSE loss**: per-channel normalised MSE, scale-invariant across flow
   quantities (see src.training.loss.nmse_loss). With the affine head the loss is
   still the dense per-pixel NMSE, but it is evaluated in closed form over
   per-leaf sufficient statistics (src.training.loss.affine_nmse_loss), so no
   [B, H, W, C] grid is ever reconstructed during training. Reconstruction is
   still used at inference and for plots (src.models.reconstruction).

4. **Tokenization is done in the DataLoader workers** (CPU) so the GPU
   only ever touches float tensors.

Learned-scorer training
-----------------------
The RefinementNet scorer is trained by **supervised regression** of its
predicted depth map against the variance-oracle depth target
(``train_scorer_supervised``), fully decoupled from the transformer.
"""

from __future__ import annotations

import sys
import time
from typing import List, Optional, Tuple

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.models.amr_model import AMRTransformer
from src.models.reconstruction import OUTPUT_BASIS_TERMS
from src.models.refinement_net import RefinementNet
from src.models.vit_model import ViT
from src.training.scheduler import WarmupCosineScheduler, WarmupScheduler
from src.training.loss import affine_nmse_loss, nmse_loss, scorer_depth_loss
from src.utils.checkpoint import save_checkpoint


# ---------------------------------------------------------------------------
# Peak GPU memory tracking (counter reset per epoch, so the reading covers one
# epoch of training + validation)
# ---------------------------------------------------------------------------
def reset_peak_gpu(device: torch.device) -> None:
    """Reset the CUDA peak-memory counter (no-op on CPU)."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_gpu_gb(device: torch.device) -> float:
    """Peak allocated GPU memory in GiB since the last reset (0.0 on CPU)."""
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / 1024 ** 3


# ---------------------------------------------------------------------------
# Weight-decay param groups (matrices decay, LayerNorm gains and biases do not)
# ---------------------------------------------------------------------------
def decay_param_groups(model: torch.nn.Module, weight_decay: float) -> List[dict]:
    """Split a model's parameters into a decayed and an undecayed group.

    Everything with two or more dimensions is a weight matrix and is decayed;
    everything else is a LayerNorm gain or a bias and is not. Shrinking a
    LayerNorm gain toward zero rescales the whole activation it normalises,
    which is a different thing from the capacity control weight decay is meant
    to be. On the 256/5/8/512 model this leaves 11,539 of 2,997,011 parameters
    undecayed.

    Args:
        model: Model whose parameters are being split.
        weight_decay: Decay applied to the matrix group; the other group gets 0.

    Returns:
        Two param-group dicts ready to hand to an optimizer.
    """
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


# ---------------------------------------------------------------------------
# Transformer training loop (mesh + per-token targets come pre-built from the
# collate, so this is identical for deterministic and learned-mesh modes)
# ---------------------------------------------------------------------------
def train_transformer(
    model: AMRTransformer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    *,
    epochs: int,
    d_model: int = 256,
    warmup_steps: int = 4000,
    schedule: str = "cosine",
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    save_path: Optional[str] = "outputs/checkpoints/transformer.pt",
    writer: Optional[SummaryWriter] = None,
) -> Tuple[List[float], List[float]]:
    """Train the transformer on packed AMR tokens.

    The adaptive mesh, packed tokens and per-token targets are built upstream in
    the collate function (``DeterministicCollateFn`` or ``LearnedCollateFn``),
    which emit the same batch dict — so this loop is shared by both mesh sources.

    Args:
        model: The ``AMRTransformer`` to train.
        train_loader / val_loader: Loaders built with an AMR collate.
        device: Device to train on.
        epochs: Number of epochs.
        d_model: Model width. Read only by the ``"noam"`` schedule, which ties
            its peak learning rate to it; ignored under ``"cosine"``.
        warmup_steps: Warmup length, in optimizer steps, for either schedule.
        schedule: ``"cosine"`` for AdamW + linear warmup + cosine decay to 1e-6
            (the default), or ``"noam"`` for plain Adam + the inverse-sqrt
            schedule every run before 2026-08-22 used. Pass ``"noam"`` to
            reproduce those runs.
        lr: Peak learning rate of the ``"cosine"`` schedule. Unused under
            ``"noam"``, whose peak is fixed at ``d_model ** -0.5 *
            warmup_steps ** -0.5``.
        weight_decay: Decoupled AdamW weight decay on the matrix param group
            (see ``decay_param_groups``). Applied only under ``"cosine"``;
            ``"noam"`` keeps the undecayed Adam it was logged with. Note the
            total shrinkage is ``exp(-weight_decay * sum(lr))``, so it depends
            on run length as well as on this value.
        save_path: Checkpoint path, overwritten on every val-loss improvement.
        writer: Optional TensorBoard writer.

    Returns:
        ``(train_loss_history, val_loss_history)``.

    Raises:
        ValueError: if ``schedule`` is neither ``"cosine"`` nor ``"noam"``.
    """
    model = model.to(device)

    if schedule == "cosine":
        optimizer = AdamW(decay_param_groups(model, weight_decay), lr=lr)
        scheduler = WarmupCosineScheduler(
            optimizer, warmup_steps=warmup_steps,
            total_steps=epochs * len(train_loader), eta_min=1e-6,
        )
        # Same schedule built from stock PyTorch (needs LinearLR, SequentialLR
        # imported alongside CosineAnnealingLR above). Differs only past the
        # last step: CosineAnnealingLR is periodic in 2 * T_max and ramps back
        # up, where WarmupCosineScheduler clamps at eta_min.
        # scheduler = SequentialLR(
        #     optimizer,
        #     [LinearLR(optimizer, start_factor=1 / warmup_steps, total_iters=warmup_steps),
        #      CosineAnnealingLR(optimizer, T_max=max(1, epochs * len(train_loader) - warmup_steps),
        #                        eta_min=1e-6)],
        #     milestones=[warmup_steps],
        # )
        print(f"Optimizer: AdamW(lr={lr:g}, weight_decay={weight_decay:g} on matrices only), "
              f"warmup {warmup_steps} steps -> cosine to 1e-6 over {epochs * len(train_loader):,} steps")
    elif schedule == "noam":
        # The pre-2026-08-22 recipe, kept verbatim so old runs reproduce.
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = WarmupScheduler(optimizer, d_model=d_model, warmup_steps=warmup_steps)
        print(f"Optimizer: Adam (no weight decay) + Noam schedule "
              f"(d_model={d_model}, warmup {warmup_steps} steps), peak lr="
              f"{d_model ** -0.5 * warmup_steps ** -0.5:.2e}")
    else:
        raise ValueError(f"Unknown schedule {schedule!r}; expected 'cosine' or 'noam'")

    best_val_loss = float('inf')
    max_gpu_gb = 0.0
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
        # Quick probe (affine mode): mean |coefficient| of each non-constant basis
        # term over the epoch. If one stays ~0 that term is dead. Empty at order 0,
        # whose only term is the constant, and there the probe never runs anyway.
        grad_abs_sum = torch.zeros(model.basis_size - 1)
        grad_abs_steps = 0
        t0 = time.time()
        reset_peak_gpu(device)

        with tqdm(train_loader, unit=" batch", leave=False, desc=f"Training Epoch {epoch}/{epochs}", disable=not interactive) as tq_loader:
            for batch in tq_loader:
                packed_tokens  = batch["packed_tokens"].to(device)
                packed_targets = batch["packed_targets"].to(device)
                tokens_per_sample = batch["tokens_per_sample"]

                batch_tokens = sum(tokens_per_sample)
                batch_mean_tokens = batch_tokens / len(tokens_per_sample)
                epoch_token_total += batch_tokens
                epoch_sample_count += len(tokens_per_sample)

                out = model(packed_tokens, tokens_per_sample)

                if model.affine_output:
                    # Dense per-pixel NMSE in closed form over the per-leaf statistics
                    # the collate cached. Same loss and same gradients as painting the
                    # [B, H, W, C] grid and scoring that, without ever building it.
                    stats = {k: v.to(device) for k, v in batch["affine_stats"].items()}
                    loss = affine_nmse_loss(out["token_preds"], stats)
                    with torch.no_grad():
                        grad_abs_sum += out["token_preds"][..., 1:].abs().mean(dim=(0, 1)).detach().cpu()
                        grad_abs_steps += 1
                else:
                    loss = nmse_loss(out["token_preds"], packed_targets)

                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()

                # Update progress bar postfix every step; log to stdout every log_every steps
                lr = scheduler.get_last_lr()[0]
                tq_loader.set_postfix(loss=f"{loss.item():.6f}", lr=f"{lr:.2e}", mean_N=f"{batch_mean_tokens:.1f}")

                # Per-step scalars recover the signal tqdm drops under nohup.
                if writer is not None:
                    writer.add_scalar("Loss/train_step", loss.item(), global_step)
                    writer.add_scalar("Tokens/mean_N_step", batch_mean_tokens, global_step)
                global_step += 1

        train_loss_history.append(epoch_loss / len(train_loader))
        epoch_mean = epoch_token_total / max(1, epoch_sample_count)
        elapsed = time.time() - t0
            
        # Validation
        val_loss = evaluate_transformer(model, val_loader, device)
        val_loss_history.append(val_loss)

        # Peak GPU memory over this epoch's train + val passes (0.0 on CPU)
        gpu_gb = peak_gpu_gb(device)
        max_gpu_gb = max(max_gpu_gb, gpu_gb)

        # Affine probe: mean |coefficient| per basis term (empty in non-affine mode)
        grad_str = ""
        if grad_abs_steps > 0:
            term_means = (grad_abs_sum / grad_abs_steps).tolist()
            grad_str = "".join(f"  mean_{name}={m:.4f}"
                               for name, m in zip(OUTPUT_BASIS_TERMS[1:], term_means))

        # Log the metrics
        # The order is in the tag because runs of different orders are meant to be
        # read side by side.
        tag = f"transformer - affine order {model.affine_output}" if model.affine_output else "transformer"
        print(f"[{tag}] epoch {epoch:03d}/{epochs}"
            f"  train_loss={train_loss_history[-1]:.6f}"
            f"  val_loss={val_loss:.6f}"
            f"  time={elapsed:.1f}s"
            f"  gpu_peak={gpu_gb:.2f}GiB{grad_str}"
        )

        # Log the metrics for TensorBoard
        if writer is not None:
            writer.add_scalar("Loss/train", train_loss_history[-1], epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)
            writer.add_scalar("Tokens/mean_N", epoch_mean, epoch)
            writer.add_scalar("Time/epoch_s", elapsed, epoch)
            writer.add_scalar("GPU/peak_mem_GiB", gpu_gb, epoch)
            if grad_abs_steps > 0:
                for name, m in zip(OUTPUT_BASIS_TERMS[1:], term_means):
                    writer.add_scalar(f"Affine/mean_abs_{name}", m, epoch)

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if save_path:
                saved = save_checkpoint(save_path, model, epoch=epoch, val_loss=val_loss)
                pad = " " * len(f"[{tag}] ")
                print(f"{pad}Saved best model to {saved.name}")

    print(f"\nTransformer training complete. Best val loss: {best_val_loss:.6f}"
          f"  Peak GPU memory: {max_gpu_gb:.2f}GiB")

    return train_loss_history, val_loss_history

@torch.no_grad()
def evaluate_transformer(
    model: AMRTransformer,
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
            stats = {k: v.to(device) for k, v in batch["affine_stats"].items()}
            total_loss += affine_nmse_loss(out["token_preds"], stats).item()
        else:
            total_loss += nmse_loss(out["token_preds"], packed_targets).item()

    model.train()
    return total_loss / len(val_loader)


# ---------------------------------------------------------------------------
# Supervised scorer training (variance-oracle target, transformer decoupled)
# ---------------------------------------------------------------------------
def train_scorer_supervised(
    scorer: RefinementNet,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    *,
    epochs: int,
    min_depth: int,
    max_depth: int,
    tv_weight: float = 0.0,
    decision_weight: float = 0.0,
    decision_margin: float = 0.0,
    decision_temp: Optional[float] = None,
    save_path: Optional[str] = "outputs/checkpoints/scorer_supervised.pt",
    writer: Optional[SummaryWriter] = None,
) -> Tuple[List[float], List[float]]:
    """Train the RefinementNet scorer by supervised regression to the oracle.

    Forward is ``grid -> scorer -> d_pred``; the loss is ``scorer_depth_loss`` against
    the precomputed oracle depth (``batch["oracle_depth"]``). This loop is fully
    decoupled — it depends only on the scorer module; the transformer, token
    packing and the quadtree build never appear here.

    Args:
        scorer: The standalone ``RefinementNet`` to train. It takes
            channel-last grids ``[B, H, W, C]`` and returns ``d_pred [B, 1, H, W]``.
        train_loader / val_loader: Loaders built with ``ScorerCollateFn`` (yield
            ``grids``, ``targets`` and ``oracle_depth``).
        epochs: Number of epochs.
        min_depth, max_depth: Quadtree depth bounds; must match the oracle and the
            mesh builder. ``max_depth`` is the reachable depth (derived from the
            patch sizes) and is used directly as the loss's ``max_depth``.
        tv_weight: Small TV regulariser weight on ``d_pred`` (0 = off).
        decision_weight: Decision-consistency term weight (0 = off, default).
        decision_margin, decision_temp: Decision-term margin / smooth-max temp.
        save_path: Full checkpoint path (e.g. ``outputs/checkpoints/scorer_supervised.pt``),
            The date in a run's filename comes from ``main.py``.
            Overwritten on every val-loss improvement. ``None`` disables checkpointing.

    Returns:
        ``(train_loss_history, val_loss_history)``.
    """
    scorer = scorer.to(device)

    optimizer = AdamW(scorer.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_loss = float("inf")
    max_gpu_gb = 0.0
    interactive = sys.stderr.isatty()
    train_loss_history: List[float] = []
    val_loss_history: List[float] = []

    for epoch in range(1, epochs + 1):
        scorer.train()
        epoch_loss = 0.0
        t0 = time.time()
        reset_peak_gpu(device)
        # Per-depth pixel histograms (predicted vs oracle), accumulated over the
        # epoch and reset each epoch. Vectorised via bincount on the rounded depth
        # maps already in scope — no mesh build, no signature change. Reveals
        # whether the scorer tracks the oracle's depth distribution or collapses
        # to a single depth.
        pred_depth_hist = torch.zeros(max_depth + 1, dtype=torch.long, device=device)
        oracle_depth_hist = torch.zeros(max_depth + 1, dtype=torch.long, device=device)

        with tqdm(train_loader, unit=" batch", leave=False, desc=f"Scorer Epoch {epoch}/{epochs}", disable=not interactive) as tq_loader:
            for batch in tq_loader:
                grids = batch["grids"].to(device)
                oracle = batch["oracle_depth"].to(device)

                d_pred = scorer(grids)                           # [B, 1, H, W]
                loss, comp = scorer_depth_loss(
                    d_pred, oracle,
                    tv_weight=tv_weight,
                    decision_weight=decision_weight,
                    min_depth=min_depth,
                    max_depth=max_depth,
                    margin=decision_margin,
                    decision_temp=decision_temp,
                )

                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(scorer.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += comp["total"]
                tq_loader.set_postfix(loss=f"{comp['total']:.4f}", reg=f"{comp['reg']:.4f}")

                # Accumulate per-depth pixel counts (rounded, clamped to the
                # reachable range) for both the prediction and the oracle target.
                with torch.no_grad():
                    pr = d_pred.detach().float().round().clamp_(0, max_depth).long().reshape(-1)
                    orc = oracle.detach().float().round().clamp_(0, max_depth).long().reshape(-1)
                    pred_depth_hist += torch.bincount(pr, minlength=max_depth + 1)
                    oracle_depth_hist += torch.bincount(orc, minlength=max_depth + 1)

        scheduler.step()
        train_loss_history.append(epoch_loss / len(train_loader))
        elapsed = time.time() - t0

        # Validation
        val_loss = evaluate_scorer(
            scorer, val_loader, device,
            max_depth=max_depth, min_depth=min_depth, tv_weight=tv_weight,
            decision_weight=decision_weight, decision_margin=decision_margin,
            decision_temp=decision_temp,
        )
        val_loss_history.append(val_loss)

        # Peak GPU memory over this epoch's train + val passes (0.0 on CPU)
        gpu_gb = peak_gpu_gb(device)
        max_gpu_gb = max(max_gpu_gb, gpu_gb)

        # Depth-distribution diagnostic (no mesh build, no signature change)
        # tokens-per-depth-per-sample = frac_d * 4^d  (each depth-d leaf tiles
        # H*W/4^d pixels); summing over depths gives an approximate mean_N
        w4 = (4.0 ** torch.arange(max_depth + 1, device=device, dtype=torch.float64))
        pf = pred_depth_hist.to(torch.float64)
        of = oracle_depth_hist.to(torch.float64)
        pred_frac = pf / pf.sum().clamp_(min=1.0)
        oracle_frac = of / of.sum().clamp_(min=1.0)
        pred_tokens_d = (pred_frac * w4).cpu()        # tokens at each depth, per sample
        oracle_tokens_d = (oracle_frac * w4).cpu()
        pred_mean_N = float(pred_tokens_d.sum())
        oracle_mean_N = float(oracle_tokens_d.sum())

        tok_str = " ".join(f"{t:.0f}" for t in pred_tokens_d.tolist())
        depth_str = (f"  mean_N~{pred_mean_N:.0f} (oracle {oracle_mean_N:.0f})"
                     f"  tokens/depth d0..d{max_depth}: {tok_str}")

        # Log the metrics
        tag = "scorer"
        print(f"[{tag}] epoch {epoch:03d}/{epochs}"
            f"  train_loss={train_loss_history[-1]:.6f}"
            f"  val_loss={val_loss:.6f}"
            f"  time={elapsed:.1f}s"
            f"  gpu_peak={gpu_gb:.2f}GiB{depth_str}"
        )

        # Log the metrics for TensorBoard
        if writer is not None:
            writer.add_scalar("Loss/train", train_loss_history[-1], epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)
            writer.add_scalar("Time/epoch_s", elapsed, epoch)
            writer.add_scalar("GPU/peak_mem_GiB", gpu_gb, epoch)
            writer.add_scalar("Tokens/mean_N_pred", pred_mean_N, epoch)
            writer.add_scalar("Tokens/mean_N_oracle", oracle_mean_N, epoch)
            for d in range(max_depth + 1):
                writer.add_scalar(f"TokensPerDepth/pred_d{d}", float(pred_tokens_d[d]), epoch)
                writer.add_scalar(f"TokensPerDepth/oracle_d{d}", float(oracle_tokens_d[d]), epoch)

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if save_path:
                saved = save_checkpoint(save_path, scorer, epoch=epoch, val_loss=val_loss)
                pad = " " * len(f"[{tag}] ")
                print(f"{pad}Saved best model to {saved.name}")

    print(f"\nScorer training complete. Best val loss: {best_val_loss:.6f}"
          f"  Peak GPU memory: {max_gpu_gb:.2f}GiB")

    return train_loss_history, val_loss_history

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


# ---------------------------------------------------------------------------
# ViT baseline training (dense prediction, NMSE only)
# ---------------------------------------------------------------------------
def train_vit(
    model: ViT,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    *,
    epochs: int,
    save_path: Optional[str] = "outputs/checkpoints/vit.pt",
    writer: Optional[SummaryWriter] = None,
) -> Tuple[List[float], List[float]]:
    """Train a ViT on dense [B, H, W, C] grids with NMSE loss only.

    This is the non-adaptive baseline: batches come from the dense collate, so
    there is no mesh, no packed tokens and no per-token weighting — the model
    sees the full uniform grid and is scored on the dense field directly.
    """
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_loss = float("inf")
    max_gpu_gb = 0.0
    interactive = sys.stderr.isatty()

    # Track loss history to see how the network behaves during training
    train_loss_history = []
    val_loss_history = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        reset_peak_gpu(device)

        with tqdm(train_loader, leave=False, desc=f"ViT epoch {epoch}/{epochs}", disable=not interactive) as tq_loader:
            for batch in tq_loader:
                grids   = batch["grids"].to(device).permute(0, 3, 1, 2).float()
                targets = batch["targets"].to(device).permute(0, 3, 1, 2).float()

                preds = model(grids)
                loss  = nmse_loss(preds, targets, channel_dim=1)

                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
        scheduler.step()

        train_loss_history.append(epoch_loss / len(train_loader))
        elapsed = time.time() - t0

        # Validation
        val_loss = evaluate_vit(model, val_loader, device)
        val_loss_history.append(val_loss)

        # Peak GPU memory over this epoch's train + val passes (0.0 on CPU)
        gpu_gb = peak_gpu_gb(device)
        max_gpu_gb = max(max_gpu_gb, gpu_gb)

        # Log the metrics
        tag = "vit"
        print(f"[{tag}] epoch {epoch:03d}/{epochs}"
            f"  train_loss={train_loss_history[-1]:.6f}"
            f"  val_loss={val_loss:.6f}"
            f"  time={elapsed:.1f}s"
            f"  gpu_peak={gpu_gb:.2f}GiB"
        )

        # Log the metrics for TensorBoard
        if writer is not None:
            writer.add_scalar("Loss/train", train_loss_history[-1], epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)
            writer.add_scalar("Time/epoch_s", elapsed, epoch)
            writer.add_scalar("GPU/peak_mem_GiB", gpu_gb, epoch)

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if save_path:
                saved = save_checkpoint(save_path, model, epoch=epoch, val_loss=val_loss)
                pad = " " * len(f"[{tag}] ")
                print(f"{pad}Saved best model to {saved.name}")

    print(f"\nViT training complete. Best val loss: {best_val_loss:.6f}"
          f"  Peak GPU memory: {max_gpu_gb:.2f}GiB")

    return train_loss_history, val_loss_history

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

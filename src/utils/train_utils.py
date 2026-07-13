from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from matplotlib import pyplot as plt
import torch

from src.model.loss import nmse_loss
from src.model.reconstruction import (
    precompute_affine_geometry,
    tokens_to_grid_affine_torch,
    tokens_to_grid_torch,
)
from src.amr.quadtree import QuadNode
from src.utils.visualization_utils import save_plot


def save_checkpoint(save_path, model, optimizer=None, scheduler=None, epoch=None, val_loss=None, prefix=""):
    """Save model, optimizer, and scheduler at their current state.

    Args:
        save_path: Full checkpoint path (outputs/checkpoints/vit.pt)
        prefix: Optional string prepended to every key of ``model.state_dict()``.
            Use ``"scorer."`` to save a bare submodule (e.g. ``RefinementNet``)
            so its keys namespace into the parent ``AdaptiveMeshAeroModel`` and
            still load via ``load_state_dict(state, strict=False)``.

    Returns:
        The full ``Path`` the checkpoint was written to (timestamp included).
    """
    timestamp = datetime.now().strftime("%Y-%m-%d")
    save_path = Path(save_path)
    save_path = save_path.with_name(f"{timestamp}_{save_path.name}")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    state = model.state_dict() if model else None
    if state is not None and prefix:
        state = {f"{prefix}.{k}": v for k, v in state.items()}

    torch.save({
        "model": state,
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "val_loss": val_loss
        }, save_path)

    return save_path


def tau_schedule(epoch: int, tau_start: float, tau_end: float, T: int) -> float:
    """
    Exponential decay of the Gumbel-Softmax temperature.

        tau(epoch) = tau_start * (tau_end / tau_start) ** (epoch / max(1, T - 1))

    Clamped at tau_end for epoch >= T - 1. Defined for epoch >= 0.

    Note: with hard Gumbel-max sampling this is a GRADIENT-SHARPNESS anneal,
    not an exploration schedule — the sampling distribution is softmax(logits)
    independent of tau; tau only scales the gradient of the soft
    (straight-through) component that flows back into the scorer.
    """
    if T <= 1:
        return tau_end
    progress = min(max(epoch, 0), T - 1) / (T - 1)
    return tau_start * (tau_end / tau_start) ** progress


def average_targets_per_token(targets: torch.Tensor, token_lists: List[List[QuadNode]]) -> torch.Tensor:
    """
    Compute per-leaf target means from a full-resolution target grid.

    For each QuadNode leaf with bbox (r0, c0, r1, c1), average
    targets[b, r0:r1, c0:c1, :] over the spatial axes to produce one
    row of the output. Leaves are consumed in the same order as
    token_lists[b], so the output row order matches the transformer's
    packed-token sequence.

    Args:
        targets     : [B, H, W, output_channels] tensor on any device
        token_lists : length-B list; each element is the per-sample list of QuadNodes produced by the score-guided mesh builder

    Returns:
        packed_targets : [total_N, output_channels] on the same device as targets
    """
    assert targets.dim() == 4, \
        f"expected [B,H,W,D], got {tuple(targets.shape)}"
    B = targets.size(0)
    assert len(token_lists) == B, \
        f"token_lists len {len(token_lists)} != B {B}"

    rows = []
    for b in range(B):
        for leaf in token_lists[b]:
            r0, c0, r1, c1 = leaf.bbox
            patch = targets[b, r0:r1, c0:c1, :]
            rows.append(patch.mean(dim=(0, 1)))

    return torch.stack(rows, dim=0)

@torch.no_grad()
def evaluate_end_to_end(scorer, model, loader, device, *, min_depth, max_depth, offset=0.0) -> Tuple[float, float]:
    """End-to-end sanity metric: scorer mesh -> frozen transformer -> dense NMSE.

    Builds meshes with the scorer, packs the tokens, runs the transformer, paints
    the token predictions back to the dense grid and reports ``(dense_nmse,
    mean_N)``. This is the one path that needs both the scorer and the transformer
    together, so it lives outside both trainers and is called by the caller that
    owns them.

    It is a diagnostic only — never backpropagated. Sets both modules to ``eval``
    and leaves them there; the scorer trainer re-asserts ``scorer.train()`` each
    epoch.

    Args:
        scorer: Trained ``RefinementNet`` (produces the predicted depth map).
        model: Transformer ``AdaptiveMeshAeroModel`` (consumes packed tokens).
        loader: DataLoader yielding ``grids`` and dense ``targets``.
        min_depth, max_depth, offset: Depth-guided mesh-builder parameters (must
            match the scorer's training bounds).
    """
    from src.amr.learned_adaptive_mesh import build_depth_guided_mesh
    from src.amr.quadtree_tokenizer import nodes_to_token_array

    scorer.eval()
    model.eval()
    total_nmse, n = 0.0, 0
    tokens, samples = 0, 0
    for batch in loader:
        grids = batch["grids"].to(device)
        targets = batch["targets"].to(device)
        H, W, C = grids.shape[1], grids.shape[2], grids.shape[3]

        # Scorer depth map -> per-sample leaves -> packed tokens.
        depth_maps = scorer(grids).squeeze(1).cpu().numpy()      # [B, H, W]
        grids_np = grids.cpu().numpy()
        token_lists, all_tokens, tokens_per_sample = [], [], []
        for b in range(grids.shape[0]):
            leaves = build_depth_guided_mesh(
                data=grids_np[b], depth_map=depth_maps[b],
                max_depth=max_depth, min_depth=min_depth, offset=offset)
            token_lists.append(leaves)
            all_tokens.append(torch.from_numpy(nodes_to_token_array(leaves, H, W, C)))
            tokens_per_sample.append(len(leaves))

        packed = torch.cat(all_tokens, dim=0).to(device)
        out = model(packed, tokens_per_sample)

        if getattr(model, "affine_output", False):
            geom = precompute_affine_geometry(token_lists, tokens_per_sample, H, W)
            dense = tokens_to_grid_affine_torch(
                out["token_preds"], geom, H, W, model.output_channels)
        else:
            dense = tokens_to_grid_torch(
                out["token_preds"], token_lists, tokens_per_sample, H, W)
        total_nmse += nmse_loss(dense, targets).item()
        n += 1
        tokens += sum(tokens_per_sample)
        samples += len(tokens_per_sample)
    return total_nmse / max(1, n), tokens / max(1, samples)


def plot_loss_curves(
    train_loss_history: List[float],
    val_loss_history: List[float],
    epochs: int,
    show: bool = True,
    save_path: Optional[str | Path] = None
):
    """ Plot the training and validation loss curves for training diagnostics """
    train_steps = torch.arange(1, epochs + 1, 1)

    fig = plt.figure(figsize=(10, 4))
    plt.plot(train_steps, train_loss_history, label="train_loss")
    plt.plot(train_steps, val_loss_history, label="val_loss")
    plt.legend()
    plt.title(f"Training Loss Curves for {epochs} Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)

    if save_path:
        save_plot(save_path, fig)

    if show:
        plt.show()
        

if __name__ == "__main__":
    from src.model.loss import smooth_loss
    # tau_schedule
    assert abs(tau_schedule(0, 5.0, 0.5, 10) - 5.0) < 1e-6
    assert abs(tau_schedule(9, 5.0, 0.5, 10) - 0.5) < 1e-6
    mid = tau_schedule(5, 5.0, 0.5, 10)
    assert 0.5 < mid < 5.0
    print(f"tau_schedule OK (mid={mid:.4f})")

    # smooth_loss
    uniform = torch.full((2, 1, 16, 16), 0.5)
    assert smooth_loss(uniform).item() < 1e-6
    checker = torch.zeros(2, 16, 16)
    checker[:, ::2, ::2] = 1.0
    checker[:, 1::2, 1::2] = 1.0
    assert smooth_loss(checker).item() > 0.4
    print("smooth_loss OK")

    # average_targets_per_token
    from src.amr.quadtree import QuadNode
    grid_targets = torch.arange(2 * 8 * 8 * 3, dtype=torch.float32).reshape(2, 8, 8, 3)
    leaf_top    = QuadNode(bbox=(0, 0, 4, 8), depth=1, is_leaf=True)
    leaf_bottom = QuadNode(bbox=(4, 0, 8, 8), depth=1, is_leaf=True)
    token_lists = [[leaf_top, leaf_bottom], [leaf_top, leaf_bottom]]
    packed = average_targets_per_token(grid_targets, token_lists)
    assert packed[0, 0] < packed[1, 0]
    print("average_targets_per_token OK")

    print("All smoke tests passed.")

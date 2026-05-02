from pathlib import Path
from typing import List, Optional

from matplotlib import pyplot as plt
import torch

from src.amr.quadtree import QuadNode
from src.utils.visualization_utils import save_plot


def save_checkpoint(checkpoint_path, checkpoint_name, model, optimizer=None, scheduler=None, epoch=None, val_loss=None):
    """ Save model, optimizer, and scheduler at their current state to checkpoint_path/checkpoint_name """
    save_path = Path(checkpoint_path) / checkpoint_name
    save_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "model": model.state_dict() if model else None,
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "val_loss": val_loss
        }, save_path)


def tau_schedule(epoch: int, tau_start: float, tau_end: float, T: int) -> float:
    """
    Exponential decay of the Gumbel-Softmax temperature.

        tau(epoch) = tau_start * (tau_end / tau_start) ** (epoch / max(1, T - 1))

    Clamped at tau_end for epoch >= T - 1. Defined for epoch >= 0
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
        targets     :[B, H, W, output_dim] tensor on any device
        token_lists : length-B list; each element is the per-sample list of QuadNodes produced by the score-guided mesh builder

    Returns:
        packed_targets : [total_N, output_dim] on the same device as targets
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
    assert packed.shape == (4, 3)
    assert packed[0, 0] < packed[1, 0]
    print("average_targets_per_token OK")

    print("All smoke tests passed.")

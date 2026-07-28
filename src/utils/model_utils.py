"""
Checkpoint I/O shared by every entry point.

``save_checkpoint`` and ``load_checkpoint`` are exact inverses and live together
so their contract cannot drift: the state dict is stored under the ``"model"``
key, with the module's own parameter names and nothing prepended. Every model in
this project (``AdaptiveMeshAeroModel``, ``RefinementNet``, ``ViT``) is trained
and loaded standalone, so no key namespacing is involved.

Loading is deliberately tolerant (``strict=False`` plus a shape-mismatch drop) so
a constant-head checkpoint can warm-start an ``affine_output`` model, but it
refuses a checkpoint that shares no parameter name with the target module —
otherwise pointing a config at the wrong file yields a randomly initialised
network and a training run that looks fine.
"""

from pathlib import Path
import torch


def save_checkpoint(save_path, model, optimizer=None, scheduler=None, epoch=None, val_loss=None):
    """Save model, optimizer, and scheduler at their current state.

    Args:
        save_path: Full checkpoint path (outputs/checkpoints/vit.pt).
        model: Module whose ``state_dict()`` is stored under the ``"model"`` key.
        optimizer: Optional optimizer whose state is stored alongside.
        scheduler: Optional LR scheduler whose state is stored alongside.
        epoch: Optional epoch index, recorded for reference.
        val_loss: Optional validation loss, recorded for reference.

    Returns:
        The ``Path`` the checkpoint was written to (identical to ``save_path``).
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "model": model.state_dict() if model else None,
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "val_loss": val_loss
        }, save_path)

    return save_path


def load_checkpoint(module, path, device="cpu"):
    """Load a checkpoint into ``module`` tolerantly (strict=False).

    Params whose shape does not match the target module are dropped and left at
    their initial values, so e.g. a constant-head checkpoint can warm-start an
    ``affine_output`` model. Raw state dicts (no ``"model"`` key) are accepted.

    Args:
        module: The module to load weights into, modified in place.
        path: Path to a checkpoint written by ``save_checkpoint``.
        device: ``map_location`` for ``torch.load``.

    Returns:
        The same ``module``, for call chaining. Its training/eval mode is
        untouched — callers doing inference must still call ``.eval()``.

    Raises:
        ValueError: If no parameter of ``module`` can be loaded — either the
            checkpoint shares no parameter name with it (a different model), or
            every shared name is shape-mismatched (the same model built with
            different hyperparameters). Both would otherwise leave the module
            fully randomly initialised.
    """
    checkpoint = torch.load(path, map_location=device)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint

    module_sd = module.state_dict()
    matched = [k for k in state if k in module_sd]
    if not matched:
        raise ValueError(
            f"Checkpoint {path} has no parameter in common with "
            f"{type(module).__name__} — it belongs to a different model.")

    dropped = [k for k in matched if state[k].shape != module_sd[k].shape]
    if len(dropped) == len(matched):
        raise ValueError(
            f"Every one of the {len(matched)} parameter(s) {type(module).__name__} shares with "
            f"checkpoint {path} is shape-mismatched, so nothing would be loaded. The model is "
            f"most likely built with different hyperparameters than it was trained with.")
    if dropped:
        state = {k: v for k, v in state.items() if k not in dropped}
        print(f"  re-initialising {len(dropped)} shape-mismatched param(s): {dropped}")

    missing, unexpected = module.load_state_dict(state, strict=False)
    print(f"Loaded checkpoint {path}  (missing: {len(missing)}, unexpected: {len(unexpected)})")

    return module

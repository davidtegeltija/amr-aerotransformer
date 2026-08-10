"""
Checkpoint I/O shared by every entry point.

``save_checkpoint`` and ``load_checkpoint`` are exact inverses: the state dict is
stored under ``"model"`` with the model's own parameter names and nothing
prepended, next to two metadata keys describing what the checkpoint is and how to
rebuild it:

    "model_class"   the saving model's class name
    "model_config"  its ``init_kwargs`` — every constructor argument it was built with

The identity lives in metadata rather than in a key prefix because a prefix is
only a label the tensors carry: the older checkpoints here have ``scorer.`` /
``transformer.`` prefixes that are submodel names from a composite model which no
longer exists, so one file claims to be both.

Loading is tolerant (``strict=False`` plus a shape-mismatch drop) so a
constant-head checkpoint can warm-start an ``affine_output`` model, but it
refuses one written by a different model class — otherwise pointing a config at
the wrong file yields a randomly initialised network and a run that looks fine.
"""

from pathlib import Path
import torch


def save_checkpoint(save_path, model, optimizer=None, scheduler=None, epoch=None, val_loss=None):
    """Save model, optimizer, and scheduler at their current state."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "model": model.state_dict() if model else None,
        "model_class": type(model).__name__ if model else None,
        "model_config": getattr(model, "init_kwargs", None) if model else None,
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "val_loss": val_loss
        }, save_path)

    return save_path


def load_checkpoint(model, path, device="cpu"):
    """Load a checkpoint into model tolerantly (strict=False).

    Params whose shape does not match the target model are dropped and left at
    their initial values, so e.g. a constant-head checkpoint can warm-start an
    ``affine_output`` model. Raw state dicts (no ``"model"`` key) are accepted.
    """
    checkpoint = torch.load(path, map_location=device)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    target_class = type(model).__name__

    # Identity check. Checkpoints written before "model_class" existed carry no
    # identity at all, so absence is not an error — those fall through to the
    # parameter-name overlap check below, which is what guarded them before.
    saved_class = checkpoint.get("model_class") if isinstance(checkpoint, dict) else None
    if saved_class is not None and saved_class != target_class:
        raise ValueError(
            f"Checkpoint {path} was written by {saved_class}, but it is being "
            f"loaded into {target_class}.")

    model_sd = model.state_dict()
    matched = [k for k in state if k in model_sd]
    if not matched:
        raise ValueError(
            f"Checkpoint {path} has no parameter in common with "
            f"{target_class} — it belongs to a different model.")

    dropped = [k for k in matched if state[k].shape != model_sd[k].shape]
    if len(dropped) == len(matched):
        raise ValueError(
            f"Every one of the {len(matched)} parameter(s) {target_class} shares with "
            f"checkpoint {path} is shape-mismatched, so nothing would be loaded. The model is "
            f"most likely built with different hyperparameters than it was trained with.")
    if dropped:
        state = {k: v for k, v in state.items() if k not in dropped}
        print(f"  re-initialising {len(dropped)} shape-mismatched param(s): {dropped}")

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded checkpoint {path}  (missing: {len(missing)}, unexpected: {len(unexpected)})")

    return model


def build_model_from_checkpoint(model_class, path, device="cpu"):
    """Rebuild a trained model from a checkpoint, weights included.

    The hyperparameters come from the checkpoint's own "model_config", so no
    config file or dataset is needed and the rebuilt model cannot disagree with
    the weights it is about to load. **load_checkpoint** still verifies that
    **model_class** is the class the checkpoint was written by.
    """
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint.get("model_config") if isinstance(checkpoint, dict) else None
    if config is None:
        raise ValueError(
            f"Checkpoint {path} records no 'model_config', so {model_class.__name__} "
            f"cannot be rebuilt from it. Construct the model explicitly and use "
            f"load_checkpoint instead.")

    model = model_class(**config).to(device)
    return load_checkpoint(model, path, device)

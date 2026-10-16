"""
========================================================================
Next-frame evaluation for the time-stepping (cavity) data.
========================================================================

Scores the rebuilt next frame in physical units with the per-sample metrics of
the AMR-Transformer code (``models/loss.py`` there): MSE, RMSE, MAE and
NMSE = MSE / mean(u_{t+1}^2), each averaged over samples. Copying ``u_t`` forward
is scored the same way, since a next-step model is only useful where it beats that.
"""

import sys

import numpy as np
import torch
from tqdm import tqdm

from src.inference.predict import predict_single_amr, predict_single_vit, resolve_mesh_source
from src.inference.predict_timestep import predict_next_timestep
from src.models.amr_model import AMRTransformer
from src.models.vit_model import ViT


CHANNEL_NAMES = ("u", "v")
METRIC_NAMES = ("mse", "rmse", "mae", "nmse")


# ---------------------------------------------------------------------------
# Next-timestep metrics — the AMR-Transformer / CFDBench scores of one sample
# ---------------------------------------------------------------------------
def next_timestep_errors(pred: np.ndarray, true: np.ndarray, eps: float = 1e-12) -> dict:
    """Per-channel and overall MSE, RMSE, MAE and NMSE of one next timestep.

    Args:
        pred: Predicted next timestep ``[H, W, C]``, physical units.
        true: True next timestep ``[H, W, C]``, same shape and units.
        eps: Floor on the NMSE denominator, for a channel that is zero over the grid.

    Returns:
        Dict mapping each of ``METRIC_NAMES`` to a ``[C + 1]`` array: one entry
        per channel, then the value over all channels together.
    """
    squared = (pred - true) ** 2
    absolute = np.abs(pred - true)
    mse = np.append(squared.mean(axis=(0, 1)), squared.mean())
    mae = np.append(absolute.mean(axis=(0, 1)), absolute.mean())
    reference = np.append((true ** 2).mean(axis=(0, 1)), (true ** 2).mean())
    return {"mse": mse, "rmse": np.sqrt(mse), "mae": mae, "nmse": mse / (reference + eps)}


# ---------------------------------------------------------------------------
# Next-timestep evaluation — model vs. copying u_t over a split
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_next_timestep(model, args, dataset, sample_indices, mesh_source=None) -> dict:
    """Next-timestep accuracy of a model over the given dataset rows.

    Args:
        model: The loaded model itself, already restored from its checkpoint.
        args: Merged config, passed on to the AMR inference path for its mesh
            settings. Unused for a ``ViT``.
        dataset: The full ``CavityDataset``, before splitting.
        sample_indices: Rows to evaluate (typically the held-out test split).
        mesh_source: The (refinement_criteria, scorer) pair, exactly as in
            ``evaluate_error_rate``.

    Returns:
        Dict with ``model`` and ``copy`` (copying ``u_t`` forward), each mapping
        the ``METRIC_NAMES`` to their ``[C + 1]`` mean over samples, and ``n_samples``.
    """
    if len(sample_indices) == 0:
        raise ValueError("no samples to evaluate; check val_split and the dataset")

    refinement_criteria, scorer = mesh_source or resolve_mesh_source(args)

    print()

    per_sample = {"model": [], "copy": []}
    for index in tqdm(sample_indices, unit=" sample", desc="Evaluating", disable=not sys.stderr.isatty()):
        if isinstance(model, ViT):
            result = predict_single_vit(model, dataset[index])
        elif isinstance(model, AMRTransformer):
            result = predict_single_amr(
                model,
                dataset[index],
                max_depth=args["max_depth"],
                min_depth=args["min_depth"],
                refinement_criteria=refinement_criteria,
                scorer=scorer,
                offset=args.get("offset", 0.0),
            )
        else:
            raise ValueError(f"Cannot evaluate {type(model).__name__}; expected ViT or AMRTransformer")

        result = predict_next_timestep(result, dataset)
        per_sample["model"].append(next_timestep_errors(result["next_timestep_pred"], result["next_timestep_true"]))
        per_sample["copy"].append(next_timestep_errors(result["current_timestep"], result["next_timestep_true"]))

    metrics: dict = {
        source: {name: np.mean([errors[name] for errors in rows], axis=0) for name in METRIC_NAMES}
        for source, rows in per_sample.items()
    }
    metrics["n_samples"] = len(sample_indices)

    print(f"Test split: {metrics['n_samples']} samples  |  {type(model).__name__}  |  next timestep, physical units")
    print(f"{'channel':>10}  {'MSE':>10}  {'RMSE':>10}  {'MAE':>10}  {'NMSE':>10}  {'NMSE copy u_t':>14}")

    for c, channel in enumerate(CHANNEL_NAMES + ("overall",)):
        m = metrics["model"]
        print(f"{channel:>10}  {m['mse'][c]:>10.3e}  {m['rmse'][c]:>10.3e}  {m['mae'][c]:>10.3e}  "
              f"{m['nmse'][c]:>10.3e}  {metrics['copy']['nmse'][c]:>14.3e}")

    return metrics

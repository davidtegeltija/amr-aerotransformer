"""
========================================================================
Next-frame rebuild for the time-stepping (cavity) data.
========================================================================

A cavity model predicts the normalised residual ``u_{t+1} - u_t``. Its error is
only comparable to the CFDBench / AMR-Transformer numbers once the next frame is
rebuilt in physical units::

    u_{t+1} = u_t + du = (input * uv_std + uv_mean) + (prediction * target_std + target_mean)
"""

import numpy as np

from src.data.cavity_dataset import CavityDataset


def current_timestep(dataset: CavityDataset, input_grid: np.ndarray) -> np.ndarray:
    """Physical ``(u_t, v_t)`` of a sample.

    Args:
        dataset: The dataset the sample came from, for its normalisation statistics.
        input_grid: Sample input ``[H, W, C]``; its first two channels are the normalised ``u_t, v_t``.

    Returns:
        ``[H, W, 2]`` current frame in physical units.
    """
    return input_grid[..., :2] * dataset.uv_std + dataset.uv_mean


def rebuild_next_timestep(dataset: CavityDataset, input_grid: np.ndarray, output: np.ndarray) -> np.ndarray:
    """Turn a field in normalised target units into the physical next frame.

    Args:
        dataset: The dataset the sample came from, for its normalisation statistics
            and whether its targets are residuals.
        input_grid: Sample input ``[H, W, C]``.
        output: A prediction or a ground-truth target ``[H, W, 2]``.

    Returns:
        ``[H, W, 2]`` next frame ``(u_{t+1}, v_{t+1})`` in physical units.
    """
    step = dataset.denormalize_target(output)
    if not dataset.predict_residual:
        return step
    return current_timestep(dataset, input_grid) + step


def predict_next_timestep(result: dict, dataset: CavityDataset) -> dict:
    """Add the physical current, true next and predicted next frame to a prediction.

    Args:
        result: Output of ``predict_single_amr`` or ``predict_single_vit``.
        dataset: The dataset the predicted sample came from.

    Returns:
        ``result`` extended with ``current_timestep``, ``next_timestep_true`` and
        ``next_timestep_pred``, each ``[H, W, 2]`` in physical units.
    """
    input_grid = np.asarray(result["input_grid"], dtype=np.float32)
    result["current_timestep"] = current_timestep(dataset, input_grid)
    result["next_timestep_true"] = rebuild_next_timestep(dataset, input_grid, result["ground_truth"])
    result["next_timestep_pred"] = rebuild_next_timestep(dataset, input_grid, result["prediction"])
    return result

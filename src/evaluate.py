import sys

import numpy as np
import torch
from tqdm import tqdm

from src.amr.refinement_criteria import CRITERIA_REGISTRY
from src.model.amr_model import AMRTransformer
from src.model.refinement_net import RefinementNet
from src.model.vit_model import ViT
from src.inference import predict_single_amr, predict_single_vit
from src.utils.model_utils import build_model_from_checkpoint
from src.utils.aero_coefficients import (
    moment_coefficients,
    skin_friction_to_xyz,
    wind_force_coefficients,
)


# The solution channels (cp, cf_tau, cf_z) are stored pre-scaled by these
# factors; anything read as a physical quantity has to divide them out first.
SOLUTION_CHANNEL_SCALES = np.array([1.0, 150.0, 300.0], dtype=np.float32)

# Magnitude of each physical channel across the SuperWing dataset — the
# denominator that turns an absolute field error into a fraction of the scale
# that channel varies over. Fixed constants rather than per-sample norms, so a
# number stays comparable between runs and splits; they are the reference values
# the original postprocessing script normalizes by. They sit well above the mean
# |field| of a single case (a peak, not an average), so these errors read lower
# than a per-sample normalization would.
SURFACE_REFERENCE_MAGNITUDES = np.array([2.35499597, 0.01597823, 0.00696571], dtype=np.float32)

# Order of the coefficients returned by ``aero_coefficients``.
COEFFICIENT_NAMES = ("CL", "CD", "CMz")

# Unit each coefficient error is quoted in: one lift count, one drag count and
# one moment count. The counterpart of SURFACE_REFERENCE_MAGNITUDES for the
# integrated coefficients, and the scale the field is conventionally read at —
# a surrogate that lands within a few drag counts is a usable one.
# The original postprocessing script writes these reciprocally, as
# coef_nondim = [1e3, 1e4, 1e3], and divides by them, which shrinks the error by
# the same factor instead of counting it; the values below count.
COEFFICIENT_COUNT_SCALES = np.array([1e-3, 1e-4, 1e-3], dtype=np.float32)


# ---------------------------------------------------------------------------
# Relative L2 error — per-channel, dimensionless, reads as a percentage
# ---------------------------------------------------------------------------
def relative_l2(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-channel relative L2 error of one reconstructed prediction.

    Args:
        pred: Predicted grid ``[H, W, C]``.
        target: Ground-truth grid ``[H, W, C]``, same shape as ``pred``.
        eps: Numerical floor on the target norm, guarding channels that are
            identically zero over the grid.

    Returns:
        Array ``[C]`` holding ``||pred - target|| / ||target||`` per channel,
        as a fraction (multiply by 100 for a percentage).
    """
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs target {target.shape}")

    err_norm = np.sqrt(((pred - target) ** 2).sum(axis=(0, 1)))
    target_norm = np.sqrt((target ** 2).sum(axis=(0, 1)))
    return err_norm / (target_norm + eps)


# ---------------------------------------------------------------------------
# Relative MAE — mean absolute error against each channel's own magnitude
# ---------------------------------------------------------------------------
def relative_mae(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-channel relative mean absolute error of one reconstructed prediction.

    The field error the original postprocessing reports: the mean absolute
    error of a channel, unscaled back to physical units and divided by that
    channel's reference magnitude. Same shape of answer as
    ``relative_l2``, but normalized by a constant instead of by the norm
    of this particular target, so a case with a nearly flat field cannot
    inflate its own error — the price is that the three channels are only
    comparable to each other as far as the reference magnitudes are
    representative.

    Args:
        pred: Predicted grid ``[H, W, C]``, in the stored (pre-scaled) units the
            model is trained on.
        target: Ground-truth grid ``[H, W, C]``, same shape and units.

    Returns:
        Array ``[C]`` holding ``mean|pred - target| / reference`` per channel,
        as a fraction (multiply by 100 for a percentage).
    """
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs target {target.shape}")

    if pred.shape[-1] != len(SURFACE_REFERENCE_MAGNITUDES):
        raise ValueError(f"expected the {len(SURFACE_REFERENCE_MAGNITUDES)} surface channels (cp, cf_tau, cf_z), got {pred.shape[-1]}")

    absolute_error = np.abs(pred - target).mean(axis=(0, 1))
    return absolute_error / (SOLUTION_CHANNEL_SCALES * SURFACE_REFERENCE_MAGNITUDES)


@torch.no_grad()
def evaluate_error_rate(
    model,
    args,
    dataset,
    sample_indices,
    error_fn=relative_l2
):
    """Mean per-channel field error of a model over the given dataset rows.

    Each row is predicted exactly as at inference time, so the reported error
    includes any reconstruction step — it is the error of the full pipeline, not
    of the raw model output. The model itself picks the path: a ``ViT`` predicts
    the dense grid directly, an ``AMRTransformer`` goes through the AMR path,
    which builds its own mesh from ``args``.

    The mean is taken over per-sample errors rather than pooling all pixels, so
    every geometry counts equally regardless of how much of the grid its wake
    occupies.

    Args:
        model: The loaded model itself, already restored from its checkpoint.
        args: Merged config, passed on to the AMR inference path for its mesh
            settings. Unused for a ``ViT``.
        dataset: The full dataset, before splitting.
        sample_indices: Rows to evaluate (typically the held-out test split).
        error_fn: Per-sample field error, ``(pred, target) -> [C]`` as a
            fraction: ``relative_l2`` (default) or ``relative_mae``.

    Returns:
        Dict with ``per_channel_error`` ``[C]`` and ``per_channel_accuracy``
        ``[C]`` (percentages), the scalar ``error`` / ``accuracy`` averaged over
        channels, the per-sample scalar spread ``std``, and ``n_samples``.
    """
    if len(sample_indices) == 0:
        raise ValueError("no samples to evaluate; check val_split and the dataset")

    refinement_criteria = CRITERIA_REGISTRY[args["refinement_criteria"]] if args.get("refinement_criteria") else None
    scorer = build_model_from_checkpoint(RefinementNet, args["checkpoint_file"]).eval() if args.get("checkpoint_file") else None

    per_sample = []
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
                offset=args.get("offset", 0.0)
            )
        else:
            raise ValueError(f"Cannot evaluate {type(model).__name__}; expected ViT or AMRTransformer")

        per_sample.append(error_fn(result["prediction"], result["ground_truth"]))

    per_sample = np.stack(per_sample)                     # [n_samples, C]
    per_channel_error = 100.0 * per_sample.mean(axis=0)   # [C]

    metrics = {
        "per_channel_error": per_channel_error,
        "per_channel_accuracy": 100.0 - per_channel_error,
        "error": float(per_channel_error.mean()),
        "accuracy": float(100.0 - per_channel_error.mean()),
        "std": float(100.0 * per_sample.mean(axis=1).std()),
        "n_samples": len(sample_indices),
    }

    print(f"\nTest split: {metrics['n_samples']} samples  |  {type(model).__name__}  |  {error_fn.__name__}")
    print(f"{'channel':>10}  {'error':>14}  {'accuracy':>10}")

    for c, (err, acc) in enumerate(zip(metrics["per_channel_error"], metrics["per_channel_accuracy"])):
        print(f"{c:>10}  {err:>13.2f}%  {acc:>9.2f}%")

    print(f"{'overall':>10}  {metrics['error']:>13.2f}%  {metrics['accuracy']:>9.2f}%"
            f"   (per-sample std {metrics['std']:.2f}%)")

    return metrics



# ---------------------------------------------------------------------------
# Aerodynamic coefficients — accuracy of what the prediction *means*
# ---------------------------------------------------------------------------
def calculate_coefficients(
    geometry: np.ndarray,
    solution: np.ndarray,
    angle_of_attack: float,
    ref_area: float = 1.0
) -> np.ndarray:
    """Integrate one surface solution into its lift, drag and pitching moment.

    The same integral the SuperWing solver reports, applied to whichever field
    is handed in — so calling it once on the ground truth and once on the
    prediction gives the pair of coefficient sets ``evaluate_aero_coefficients``
    scores. Reproduces the solver's own CL within ~1% and CD within a few
    percent on the stored fields; that offset is a property of the integration,
    not of the model, and cancels when true and predicted fields are compared.

    Args:
        geometry: Node coordinates ``(x, y, z)`` of the wing surface,
            shape ``[H, W, 3]``, i.e. the first three channels of a dataset
            sample's ``input``.
        solution: Surface solution ``(cp, cf_tau, cf_z)``, shape ``[H, W, 3]``,
            still carrying the stored channel scaling — a dataset ``target`` or
            a model prediction of one.
        angle_of_attack: Angle of attack of the case, in degrees.
        ref_area: Reference area of the wing (column 4 of the SuperWing index
            file). Defaults to 1, which leaves the coefficients unnormalized.

    Returns:
        Array ``[3]`` holding ``(CL, CD, CMz)``, the moment taken about the
        quarter-chord point.
    """
    if geometry.shape[-1] != 3 or solution.shape[-1] != 3:
        raise ValueError(f"expected 3 channels each, got geometry {geometry.shape} and solution {solution.shape}")

    # Dataset grids run [chordwise, spanwise, channel]; the surface integrals
    # index the mesh the other way round, so the two leading axes swap. Getting
    # this wrong flips every cell normal and the forces come out inward.
    geometry = torch.from_numpy(np.asarray(geometry, dtype=np.float32).transpose(1, 0, 2))
    solution = torch.from_numpy(np.asarray(solution, dtype=np.float32).transpose(1, 0, 2))
    solution = solution / torch.from_numpy(SOLUTION_CHANNEL_SCALES)

    # The solution sits on mesh nodes, the integrals want it at the centre of
    # every cell: average the four corner nodes each cell spans.
    cells = 0.25 * (solution[:-1, :-1] + solution[:-1, 1:] + solution[1:, 1:] + solution[1:, :-1])
    cp, cf = cells[..., 0], skin_friction_to_xyz(geometry, cells[..., 1:])

    # wind_force_coefficients rotates a batch of samples, hence the leading axis.
    angle_of_attack = torch.tensor([angle_of_attack], dtype=torch.float32)
    drag, lift, _ = wind_force_coefficients(geometry, angle_of_attack, cp.unsqueeze(0), cf.unsqueeze(0))[0] / ref_area
    moment_z = moment_coefficients(geometry, cp, cf)[2] / ref_area

    return np.array([lift.item(), drag.item(), moment_z.item()], dtype=np.float32)


@torch.no_grad()
def evaluate_aero_coefficients(
    model,
    args,
    dataset,
    sample_indices,
    ref_areas=None
):
    """Accuracy of the predicted lift, drag and pitching moment over a set of rows.

    A prediction can score well pointwise and still integrate to the wrong
    forces, which is what a surrogate is ultimately used for. Each row is
    predicted exactly as at inference time and both the predicted and the
    ground-truth field are pushed through the same surface integral, so the
    reported gap is the model's, not the integrator's.

    Errors are reported as a mean absolute error normalized by the mean
    magnitude of the true coefficient over the split, rather than as a mean of
    per-sample relative errors: CL and CMz cross zero as the angle of attack
    sweeps, and a per-sample ratio explodes on those cases.

    Args:
        model: The loaded model itself, which selects the inference path exactly
            as in ``evaluate_error_rate``.
        args: Merged config, passed on to the AMR inference path for its mesh
            settings. Unused for a ``ViT``.
        dataset: The full dataset, before splitting.
        sample_indices: Rows to evaluate (typically the held-out test split).
        ref_areas: Reference area of every dataset row, i.e. column 4 of the
            SuperWing index file (``np.load(index_file)[:, 4]``). Optional: left
            out, the coefficients stay unnormalized, which rescales each sample
            by its own wing area but leaves the reported accuracy meaningful.

    Returns:
        Dict with ``mae`` ``[3]``, the same error in counts ``mae_counts``
        ``[3]``, ``reference`` ``[3]`` (mean magnitude of the
        true coefficients), ``per_coefficient_error`` / ``per_coefficient_accuracy``
        ``[3]`` (percentages), the scalar ``error`` / ``accuracy`` averaged over
        the three coefficients, the raw ``coefficients_true`` / ``coefficients_pred``
        ``[n_samples, 3]``, and ``n_samples``.
    """
    if len(sample_indices) == 0:
        raise ValueError("no samples to evaluate; check val_split and the dataset")

    refinement_criteria = CRITERIA_REGISTRY[args["refinement_criteria"]] if args.get("refinement_criteria") else None
    scorer = build_model_from_checkpoint(RefinementNet, args["checkpoint_file"]).eval() if args.get("checkpoint_file") else None

    true, predicted = [], []
    for index in tqdm(sample_indices, unit=" sample", desc="Integrating", disable=not sys.stderr.isatty()):
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
                offset=args.get("offset", 0.0)
            )
        else:
            raise ValueError(f"Cannot evaluate {type(model).__name__}; expected ViT or AMRTransformer")

        # Channels 0-2 of the input are the surface nodes and channel 3 is the
        # angle of attack, broadcast over the grid by WingDataset.
        input_grid = np.asarray(result["input_grid"])
        geometry = input_grid[..., :3]
        angle_of_attack = float(input_grid[0, 0, 3])
        ref_area = 1.0 if ref_areas is None else float(ref_areas[index])

        true.append(calculate_coefficients(geometry, result["ground_truth"], angle_of_attack, ref_area))
        predicted.append(calculate_coefficients(geometry, result["prediction"], angle_of_attack, ref_area))

    true = np.stack(true)                                   # [n_samples, 3]
    predicted = np.stack(predicted)                         # [n_samples, 3]

    mae = np.abs(predicted - true).mean(axis=0)             # [3]
    reference = np.abs(true).mean(axis=0)                   # [3]
    per_coefficient_error = 100.0 * mae / reference

    metrics = {
        "mae": mae,
        "mae_counts": mae / COEFFICIENT_COUNT_SCALES,
        "reference": reference,
        "per_coefficient_error": per_coefficient_error,
        "per_coefficient_accuracy": 100.0 - per_coefficient_error,
        "error": float(per_coefficient_error.mean()),
        "accuracy": float(100.0 - per_coefficient_error.mean()),
        "coefficients_true": true,
        "coefficients_pred": predicted,
        "n_samples": len(sample_indices),
    }

    print(f"\nTest split: {metrics['n_samples']} samples  |  {type(model).__name__}  |  aero coefficients"
          f"{'' if ref_areas is not None else '  (unnormalized: no ref_areas given)'}")
    print(f"{'coef.':>10}  {'MAE':>12}  {'MAE (counts)':>13}  {'mean |true|':>12}  {'error':>10}  {'accuracy':>10}")

    for name, err, acc, err_abs, counts, ref in zip(COEFFICIENT_NAMES, metrics["per_coefficient_error"],
                                                    metrics["per_coefficient_accuracy"], mae,
                                                    metrics["mae_counts"], reference):
        print(f"{name:>10}  {err_abs:>12.5f}  {counts:>13.1f}  {ref:>12.5f}  {err:>9.2f}%  {acc:>9.2f}%")

    print(f"{'overall':>10}  {'':>12}  {'':>13}  {'':>12}  {metrics['error']:>9.2f}%  {metrics['accuracy']:>9.2f}%")

    return metrics
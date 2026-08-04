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

# Where the SuperWing index file keeps the numbers this module reads per case:
# the angle of attack, the reference area every coefficient is normalized by, and
# the solver's own (CL, CD, CMz) — the values the whole pipeline is scored
# against. See https://huggingface.co/datasets/yunplus/SuperWing.
INDEX_GEOMETRY_COLUMN = 0
INDEX_ANGLE_OF_ATTACK_COLUMN = 2
INDEX_REF_AREA_COLUMN = 4
INDEX_COEFFICIENT_COLUMNS = [6, 7, 8]

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
    error_fn="l2"
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
            raise ValueError(f"Cannot evaluate {type(model).__name__}; expected **ViT** or **AMRTransformer**")

        if error_fn == "l2":
            error = relative_l2(result["prediction"], result["ground_truth"])
        elif error_fn == "mae":
            error = relative_mae(result["prediction"], result["ground_truth"])
        else:
            raise ValueError(f"Only **l2** or **mae** are valid error functions")
        
        per_sample.append(error)

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

    print(f"\nTest split: {metrics['n_samples']} samples  |  {type(model).__name__}  |  {error_fn}")
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
    scores. On the stored fields it reproduces the solver's own coefficients
    (columns 6-8 of the index file) to within a few parts in ten thousand, so
    the gap it reports between a prediction and the reference is the model's
    rather than the integrator's.

    Args:
        geometry: Node coordinates ``(x, y, z)`` of the wing surface, shape
            ``[H + 1, W + 1, 3]``. The solution lives on the cells this node
            grid spans, so it carries one extra point in each direction — it is
            a row of the original geometry file (``origingeom.npy``), laid out
            channels-last like a dataset sample.
        solution: Surface solution ``(cp, cf_tau, cf_z)`` at cell centres, shape
            ``[H, W, 3]``, still carrying the stored channel scaling — a dataset
            ``target`` or a model prediction of one.
        angle_of_attack: Angle of attack of the case, in degrees.
        ref_area: Reference area of the wing (column 4 of the SuperWing index
            file). Defaults to 1, which leaves the coefficients unnormalized.

    Returns:
        Array ``[3]`` holding ``(CL, CD, CMz)``, the moment taken about the
        quarter-chord point.
    """
    if geometry.shape[-1] != 3 or solution.shape[-1] != 3:
        raise ValueError(f"Expected 3 channels each, got geometry {geometry.shape} and solution {solution.shape}")

    if geometry.shape[:2] != (solution.shape[0] + 1, solution.shape[1] + 1):
        raise ValueError(f"Geometry must be the node grid the solution cells span, i.e. one point larger in each direction: got geometry {geometry.shape} for solution {solution.shape}")

    # Dataset grids run [chordwise, spanwise, channel]; the surface integrals
    # index the mesh the other way round, so the two leading axes swap. Getting
    # this wrong flips every cell normal and the forces come out inward.
    geometry = torch.from_numpy(np.asarray(geometry, dtype=np.float32).transpose(1, 0, 2))
    solution = torch.from_numpy(np.asarray(solution, dtype=np.float32).transpose(1, 0, 2))
    solution = solution / torch.from_numpy(SOLUTION_CHANNEL_SCALES)

    cp, cf = solution[..., 0], skin_friction_to_xyz(geometry, solution[..., 1:])

    # wind_force_coefficients rotates a batch of samples, hence the leading axis.
    angle_of_attack = torch.tensor([angle_of_attack], dtype=torch.float32)
    drag, lift, _ = wind_force_coefficients(geometry, angle_of_attack, cp.unsqueeze(0), cf.unsqueeze(0))[0] / ref_area
    moment_z = moment_coefficients(geometry, cp, cf)[2] / ref_area

    return np.array([lift.item(), drag.item(), moment_z.item()], dtype=np.float32)


# ---------------------------------------------------------------------------
# Scoring one set of coefficients against another
# ---------------------------------------------------------------------------
def coefficient_errors(estimate: np.ndarray, reference: np.ndarray) -> dict:
    """Score a set of coefficients against the set it should have reproduced.

    Errors are reported as a mean absolute error normalized by the mean
    magnitude of the reference over the split, rather than as a mean of
    per-sample relative errors: CL and CMz cross zero as the angle of attack
    sweeps, and a per-sample ratio explodes on those cases.

    Args:
        estimate: Coefficients under test, shape ``[n_samples, 3]``.
        reference: Coefficients they are compared against, same shape.

    Returns:
        Dict with ``mae`` ``[3]``, the same error in counts ``mae_counts``
        ``[3]``, ``reference`` ``[3]`` (mean magnitude of the reference),
        ``per_coefficient_error`` / ``per_coefficient_accuracy`` ``[3]``
        (percentages), and the scalar ``error`` / ``accuracy`` averaged over the
        three coefficients.
    """
    mae = np.abs(estimate - reference).mean(axis=0)          # [3]
    magnitude = np.abs(reference).mean(axis=0)               # [3]
    per_coefficient_error = 100.0 * mae / magnitude

    return {
        "mae": mae,
        "mae_counts": mae / COEFFICIENT_COUNT_SCALES,
        "reference": magnitude,
        "per_coefficient_error": per_coefficient_error,
        "per_coefficient_accuracy": 100.0 - per_coefficient_error,
        "error": float(per_coefficient_error.mean()),
        "accuracy": float(100.0 - per_coefficient_error.mean()),
    }


def print_coefficient_errors(title: str, errors: dict) -> None:
    """Print one ``coefficient_errors`` block as a per-coefficient table.

    Args:
        title: Line printed above the table, naming what is compared to what.
        errors: The dict returned by ``coefficient_errors``.
    """
    print(f"\n{title}")
    print(f"{'coef.':>10}  {'MAE':>12}  {'MAE (counts)':>13}  {'mean |ref|':>12}  {'error':>10}  {'accuracy':>10}")

    for name, err_abs, counts, ref, err, acc in zip(COEFFICIENT_NAMES, errors["mae"], errors["mae_counts"],
                                                    errors["reference"], errors["per_coefficient_error"],
                                                    errors["per_coefficient_accuracy"]):
        print(f"{name:>10}  {err_abs:>12.5f}  {counts:>13.1f}  {ref:>12.5f}  {err:>9.2f}%  {acc:>9.2f}%")

    print(f"{'overall':>10}  {'':>12}  {'':>13}  {'':>12}  {errors['error']:>9.2f}%  {errors['accuracy']:>9.2f}%")


@torch.no_grad()
def evaluate_aero_coefficients(
    model,
    args,
    dataset,
    sample_indices,
    index_array,
    geometry_array
):
    """Accuracy of the predicted lift, drag and pitching moment over a set of rows.

    A prediction can score well pointwise and still integrate to the wrong
    forces, which is what a surrogate is ultimately used for. Each row is
    predicted exactly as at inference time, and both the predicted and the
    ground-truth field are pushed through the same surface integral.

    The solver's own coefficients sit in the index file, so all three sets are
    scored against each other:

        prediction vs solver : the number that matters — how far the model's
                               field lands from the coefficients the CFD run
                               reported, integration error included.
        truth vs solver      : the integrator's own error floor, obtained by
                               feeding it the stored ground-truth field. Small
                               (a few parts in ten thousand), and what is left
                               of "prediction vs solver" if the model were
                               perfect.
        prediction vs truth  : the model's error alone, with the integration
                               offset cancelled out because both sides go
                               through the same integral.

    Args:
        model: The loaded model itself, which selects the inference path exactly
            as in ``evaluate_error_rate``.
        args: Merged config, passed on to the AMR inference path for its mesh
            settings. Unused for a ``ViT``.
        dataset: The full dataset, before splitting.
        sample_indices: Rows to evaluate (typically the held-out test split).
        index_array: The SuperWing index file the dataset was built from
            (``np.load(args["index_file"])``), one row per dataset row. Supplies
            the angle of attack, the reference area, the solver's coefficients,
            and the geometry row each case was run on.
        geometry_array: The original geometry file (``origingeom.npy``), one row
            per wing, holding the node grid the solution cells span. Indexed by
            column 0 of ``index_array``, not by the dataset row.

    Returns:
        Dict with one ``coefficient_errors`` block per comparison
        (``prediction_vs_solver``, ``truth_vs_solver``, ``prediction_vs_truth``),
        the raw ``coefficients_solver`` / ``coefficients_true`` /
        ``coefficients_pred`` ``[n_samples, 3]``, and ``n_samples``.
    """
    if len(sample_indices) == 0:
        raise ValueError("no samples to evaluate; check val_split and the dataset")

    refinement_criteria = CRITERIA_REGISTRY[args["refinement_criteria"]] if args.get("refinement_criteria") else None
    scorer = build_model_from_checkpoint(RefinementNet, args["checkpoint_file"]).eval() if args.get("checkpoint_file") else None

    solver, true, predicted = [], [], []
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

        # The mesh the integrals run on is the *node* grid of the original
        # geometry file, one point larger in each direction than the solution
        # and shared by every case flown on that wing — hence the lookup through
        # the geometry column rather than the dataset row. Both grids are laid
        # out channels-last, like a dataset sample.
        geometry = np.asarray(geometry_array[int(index_array[index, INDEX_GEOMETRY_COLUMN])]).transpose(2, 1, 0)
        angle_of_attack = float(index_array[index, INDEX_ANGLE_OF_ATTACK_COLUMN])
        ref_area = float(index_array[index, INDEX_REF_AREA_COLUMN])

        solver.append(np.asarray(index_array[index, INDEX_COEFFICIENT_COLUMNS], dtype=np.float32))
        true.append(calculate_coefficients(geometry, result["ground_truth"], angle_of_attack, ref_area))
        predicted.append(calculate_coefficients(geometry, result["prediction"], angle_of_attack, ref_area))

    solver = np.stack(solver)                               # [n_samples, 3]
    true = np.stack(true)                                   # [n_samples, 3]
    predicted = np.stack(predicted)                         # [n_samples, 3]

    metrics = {
        "prediction_vs_solver": coefficient_errors(predicted, solver),
        "truth_vs_solver": coefficient_errors(true, solver),
        "prediction_vs_truth": coefficient_errors(predicted, true),
        "coefficients_solver": solver,
        "coefficients_true": true,
        "coefficients_pred": predicted,
        "n_samples": len(sample_indices),
    }

    print(f"\nTest split: {metrics['n_samples']} samples  |  {type(model).__name__}  |  aero coefficients")
    print_coefficient_errors("prediction vs solver  (the model, integration error included)", metrics["prediction_vs_solver"])
    print_coefficient_errors("ground truth vs solver  (the integrator's own error floor)", metrics["truth_vs_solver"])
    print_coefficient_errors("prediction vs ground truth  (the model, integration cancelled)", metrics["prediction_vs_truth"])

    return metrics
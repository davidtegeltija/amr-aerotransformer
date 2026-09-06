import sys

import numpy as np
import torch
from tqdm import tqdm

from src.models.amr_model import AMRTransformer
from src.models.vit_model import ViT
from src.evaluation.metrics import l2_error, mae_error, calculate_coefficients, coefficient_errors
from src.inference.predict import resolve_mesh_source, predict_single_amr, predict_single_vit


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


@torch.no_grad()
def evaluate_error_rate(
    model,
    args,
    dataset,
    sample_indices,
    error_fn="l2",
    mesh_source=None
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
            fraction: ``l2_error`` (default) or ``mae_error``.
        mesh_source: The (refinement_criteria, scorer) pair the AMR path
            builds its mesh with. Resolved from args when omitted; pass the
            pair from resolve_mesh_source to load the scorer once across
            several evaluations. Unused for a ViT.

    Returns:
        Dict with ``per_channel_error`` ``[C]`` and ``per_channel_accuracy``
        ``[C]`` (percentages), the absolute pair those percentages come from,
        ``per_channel_absolute_error`` ``[C]`` and ``per_channel_reference``
        ``[C]`` (units depend on ``error_fn``, see the two metric functions),
        the scalar ``error`` / ``accuracy`` averaged over channels, the
        per-sample scalar spread ``std``, and ``n_samples``.
    """
    if len(sample_indices) == 0:
        raise ValueError("no samples to evaluate; check val_split and the dataset")

    refinement_criteria, scorer = mesh_source or resolve_mesh_source(args)

    print()

    per_sample_relative, per_sample_absolute, per_sample_reference = [], [], []
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
            relative, absolute, reference = l2_error(result["prediction"], result["ground_truth"])
        elif error_fn == "mae":
            relative, absolute, reference = mae_error(result["prediction"], result["ground_truth"])
        else:
            raise ValueError(f"Only **l2** or **mae** are valid error functions")

        per_sample_relative.append(relative)
        per_sample_absolute.append(absolute)
        per_sample_reference.append(reference)

    per_sample_relative = np.stack(per_sample_relative)             # [n_samples, C]
    per_channel_error = 100.0 * per_sample_relative.mean(axis=0)    # [C]

    metrics = {
        "per_channel_error": per_channel_error,
        "per_channel_accuracy": 100.0 - per_channel_error,
        "per_channel_absolute_error": np.stack(per_sample_absolute).mean(axis=0),
        "per_channel_reference": np.stack(per_sample_reference).mean(axis=0),
        "error": float(per_channel_error.mean()),
        "accuracy": float(100.0 - per_channel_error.mean()),
        "std": float(100.0 * per_sample_relative.mean(axis=1).std()),
        "n_samples": len(sample_indices),
    }

    # The l2 normalizer is this sample's own target norm, so the two absolute
    # columns are themselves averages and their ratio only approximates the
    # error column. The mae normalizer is a constant, so there the ratio is exact.
    absolute_label, reference_label = ("RMSE", "RMS |ref|") if error_fn == "l2" else ("MAE", "mean |ref|")

    print(f"Test split: {metrics['n_samples']} samples  |  {type(model).__name__}  |  {error_fn}")
    print(f"{'channel':>10}  {absolute_label:>12}  {reference_label:>12}  {'error':>10}  {'accuracy':>10}")

    for c, (abs_err, ref, err, acc) in enumerate(zip(metrics["per_channel_absolute_error"], metrics["per_channel_reference"],
                                                     metrics["per_channel_error"], metrics["per_channel_accuracy"])):
        print(f"{c:>10}  {abs_err:>12.6f}  {ref:>12.6f}  {err:>9.2f}%  {acc:>9.2f}%")

    print(f"{'overall':>10}  {'':>12}  {'':>12}  {metrics['error']:>9.2f}%  {metrics['accuracy']:>9.2f}%"
            f"   (per-sample std {metrics['std']:.2f}%)")

    return metrics


@torch.no_grad()
def evaluate_aero_coefficients(
    model,
    args,
    dataset,
    sample_indices,
    index_array,
    geometry_array,
    mesh_source=None
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
        mesh_source: The (refinement_criteria, scorer) pair, exactly as in
            evaluate_error_rate.

    Returns:
        Dict with one ``coefficient_errors`` block per comparison
        (``prediction_vs_solver``, ``truth_vs_solver``, ``prediction_vs_truth``),
        the raw ``coefficients_solver`` / ``coefficients_true`` /
        ``coefficients_pred`` ``[n_samples, 3]``, and ``n_samples``.
    """
    if len(sample_indices) == 0:
        raise ValueError("no samples to evaluate; check val_split and the dataset")

    refinement_criteria, scorer = mesh_source or resolve_mesh_source(args)

    print()

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
        # "truth_vs_solver": coefficient_errors(true, solver),
        # "prediction_vs_truth": coefficient_errors(predicted, true),
        "coefficients_solver": solver,
        "coefficients_true": true,
        "coefficients_pred": predicted,
        "n_samples": len(sample_indices),
    }

    print(f"Test split: {metrics['n_samples']} samples  |  {type(model).__name__}  |  aero coefficients")
    print_coefficient_errors("prediction vs solver  (the model, integration error included)", metrics["prediction_vs_solver"])
    # print_coefficient_errors("ground truth vs solver  (the integrator's own error floor)", metrics["truth_vs_solver"])
    # print_coefficient_errors("prediction vs ground truth  (the model, integration cancelled)", metrics["prediction_vs_truth"])

    return metrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def print_coefficient_errors(title: str, errors: dict) -> None:
    """Print one ``coefficient_errors`` block as a per-coefficient table.

    Args:
        title: Line printed above the table, naming what is compared to what.
        errors: The dict returned by ``coefficient_errors``.
    """
    print(f"{title}")
    print(f"{'coef.':>10}  {'MAE':>12}  {'MAE (counts)':>13}  {'mean |ref|':>12}  {'error':>10}  {'accuracy':>10}")

    for name, err_abs, counts, ref, err, acc in zip(COEFFICIENT_NAMES, errors["mae"], errors["mae_counts"],
                                                    errors["reference"], errors["per_coefficient_error"],
                                                    errors["per_coefficient_accuracy"]):
        print(f"{name:>10}  {err_abs:>12.5f}  {counts:>13.1f}  {ref:>12.5f}  {err:>9.2f}%  {acc:>9.2f}%")

    print(f"{'overall':>10}  {'':>12}  {'':>13}  {'':>12}  {errors['error']:>9.2f}%  {errors['accuracy']:>9.2f}%\n")

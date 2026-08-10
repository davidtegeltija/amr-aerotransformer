import numpy as np
import torch

from src.evaluation.aero_coefficients import moment_coefficients, skin_friction_to_xyz, wind_force_coefficients

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
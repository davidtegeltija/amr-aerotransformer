"""
========================================================================
Physics-aware refinement metrics for adaptive mesh generation.
========================================================================

Each function operates on a local region extracted from the full field:

    region : np.ndarray  shape (H_cell, W_cell, C)
                
C channels of physical variables (e.g., velocity_x, velocity_y, ...).
The functions return scalar float values indicating "how complex" the
region is.  Higher values -> more variation -> candidate for subdivision.

Metric catalogue (matching AMR-Transformer paper equations):
    1.  velocity_gradient  -- ||grad(u)||       (Eq. 2)
    2.  vorticity          -- |dv/dx - du/dy|   (Eq. 3)
    3.  momentum           -- |integral(u)|     (Eq. 4)
    4.  kh_shear           -- |du/dy - dv/dx|   (Eq. 5)
    5.  variance           -- mean channel variance
    6.  entropy            -- Shannon entropy

Design decisions
----------------
- All functions accept arbitrary channel counts C >= 1.
- Velocity channels are assumed to be channels 0 and 1.
  If only one channel is present, vorticity and KH-shear fall back to
  the single-channel gradient magnitude.
- Pure numpy; no torch dependency.

RefinementCriteria
----------------
A single dataclass that holds the threshold for each metric.  Set a
threshold to None to disable that metric entirely.
 
    cfg = RefinementCriteria(grad_threshold=0.05, vorticity_threshold=None)
 
Two ready-made configs are provided as module-level constants:
 
    AERODYNAMIC_CONFIG  -- tuned for steady aero fields
    DEFAULT_CONFIG      -- balanced generic config
"""

from __future__ import annotations
 
from dataclasses import dataclass
from typing import Dict, Optional
 
import numpy as np


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------

def compute_velocity_gradient(region: np.ndarray) -> float:
    """
    Mean velocity gradient magnitude over the region.
        G = (1/|m|) * integral( sqrt(|grad u|^2 + |grad v|^2) dm)

    For C >= 2, channels 0 and 1 are treated as u and v.
    For C == 1, channel 0 is used for both u and v (reduces to |grad u|).

    Parameters
    ----------
    region : (H, W, C)

    Returns
    ----------
    float  Mean gradient magnitude in [0, Inf).
    """
    if region.ndim != 3 or region.size == 0:
        return 0.0

    u = region[:, :, 0]
    v = region[:, :, 1] if region.shape[2] >= 2 else u

    du_dx, du_dy = _gradient_xy(u)
    dv_dx, dv_dy = _gradient_xy(v)

    grad_u = np.sqrt(du_dx ** 2 + du_dy ** 2)
    grad_v = np.sqrt(dv_dx ** 2 + dv_dy ** 2)
    grad_mag = np.sqrt(grad_u ** 2 + grad_v ** 2)

    return _safe_mean(grad_mag)


def compute_vorticity(region: np.ndarray) -> float:
    """
    Mean vorticity magnitude over the region.
        omega = (1/|m|) * integral( |dv/dx - du/dy| dm)

    Falls back to gradient magnitude when C == 1.

    Parameters
    ----------
    region : (H, W, C)

    Returns
    ----------
    float  Mean |vorticity| in [0, Inf).
    """
    if region.ndim != 3 or region.size == 0:
        return 0.0

    u = region[:, :, 0]

    if region.shape[2] >= 2:
        v = region[:, :, 1]
        _, du_dy = _gradient_xy(u)
        dv_dx, _ = _gradient_xy(v)
        vorticity = np.abs(dv_dx - du_dy)
    else:
        # Single channel: use gradient magnitude as proxy
        du_dx, du_dy = _gradient_xy(u)
        vorticity = np.sqrt(du_dx ** 2 + du_dy ** 2)

    return _safe_mean(vorticity)


def compute_momentum_magnitude(region: np.ndarray) -> float:
    """
    Normalised momentum magnitude per unit area.
        M = (1/|m|) * sqrt( (integral u dm)^2 + (integral v dm)^2 )

    Parameters
    ----------
    region : (H, W, C)

    Returns
    ----------
    float
    """
    if region.ndim != 3 or region.size == 0:
        return 0.0

    area = region.shape[0] * region.shape[1]
    u_sum = float(np.sum(region[:, :, 0]))
    v_sum = float(np.sum(region[:, :, 1])) if region.shape[2] >= 2 else u_sum

    return np.sqrt(u_sum ** 2 + v_sum ** 2) / max(area, 1)


def compute_kelvin_helmholtz_shear(region: np.ndarray) -> float:
    """
    Maximum shear strength associated with KH instability.
        S = max( |du/dy - dv/dx| )

    Parameters
    ----------
    region : (H, W, C)

    Returns
    -------
    float
    """
    if region.ndim != 3 or region.size == 0:
        return 0.0

    u = region[:, :, 0]

    if region.shape[2] >= 2:
        v = region[:, :, 1]
        _, du_dy = _gradient_xy(u)
        dv_dx, _ = _gradient_xy(v)
        shear = np.abs(du_dy - dv_dx)
    else:
        _, du_dy = _gradient_xy(u)
        shear = np.abs(du_dy)

    return float(np.max(shear)) if shear.size > 0 else 0.0


def compute_channel_variance(region: np.ndarray) -> float:
    """
    Mean per-channel variance (averaged across all C channels).
    High variance in any channel indicates heterogeneous physics.

    Parameters
    ----------
    region : (H, W, C)

    Returns
    -------
    float
    """
    if region.ndim != 3 or region.size == 0:
        return 0.0

    H, W, C = region.shape
    flat = region.reshape(-1, C)      # (H*W, C)
    variances = np.var(flat, axis=0)  # (C,)
    return _safe_mean(variances)


def compute_channel_entropy(region: np.ndarray, bins: int = 64) -> float:
    """
    Mean Shannon entropy of the pixel value distribution (averaged 
    across all C channels). Inspired by the APT paper:
        H = -sum( p_i * log2(p_i) )

    Parameters
    ----------
    region : (H, W, C)
    bins   : number of histogram bins

    Returns
    -------
    float  entropy in [0, log2(bins)]
    """
    if region.ndim != 3 or region.size == 0:
        return 0.0

    H, W, C = region.shape
    entropies = []

    for c in range(C):
        channel = region[:, :, c].ravel()
        if channel.size < 2:
            entropies.append(0.0)
            continue

        counts, _ = np.histogram(channel, bins=bins)
        probs = counts.astype(float) / counts.sum()

        mask = probs > 0
        ent = -float(np.sum(probs[mask] * np.log2(probs[mask])))
        entropies.append(ent)

    return float(np.mean(entropies))


# ---------------------------------------------------------------------------
# Combined metrics dict
# ---------------------------------------------------------------------------

def compute_all_metrics(region: np.ndarray) -> Dict[str, float]:
    """
    Compute every physics metric for a given region in one call.
    Only metrics whose thresholds are not None will actually be checked
    by should_subdivide, but computing all of them here is cheap and
    keeps the metrics dict on each QuadNode complete for inspection.

    Parameters
    ----------
    region : (H, W, C)

    Returns
    -------
    dict with keys:
        velocity_gradient, vorticity, momentum, kh_shear, variance, entropy
    """
    return {
        "velocity_gradient": compute_velocity_gradient(region),
        "vorticity":         compute_vorticity(region),
        "momentum":          compute_momentum_magnitude(region),
        "kh_shear":          compute_kelvin_helmholtz_shear(region),
        "variance":          compute_channel_variance(region),
        "entropy":           compute_channel_entropy(region),
    }


# ---------------------------------------------------------------------------
# RefinementCriteria
# ---------------------------------------------------------------------------
 
@dataclass
class RefinementCriteria:
    """
    Thresholds that control which cells get subdivided.
 
    Each field corresponds to one physics metric.  Set a field to None to
    disable that metric entirely (it will not be computed or checked).
 
    OR-logic applies: a cell is subdivided if *any* enabled metric exceeds
    its threshold (Eq. 6 of AMR-Transformer).
 
    Fields
    ------
    grad_threshold      : velocity gradient magnitude (Eq. 2).
                          Primary signal for boundary layers, shocks, wakes.
    vorticity_threshold : vorticity magnitude (Eq. 3).
                          Detects rotating flow, vortex cores.
    momentum_threshold  : momentum per unit area (Eq. 4).
                          Useful for high-speed regions.
    kh_shear_threshold  : max Kelvin-Helmholtz shear (Eq. 5).
                          Detects shear-layer instabilities.
    variance_threshold  : mean per-channel variance.
                          Generic signal for any kind of local variation.
    entropy_threshold   : Shannon entropy of value distribution.
                          Detects information-rich patches regardless of
                          which physical quantity drives the complexity.
    use_global_scaling  : if True, each threshold is multiplied by the
                          domain-wide mean of its metric before comparison
                          (Eq. 7).  Makes thresholds relative rather than
                          absolute, useful when field magnitudes are unknown.
    global_scale_factor : additional multiplier applied on top of global
                          scaling.  Only used when use_global_scaling=True.
 
    Examples
    --------
    # Use only velocity gradient and vorticity:
    cfg = RefinementCriteria(
        grad_threshold=0.05,
        vorticity_threshold=0.04,
        momentum_threshold=None,
        kh_shear_threshold=None,
        variance_threshold=None,
        entropy_threshold=None,
    )
 
    # Coarser mesh (raise all thresholds by 2x):
    cfg = AERODYNAMIC_CRITERIA.scale(2.0)
    """
 
    grad_threshold:      Optional[float] = 0.08
    vorticity_threshold: Optional[float] = 0.06
    momentum_threshold:  Optional[float] = 0.80
    kh_shear_threshold:  Optional[float] = 0.60
    variance_threshold:  Optional[float] = 0.02
    entropy_threshold:   Optional[float] = 2.50
 
    def scale(self, factor: float) -> "RefinementCriteria":
        """
        Return a new config with all enabled thresholds multiplied by factor.
 
        factor < 1.0 -> lower thresholds -> finer mesh (more tokens)
        factor > 1.0 -> higher thresholds -> coarser mesh (fewer tokens)
        """
        def _s(v):
            return v * factor if v is not None else None
        return RefinementCriteria(
            grad_threshold=      _s(self.grad_threshold),
            vorticity_threshold= _s(self.vorticity_threshold),
            momentum_threshold=  _s(self.momentum_threshold),
            kh_shear_threshold=  _s(self.kh_shear_threshold),
            variance_threshold=  _s(self.variance_threshold),
            entropy_threshold=   _s(self.entropy_threshold),
        )
 
    def to_kwargs(self) -> Dict:
        """
        Return a dict of keyword arguments accepted by build_adaptive_mesh()
        and should_subdivide().  None-valued thresholds are included so
        those metrics are explicitly skipped.
        """
        return {
            "grad_threshold":      self.grad_threshold,
            "vorticity_threshold": self.vorticity_threshold,
            "momentum_threshold":  self.momentum_threshold,
            "kh_shear_threshold":  self.kh_shear_threshold,
            "variance_threshold":  self.variance_threshold,
            "entropy_threshold":   self.entropy_threshold,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gradient_xy(field_2d: np.ndarray):
    """
    Compute spatial gradients of a 2-D scalar field using central differences
    (np.gradient is second-order accurate, unlike np.diff which is first-order).

    Parameters
    ----------
    field_2d : (H, W)

    Returns
    -------
    dfdx : (H, W)   gradient along columns (x direction)
    dfdy : (H, W)   gradient along rows    (y direction)
    """
    dfdx = np.gradient(field_2d, axis=1)   # ∂f/∂x  (column axis)
    dfdy = np.gradient(field_2d, axis=0)   # ∂f/∂y  (row    axis)
    return dfdx, dfdy


def _safe_mean(arr: np.ndarray) -> float:
    """Return mean of array; 0.0 if empty."""
    if arr.size == 0:
        return 0.0
    return float(np.mean(arr))


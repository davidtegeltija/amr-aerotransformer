"""
========================================================================
Geometry-aware refinement metrics for adaptive mesh generation.
========================================================================

Companion module to ``physics_metrics.py``.  While physics metrics
operate on flow-field channels (velocity, pressure, …), geometry
metrics operate on the **surface mesh coordinates** themselves and
detect regions where the shape is complex enough to warrant finer
tokenisation — *before* any flow solution is available.

Each function operates on a local region extracted from the full
coordinate field:

    region : np.ndarray  shape (H_cell, W_cell, 3)

The three channels are the (x, y, z) cell-center coordinates of the
structured surface mesh.  The i-direction (axis 0, H) wraps around
the airfoil circumference and the j-direction (axis 1, W) runs along
the span.

Higher return values -> more geometric complexity -> candidate for
subdivision.

Metric catalogue
----------------
1.  surface_curvature     - discrete mean curvature magnitude
2.  leading_trailing_edge - LE/TE proximity detector
3.  thickness_gradient    - |grad(thickness)| across the surface
4.  distance_to_wall      - inverse min distance to nearest wall

Design decisions
----------------
- All functions accept (H, W, 3) coordinate arrays.
- Pure numpy; no torch dependency.
- Fallbacks are provided when the region is too small for finite
  differences (returns 0.0).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Individual geometry metrics
# ---------------------------------------------------------------------------

def compute_surface_curvature(region: np.ndarray) -> float:
    """
    Mean discrete curvature magnitude over the region.

    For each interior point on the surface mesh we estimate the
    curvature from the second-order finite differences of the
    coordinate field.  On a parametric surface r(i, j) = (x, y, z),
    the discrete mean curvature is approximated via the norm of the
    Laplacian of the position vector:

        kappa ≈ ||Δr|| = sqrt( (∂²x/∂i² + ∂²x/∂j²)² + … )

    This is a standard approach for structured surface meshes in CFD
    pre-processing.
    """
    if not _valid_region(region):
        return 0.0

    curvature_mag = _discrete_mean_curvature(region)
    return _safe_mean(curvature_mag)


def compute_leading_trailing_edge(region: np.ndarray) -> float:
    """
    Leading / trailing edge proximity indicator. Higher values indicate 
    that the region contains or is close to a leading or trailing edge.

    On a structured airfoil mesh the i-direction wraps around the
    profile.  The LE and TE are characterised by *high curvature in
    the chordwise direction* **and** a local minimum in x-coordinate
    (LE) or a local convergence of upper/lower surfaces (TE).

    We use a lightweight proxy: for each spanwise station j we
    compute the chordwise curvature (second derivative of the
    coordinate vector along axis 0).  The metric returns the mean
    of the *maximum* chordwise curvature at each spanwise station
    inside the region, which naturally peaks at LE and TE.
    """
    if not _valid_region(region):
        return 0.0

    H, W, _ = region.shape
    if H < 3:
        return 0.0

    # Second derivative along chordwise direction (axis 0) for each
    # coordinate component, then magnitude.
    d2r = np.zeros((H, W, 3))
    for c in range(3):
        d2r[:, :, c] = np.gradient(np.gradient(region[:, :, c], axis=0), axis=0)

    curvature_chord = np.sqrt(np.sum(d2r ** 2, axis=-1))  # (H, W)

    # Per-spanwise-station maximum
    max_per_span = np.max(curvature_chord, axis=0)  # (W,)
    return _safe_mean(max_per_span)


def compute_thickness_gradient(region: np.ndarray) -> float:
    """
    Mean thickness-gradient magnitude over the region.

    Thickness is estimated per spanwise station as the range of the
    y-coordinate (or, more generally, the coordinate normal to the
    chord plane) along the chordwise wrap.  We then compute the
    spatial gradient of this thickness distribution along the span
    (axis 1) and along the chord (axis 0 — detecting rapid local
    thickness changes such as at flap deflections or blunt TEs).

    For a single-station region (W == 1), only the chordwise
    component is used.
    """
    if not _valid_region(region):
        return 0.0

    H, W, _ = region.shape

    # --- local thickness proxy per cell ---
    # Use the y-coordinate as the primary normal direction.
    # For each (i, j), thickness is estimated from the local spread
    # of y in a small chordwise neighbourhood.  A simpler and more
    # robust approach: compute the range of y per spanwise station
    # and distribute it as a 1-D thickness array, then differentiate.
    #
    # However, to keep the metric *local* (cell-level), we instead
    # compute the chordwise gradient of y as a proxy for how fast
    # thickness changes locally.

    y = region[:, :, 1]  # (H, W)

    dy_di = np.gradient(y, axis=0)  # chordwise thickness change
    dy_dj = np.gradient(y, axis=1) if W >= 2 else np.zeros_like(y)  # spanwise

    grad_mag = np.sqrt(dy_di ** 2 + dy_dj ** 2)

    # Additionally, second derivative captures *change* in thickness
    # gradient, which spikes at rapid transitions (LE radius, flaps).
    d2y_di = np.gradient(dy_di, axis=0)
    thickness_signal = np.sqrt(grad_mag ** 2 + d2y_di ** 2)

    return _safe_mean(thickness_signal)


def compute_distance_to_wall(region: np.ndarray, wall_points: Optional[np.ndarray] = None) -> float:
    """
    Inverse minimum distance to the nearest wall point.
    (Higher -> closer to wall -> candidate for refinement.)

    If explicit wall points are not supplied, the metric falls back
    to an **implicit wall-proximity indicator**: on a body-fitted
    structured mesh the first and last rows in the chordwise
    direction (axis 0) *are* the wall.  The cells closest to i = 0
    and i = H-1 have the smallest wall distance.  We return the
    inverse of the mean Euclidean distance of each cell to the
    nearest boundary row, so that cells *near* the wall score high
    and cells far away score low.

    When "wall_points (N, 3)" is provided, the exact minimum Euclidean 
    distance from each cell center to the wall point cloud is used instead.
    """
    if not _valid_region(region):
        return 0.0

    H, W, _ = region.shape

    if wall_points is not None and wall_points.ndim == 2 and wall_points.shape[1] == 3:
        # Exact wall distance
        coords = region.reshape(-1, 3)  # (H*W, 3)
        # Brute-force nearest-neighbour (fine for small regions)
        dists = np.min(
            np.linalg.norm(coords[:, None, :] - wall_points[None, :, :], axis=-1),
            axis=1,
        )  # (H*W,)
        mean_dist = _safe_mean(dists)
    else:
        # Implicit: distance to nearest boundary row (i=0 or i=H-1)
        wall_top = region[0, :, :]      # (W, 3)
        wall_bot = region[-1, :, :]     # (W, 3)

        dists_top = np.linalg.norm(region - wall_top[None, :, :], axis=-1)  # (H, W)
        dists_bot = np.linalg.norm(region - wall_bot[None, :, :], axis=-1)  # (H, W)
        min_wall_dist = np.minimum(dists_top, dists_bot)  # (H, W)
        mean_dist = _safe_mean(min_wall_dist)

    # Invert so that *small* distance -> *high* metric value.
    # Add epsilon to avoid division by zero.
    return 1.0 / (mean_dist + 1e-12)


# ---------------------------------------------------------------------------
# Combined geometry metrics dict
# ---------------------------------------------------------------------------

def compute_all_geometry_metrics(
    region: np.ndarray,
    wall_points: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute every geometry metric for a given region in one call.

    Parameters
    ----------
    region      : (H, W, 3)   cell-center coordinates
    wall_points : (N, 3), optional   explicit wall point cloud

    Returns
    -------
    dict with keys:
        surface_curvature, leading_trailing_edge, thickness_gradient, distance_to_wall
    """
    return {
        "surface_curvature":     compute_surface_curvature(region),
        "leading_trailing_edge": compute_leading_trailing_edge(region),
        "thickness_gradient":    compute_thickness_gradient(region),
        "distance_to_wall":      compute_distance_to_wall(region, wall_points),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid_region(region: np.ndarray) -> bool:
    """Check that region has shape (H, W, 3) and is non-empty."""
    return region.ndim == 3 and region.shape[2] == 3 and region.size > 0


def _discrete_mean_curvature(region: np.ndarray) -> np.ndarray:
    """
    Approximate discrete mean curvature at each interior cell of a
    structured surface mesh via the Laplacian of the position vector.

        kappa(i,j) ≈ ||∂²r/∂i² + ∂²r/∂j²||
    """
    laplacian = np.zeros_like(region)
    for c in range(3):
        d2_di = np.gradient(np.gradient(region[:, :, c], axis=0), axis=0)
        d2_dj = np.gradient(np.gradient(region[:, :, c], axis=1), axis=1)
        laplacian[:, :, c] = d2_di + d2_dj

    # Curvature magnitude at each cell
    return np.sqrt(np.sum(laplacian ** 2, axis=-1))  # (H, W)


def _safe_mean(arr: np.ndarray) -> float:
    """Return mean of array; 0.0 if empty."""
    if isinstance(arr, (int, float)):
        return float(arr)
    if arr.size == 0:
        return 0.0
    return float(np.mean(arr))
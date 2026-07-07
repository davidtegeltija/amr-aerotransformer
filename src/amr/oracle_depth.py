"""
========================================================================
Variance (wavelet) oracle — the supervised target for the scorer.
========================================================================

This module manufactures the supervised target the RefinementNet scorer is
trained against. The idea is that for the dense painting loss, the optimal
mesh under a *perfect constant-per-cell predictor* refines a cell exactly
while the within-cell residual (the sum-of-squared-deviations from the cell
mean) stays above a tolerance. So the "correct" quadtree depth at each pixel
is the SHALLOWEST depth whose CONTAINING cell already has within-cell error <= tol.

Correctness notes:

* Cells follow the EXACT rectangular quadtree geometry of the mesh builder
  (on a 256x128 domain the cells are 2:1 rectangles).
* Using the containing-cell test (not a free per-pixel detector) makes the
  depth map piecewise-constant over the leaves of the mesh it induces, which is
  what makes the closed-form leaf count exact.
* Error is measured on per-channel scale-normalised target channels so no single field
  dominates through its units.
* ``max_depth`` is the reachable depth derived at config load from
  ``min_patch_size`` by ``patch_sizes_to_depth_bounds``. Every cell subdivides
  uniformly to it, so at every depth the cells tile the whole domain and each
  pixel reaches exactly ``max_depth``.

The output is a small deterministic ``[H, W]`` int array, cheap to cache.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def calibrate_global_tolerance(
    targets: Iterable[np.ndarray],
    *,
    n_target: int,
    min_depth: int,
    max_depth: int,
    valid_masks: Optional[Iterable[Optional[np.ndarray]]] = None,
    iters: int = 40,
    rel_tol: float = 1e-3,
) -> float:
    """Find ONE global tolerance so the mean leaf count lands near ``n_target``.

    A single global ``tol`` keeps the error *scale* fixed and lets per-sample
    leaf counts vary with geometry complexity (the physically meaningful behaviour,
    unlike forcing every sample to an identical count).

    Leaf count is monotonically non-increasing in ``tol`` (a looser tolerance
    accepts shallower cells), so a bisection on ``tol`` lands the budget. The
    per-sample cell errors are built once and reused across the search.

    Args:
        targets: Iterable of ``[H, W, C]`` target fields (a calibration subset
            of the train split).
        n_target: Desired mean leaf (token) count.
        min_depth, max_depth: Quadtree depth bounds.
        valid_masks: Optional iterable of per-sample ``[H, W]`` bool masks,
            aligned with ``targets``.
        iters: Max bisection iterations.
        rel_tol: Stop once ``|mean_count - n_target| <= rel_tol * n_target``.

    Returns:
        The calibrated global tolerance.
    """
    targets = list(targets)
    if not targets:
        raise ValueError("calibrate_global_tolerance needs at least one target")
    if valid_masks is None:
        masks: List[Optional[np.ndarray]] = [None] * len(targets)
    else:
        masks = list(valid_masks)

    per_sample_errors = [
        build_cell_errors(t, max_depth=max_depth, valid_mask=m)
        for t, m in zip(targets, masks)
    ]

    def mean_count(tol: float) -> float:
        counts = [
            leaf_count_from_oracle(oracle_from_cell_errors(ssd, tol, min_depth), area)
            for ssd, area in per_sample_errors
        ]
        return float(np.mean(counts))

    # Bracket: tol = 0 -> finest (most leaves); tol = max SSD -> coarsest. The
    # root cell's SSD (constant over ssd[0]) is the maximum over all depths by
    # construction, since error is non-increasing with depth.
    hi = max(float(ssd[0, 0, 0]) for ssd, _ in per_sample_errors)
    hi = max(hi, 1e-12)
    lo = 0.0

    # mean_count(lo=0) >= n_target >= mean_count(hi) by monotonicity; if the
    # request is outside the reachable range we return the closest bound.
    if mean_count(hi) >= n_target:
        return hi
    # mean_count(lo=0) is the finest reachable mesh. If n_target meets or exceeds
    # it, the bracket collapses to tol=0, which is degenerate: ssd <= 0 is only
    # satisfied by perfectly-constant cells, so every other pixel pins to its max
    # reachable depth and the oracle becomes a near-constant "refine everything"
    # map. The scorer then trivially regresses it and the loss collapses to ~0
    # while learning nothing. Refuse rather than return tol=0.
    finest_count = mean_count(lo)
    if finest_count <= n_target:
        raise ValueError(
            f"n_target={n_target} >= max reachable mean leaf count "
            f"({finest_count:.1f}); tolerance would pin to tol=0, producing a "
            f"degenerate near-constant oracle (every pixel at max depth). "
            f"Lower n_target below {finest_count:.1f} (e.g. by reducing it or "
            f"increasing min_patch_size / max_patch_size to lower max_depth)."
        )

    tol = hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        c = mean_count(mid)
        tol = mid
        if abs(c - n_target) <= rel_tol * n_target:
            break
        if c > n_target:
            # too many leaves -> loosen tolerance (raise it)
            lo = mid
        else:
            hi = mid
    return tol


def compute_oracle_depth(
    target: np.ndarray,
    *,
    tol: float,
    min_depth: int,
    max_depth: int,
    valid_mask: Optional[np.ndarray] = None,
    channel_scale: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Convenience: build the cell errors and return the oracle depth map.

    Args:
        target: ``[H, W, C]`` ground-truth field.
        tol: Error tolerance on scaled channels (use a single global value
            calibrated by :func:`calibrate_global_tolerance`).
        min_depth, max_depth: Quadtree depth bounds (derived at config load from
            the patch sizes by ``patch_sizes_to_depth_bounds``).
        valid_mask: Optional ``[H, W]`` bool mask of valid pixels.
        channel_scale: Optional ``[C]`` per-channel scale (defaults to per-sample
            std).

    Returns:
        ``[H, W]`` int64 oracle depth map.
    """
    ssd, _ = build_cell_errors(
        target,
        max_depth=max_depth,
        valid_mask=valid_mask,
        channel_scale=channel_scale,
    )
    return oracle_from_cell_errors(ssd, tol, min_depth)


# ---------------------------------------------------------------------------
# Per-depth cell errors (within-cell SSD on the exact quadtree geometry)
# ---------------------------------------------------------------------------

def build_cell_errors(
    target: np.ndarray,
    *,
    max_depth: int,
    valid_mask: Optional[np.ndarray] = None,
    channel_scale: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build the per-depth within-cell errors for one dense target field.

    Every cell subdivides uniformly to ``max_depth`` on an evenly-halving grid,
    so at every depth the cells tile the whole ``[0, H) x [0, W)`` domain and
    each entry of the returned arrays is written exactly once.

    Args:
        target: ``[H, W, C]`` ground-truth field.
        max_depth: Reachable depth cap (derived once at config load from
            ``min_patch_size`` by ``patch_sizes_to_depth_bounds``).
        valid_mask: Optional ``[H, W]`` bool mask of valid pixels. Invalid
            pixels are excluded from cell means and SSD; a cell with no valid
            pixels gets SSD 0 (it never demands refinement).
        channel_scale: Optional ``[C]`` per-channel scale. Defaults to this
            sample's per-channel std over valid pixels.

    Returns:
        ``(ssd, area)``, two ``[max_depth + 1, H, W]`` float64 arrays.
        ``ssd[d, r, c]`` is the within-cell sum-of-squared-deviations (over
        scaled channels and valid pixels) of the depth-``d`` cell containing
        pixel ``(r, c)``; it is non-increasing with depth (depth 0 = root =
        largest error). ``area[d, r, c]`` is that cell's pixel area
        (height*width), used for the closed-form leaf count.
    """
    assert target.ndim == 3, f"expected [H, W, C], got {target.shape}"
    H, W, C = target.shape

    if channel_scale is None:
        channel_scale = _per_channel_scale(target, valid_mask)
    scaled = target.astype(np.float64) / channel_scale.reshape(1, 1, C)

    if valid_mask is None:
        mask = np.ones((H, W), dtype=bool)
    else:
        mask = valid_mask.astype(bool)

    ssd = np.zeros((max_depth + 1, H, W), dtype=np.float64)
    area = np.zeros((max_depth + 1, H, W), dtype=np.float64)

    def recurse(r0: int, c0: int, r1: int, c1: int, depth: int) -> None:
        h, w = r1 - r0, c1 - c0
        m = mask[r0:r1, c0:c1]
        if not m.any():
            cell_ssd = 0.0
        else:
            vals = scaled[r0:r1, c0:c1, :][m]          # [n_valid, C]
            cell_ssd = float(((vals - vals.mean(axis=0)) ** 2).sum())

        ssd[depth, r0:r1, c0:c1] = cell_ssd
        area[depth, r0:r1, c0:c1] = float(h * w)

        if depth == max_depth:
            return

        r_mid = r0 + h // 2
        c_mid = c0 + w // 2
        recurse(r0, c0, r_mid, c_mid, depth + 1)   # top-left
        recurse(r0, c_mid, r_mid, c1, depth + 1)   # top-right
        recurse(r_mid, c0, r1, c_mid, depth + 1)   # bottom-left
        recurse(r_mid, c_mid, r1, c1, depth + 1)   # bottom-right

    recurse(0, 0, H, W, 0)
    return ssd, area


# ---------------------------------------------------------------------------
# Oracle depth
# ---------------------------------------------------------------------------

def oracle_from_cell_errors(ssd: np.ndarray, tol: float, min_depth: int) -> np.ndarray:
    """Per-pixel oracle depth from the per-depth cell errors and a tolerance.

    Oracle depth = the shallowest depth ``>= min_depth`` whose containing cell
    has error ``<= tol``. Pixels never acceptable fall back to ``max_depth``
    (the deepest level, which is forced to be a candidate).

    Args:
        ssd: ``[max_depth + 1, H, W]`` within-cell SSD from
            :func:`build_cell_errors`.
        tol: Error tolerance (on scaled channels).
        min_depth: Depth floor; oracle is never shallower than this.

    Returns:
        ``[H, W]`` int64 oracle depth map.
    """
    cand = ssd <= tol                  # [max_depth + 1, H, W]
    cand[:min_depth] = False           # start the search at min_depth
    cand[-1] = True                    # deepest depth is the fallback
    return cand.argmax(axis=0).astype(np.int64)


# ---------------------------------------------------------------------------
# Leaf count
# ---------------------------------------------------------------------------

def leaf_count_from_oracle(oracle: np.ndarray, area: np.ndarray) -> float:
    """Closed-form leaf count of an oracle depth map.

    The leaf count of a tiling-consistent depth map is ``sum_pixels 1 / cell_area``
    (each leaf of pixel-area ``A`` contributes ``A * (1/A) = 1``). Because the
    containing-cell oracle is constant within every induced leaf, this is an
    exact integer (up to float rounding) and matches the mesh builder.

    Args:
        oracle: ``[H, W]`` oracle depth map.
        area: ``[max_depth + 1, H, W]`` cell-area array from
            :func:`build_cell_errors`.

    Returns:
        Leaf count as a float (round for the integer count).
    """
    H, W = oracle.shape
    rr, cc = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    return float((1.0 / area[oracle, rr, cc]).sum())


# ---------------------------------------------------------------------------
# Channel scaling
# ---------------------------------------------------------------------------

def _per_channel_scale(
    target: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    eps: float = 1e-8,
) -> np.ndarray:
    """Per-channel standard deviation used to scale-normalise the target.

    Args:
        target: ``[H, W, C]`` ground-truth field.
        valid_mask: Optional ``[H, W]`` bool mask; std is computed over valid
            pixels only.
        eps: Floor added to each channel's std so a constant channel does not
            divide by zero.

    Returns:
        ``[C]`` float64 array of per-channel scales (std + eps).
    """
    if valid_mask is None:
        flat = target.reshape(-1, target.shape[-1])
    else:
        flat = target[valid_mask]
    if flat.shape[0] == 0:
        return np.ones(target.shape[-1], dtype=np.float64)
    return flat.std(axis=0).astype(np.float64) + eps

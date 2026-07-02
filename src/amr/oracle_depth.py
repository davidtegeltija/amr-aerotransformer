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

Key correctness properties:

* The error pyramid uses the EXACT rectangular quadtree geometry of the mesh
  builder (``QuadNode.compute_child_bboxes``: ``mid = lo + size // 2``), not a
  square 2x2 pooling. On a 256x128 domain the cells are 2:1 rectangles.

* Oracle depth is defined by the nested CONTAINING-CELL test, never a free
  per-pixel detector. Because a containing cell's error includes any hot
  sub-region, two pixels that share a depth-``d`` cell share every error test
  down to depth ``d`` and therefore get the *same* oracle depth whenever that
  depth is ``<= d``. This makes the depth map piecewise-constant over the
  leaves of the mesh it induces ("tiling-consistent"), which is what makes the
  closed-form leaf count exact and the inference-reconstructs-oracle invariant
  hold.

* Error is measured on per-channel scale-normalised target channels so no
  single field dominates through its units.

* ``max_depth`` is the reachable depth derived once at config load from
  ``min_patch_size`` by ``patch_sizes_to_depth_bounds``, never assumed equal to
  any raw configured depth.

The output is a small deterministic integer array ``[H, W]`` and is cheap to
cache per sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Channel scaling
# ---------------------------------------------------------------------------

def per_channel_scale(
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


# ---------------------------------------------------------------------------
# Error pyramid (per-depth within-cell SSD on the exact quadtree geometry)
# ---------------------------------------------------------------------------

@dataclass
class ErrorPyramid:
    """Per-depth within-cell error pyramid over the exact quadtree geometry.

    Attributes:
        ssd: ``[D+1, H, W]`` float64. ``ssd[d, r, c]`` is the within-cell
            sum-of-squared-deviations (over scaled channels and valid pixels)
            of the depth-``d`` cell containing pixel ``(r, c)``. Entries deeper
            than a pixel's reachable depth are ``+inf`` (the pixel has no cell
            there). Depth 0 = root (largest error); error is non-increasing with
            depth.
        area: ``[D+1, H, W]`` float64. ``area[d, r, c]`` is the pixel area
            (height*width) of the same containing cell; ``+inf`` past the
            pixel's reachable depth. Used for the closed-form leaf count.
        pixel_max_depth: ``[H, W]`` int — the deepest cell depth each pixel can
            reach (its leaf depth in a fully-refined tree).
        reachable: Global ``max_depth_reachable`` (= ``D``).
    """

    ssd: np.ndarray
    area: np.ndarray
    pixel_max_depth: np.ndarray
    reachable: int


def build_error_pyramid(
    target: np.ndarray,
    *,
    max_depth: int,
    valid_mask: Optional[np.ndarray] = None,
    channel_scale: Optional[np.ndarray] = None,
) -> ErrorPyramid:
    """Build the within-cell error pyramid for one dense target field.

    Args:
        target: ``[H, W, C]`` ground-truth field.
        max_depth: Reachable depth cap (derived once at config load from
            ``min_patch_size`` by ``patch_sizes_to_depth_bounds``). Every cell
            subdivides uniformly to this depth on an evenly-halving grid.
        valid_mask: Optional ``[H, W]`` bool mask of valid pixels. Invalid
            pixels are excluded from cell means and SSD; a cell with no valid
            pixels gets SSD 0 (it never demands refinement).
        channel_scale: Optional ``[C]`` per-channel scale. Defaults to this
            sample's per-channel std over valid pixels.

    Returns:
        An :class:`ErrorPyramid`.
    """
    assert target.ndim == 3, f"expected [H, W, C], got {target.shape}"
    H, W, C = target.shape
    reachable = max_depth

    if channel_scale is None:
        channel_scale = per_channel_scale(target, valid_mask)
    scaled = target.astype(np.float64) / channel_scale.reshape(1, 1, C)

    if valid_mask is None:
        mask = np.ones((H, W), dtype=bool)
    else:
        mask = valid_mask.astype(bool)

    ssd = np.full((reachable + 1, H, W), np.inf, dtype=np.float64)
    area = np.full((reachable + 1, H, W), np.inf, dtype=np.float64)
    pixel_max_depth = np.zeros((H, W), dtype=np.int64)

    def recurse(r0: int, c0: int, r1: int, c1: int, depth: int) -> None:
        h, w = r1 - r0, c1 - c0
        m = mask[r0:r1, c0:c1]
        n_valid = int(m.sum())
        if n_valid == 0:
            cell_ssd = 0.0
        else:
            vals = scaled[r0:r1, c0:c1, :][m]          # [n_valid, C]
            cell_mean = vals.mean(axis=0)
            cell_ssd = float(((vals - cell_mean) ** 2).sum())

        ssd[depth, r0:r1, c0:c1] = cell_ssd
        area[depth, r0:r1, c0:c1] = float(h * w)

        can_split = depth < max_depth
        if not can_split:
            pixel_max_depth[r0:r1, c0:c1] = depth
            return

        r_mid = r0 + h // 2
        c_mid = c0 + w // 2
        recurse(r0, c0, r_mid, c_mid, depth + 1)   # top-left
        recurse(r0, c_mid, r_mid, c1, depth + 1)   # top-right
        recurse(r_mid, c0, r1, c_mid, depth + 1)   # bottom-left
        recurse(r_mid, c_mid, r1, c1, depth + 1)   # bottom-right

    recurse(0, 0, H, W, 0)
    return ErrorPyramid(ssd=ssd, area=area, pixel_max_depth=pixel_max_depth, reachable=reachable)


def per_depth_total_error(pyr: ErrorPyramid) -> np.ndarray:
    """Total within-cell error at each depth (summed over the distinct cells).

    ``ssd[d]`` repeats a cell's error across all its pixels, so the per-cell
    sum is recovered by dividing by the per-pixel cell area before summing.
    Used to verify pyramid orientation (error non-increasing with depth).

    Returns:
        ``[D+1]`` array; entry ``d`` is the sum of within-cell SSD over all
        depth-``d`` cells. Monotonically non-increasing in ``d``.
    """
    out = np.zeros(pyr.reachable + 1, dtype=np.float64)
    for d in range(pyr.reachable + 1):
        filled = np.isfinite(pyr.ssd[d])
        out[d] = float((pyr.ssd[d][filled] / pyr.area[d][filled]).sum())
    return out


# ---------------------------------------------------------------------------
# Oracle depth + leaf count
# ---------------------------------------------------------------------------

def oracle_from_pyramid(pyr: ErrorPyramid, tol: float, min_depth: int) -> np.ndarray:
    """Per-pixel oracle depth from an error pyramid and a tolerance.

    Oracle depth = the shallowest depth ``>= min_depth`` whose containing cell
    has error ``<= tol``, clamped to the pixel's reachable depth. Pixels never
    acceptable go to their maximum reachable depth.

    Args:
        pyr: The error pyramid from :func:`build_error_pyramid`.
        tol: Error tolerance (on scaled channels).
        min_depth: Depth floor; oracle is never shallower than this.

    Returns:
        ``[H, W]`` int64 oracle depth map.
    """
    cand = pyr.ssd <= tol                      # [D+1, H, W] (inf entries are False)
    if min_depth > 0:
        cand[:min_depth] = False               # force search to start at min_depth
    has = cand.any(axis=0)
    first = cand.argmax(axis=0)                # shallowest accepting depth (0 if none)
    oracle = np.where(has, first, pyr.pixel_max_depth)
    # Clamp into [min_depth, pixel_max_depth]; np.minimum first so a min_depth
    # floor never exceeds a (rare) shallower reachable depth.
    oracle = np.minimum(oracle, pyr.pixel_max_depth)
    oracle = np.maximum(oracle, np.minimum(min_depth, pyr.pixel_max_depth))
    return oracle.astype(np.int64)


def leaf_count_from_oracle(oracle: np.ndarray, pyr: ErrorPyramid) -> float:
    """Closed-form leaf count of an oracle depth map.

    The leaf count of a tiling-consistent depth map is ``sum_pixels 1 / cell_area``
    (each leaf of pixel-area ``A`` contributes ``A * (1/A) = 1``). Because the
    containing-cell oracle is constant within every induced leaf, this is an
    exact integer (up to float rounding) and matches the mesh builder.

    Args:
        oracle: ``[H, W]`` oracle depth map.
        pyr: The pyramid the oracle was derived from (provides cell areas).

    Returns:
        Leaf count as a float (round for the integer count).
    """
    H, W = oracle.shape
    rr, cc = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    areas = pyr.area[oracle, rr, cc]           # per-pixel containing-cell area
    return float((1.0 / areas).sum())


def compute_oracle_depth(
    target: np.ndarray,
    *,
    tol: float,
    min_depth: int,
    max_depth: int,
    valid_mask: Optional[np.ndarray] = None,
    channel_scale: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Convenience: build the pyramid and return the oracle depth map.

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
    pyr = build_error_pyramid(
        target,
        max_depth=max_depth,
        valid_mask=valid_mask,
        channel_scale=channel_scale,
    )
    return oracle_from_pyramid(pyr, tol, min_depth)


# ---------------------------------------------------------------------------
# Global tolerance calibration (single tol across the train split)
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
    per-sample pyramids are built once and reused across the search.

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

    pyramids = [
        build_error_pyramid(t, max_depth=max_depth, valid_mask=m)
        for t, m in zip(targets, masks)
    ]

    def mean_count(tol: float) -> float:
        counts = [
            leaf_count_from_oracle(oracle_from_pyramid(p, tol, min_depth), p)
            for p in pyramids
        ]
        return float(np.mean(counts))

    # Bracket: tol = 0 -> finest (most leaves); tol = max SSD -> coarsest.
    hi = max(float(np.nanmax(p.ssd[np.isfinite(p.ssd)])) for p in pyramids)
    hi = max(hi, 1e-12)
    lo = 0.0

    # mean_count(lo) >= n_target >= mean_count(hi) by monotonicity; if the
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

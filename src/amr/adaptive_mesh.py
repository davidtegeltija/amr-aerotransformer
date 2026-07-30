"""
========================================================================
Physics-aware Adaptive Mesh Refinement (AMR) pipeline.
========================================================================

Combines:
  - Quadtree hierarchy          (quadtree.py)
  - Physics-based refinement    (physics_metrics.py)

Main public API
---------------
build_adaptive_mesh(data, max_depth, refinement_criteria, ...) -> List[QuadNode]
    Takes a single physical field [C, H, W] or [H, W, C] and returns a
    flat list of leaf QuadNodes of the adaptive mesh.

process_batch(data, max_depth, refinement_criteria, ...) -> List[List[QuadNode]]
    Convenience wrapper that processes each sample in a [B, C, H, W] batch
    independently and returns one mesh list per sample.

mesh_statistics(mesh) -> dict
    Summary statistics (patch count, depth range, size distribution).
"""

from __future__ import annotations

from functools import partial
from typing import Dict, List, Optional

import numpy as np

from src.amr.refinement_criteria import GEOMETRY_ONLY_COMBINED_CONFIG, RefinementCriteria
from src.amr.quadtree import QuadNode, build_tree, collect_leaves


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_adaptive_mesh(
    data: np.ndarray,
    refinement_criteria: RefinementCriteria,
    *,
    max_depth: int = 6,
    min_depth: int = 1,
    uniform_cell_size: Optional[int] = None,
) -> List[QuadNode]:
    """
    Build an adaptive mesh over a single physical field.

    Parameters
    ----------
    data : np.ndarray, shape (C, H, W) or (H, W, C)
        Physical field.  Channel-first layout is auto-detected.
    max_depth : int
        Maximum subdivision depth (root = depth 0). Derived from the
        configured patch sizes by patch_sizes_to_depth_bounds so a
        depth-max_depth cell is exactly the finest allowed patch.
    min_depth : int
        Depth floor: cells shallower than this always subdivide, regardless
        of the criteria, so the leaves tile the grid at depth >= min_depth
        (matching build_depth_guided_mesh in the learned path).
    refinement_criteria : RefinementCriteria, optional
        Thresholds controlling subdivision.  Defaults to AERODYNAMIC_CONFIG.
        Use config.scale(factor) to uniformly loosen or tighten the mesh.
        Set individual thresholds to None to disable specific metrics.
    uniform_cell_size : int, optional
        When set, skip adaptive refinement and return a regular grid of
        cells of exactly this pixel size (e.g. 4 for a 4x4 uniform mesh).
        Cells at the right/bottom edge are clamped to the grid boundary.

    Returns
    -------
    List[QuadNode]
        Flat list of leaf nodes representing the adaptive mesh.
    """

    # Input validation and layout normalisation
    if data.ndim == 2:
        data = data[:, :, np.newaxis]

    if data.ndim == 3:
        # Detect channel-first (C, H, W) and transpose to (H, W, C)
        if data.shape[0] < data.shape[1] and data.shape[0] < data.shape[2]:
            data = data.transpose(1, 2, 0) # (C, H, W) -> (H, W, C)
    else:
        raise ValueError(f"Expected 2-D or 3-D input, got shape {data.shape}")

    data = data.astype(np.float64)
    H, W, C = data.shape

    if uniform_cell_size is not None:
        return _build_uniform_mesh(data, uniform_cell_size)

    # Build the quadtree starting with the whole field. partial binds the
    # per-build config, leaving the (node) predicate build_tree expects.
    root = QuadNode(bbox=(0, 0, H, W), depth=0)
    should_subdivide = partial(
        _should_subdivide,
        data=data,
        refinement_criteria=refinement_criteria,
        min_depth=min_depth,
        max_depth=max_depth,
    )
    build_tree(data, root, should_subdivide)
    return collect_leaves(root)


# ---------------------------------------------------------------------------
# Uniform mesh builder
# ---------------------------------------------------------------------------

def _build_uniform_mesh(data: np.ndarray, cell_size: int) -> List[QuadNode]:
    """
    Partition the grid into a regular cell_size × cell_size mesh.

    Parameters
    ----------
    data      : (H, W, C) float64 array
    cell_size : side length of each cell in pixels

    Returns
    -------
    List[QuadNode]  Row-major ordered leaf nodes, one per cell.
    """
    H, W, _ = data.shape
    if cell_size < 1:
        raise ValueError(f"uniform_cell_size must be >= 1, got {cell_size}")

    import math
    depth = max(0, int(round(math.log2(max(H, W) / cell_size))))

    leaves: List[QuadNode] = []
    r = 0
    while r < H:
        r1 = min(r + cell_size, H)
        c = 0
        while c < W:
            c1 = min(c + cell_size, W)
            node = QuadNode(bbox=(r, c, r1, c1), depth=depth)
            node.is_leaf = True
            region = data[r:r1, c:c1, :]
            node.features = region.mean(axis=(0, 1))
            node.metrics = {}
            leaves.append(node)
            c = c1
        r = r1

    return leaves


# ---------------------------------------------------------------------------
# Subdivision decision
# ---------------------------------------------------------------------------

def _should_subdivide(
    node: QuadNode,
    *,
    data: np.ndarray,
    refinement_criteria: RefinementCriteria,
    min_depth: int,
    max_depth: int,
) -> bool:
    """
    Criteria-based subdivision test for a single cell (the ``build_tree``
    predicate; ``functools.partial`` binds the field, criteria and depth bounds).

    Computes only the metrics whose thresholds are enabled, for the x, y, z
    channels (Storage AVG step, Fig. 2 of the AMR-Transformer paper), stores them on the
    node for inspection, then returns True iff the cell should subdivide:
      * forced stop when the cell is at ``max_depth``;
      * forced split below ``min_depth``;
      * otherwise OR-logic: subdivide iff any enabled metric exceeds its
        threshold, matching Eq. 6 of the AMR-Transformer paper. A metric is
        disabled by setting its threshold to None in the criteria.

    Parameters
    ----------
    node : QuadNode  candidate cell (``node.metrics`` is populated here)
    data : np.ndarray  (H, W, C) field; the cell region is sliced by node.bbox
    refinement_criteria  : RefinementCriteria thresholds and scaling flags
    min_depth : int  depth floor (cells shallower than this always subdivide)
    max_depth : int  hard depth cap (cells at this depth never subdivide)

    Returns
    -------
    bool  True -> subdivide this cell.
    """
    region = data[node.r0:node.r1, node.c0:node.c1, :]
    metrics = refinement_criteria.compute_enabled_metrics(region[:, :, :3])
    node.metrics = metrics

    depth = node.depth
    if depth >= max_depth:
        return False
    if depth < min_depth:
        return True
    for metric_name, threshold in refinement_criteria.threshold_checks():
        if metrics.get(metric_name, 0.0) > threshold:
            return True
    return False


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def process_batch(
    data: np.ndarray,
    max_depth: int = 6,
    refinement_criteria: Optional[RefinementCriteria] = None,
) -> List[List[QuadNode]]:
    """
    Process a batch of physical fields independently.

    Parameters
    ----------
    data : (B, C, H, W)
    max_depth : int
    config : RefinementCriteria, optional

    Returns
    -------
    List[List[QuadNode]]  One mesh per batch element.
    """
    if data.ndim != 4:
        raise ValueError(f"Expected 4-D input [B, C, H, W], got shape {data.shape}")

    return [
        build_adaptive_mesh(data[b], max_depth=max_depth, refinement_criteria=refinement_criteria)
        for b in range(data.shape[0])
    ]


# ---------------------------------------------------------------------------
# Mesh statistics
# ---------------------------------------------------------------------------

def mesh_statistics(mesh: List[QuadNode]) -> Dict:
    """
    Summary statistics for a generated mesh.

    Parameters
    ----------
    mesh : List[QuadNode]  Output of build_adaptive_mesh.

    Returns
    -------
    dict with keys:
        total_patches, depth_distribution, min_patch_size,
        max_patch_size, mean_patch_area, depth_range
    """
    if not mesh:
        return {}

    depths  = [p.depth   for p in mesh]
    heights = [p.height  for p in mesh]
    widths  = [p.width   for p in mesh]
    areas   = [h * w for h, w in zip(heights, widths)]

    depth_dist: Dict[int, int] = {}
    for d in depths:
        depth_dist[d] = depth_dist.get(d, 0) + 1

    return {
        "total_patches":      len(mesh),
        "depth_distribution": depth_dist,
        "min_patch_size":     (min(heights), min(widths)),
        "max_patch_size":     (max(heights), max(widths)),
        "mean_patch_area":    float(np.mean(areas)),
        "depth_range":        (min(depths), max(depths)),
    }

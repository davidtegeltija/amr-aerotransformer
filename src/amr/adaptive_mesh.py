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

from typing import Dict, List, Optional, Tuple

import numpy as np

from src.amr.refinement_criteria import GEOMETRY_ONLY_COMBINED_CONFIG, RefinementCriteria
from src.amr.quadtree import QuadNode, collect_leaves


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_adaptive_mesh(
    data: np.ndarray,
    max_depth: int = 6,
    refinement_criteria: Optional[RefinementCriteria] = None,
    uniform_cell_size: Optional[int] = None,
) -> List[QuadNode]:
    """
    Build an adaptive mesh over a single physical field.

    Parameters
    ----------
    data : np.ndarray, shape (C, H, W) or (H, W, C)
        Physical field.  Channel-first layout is auto-detected.
    max_depth : int
        Maximum subdivision depth (root = depth 0). The sole leaf floor;
        derived from the configured patch sizes by patch_sizes_to_depth_bounds
        so a depth-max_depth cell is exactly the finest allowed patch.
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
    if refinement_criteria is None:
        refinement_criteria = GEOMETRY_ONLY_COMBINED_CONFIG

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

    # Build the quadtree starting with the whole field
    root = QuadNode(bbox=(0, 0, H, W), depth=0)
    _build_node(data=data, node=root, max_depth=max_depth, refinement_criteria=refinement_criteria)

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
# Core recursive builder
# ---------------------------------------------------------------------------

def _build_node(
    data: np.ndarray,
    node: QuadNode,
    max_depth: int,
    refinement_criteria: RefinementCriteria,
) -> None:
    """
    Recursively process a single QuadNode.

    Steps
    -----
    1. Extract the data region for this cell.
    2. Compute mean features for storage.
    3. Compute geometry and physics metrics.
    4. Decide whether to subdivide.
    5. If subdividing: create children and recurse.
    """
    # 1. Extract region data
    region = _extract_region(data, node.bbox)

    if region.size == 0:
        node.features = np.zeros(data.shape[2])
        return

    # 2. Compute per-channel mean features (Storage AVG step, Fig. 2 of AMR-Transformer)
    node.features = region.mean(axis=(0, 1))  # (C,)

    # 3. Compute only the metrics whose thresholds are enabled for the x, y, z channels
    metrics = refinement_criteria.compute_enabled_metrics(region[:, :, :3])
    node.metrics = metrics

    # 4. Check stop condition (depth cap is the sole leaf floor)
    if node.depth >= max_depth:
        node.is_leaf = True
        return

    # 5. Subdivision decision via RefinementCriteria
    if not _should_subdivide(refinement_criteria, metrics):
        node.is_leaf = True
        return

    # 6. Subdivide into four children and recurse
    for child in node.subdivide(depth=node.depth + 1):
        _build_node(data=data, node=child, max_depth=max_depth, refinement_criteria=refinement_criteria)


# ---------------------------------------------------------------------------
# Subdivision decision
# ---------------------------------------------------------------------------

def _should_subdivide(
    refinement_criteria: RefinementCriteria,
    metrics: Dict[str, float],
) -> bool:
    """
    Decide whether a region should be subdivided.

    OR-logic: subdivision triggers if any enabled metric exceeds its threshold,
    matching Eq. 6 of the AMR-Transformer paper.
    A metric is disabled by setting its threshold to None in the criteria.

    Parameters
    ----------
    region  : (H, W, C)  raw data for the candidate cell
    refinement_criteria  : RefinementCriteria thresholds and scaling flags
    metrics : dict, optional  pre-computed metrics dict (avoids recomputation)

    Returns
    -------
    bool  True -> subdivide this cell.
    """
    for metric_name, threshold in refinement_criteria.threshold_checks():
        if metrics.get(metric_name, 0.0) > threshold:
            return True
    return False


def _extract_region(data: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Crop a bounding box from a [H, W, C] array.

    Parameters
    ----------
    data : (H, W, C)
    bbox : (r0, c0, r1, c1)

    Returns
    -------
    np.ndarray  shape (r1-r0, c1-c0, C)
    """
    r0, c0, r1, c1 = bbox
    return data[r0:r1, c0:c1, :]


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

"""
========================================================================
Physics-aware Adaptive Mesh Refinement (AMR) pipeline.
========================================================================

Combines:
  - Quadtree hierarchy          (quadtree.py)
  - Physics-based refinement    (physics_metrics.py)

Main public API
---------------
build_adaptive_mesh(data, max_depth, refinement_criteria, ...) -> List[dict]
    Takes a single physical field [C, H, W] or [H, W, C] and returns a
    list of patch dicts representing the leaf cells of the adaptive mesh.

process_batch(data, max_depth, refinement_criteria, ...) -> List[List[dict]]
    Convenience wrapper that processes each sample in a [B, C, H, W] batch
    independently and returns one mesh list per sample.

mesh_statistics(mesh) -> dict
    Summary statistics (patch count, depth range, size distribution).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from src.amr.configs import GEOMETRY_ONLY_COMBINED_CONFIG
from src.amr.refinement_criteria import RefinementCriteria
from src.amr.quadtree import QuadNode, collect_leaves


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_adaptive_mesh(
    data: np.ndarray,
    max_depth: int = 6,
    min_cell_size: int = 4,
    refinement_criteria: Optional[RefinementCriteria] = None,
    return_tree: bool = False,
) -> List[dict]:
    """
    Build an adaptive mesh over a single physical field.

    Parameters
    ----------
    data : np.ndarray, shape (C, H, W) or (H, W, C)
        Physical field.  Channel-first layout is auto-detected.
    max_depth : int
        Maximum subdivision depth (root = depth 0).
    min_cell_size : int
        Cells smaller than this (in either dimension) are never subdivided.
    refinement_criteria : RefinementCriteria, optional
        Thresholds controlling subdivision.  Defaults to AERODYNAMIC_CONFIG.
        Use config.scale(factor) to uniformly loosen or tighten the mesh.
        Set individual thresholds to None to disable specific metrics.
    return_tree : bool
        If True, also return the QuadNode root for inspection.

    Returns
    -------
    List[dict]
        One patch dict per leaf node.  Keys:
            bbox          : (r0, c0, r1, c1)
            depth         : int
            mean_features : list[float]  length C
            center        : (row_center, col_center)
            size          : (height, width)
            metrics       : dict[str, float]
    QuadNode (only if return_tree=True)
        Root of the quadtree for inspection / visualisation.
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

    # Build the quadtree starting with the whole field
    root = QuadNode(bbox=(0, 0, H, W), depth=0)
    _build_node(data=data, node=root, max_depth=max_depth, min_cell_size=min_cell_size, refinement_criteria=refinement_criteria)

    patches = [leaf.to_patch_dict() for leaf in collect_leaves(root)]

    if return_tree:
        return patches, root
    return patches


# ---------------------------------------------------------------------------
# Core recursive builder
# ---------------------------------------------------------------------------

def _build_node(
    data: np.ndarray,
    node: QuadNode,
    max_depth: int,
    min_cell_size: int,
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

    # 3. Compute only the metrics whose thresholds are enabled
    metrics = refinement_criteria.compute_enabled_metrics(region)
    node.metrics = metrics

    # 4. Check stop conditions
    cell_too_small = (
        node.height // 2 < min_cell_size  or
        node.width  // 2 < min_cell_size
    )
    at_max_depth = node.depth >= max_depth

    if at_max_depth or cell_too_small:
        node.is_leaf = True
        return

    # 5. Subdivision decision via RefinementCriteria
    if not should_subdivide(region, refinement_criteria, metrics=metrics):
        node.is_leaf = True
        return

    # 6. Subdivide into four children and recurse
    for child in node.subdivide(depth=node.depth + 1):
        _build_node(data=data, node=child, max_depth=max_depth, min_cell_size=min_cell_size, refinement_criteria=refinement_criteria)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def should_subdivide(
    region: np.ndarray,
    refinement_criteria: RefinementCriteria,
    metrics: Optional[Dict[str, float]] = None,
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
    if metrics is None:
        metrics = refinement_criteria.compute_enabled_metrics(region)
        
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
    min_cell_size: int = 4,
    refinement_criteria: Optional[RefinementCriteria] = None,
) -> List[List[dict]]:
    """
    Process a batch of physical fields independently.

    Parameters
    ----------
    data : (B, C, H, W)
    max_depth : int
    min_cell_size : int
    config : RefinementCriteria, optional

    Returns
    -------
    List[List[dict]]  One mesh per batch element.
    """
    if data.ndim != 4:
        raise ValueError(f"Expected 4-D input [B, C, H, W], got shape {data.shape}")

    return [
        build_adaptive_mesh(data[b], max_depth=max_depth, min_cell_size=min_cell_size, refinement_criteria=refinement_criteria)
        for b in range(data.shape[0])
    ]


# ---------------------------------------------------------------------------
# Mesh statistics
# ---------------------------------------------------------------------------

def mesh_statistics(mesh: List[dict]) -> Dict:
    """
    Summary statistics for a generated mesh.

    Parameters
    ----------
    mesh : List[dict]  Output of build_adaptive_mesh.

    Returns
    -------
    dict with keys:
        total_patches, depth_distribution, min_patch_size,
        max_patch_size, mean_patch_area, depth_range
    """
    if not mesh:
        return {}

    depths  = [p["depth"]    for p in mesh]
    heights = [p["size"][0]  for p in mesh]
    widths  = [p["size"][1]  for p in mesh]
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

"""
========================================================================
Thin adapter between the AMR mesh pipeline and the Transformer pipeline.
========================================================================

This module is the only file the rest of the pipeline (model.py, train.py,
reconstruction.py, visualization.py) needs to import.  It re-exports the
same public symbols as the original implementation so that no other file
needs to change.

Architecture
------------
The actual patch generation is now handled by three dedicated modules:

    quadtree.py        - QuadNode tree data structure and traversal
    physics_metrics.py - Six physics-aware refinement metrics
                         (velocity gradient, vorticity, momentum, KH shear,
                          variance, entropy) matching AMR-Transformer Eqs 2-5
    adaptive_mesh.py   - build_adaptive_mesh() pipeline that drives the tree

This module's only job is to:
1. Call build_adaptive_mesh() with a RefinementCriteria and collect leaf QuadNodes.
2. Stack the leaf nodes into the [N, C+3] float32 token array the Transformer consumes.
3. Pass the leaf nodes directly to reconstruction.py and visualization.py.
   QuadNode already carries everything those modules need (bbox, depth, features).

Public API
--------------------------------------------
  QuadNode               - dataclass holding one cell's data + bbox
  QuadtreeTokenizer      - tokenizes a [H, W, C] grid to [N, C+3]
  RefinementCriteria     - base class for custom criteria
"""

from __future__ import annotations

from typing import List, Optional, Tuple
 
import numpy as np
 
from src.amr.refinement_criteria import AERODYNAMIC_CRITERIA, RefinementCriteria
from src.amr.adaptive_mesh import build_adaptive_mesh
from src.amr.quadtree import QuadNode


class QuadtreeTokenizer:
    """
    Tokenizes a spatial [H, W, C] grid into adaptive mesh tokens using the
    physics-aware AMR pipeline from adaptive_mesh.py.

    Drives build_adaptive_mesh() and converts its List[dict] patch output
    into the [N, C+3] token array and List[QuadNode] metadata that the
    Transformer pipeline expects.

    Args
    ----
    min_depth               : minimum quadtree depth (cells always subdivided to here)
    max_depth               : maximum quadtree depth (hard upper limit, and sole leaf
                              floor; derived from the configured patch sizes by
                              patch_sizes_to_depth_bounds)
    refinement_criteria     : RefinementCriteria instance controlling physics thresholds.
                              Defaults to AERODYNAMIC_CONFIG from physics_metrics.py.
    """

    def __init__(
        self,
        min_depth:           int = 2,
        max_depth:           int = 6,
        refinement_criteria: Optional[RefinementCriteria] = None,
    ):
        self.min_depth           = min_depth
        self.max_depth           = max_depth
        self.refinement_criteria = refinement_criteria or AERODYNAMIC_CRITERIA

    def tokenize(self, grid: np.ndarray) -> Tuple[np.ndarray, List[QuadNode]]:
        """
        Tokenize a single spatial grid.

        Args
        ----
        grid : [H, W, C] float32 numpy array

        Returns
        -------
        token_array : [N, C+3] float32 array
                      columns: [feat_0...feat_{C-1}, x_c, y_c, cell_size]
        token_list  : List[QuadNode] with bounding boxes for reconstruction
        """
        assert grid.ndim == 3, f"Expected [H, W, C], got shape {grid.shape}"
        H, W, C = grid.shape

        node_list: List[QuadNode] = build_adaptive_mesh(
            grid,
            max_depth=self.max_depth,
            min_depth=self.min_depth,
            refinement_criteria=self.refinement_criteria,
        )

        token_array = nodes_to_token_array(node_list, H, W, C)
        return token_array, node_list


# ------------------------------------------------------------------
# Convert Quadtree leaves to token array helper
# ------------------------------------------------------------------

def nodes_to_token_array(nodes: List[QuadNode], H: int, W: int, C: int) -> np.ndarray:
    """
    Stack leaf QuadNodes into a [N, C+3] float32 token array.

    Columns:
        0..C-1  : per-channel mean features (from node.features)
        C       : x_center  -- normalised column centre  = x_center / W
        C+1     : y_center  -- normalised row centre     = y_center / H
        C+2     : cell_size -- normalised max dimension  = max(width/W, height/H)

    Args:
        nodes: Leaf ``QuadNode`` s to tokenize.
        H: Grid height (rows), used to normalise the row centre.
        W: Grid width (columns), used to normalise the column centre.
        C: Number of feature channels.

    Returns:
        ``[N, C+3]`` float32 array, one row per node.
    """
    N = len(nodes)
    arr = np.empty((N, C + 3), dtype=np.float32)
    for i, node in enumerate(nodes):
        y_center, x_center = node.center
        arr[i, :C] = node.features if node.features is not None else 0.0
        arr[i, C] = x_center / W
        arr[i, C+1] = y_center / H
        arr[i, C+2] = max(node.width / W, node.height / H)
    return arr

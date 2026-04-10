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
1. Call build_adaptive_mesh() with with a RefinementCriteria and collect leaf QuadNodes.
2. Stack the leaf nodes into the [N, C+4] float32 token array the Transformer consumes.
3. Pass the leaf nodes directly to reconstruction.py and visualization.py.
   QuadNode already carries everything those modules need (bbox, depth, features).

Public API (unchanged from previous version)
--------------------------------------------
  QuadtreeToken               - dataclass holding one cell's data + bbox
  QuadtreeTokenizer           - tokenizes a [H, W, C] grid to [N, C+4]
  RefinementCriteria          - base class for custom criteria
"""

from __future__ import annotations

from typing import List, Optional, Tuple
 
import numpy as np
import torch
 
from src.amr.configs import AERODYNAMIC_CRITERIA
from src.amr.refinement_criteria import RefinementCriteria
from src.amr.adaptive_mesh import build_adaptive_mesh
from src.amr.quadtree import QuadNode, collect_leaves


# ---------------------------------------------------------------------------
# QuadtreeTokenizer
# ---------------------------------------------------------------------------

class QuadtreeTokenizer:
    """
    Tokenizes a spatial [H, W, C] grid into adaptive mesh tokens using the
    physics-aware AMR pipeline from adaptive_mesh.py.

    Drives build_adaptive_mesh() and converts its List[dict] patch output
    into the [N, C+4] token array and List[QuadtreeToken] metadata that the
    Transformer pipeline expects.

    Args
    ----
    min_depth               : minimum quadtree depth (cells always subdivided to here)
    max_depth               : maximum quadtree depth (hard upper limit)
    min_cell_size           : minimum pixel side-length; cells below this are not split.
    refinement_criteria     : RefinementCriteria instance controlling physics thresholds.
                              Defaults to AERODYNAMIC_CONFIG from physics_metrics.py. 
    """

    def __init__(
        self,
        min_depth:           int = 2,
        max_depth:           int = 6,
        min_cell_size:       int = 4,
        refinement_criteria: Optional[RefinementCriteria] = None,
    ):
        self.min_depth           = min_depth
        self.max_depth           = max_depth
        self.min_cell_size       = min_cell_size
        self.refinement_criteria = refinement_criteria or AERODYNAMIC_CRITERIA

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tokenize(self, grid: np.ndarray) -> Tuple[np.ndarray, List[QuadNode]]:
        """
        Tokenize a single spatial grid.

        Args
        ----
        grid : [H, W, C] float32 numpy array

        Returns
        -------
        token_array : [N, C+4] float32 array
                      columns: [feat_0...feat_{C-1}, x_c, y_c, cell_size, depth_norm]
        token_list  : List[QuadNode] with bounding boxes for reconstruction
        """
        assert grid.ndim == 3, f"Expected [H, W, C], got shape {grid.shape}"
        H, W, C = grid.shape

        _, root = build_adaptive_mesh(
            grid,
            max_depth=self.max_depth,
            min_cell_size=self.min_cell_size,
            refinement_criteria = self.refinement_criteria,
            return_tree=True,
        )
 
        token_list: List[QuadNode] = collect_leaves(root)

        # Discard patches from shallower levels
        if self.min_depth > 0:
            filtered = [t for t in token_list if t.depth >= self.min_depth]
            # Guard against empty result on very small grids
            token_list = filtered if filtered else token_list

        token_array = self._tokens_to_array(token_list, H, W, C)
        return token_array, token_list

    def tokenize_tensor(self, grid: torch.Tensor) -> Tuple[torch.Tensor, List[QuadNode]]:
        """Convenience wrapper that accepts and returns torch.Tensor."""
        arr = grid.cpu().numpy() if isinstance(grid, torch.Tensor) else grid
        token_np, token_list = self.tokenize(arr)
        return torch.from_numpy(token_np), token_list

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tokens_to_array(tokens: List[QuadNode], H: int, W: int, C: int, max_depth: int = 20) -> np.ndarray:
        """
        Stack leaf QuadNodes (tokens) into a [N, C+4] float32 array.
        
        Columns:
            0..C-1  : per-channel mean features (from node.features)
            C       : x_center  -- normalised column centre  = (c0+c1)/2 / W
            C+1     : y_center  -- normalised row centre     = (r0+r1)/2 / H
            C+2     : cell_size -- normalised max dimension  = max(width/W, height/H)
            C+3     : depth_norm -- depth / max_depth
        """
        N = len(tokens)
        arr = np.empty((N, C + 4), dtype=np.float32)
        for i, token in enumerate(tokens):
            arr[i, :C]  = token.features if token.features is not None else 0.0
            arr[i, C]   = (token.c0 + token.c1) / 2.0 / W   # x_center
            arr[i, C+1] = (token.r0 + token.r1) / 2.0 / H   # y_center
            arr[i, C+2] = max(token.width / W, token.height / H)  # cell_size
            arr[i, C+3] = token.depth / max_depth
        return arr


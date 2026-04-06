"""
========================================================================
Core quadtree data structure for adaptive mesh refinement.
========================================================================

Implements a QuadNode class and recursive subdivision logic.
Each node covers a rectangular region of the physical domain and
can be split into four equal children when a refinement criterion fires.

Coordinate convention (row-major, matching numpy/image layout):
    bbox = (r0, c0, r1, c1)   # inclusive top-left, exclusive bottom-right
    r = row index (height axis)
    c = column index (width axis)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

@dataclass
class QuadNode:
    """
    A single node in the quadtree.

    Attributes
    ----------
    bbox : (r0, c0, r1, c1)
        Bounding box of this cell in pixel coordinates.
        r0, c0 are inclusive; r1, c1 are exclusive.
    depth : int
        Depth of this node (root = 0).
    children : list of QuadNode
        Four children produced after subdivision (empty if leaf).
    features : np.ndarray or None
        Per-channel mean values inside this cell.  Shape: (C,)
    metrics : dict
        Dictionary of computed physics metrics (for inspection / debugging).
    is_leaf : bool
        True if this node was not further subdivided.
    """

    bbox: Tuple[int, int, int, int]       # (r0, c0, r1, c1)
    depth: int = 0
    children: List["QuadNode"] = field(default_factory=list)
    features: Optional[np.ndarray] = None
    metrics: dict = field(default_factory=dict)
    is_leaf: bool = False

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    @property
    def r0(self) -> int:
        return self.bbox[0]

    @property
    def c0(self) -> int:
        return self.bbox[1]

    @property
    def r1(self) -> int:
        return self.bbox[2]

    @property
    def c1(self) -> int:
        return self.bbox[3]

    @property
    def height(self) -> int:
        return self.r1 - self.r0

    @property
    def width(self) -> int:
        return self.c1 - self.c0

    @property
    def center(self) -> Tuple[float, float]:
        """(row_center, col_center) in pixel coordinates."""
        return (self.r0 + self.r1) / 2.0, (self.c0 + self.c1) / 2.0

    def area(self) -> int:
        return self.height * self.width

    # ------------------------------------------------------------------
    # Child generation
    # ------------------------------------------------------------------

    def compute_child_bboxes(self) -> List[Tuple[int, int, int, int]]:
        """
        Split this node into four equal quadrants.

        Returns a list of four (r0, c0, r1, c1) tuples:
            [top-left, top-right, bottom-left, bottom-right]
        """
        r_mid = self.r0 + self.height // 2
        c_mid = self.c0 + self.width // 2

        return [
            (self.r0, self.c0, r_mid,    c_mid   ),   # top-left
            (self.r0, c_mid,   r_mid,    self.c1 ),   # top-right
            (r_mid,   self.c0, self.r1,  c_mid   ),   # bottom-left
            (r_mid,   c_mid,   self.r1,  self.c1 ),   # bottom-right
        ]

    def subdivide(self, depth: int) -> List["QuadNode"]:
        """
        Create four child nodes and attach them.

        Parameters
        ----------
        depth : int  Depth of the children (self.depth + 1).

        Returns
        -------
        list of QuadNode
        """
        self.is_leaf = False
        child_bboxes = self.compute_child_bboxes()
        self.children = [QuadNode(bbox=bb, depth=depth) for bb in child_bboxes]
        return self.children

    # ------------------------------------------------------------------
    # Patch dict export
    # ------------------------------------------------------------------

    def to_patch_dict(self) -> dict:
        """
        Export this node as the standard patch dictionary format.

        Returns
        -------
        dict with keys:
            bbox         : (r0, c0, r1, c1)
            depth        : int
            mean_features: list of floats (one per channel)
            center       : (row_center, col_center)
            size         : (height, width)
            metrics      : dict of scalar physics metrics
        """
        return {
            "bbox":          self.bbox,
            "depth":         self.depth,
            "mean_features": self.features.tolist() if self.features is not None else [],
            "center":        self.center,
            "size":          (self.height, self.width),
            "metrics":       self.metrics,
        }

    def __repr__(self) -> str:
        return (
            f"QuadNode(depth={self.depth}, bbox={self.bbox}, "
            f"size=({self.height}x{self.width}), leaf={self.is_leaf})"
        )


# ---------------------------------------------------------------------------
# Tree traversal utilities
# ---------------------------------------------------------------------------

def collect_leaves(root: QuadNode) -> List[QuadNode]:
    """
    DFS traversal: return all leaf nodes of the quadtree.
    Leaf nodes correspond to the final adaptive mesh cells.
    """
    leaves: List[QuadNode] = []
    stack = [root]
    while stack:
        node = stack.pop()
        if node.is_leaf:
            leaves.append(node)
        else:
            stack.extend(node.children)
    return leaves


def collect_all_nodes(root: QuadNode) -> List[QuadNode]:
    """DFS traversal: return every node at every depth level."""
    nodes: List[QuadNode] = []
    stack = [root]
    while stack:
        node = stack.pop()
        nodes.append(node)
        stack.extend(node.children)
    return nodes


def collect_nodes_at_depth(root: QuadNode, target_depth: int) -> List[QuadNode]:
    """Return all nodes at a specific depth level."""
    result: List[QuadNode] = []
    stack = [root]
    while stack:
        node = stack.pop()
        if node.depth == target_depth:
            result.append(node)
        elif node.depth < target_depth:
            stack.extend(node.children)
    return result
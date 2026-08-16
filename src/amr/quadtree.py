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
import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple
import numpy as np


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


# ---------------------------------------------------------------------------
# Shared recursive builder
# ---------------------------------------------------------------------------

def build_tree(
    data: np.ndarray,
    node: QuadNode,
    should_subdivide: Callable[[QuadNode], bool],
) -> None:
    """Recursively build the quadtree rooted at ``node``, in place.

    The skeleton shared by the deterministic (criteria) and learned (depth-map)
    builders: extract the cell region, guard zero-area cells, store the mean
    features, then delegate the split/stop decision to ``should_subdivide`` and
    recurse. The predicate owns everything mode-specific — metric computation,
    storing ``node.metrics``, and the actual threshold/depth test; it reads the
    cell extent from ``node.bbox``.

    Populates the tree in place: each final cell is flagged ``is_leaf`` so that
    ``collect_leaves(root)`` afterwards returns the mesh patches.

    Steps
    -----
    1. Extract the data region for this cell.
    2. Compute mean features for storage.
    3. Decide whether to subdivide (delegated to ``should_subdivide``, which
       also computes and stores any mode-specific metrics).
    4. If subdividing: create children and recurse; else mark as leaf.

    Args:
        data: ``[H, W, C]`` field. Each cell's per-channel mean over its bbox
            becomes ``node.features``.
        node: Cell to process; call with the seeded root (bbox covering the
            whole grid, depth 0) to build the full tree.
        should_subdivide: Callable ``(node) -> bool`` invoked on each
            non-degenerate cell after its features are set. Returns True to
            split the cell into four children and recurse.
    """
    # 1. Extract region data
    r0, c0, r1, c1 = node.bbox
    region = data[r0:r1, c0:c1, :]

    if region.size == 0:
        # Zero-area cell: not marked as leaf, so collect_leaves drops it
        # (a degenerate token would yield NaN per-token targets downstream).
        node.features = np.zeros(data.shape[2], dtype=data.dtype)
        return

    # 2. Compute per-channel mean features (Storage AVG step, Fig. 2 of the AMR-Transformer paper)
    node.features = region.mean(axis=(0, 1))  # (C,)

    # 3. and 4. Subdivide into four children and recurse, or mark as leaf
    if should_subdivide(node):
        for child in node.subdivide(depth=node.depth + 1):
            build_tree(data, child, should_subdivide)
    else:
        node.is_leaf = True


# ---------------------------------------------------------------------------
# Quadtree leaves -> token array
# ---------------------------------------------------------------------------

def nodes_to_token_array(nodes: List[QuadNode], H: int, W: int, C: int) -> np.ndarray:
    """Stack leaf QuadNodes into a ``[N, C+3]`` float32 token array.

    Columns:
        0..C-1  : per-channel mean features (from node.features)
        C       : x_center   -- normalised column centre = x_center / W
        C+1     : y_center   -- normalised row centre    = y_center / H
        C+2     : cell_level -- refinement depth = -log2(max(width/W, height/H))

    The size channel is stored as a log2 level rather than the raw normalised
    extent because cell size is inherently a power of two. A leaf at quadtree
    depth d has normalised extent 2**-d, so this column is just d, giving evenly
    spaced integers 0, 1, 2, ... instead of the geometrically bunched 1, 1/2,
    1/4, ... The consumer is the Fourier positional encoding in AMRTransformer,
    whose fixed frequency bank is shared across all three meta channels; raw
    extents crowd every refinement level into a narrow band near zero, where the
    low frequencies are near-linear and the high ones alias, so neither end of
    the bank separates a coarse cell from a fine one cleanly. Levels spread the
    same information across a range the bank resolves. Non-power-of-two extents
    (partial edge cells from a uniform mesh) stay well defined as a float level.

    Note:
        This is the *model's* view of cell size. Anything doing geometry with the
        cell -- e.g. the affine ramp normalisation in models/reconstruction.py --
        needs the linear extent and computes it from the leaf directly.

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
        arr[i, C + 1] = y_center / H
        arr[i, C + 2] = -math.log2(max(node.width / W, node.height / H))
    return arr
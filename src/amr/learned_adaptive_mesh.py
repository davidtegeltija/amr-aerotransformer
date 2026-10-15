"""
========================================================================
Depth-guided quadtree builder for the variance-supervised scorer.
========================================================================

The RefinementNet emits a single scalar field over the grid, interpreted as a
*predicted target depth* ``d_pred(p)`` (NOT a sign-thresholded logit). The mesh
builder turns it into leaves with one rule:

    a depth-``d`` cell subdivides  iff  mean(d_pred over the cell) > d + offset

``offset`` is a single global budget knob that shifts the
whole mesh coarser (positive) or finer (negative).

Why a running-depth comparison rather than a fixed ``logit > 0`` threshold. A
static sign map provably cannot represent an intermediate-depth plateau — a
depth-``d`` cell that must stop pins all of its pixels to the "stop" side, which
forces its parent (four such cells) to stop one level too shallow. Comparing a
predicted value against the moving depth removes that wall while keeping a
single map.

Feeding the oracle-depth map (see ``src/amr/oracle_depth.py``) in place of the
scorer output reconstructs the oracle mesh exactly — the consistency invariant
that ties inference to the supervised target. The scorer is now trained by 
supervised regression, decoupled from the transformer.

A quadtree split keeps the parent's aspect ratio, so the leaf shape is set by
the root tiles (``root_bboxes``). ``patch_sides_ratio`` is the leaf height / width: on
the 256x128 wing grid 1 gives two square roots; None keeps the whole grid as
the one root (2:1 leaves).

bbox convention (matched to ``quadtree.py``): ``(r0, c0, r1, c1)`` with
``r0, c0`` inclusive and ``r1, c1`` exclusive.
"""

from __future__ import annotations

from functools import partial
from typing import List, Optional

import numpy as np
import torch

from src.amr.quadtree import QuadNode, build_tree, collect_leaves, root_bboxes


# ---------------------------------------------------------------------------
# Public API — depth-guided builder
# ---------------------------------------------------------------------------

def build_depth_guided_mesh(
    data: np.ndarray,
    depth_map: np.ndarray,
    *,
    max_depth: int = 5,
    min_depth: int = 2,
    offset: float = 0.0,
    patch_sides_ratio: Optional[float] = None,
) -> List[QuadNode]:
    """Build a quadtree by comparing a predicted depth map against running depth.

    Args:
        data: ``[H, W, C]`` raw geometry grid (used only for ``node.features``,
            the per-channel cell mean).
        depth_map: ``[H, W]`` predicted depth ``d_pred`` (any real values). A
            ``torch.Tensor`` is accepted and detached to numpy — the builder is
            inference-only and carries no gradient.
        max_depth: Hard depth cap (cells at this depth never subdivide). Derived
            at config load from the patch sizes; on an evenly-halving grid a
            depth-``max_depth`` cell is exactly the finest allowed patch.
        min_depth: Depth floor (cells shallower than this always subdivide).
        offset: Global budget offset added to the running depth in the
            comparison. Positive -> coarser mesh, negative -> finer.
        patch_sides_ratio: Leaf height / width (see ``root_shape``).

    Returns:
        Flat list of leaf ``QuadNode`` s, each with ``features`` populated, that
        tile the full ``H x W`` grid.
    """
    if data.ndim != 3:
        raise ValueError(f"Expected data with shape (H, W, C), got {data.shape}")
    if isinstance(depth_map, torch.Tensor):
        depth_map = depth_map.detach().cpu().numpy()
    depth_map = np.asarray(depth_map, dtype=np.float64)
    if depth_map.ndim != 2:
        raise ValueError(f"Expected depth_map with shape (H, W), got {depth_map.shape}")

    H, W, _C = data.shape
    if depth_map.shape != (H, W):
        raise ValueError(
            f"depth_map shape must equal data H, W = {(H, W)}; got {depth_map.shape}"
        )

    # partial binds the depth map and budget, leaving the (node) predicate
    # build_tree expects.
    should_subdivide = partial(
        _should_subdivide_depth,
        depth_map=depth_map,
        min_depth=min_depth,
        max_depth=max_depth,
        offset=offset,
    )
    leaves = []
    for bbox in root_bboxes(H, W, patch_sides_ratio):
        root = QuadNode(bbox=bbox, depth=0)
        build_tree(data, root, should_subdivide)
        leaves += collect_leaves(root)
    return leaves


# ---------------------------------------------------------------------------
# Subdivision decision
# ---------------------------------------------------------------------------

def _should_subdivide_depth(
    node: QuadNode,
    *,
    depth_map: np.ndarray,
    min_depth: int,
    max_depth: int,
    offset: float,
) -> bool:
    """Running-depth subdivision test for a single cell (the ``build_tree``
    predicate; ``functools.partial`` binds the depth map and budget).

    Returns True iff the cell should subdivide:
      * forced stop when the cell is at ``max_depth``;
      * forced split below ``min_depth``;
      * otherwise subdivide iff ``mean(d_pred over cell) > depth + offset``.
    """
    depth = node.depth
    if depth >= max_depth:
        return False
    if depth < min_depth:
        return True
    r0, c0, r1, c1 = node.bbox
    cell_mean = round(float(depth_map[r0:r1, c0:c1].mean()))
    return cell_mean > depth + offset


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import numpy as np

    H, W = 256, 128
    data = np.random.default_rng(0).standard_normal((H, W, 3)).astype(np.float32)

    # Each root holds 4**depth leaves at a uniform depth, all of the root's aspect.
    for patch_sides_ratio in (None, 2.0, 1.0, 0.5, 4.0):
        n_roots = len(root_bboxes(H, W, patch_sides_ratio))
        aspect = patch_sides_ratio or H / W
        for depth_value, depth in ((2.0, 2), (99.0, 4)):
            leaves = build_depth_guided_mesh(
                data, np.full((H, W), depth_value), max_depth=4, min_depth=2, patch_sides_ratio=patch_sides_ratio
            )
            assert len(leaves) == n_roots * 4 ** depth, f"aspect {patch_sides_ratio}: got {len(leaves)} leaves"
            assert all(n.height == aspect * n.width for n in leaves), f"aspect {patch_sides_ratio}: wrong leaf shape"
    print("Smoke OK: leaf counts and shapes match every patch_sides_ratio.")

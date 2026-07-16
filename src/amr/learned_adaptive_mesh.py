"""
========================================================================
Depth-guided quadtree builder for the variance-supervised scorer.
========================================================================

The RefinementNet emits a single scalar field over the grid, interpreted as a
*predicted target depth* ``d_pred(p)`` (NOT a sign-thresholded logit). The mesh
builder turns it into leaves with one rule:

    a depth-``d`` cell subdivides  iff  max(d_pred over the cell) > d + offset

``offset`` is a single global budget knob that shifts the
whole mesh coarser (positive) or finer (negative).

Why a running-depth comparison rather than a fixed ``logit > 0`` threshold. A
static sign map under the max-OR subdivision rule provably cannot represent an
intermediate-depth plateau — if a depth-``d`` cell must stop, all of its pixels
sit on the "stop" side, which forces its parent (four such cells) to stop one
level too shallow. Comparing a predicted value against the moving depth removes
that wall while keeping a single map.

Feeding the oracle-depth map (see ``src/amr/oracle_depth.py``) in place of the
scorer output reconstructs the oracle mesh exactly — the consistency invariant
that ties inference to the supervised target. The scorer is now trained by 
supervised regression, decoupled from the transformer.

bbox convention (matched to ``quadtree.py``): ``(r0, c0, r1, c1)`` with
``r0, c0`` inclusive and ``r1, c1`` exclusive.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch

from src.amr.quadtree import QuadNode, collect_leaves


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

    root = QuadNode(bbox=(0, 0, H, W), depth=0)
    _build_node_depth(
        data=data,
        node=root,
        depth_map=depth_map,
        max_depth=max_depth,
        min_depth=min_depth,
        offset=offset,
    )
    return collect_leaves(root)




# ---------------------------------------------------------------------------
# Core recursive builder
# ---------------------------------------------------------------------------

def _build_node_depth(
    data: np.ndarray,
    node: QuadNode,
    depth_map: np.ndarray,
    max_depth: int,
    min_depth: int,
    offset: float,
) -> None:
    """Populate ``node`` (features + children) in-place by depth-guided recursion."""
    r0, c0, r1, c1 = node.bbox
    region = data[r0:r1, c0:c1, :]

    if region.size == 0:
        # Zero-area cell: not marked as leaf, so collect_leaves drops it
        # (a degenerate token would yield NaN per-token targets downstream).
        node.features = np.zeros(data.shape[2], dtype=data.dtype)
        return

    node.features = region.mean(axis=(0, 1))
    node.metrics = {}

    if _should_subdivide_depth(
        depth_map=depth_map,
        r0=r0, c0=c0, r1=r1, c1=c1,
        depth=node.depth,
        min_depth=min_depth,
        max_depth=max_depth,
        offset=offset,
    ):
        for child in node.subdivide(depth=node.depth + 1):
            _build_node_depth(
                data=data,
                node=child,
                depth_map=depth_map,
                max_depth=max_depth,
                min_depth=min_depth,
                offset=offset,
            )
    else:
        node.is_leaf = True


def _should_subdivide_depth(
    depth_map: np.ndarray,
    r0: int, c0: int, r1: int, c1: int,
    depth: int,
    min_depth: int,
    max_depth: int,
    offset: float,
) -> bool:
    """Running-depth subdivision test for a single cell.

    Returns True iff the cell should subdivide:
      * forced stop when the cell is at ``max_depth``;
      * forced split below ``min_depth``;
      * otherwise subdivide iff ``max(d_pred over cell) > depth + offset``.
    """
    if depth >= max_depth:
        return False
    if depth < min_depth:
        return True
    cell_max = round(float(depth_map[r0:r1, c0:c1].mean()))
    return cell_max > depth + offset


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import numpy as np
    from src.utils.geometry_utils import mesh_token_bounds

    H, W = 256, 128
    floor, cap = mesh_token_bounds(H, W, min_depth=2, max_depth=5)
    assert (floor, cap) == (16, 1024), f"unexpected bounds: {(floor, cap)}"

    data = np.random.default_rng(0).standard_normal((H, W, 3)).astype(np.float32)

    # Uniform depth map at the floor -> floor leaves.
    leaves = build_depth_guided_mesh(
        data, np.full((H, W), 2.0), max_depth=5, min_depth=2
    )
    assert len(leaves) == floor, f"expected {floor} leaves, got {len(leaves)}"

    # Uniform depth map above the cap -> cap leaves.
    leaves = build_depth_guided_mesh(
        data, np.full((H, W), 99.0), max_depth=5, min_depth=2
    )
    assert len(leaves) == cap, f"expected {cap} leaves, got {len(leaves)}"
    print(f"Smoke OK: bounds=({floor},{cap}); uniform floor/cap reproduce them.")

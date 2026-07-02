from typing import Tuple


def max_reachable_depth(H: int, W: int, min_cell_size: int, max_depth: int) -> int:
    """Deepest quadtree depth any cell reaches under the builder's stop rules.

    Mirrors ``_should_subdivide`` in the mesh builders: a cell is a forced leaf
    once the next split would drop either axis below ``min_cell_size``, or once
    it hits ``max_depth``. The reachable depth is the maximum leaf depth when
    *every* cell subdivides as far as allowed.

    This is the value the oracle, the loss, and the mesh builder must all share.
    It can be strictly smaller than the configured ``max_depth``
    (e.g. width 128, ``min_cell_size`` 4 -> depth 5, not 6).

    Args:
        H, W: Grid dimensions in pixels.
        min_cell_size: A cell whose next split would drop either axis below this
            size is a forced leaf.
        max_depth: Configured hard depth cap.

    Returns:
        The deepest reachable depth (an int in ``[0, max_depth]``).
    """
    def deepest(h: int, w: int, depth: int) -> int:
        if depth >= max_depth:
            return depth
        if h // 2 < min_cell_size or w // 2 < min_cell_size:
            return depth
        # Child sizes follow QuadNode.compute_child_bboxes (size//2 and size - size//2).
        return max(
            deepest(hh, ww, depth + 1)
            for hh in (h // 2, h - h // 2)
            for ww in (w // 2, w - w // 2)
        )

    return deepest(H, W, 0)


def mesh_token_bounds(
    H: int,
    W: int,
    min_depth: int,
    max_depth: int,
) -> Tuple[int, int]:
    """
    Reachable leaf-count bounds for the depth-guided builder.

    The floor is the leaf count when every non-forced decision says "stop"
    (collapse to the ``min_depth`` floor); the cap is the leaf count when every
    non-forced decision says "subdivide" (max refinement under ``max_depth``).
    Used to annotate the end-to-end sanity log with the token range the mesh can
    occupy.

    Args:
        H, W: Grid dimensions in pixels.
        min_depth: Depth below which subdivision is forced.
        max_depth: Depth at which subdivision stops unconditionally (the reachable
            depth derived from the patch sizes).

    Returns:
        (floor_tokens, max_tokens) leaf counts.
    """
    def count(h: int, w: int, depth: int, always_subdivide: bool) -> int:
        if depth >= max_depth:
            return 1
        if depth >= min_depth and not always_subdivide:
            return 1
        # Child sizes follow QuadNode.compute_child_bboxes (h//2 and h - h//2).
        return sum(
            count(hh, ww, depth + 1, always_subdivide)
            for hh in (h // 2, h - h // 2)
            for ww in (w // 2, w - w // 2)
        )

    return count(H, W, 0, False), count(H, W, 0, True)


def patch_sizes_to_depth_bounds(
    H: int,
    W: int,
    min_patch_size: int,
    max_patch_size: int,
) -> Tuple[int, int]:
    """Convert (min_patch_size, max_patch_size) pixel bounds to integer (min_depth, max_depth) quadtree depth bounds.

    Patch-size bounds are resolution-independent; the rest of the pipeline keys on
    integer quadtree depth, so this is applied once at config load and the depth
    bounds drive everything downstream.

    Conventions (matched to the builder geometry — ``QuadNode.compute_child_bboxes``
    splits an axis into ``size // 2`` and ``size - size // 2``):

    * ``max_depth`` (fine cap): the deepest depth before the next split would drop
      EITHER axis below ``min_patch_size``. This is exactly the builder's forced-leaf
      rule, i.e. ``max_reachable_depth`` with no extra depth cap. It binds on the
      SMALLER axis (the first to underflow).
    * ``min_depth`` (coarse floor): the shallowest depth at which every cell has both
      axes <= ``max_patch_size``. It binds on the LARGER axis (the last to fit),
      tracked by repeated ceil-halving (the larger child is ``size - size // 2``).

    The iterative form mirrors the builder's actual splits, so it stays exact on
    non-square / non-power-of-two grids (a ``log2`` closed form would not) and the
    ``depth < max_depth`` guard keeps ``min_depth <= max_depth`` for any inputs.

    Args:
        H, W: Grid dimensions in pixels.
        min_patch_size: Smallest allowed leaf patch (per axis), in pixels.
        max_patch_size: Largest allowed leaf patch (per axis), in pixels. A cell whose
            larger axis exceeds this is force-subdivided.

    Returns:
        ``(min_depth, max_depth)`` integer quadtree depth bounds with
        ``0 <= min_depth <= max_depth``.

    Raises:
        ValueError: if either size is < 1, or ``min_patch_size > max_patch_size``.
    """
    if min_patch_size is None or max_patch_size is None:
        raise SystemExit(
            "Config must define min and max patch size (it replaces min_depth/max_depth). "
        )
    if min_patch_size < 1 or max_patch_size < 1:
        raise ValueError(
            f"patch sizes must be >= 1, got min_patch_size={min_patch_size}, "
            f"max_patch_size={max_patch_size}"
        )
    if min_patch_size > max_patch_size:
        raise ValueError(
            f"min_patch_size ({min_patch_size}) must be <= max_patch_size ({max_patch_size})"
        )

    # Fine cap: deepest depth before the next split underflows min_patch_size
    # (binds on the smaller axis). Identical to the builder's forced-leaf rule.
    max_depth = max_reachable_depth(H, W, min_patch_size, 64)

    # Coarse floor: force-subdivide while the largest cell dimension exceeds
    # max_patch_size. The larger child of an axis is ``size - size // 2`` (ceil half).
    h, w, depth = H, W, 0
    while max(h, w) > max_patch_size and depth < max_depth:
        h -= h // 2
        w -= w // 2
        depth += 1
    min_depth = depth

    # Guard: the learned/oracle path subdivides on integer depth alone (the
    # per-cell min_patch_size safety check was removed). That is exact only when
    # the grid halves evenly down to max_depth, so every depth-max_depth cell is
    # uniform and >= min_patch_size. Otherwise depth-only subdivision would create
    # sub-min_patch_size cells. Fail loudly here rather than silently.
    if H % (2 ** max_depth) != 0 or W % (2 ** max_depth) != 0:
        raise SystemExit(
            f"Grid {H}x{W} does not halve evenly to depth {max_depth} "
            f"(H,W must be divisible by 2**max_depth = {2 ** max_depth}). "
            f"Depth-based subdivision would produce cells smaller than "
            f"min_patch_size={min_patch_size}. Use a power-of-two-compatible grid "
            f"or patch sizes."
        )
    
    return min_depth, max_depth

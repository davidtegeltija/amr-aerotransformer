"""
========================================================================
Score-guided quadtree builder driven by a RefinementNet score map.
========================================================================

Replaces the threshold-based OR-logic in ``adaptive_mesh.py`` with a
Gumbel-Softmax subdivision decision sampled from a pre-computed per-pixel
importance map. The builder returns:

    leaves:  flat list of leaf QuadNodes (same structure as the threshold-based
             builder so downstream tokenization can reuse QuadNode.features).
    soft_N:  0-dim torch.Tensor carrying gradients back into the score tensor.
             Equal to sum over all Gumbel decisions of soft_prob_subdivide —
             a differentiable approximation of the fractional token count
             above the min_depth floor.

Design conventions (matched to ``adaptive_mesh.py`` / ``quadtree.py``):
    * bbox is row/col: ``(r0, c0, r1, c1)`` with ``r0, c0`` inclusive and
      ``r1, c1`` exclusive.
    * Stop conditions mirror the threshold-based builder:
        - ``cell_h // 2 < min_cell_size`` OR ``cell_w // 2 < min_cell_size`` -> stop
        - ``depth >= max_depth`` -> stop
        - ``depth < min_depth`` -> force subdivide, no Gumbel draw
    * Per-channel mean features are computed inline (no cross-module import
      of the threshold builder's helpers — keeps coupling loose per the plan).

Gradient contract:
    The caller passes ``score_tensor`` as a CPU tensor with gradients still
    connected to the CNN scorer (i.e. **not** detached). This function
    slices into it to build the Gumbel logits, so ``soft_N.backward()``
    reaches the CNN. The companion ``score_map`` numpy array is only used
    for the hard decision and never participates in autograd.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.amr.quadtree import QuadNode, collect_leaves


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_learned_adaptive_mesh(
    data: np.ndarray,
    score_map: np.ndarray,
    score_tensor: torch.Tensor,
    max_depth: int = 5,
    min_depth: int = 2,
    min_cell_size: int = 4,
    training: bool = False,
    tau: float = 1.0,
) -> Tuple[List[QuadNode], torch.Tensor]:
    """
    Build a quadtree by Gumbel-Softmax sampling from a score map.

    Parameters
    ----------
    data : np.ndarray, shape (H, W, C)
        Raw geometry grid (used only for ``node.features`` — per-channel mean).
    score_map : np.ndarray, shape (H, W)
        Detached per-pixel importance scores in (0, 1). Drives the hard
        subdivide/stop decision. Not used for gradient.
    score_tensor : torch.Tensor, shape (H, W)
        CPU tensor carrying the *same* values as ``score_map`` but with
        gradient flow intact back to the CNN scorer. The caller must **not**
        detach this tensor — otherwise L_budget will have zero gradient.
    max_depth : int, default 5
        Upper depth cap. Cells at this depth are never subdivided.
    min_depth : int, default 2
        Lower depth floor. Cells above ``min_depth`` always subdivide
        (no Gumbel draw — matches the threshold-based ``min_depth`` semantics).
    min_cell_size : int, default 4
        A cell is forced to be a leaf once the next split would drop either
        axis below this size.
    training : bool, default False
        If True, sample subdivide decisions via Gumbel-Softmax straight-through.
        If False, use a deterministic 0.5 threshold on the max-pooled score.
    tau : float, default 1.0
        Gumbel-Softmax temperature.

    Returns
    -------
    leaves : list[QuadNode]
        Flat list of leaf nodes, each with ``features`` populated.
    soft_N : torch.Tensor
        0-dim differentiable scalar equal to the sum of Gumbel soft-probs of
        "subdivide" over every non-forced decision in the tree. Serves as a
        differentiable surrogate for the fractional token count above the
        min_depth floor. Stays exactly ``0.0`` if every decision was forced
        (for example if the scorer drives every cell below threshold and the
        tree stops at min_depth on all branches) — callers should clamp /
        add an L_budget floor in Phase 7 rather than this builder.
    """
    if data.ndim != 3:
        raise ValueError(f"Expected data with shape (H, W, C), got {data.shape}")
    if score_map.ndim != 2:
        raise ValueError(f"Expected score_map with shape (H, W), got {score_map.shape}")
    if score_tensor.ndim != 2:
        raise ValueError(f"Expected score_tensor with shape (H, W), got {score_tensor.shape}")

    H, W, _C = data.shape
    if score_map.shape != (H, W) or tuple(score_tensor.shape) != (H, W):
        raise ValueError(
            f"score_map / score_tensor shape must equal data H, W = {(H, W)}; "
            f"got {score_map.shape}, {tuple(score_tensor.shape)}"
        )

    root = QuadNode(bbox=(0, 0, H, W), depth=0)
    soft_N_accum: List[torch.Tensor] = [torch.tensor(0.0, dtype=torch.float32)]

    _build_node_score(
        data=data,
        node=root,
        score_tensor=score_tensor,
        max_depth=max_depth,
        min_depth=min_depth,
        min_cell_size=min_cell_size,
        training=training,
        tau=tau,
        soft_N_accum=soft_N_accum,
    )

    return collect_leaves(root), soft_N_accum[0]


# ---------------------------------------------------------------------------
# Core recursive builder
# ---------------------------------------------------------------------------

def _build_node_score(
    data: np.ndarray,
    node: QuadNode,
    score_tensor: torch.Tensor,
    max_depth: int,
    min_depth: int,
    min_cell_size: int,
    training: bool,
    tau: float,
    soft_N_accum: List[torch.Tensor],
) -> None:
    """Populate ``node`` (features + children) in-place by score-guided recursion."""
    r0, c0, r1, c1 = node.bbox
    region = data[r0:r1, c0:c1, :]

    if region.size == 0:
        node.features = np.zeros(data.shape[2], dtype=data.dtype)
        node.is_leaf = True
        return

    # Per-channel mean features — same convention as adaptive_mesh.py.
    node.features = region.mean(axis=(0, 1))
    node.metrics = {}

    cell_h = node.height
    cell_w = node.width

    hard_decision, soft_prob = _should_subdivide_score(
        score_tensor=score_tensor,
        r0=r0, c0=c0, r1=r1, c1=c1,
        depth=node.depth,
        min_depth=min_depth,
        max_depth=max_depth,
        min_cell_size=min_cell_size,
        cell_h=cell_h, cell_w=cell_w,
        training=training, tau=tau,
    )

    if soft_prob is not None:
        soft_N_accum[0] = soft_N_accum[0] + soft_prob

    if not hard_decision:
        node.is_leaf = True
        return

    for child in node.subdivide(depth=node.depth + 1):
        _build_node_score(
            data=data,
            node=child,
            score_tensor=score_tensor,
            max_depth=max_depth,
            min_depth=min_depth,
            min_cell_size=min_cell_size,
            training=training,
            tau=tau,
            soft_N_accum=soft_N_accum,
        )


# ---------------------------------------------------------------------------
# Subdivision decision
# ---------------------------------------------------------------------------

def _should_subdivide_score(
    score_tensor: torch.Tensor,
    r0: int, c0: int, r1: int, c1: int,
    depth: int,
    min_depth: int,
    max_depth: int,
    min_cell_size: int,
    cell_h: int,
    cell_w: int,
    training: bool,
    tau: float,
) -> Tuple[bool, Optional[torch.Tensor]]:
    """
    Decide whether the cell at ``(r0, c0, r1, c1)`` should subdivide.

    Returns
    -------
    hard_decision : bool
        True -> subdivide this cell.
    soft_prob_subdivide : torch.Tensor or None
        The "subdivide" component of the Gumbel-Softmax sample (keeps
        gradient to ``score_tensor``). None when no Gumbel draw was made
        (forced stop or forced subdivide).
    """
    # Forced stop: any further split would violate min_cell_size in *either* axis.
    if cell_h // 2 < min_cell_size or cell_w // 2 < min_cell_size:
        return False, None
    if depth >= max_depth:
        return False, None

    # Forced subdivide below min_depth — no Gumbel draw (no gradient needed).
    if depth < min_depth:
        return True, None

    # Max-pool the score over the cell region. Slicing preserves grad flow.
    importance = score_tensor[r0:r1, c0:c1].max()
    # Clamp for numerical safety in log(). log(0) and log(1 - 1) would nan.
    importance = importance.clamp(min=1e-6, max=1.0 - 1e-6)

    if training:
        logits = torch.stack([
            torch.log(1.0 - importance),
            torch.log(importance),
        ], dim=-1)
        soft = F.gumbel_softmax(logits, tau=tau, hard=True)
        hard_decision = bool(soft[1].item() > 0.5)
        return hard_decision, soft[1]
    else:
        # Deterministic threshold at 0.5 when not training (no grad needed).
        return (importance.item() > 0.5), None




# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    torch.manual_seed(0)

    # ---- Test A: shape / no-training path ---------------------------------
    data = rng.standard_normal((256, 128, 3)).astype(np.float32)
    score_map_np = rng.random((256, 128)).astype(np.float32)
    score_t = torch.from_numpy(score_map_np).clone()

    leaves, soft_N = build_learned_adaptive_mesh(
        data, score_map_np, score_t,
        max_depth=5, min_depth=2, min_cell_size=4, training=False,
    )
    assert 1 <= len(leaves) <= 1024, f"leaves out of range: {len(leaves)}"
    assert isinstance(soft_N, torch.Tensor) and soft_N.ndim == 0, \
        f"soft_N must be 0-dim tensor, got {soft_N}"
    print(f"Test A OK: {len(leaves)} leaves, soft_N={soft_N.item():.2f}")

    # ---- Test B: focused score -> refinement concentrates in circle ------
    H, W = 256, 128
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    cy, cx, radius = 64, 64, 20
    inside = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
    score_focus = np.where(inside, 0.99, 0.01).astype(np.float32)
    score_focus_t = torch.from_numpy(score_focus).clone()
    data_B = rng.standard_normal((H, W, 3)).astype(np.float32)

    leaves_B, _ = build_learned_adaptive_mesh(
        data_B, score_focus, score_focus_t,
        max_depth=5, min_depth=2, min_cell_size=4, training=False,
    )

    # Leaves whose bbox is entirely inside the circle should be at max_depth.
    # Leaves in regions no ancestor cell ever touches the circle should stay
    # at min_depth. (A leaf that is "just outside" the circle can legitimately
    # be at depths 3-5: its depth-2 ancestor straddled the circle edge and
    # subdivided, producing a fully-outside child at depth 3 which *also*
    # sees max-score 0.01 and stops. So we test only the far-outside region.)
    # The circle lives in rows 44..84, cols 44..84; anything beyond that is
    # covered by depth-2 cells (64x32) that are themselves fully outside.
    inside_depths: List[int] = []
    far_outside_depths: List[int] = []
    far_outside_r_min = 128  # well below the circle
    for lf in leaves_B:
        r0, c0, r1, c1 = lf.bbox
        region_mask = inside[r0:r1, c0:c1]
        if region_mask.all():
            inside_depths.append(lf.depth)
        if r0 >= far_outside_r_min:
            far_outside_depths.append(lf.depth)

    assert inside_depths, "no leaves entirely inside the circle"
    assert far_outside_depths, "no leaves found in the far-outside region"
    assert all(d == 5 for d in inside_depths), \
        f"expected all inside leaves at max_depth=5, got {sorted(set(inside_depths))}"
    assert all(d == 2 for d in far_outside_depths), \
        f"expected all far-outside leaves at min_depth=2, got {sorted(set(far_outside_depths))}"
    print(
        f"Test B OK: inside leaves all at depth {set(inside_depths).pop()}, "
        f"far-outside leaves all at depth {set(far_outside_depths).pop()}"
    )

    # ---- Test C: gradient flow through soft_N -----------------------------
    score_t = torch.rand(256, 128, requires_grad=True)
    score_np = score_t.detach().numpy()
    data_C = rng.standard_normal((256, 128, 3)).astype(np.float32)

    leaves_C, soft_N_C = build_learned_adaptive_mesh(
        data_C, score_np, score_t,
        training=True, tau=1.0,
    )
    soft_N_C.backward()
    assert score_t.grad is not None and (score_t.grad != 0).any(), \
        "soft_N.backward() produced no gradient on score_tensor"
    nonzero_pixels = int((score_t.grad != 0).sum().item())
    print(f"Test C OK: grad nonzero on {nonzero_pixels} pixels, "
          f"soft_N={soft_N_C.item():.2f}, leaves={len(leaves_C)}")

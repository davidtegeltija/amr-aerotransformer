from typing import Any, Dict, List, Optional

import numpy as np
import torch

from src.amr.adaptive_mesh import build_adaptive_mesh
from src.amr.learned_adaptive_mesh import build_depth_guided_mesh
from src.amr.oracle_depth import compute_oracle_depth
from src.amr.quadtree import QuadNode, nodes_to_token_array
from src.amr.refinement_criteria import RefinementCriteria
from src.models.reconstruction import basis_size, cell_basis


# ---------------------------------------------------------------------------
# Per-leaf reductions of the dense target
# ---------------------------------------------------------------------------

def _per_token_targets(target: np.ndarray, leaves: List[QuadNode]) -> np.ndarray:
    """Average the dense target over each leaf's bbox -> [N, output_channels].

    Row order matches ``leaves`` (and therefore the packed-token order), so the
    result lines up with the transformer's per-token predictions.
    """
    output_channels = target.shape[-1]
    token_target = np.zeros((len(leaves), output_channels), dtype=np.float32)
    for i, node in enumerate(leaves):
        token_target[i] = target[node.r0:node.r1, node.c0:node.c1].mean(axis=(0, 1))
    return token_target


def _affine_leaf_stats(target: np.ndarray, leaves: List[QuadNode], H: int, W: int,
                       order: int) -> Dict[str, np.ndarray]:
    """Per-leaf sufficient statistics for the closed-form quadratic dense NMSE.

    The head paints leaf ``i`` with ``v + gx*dx + gy*dy + gxx*dxx + gxy*dxy + gyy*dyy``
    over its pixels. The six terms are mutually orthogonal over a cell (``cell_basis``
    centres ``dxx``/``dyy`` to make that true), so every cross term drops and the
    cell's squared error collapses to

        SSE = num_pixels*(v - mean_target)^2
              + sum_xx*gx^2     - 2*gx*sum_target_dx
              + sum_yy*gy^2     - 2*gy*sum_target_dy
              + sum_xxxx*gxx^2  - 2*gxx*sum_target_dxx
              + sum_xyxy*gxy^2  - 2*gxy*sum_target_dxy
              + sum_yyyy*gyy^2  - 2*gyy*sum_target_dyy
              + sum_sq_resid

    in which every term that touches the target is a constant of ``(mesh, target)``.
    Those constants are what this function returns, so ``affine_nmse_loss`` can
    evaluate the dense per-pixel loss without ever building the ``[B,H,W,C]`` grid
    (see src/training/loss.py:affine_nmse_loss). This is the affine version of the
    same expression with three more terms.

    The terms come from ``src/models/reconstruction.py:cell_basis``, which is also
    what paints the pixels, so the two paths cannot disagree about what ``dxx``
    means. A term the cell is too small to resolve is an all-zero column, so its
    norm and its target product are both zero and it leaves the sum untouched --
    the same way ``sum_xx = 0`` handled a one-pixel-wide cell before.

    At order 1 the last three terms do not exist, so their statistics are neither
    computed nor returned and the expression stops after ``gy``.

    Accumulation is float64 and ``sum_sq_resid`` is centred on the cell mean, which
    keeps the subtraction out of float32.

    Args:
        target: ``[H, W, output_channels]`` dense ground truth for one sample.
        leaves: The sample's leaf ``QuadNode`` s, in packed-token order. They must
            tile the full grid.
        H: Grid height (rows).
        W: Grid width (columns).
        order: The head's ``affine_output`` order, 1 or 2.

    Returns:
        Dict of float32 arrays, rows aligned with ``leaves``:
        ``num_pixels`` [N] pixel count,
        ``sum_xx``/``sum_yy``/``sum_xxxx``/``sum_xyxy``/``sum_yyyy`` [N] term norms
        (cell-shape only),
        ``mean_target`` [N, C] cell mean,
        ``sum_target_dx``/``_dy``/``_dxx``/``_dxy``/``_dyy`` [N, C] target-term products,
        ``sum_sq_resid`` [N, C] within-cell sum of squares about the mean.
        At order 1 the six second-order arrays are present but have no rows.

    Raises:
        ValueError: if the leaves do not tile the grid exactly, which would leave
            pixels out of the loss instead of merely mis-weighting them.
    """
    N, C = len(leaves), target.shape[-1]
    n_taylor_terms = basis_size(order)
    second_order = order == 2
    # Rows of the second-order arrays. At order 1 those terms do not exist, so the
    # arrays are allocated with no rows and stay empty: the dict has the same keys
    # at both orders, and nothing is computed for a term the head does not have.
    N2 = N if second_order else 0

    num_pixels = np.empty(N, dtype=np.float32)
    sum_xx = np.empty(N, dtype=np.float32)
    sum_yy = np.empty(N, dtype=np.float32)
    sum_xxxx = np.empty(N2, dtype=np.float32)
    sum_xyxy = np.empty(N2, dtype=np.float32)
    sum_yyyy = np.empty(N2, dtype=np.float32)
    mean_target = np.empty((N, C), dtype=np.float32)
    sum_target_dx = np.empty((N, C), dtype=np.float32)
    sum_target_dy = np.empty((N, C), dtype=np.float32)
    sum_target_dxx = np.empty((N2, C), dtype=np.float32)
    sum_target_dxy = np.empty((N2, C), dtype=np.float32)
    sum_target_dyy = np.empty((N2, C), dtype=np.float32)
    sum_sq_resid = np.empty((N, C), dtype=np.float32)

    target = np.asarray(target, dtype=np.float64)
    # The basis terms depend only on the cell shape, and a quadtree has one shape
    # per depth, so a handful of entries covers every leaf of every sample.
    terms: Dict[tuple, tuple] = {}
    covered = 0

    for i, node in enumerate(leaves):
        r0, c0, r1, c1 = node.r0, node.c0, node.r1, node.c1
        h, w = r1 - r0, c1 - c0

        cached = terms.get((h, w))
        if cached is None:
            # cell_basis is the one definition of the terms and of the centring
            # that makes dx^2 / dy^2 orthogonal to the constant; take the columns
            # from it rather than rebuilding them here.
            basis = cell_basis(h, w, n_taylor_terms)
            cached = (basis.reshape(h, w, n_taylor_terms), (basis ** 2).sum(axis=0))
            terms[(h, w)] = cached
        cols, norms = cached
        dx, dy = cols[:, :, 1], cols[:, :, 2]

        t = target[r0:r1, c0:c1]                  # [h, w, C]
        mean = t.mean(axis=(0, 1))
        num_pixels[i] = h * w
        sum_xx[i], sum_yy[i] = norms[1], norms[2]
        mean_target[i] = mean
        sum_target_dx[i] = (t * dx[:, :, None]).sum(axis=(0, 1))
        sum_target_dy[i] = (t * dy[:, :, None]).sum(axis=(0, 1))
        sum_sq_resid[i] = ((t - mean) ** 2).sum(axis=(0, 1))
        covered += h * w

        if second_order:
            dxx, dxy, dyy = cols[:, :, 3], cols[:, :, 4], cols[:, :, 5]
            sum_xxxx[i], sum_xyxy[i], sum_yyyy[i] = norms[3], norms[4], norms[5]
            sum_target_dxx[i] = (t * dxx[:, :, None]).sum(axis=(0, 1))
            sum_target_dxy[i] = (t * dxy[:, :, None]).sum(axis=(0, 1))
            sum_target_dyy[i] = (t * dyy[:, :, None]).sum(axis=(0, 1))

    if covered != H * W:
        raise ValueError(
            f"leaves cover {covered} of {H * W} pixels; they must tile the grid"
        )

    return {
        "num_pixels": num_pixels,
        "sum_xx": sum_xx,
        "sum_yy": sum_yy,
        "sum_xxxx": sum_xxxx,
        "sum_xyxy": sum_xyxy,
        "sum_yyyy": sum_yyyy,
        "mean_target": mean_target,
        "sum_target_dx": sum_target_dx,
        "sum_target_dy": sum_target_dy,
        "sum_target_dxx": sum_target_dxx,
        "sum_target_dxy": sum_target_dxy,
        "sum_target_dyy": sum_target_dyy,
        "sum_sq_resid": sum_sq_resid,
    }


def _stack_affine_stats(per_sample: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
    """Concatenate per-sample ``_affine_leaf_stats`` along the packed token axis.

    Row order matches the packed tokens, so ``affine_nmse_loss`` can index the
    model's per-token predictions with these directly.
    """
    return {key: torch.from_numpy(np.concatenate([s[key] for s in per_sample], axis=0))
            for key in per_sample[0]}


class DeterministicCollateFn:
    """
    Picklable collate callable for DataLoader with num_workers > 0.
 
    For each sample in the batch, builds the physics-based adaptive mesh on the
    CPU worker, then concatenates all token sequences into a single packed
    tensor (the transformer pads to the per-batch max internally, around its
    encoder only).

    Must be a top-level class (not a closure) to be picklable by
    Python's multiprocessing.

    Batch dict keys:
        packed_tokens     : [total_N, C+3]             concatenated tokenized inputs
        packed_targets    : [total_N, output_channels] per-token averaged ground truth
        tokens_per_sample : List[int]                  token count per sample
        affine_stats      : Dict[str, Tensor]          per-leaf stats for the affine
                                                       per-pixel loss (only when affine_output)

    Args:
        refinement_criteria: Thresholds driving the physics-based subdivision.
        min_depth: Depth floor; cells shallower than this always subdivide.
        max_depth: Hard depth cap; cells at this depth never subdivide.
        affine_input: 1 if each token also carries its cell's (gx, gy), which
            widens it to token_feature_width(C) + 3; 0 for the mean alone.
        affine_output: The model's output order (0, 1 or 2). Non-zero means the
            affine head, so the collate also builds the per-leaf statistics the
            closed-form affine loss needs at that order (see ``_affine_leaf_stats``).
            At 0 they are neither computed nor cached, since the constant head is
            scored on ``packed_targets``.
    """

    def __init__(self, refinement_criteria: RefinementCriteria, min_depth: int, max_depth: int,
                 affine_input: int = 0, affine_output: int = 0):
        self.refinement_criteria = refinement_criteria
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.affine_input = affine_input
        self.affine_output = affine_output
        # Per-sample cache keyed by dataset index. The quadtree build and the
        # target reductions are deterministic, so the first epoch fills this and
        # every later epoch is a lookup. Only persists with num_workers=0 (workers
        # get their own copy that is discarded after each batch). The QuadNode
        # leaves are deliberately NOT kept: everything downstream needs is reduced
        # to flat arrays here, and the objects cost ~10x more memory than they do.
        self._cache: Dict[int, tuple] = {}

    def __call__(self, samples: List[Dict]) -> Dict:
        all_tokens = []
        all_targets = []
        tokens_per_sample = []
        all_stats = []

        for s in samples:
            input = s["input"]   # [H, W, C]
            target = s["target"]  # [H, W, output_channels]

            cached = self._cache.get(s["index"])
            if cached is None:
                H, W, C = np.asarray(input).shape
                leaves = build_adaptive_mesh(
                    input,
                    max_depth=self.max_depth,
                    min_depth=self.min_depth,
                    refinement_criteria=self.refinement_criteria,
                )
                token_array = nodes_to_token_array(leaves, H, W, C, self.affine_input)
                token_target = _per_token_targets(target, leaves)
                stats = _affine_leaf_stats(target, leaves, H, W, self.affine_output) if self.affine_output else None
                cached = (token_array, len(leaves), token_target, stats)
                self._cache[s["index"]] = cached

            token_array, N, token_target, stats = cached
            all_tokens.append(torch.from_numpy(token_array))
            all_targets.append(torch.from_numpy(token_target))
            tokens_per_sample.append(N)
            if self.affine_output:
                all_stats.append(stats)

        batch = {
            "packed_tokens": torch.cat(all_tokens,  dim=0),
            "packed_targets": torch.cat(all_targets, dim=0),
            "tokens_per_sample": tokens_per_sample,
        }
        if self.affine_output:
            batch["affine_stats"] = _stack_affine_stats(all_stats)
        return batch


class ScorerCollateFn:
    """Collate for supervised scorer training (variance-oracle target).

    Stacks the input/target grids and, per sample, computes the oracle depth
    map from the dense target with a single GLOBAL tolerance (calibrated once on
    the train split — see ``calibrate_global_tolerance``). The oracle is a small
    deterministic integer array, so computing it in the worker mirrors how
    ``DeterministicCollateFn`` precomputes per-token targets.

    Batch dict keys:
        grids        : [B, H, W, C]                  input geometry
        targets      : [B, H, W, output_channels]    dense ground truth
        oracle_depth : [B, 1, H, W] long             per-pixel oracle depth
    """

    def __init__(
        self,
        tol: float,
        min_depth: int,
        max_depth: int,
        channel_scale: Optional[np.ndarray] = None,
    ):
        self.tol = tol
        self.min_depth = min_depth
        self.max_depth = max_depth
        # Optional fixed per-channel scale shared across samples; None -> each
        # sample is normalised by its own per-channel std (the default).
        self.channel_scale = channel_scale
        # Per-sample oracle cache keyed by dataset index. The oracle is a
        # deterministic function of (target, tol, depths), so the first epoch
        # fills this and every later epoch is a lookup. Only persists with
        # num_workers=0 (workers get their own copy, discarded after each batch).
        self._cache: Dict[int, np.ndarray] = {}

    def __call__(self, samples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        grids = torch.stack([torch.from_numpy(np.asarray(s["input"], dtype=np.float32)) for s in samples])
        targets = torch.stack([torch.from_numpy(np.asarray(s["target"], dtype=np.float32)) for s in samples])

        oracle_maps = []
        for s in samples:
            oracle = self._cache.get(s["index"])
            if oracle is None:
                target = np.asarray(s["target"], dtype=np.float32)
                oracle = compute_oracle_depth(
                    target,
                    tol=self.tol,
                    min_depth=self.min_depth,
                    max_depth=self.max_depth,
                    channel_scale=self.channel_scale,
                )
                self._cache[s["index"]] = oracle
            oracle_maps.append(torch.from_numpy(oracle))

        oracle_depth = torch.stack(oracle_maps).unsqueeze(1).long()   # [B, 1, H, W]
        return {
            "grids": grids,
            "targets": targets,
            "oracle_depth": oracle_depth,
        }


class LearnedCollateFn:
    """Builds the adaptive mesh from a frozen scorer, in the DataLoader worker.

    The learned-mesh twin of ``DeterministicCollateFn``: it produces the exact
    same packed batch dict, so the transformer and the training loop are shared.
    The only difference is where the leaves come from — here a frozen
    ``RefinementNet`` scores the grid, and ``build_depth_guided_mesh`` turns that
    predicted depth map into leaves.

    The scorer is frozen, so the mesh (token array + per-token targets) is a
    deterministic function of the sample: the per-index cache fills on the first
    epoch and every later epoch is a lookup (persists only with num_workers=0;
    workers get their own copy, discarded after each batch). The scorer runs on
    CPU here — the whole point is to keep this work off the training step.

    Batch dict keys (identical to ``DeterministicCollateFn``):
        packed_tokens     : [total_N, C+3]             concatenated tokenized inputs
        packed_targets    : [total_N, output_channels] per-token averaged ground truth
        tokens_per_sample : List[int]                  token count per sample
        affine_stats      : Dict[str, Tensor]          per-leaf stats for the affine
                                                       per-pixel loss (only when affine_output)

    Args:
        scorer: Trained ``RefinementNet``; frozen here and used only to build meshes.
        min_depth: Depth floor; cells shallower than this always subdivide.
        max_depth: Hard depth cap; cells at this depth never subdivide.
        offset: Mesh budget offset passed to ``build_depth_guided_mesh``.
        affine_input: 1 if each token also carries its cell's (gx, gy), which
            widens it to token_feature_width(C) + 3; 0 for the mean alone.
        affine_output: The model's output order (0, 1 or 2). Non-zero means the
            affine head, so the collate also builds the per-leaf statistics the
            closed-form affine loss needs at that order (see ``_affine_leaf_stats``).
            At 0 they are neither computed nor cached.
    """

    def __init__(self, scorer, min_depth: int, max_depth: int, offset: float = 0.0,
                 affine_input: int = 0, affine_output: int = 0):
        # Freeze the scorer: it only builds meshes here, it is never trained.
        self.scorer = scorer.eval().requires_grad_(False)
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.offset = offset
        self.affine_input = affine_input
        self.affine_output = affine_output
        # See DeterministicCollateFn._cache: same contract, and the QuadNode leaves
        # are likewise reduced to flat arrays rather than cached as objects.
        self._cache: Dict[int, tuple] = {}

    @torch.no_grad()
    def __call__(self, samples: List[Dict[str, Any]]) -> Dict:
        # Score every uncached sample in one batched CPU forward, then build the
        # per-sample mesh from its predicted depth map.
        todo = [s for s in samples if s["index"] not in self._cache]
        if todo:
            grids = torch.stack(
                [torch.from_numpy(np.asarray(s["input"], dtype=np.float32)) for s in todo])
            depth_maps = self.scorer(grids).squeeze(1).numpy()   # [len(todo), H, W]
            for s, depth_map in zip(todo, depth_maps):
                input = np.asarray(s["input"], dtype=np.float32)   # [H, W, C]
                H, W, C = input.shape
                leaves = build_depth_guided_mesh(
                    data=input,
                    depth_map=depth_map,
                    max_depth=self.max_depth,
                    min_depth=self.min_depth,
                    offset=self.offset,
                )
                target = np.asarray(s["target"], dtype=np.float32)
                token_array = nodes_to_token_array(leaves, H, W, C, self.affine_input)
                token_target = _per_token_targets(target, leaves)
                stats = (_affine_leaf_stats(target, leaves, H, W, self.affine_output)
                         if self.affine_output else None)
                self._cache[s["index"]] = (token_array, len(leaves), token_target, stats)

        all_tokens, all_targets, tokens_per_sample, all_stats = [], [], [], []
        for s in samples:
            token_array, N, token_target, stats = self._cache[s["index"]]
            all_tokens.append(torch.from_numpy(token_array))
            all_targets.append(torch.from_numpy(token_target))
            tokens_per_sample.append(N)
            if self.affine_output:
                all_stats.append(stats)

        batch = {
            "packed_tokens": torch.cat(all_tokens, dim=0),
            "packed_targets": torch.cat(all_targets, dim=0),
            "tokens_per_sample": tokens_per_sample,
        }
        if self.affine_output:
            batch["affine_stats"] = _stack_affine_stats(all_stats)
        return batch


class VitCollateFn:
    """Stacks per-sample input/target grids into a batch. No tokenization.

    Used by the ViT baseline, which consumes dense grids directly.
    """

    def __call__(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        grids = torch.stack([torch.from_numpy(np.asarray(s["input"], dtype=np.float32)) for s in samples])    # [B, H, W, C]
        targets = torch.stack([torch.from_numpy(np.asarray(s["target"], dtype=np.float32)) for s in samples]) # [B, H, W, output_channels]

        return {
            "grids": grids,
            "targets": targets,
        }
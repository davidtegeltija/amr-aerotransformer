from typing import Any, Dict, List, Optional

import numpy as np
import torch

from src.amr.learned_adaptive_mesh import build_depth_guided_mesh
from src.amr.oracle_depth import compute_oracle_depth
from src.amr.quadtree import QuadNode
from src.amr.quadtree_tokenizer import QuadtreeTokenizer, nodes_to_token_array


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


class DeterministicCollateFn:
    """
    Picklable collate callable for DataLoader with num_workers > 0.
 
    For each sample in the batch, tokenizes the input grid on the CPU worker,
    then concatenates all token sequences into a single packed tensor
    (sequence packing strategy - no padding, no wasted compute).
 
    Must be a top-level class (not a closure) to be picklable by
    Python's multiprocessing.
 
    Batch dict keys:
        packed_tokens     : [total_N, C+3]             concatenated tokenized inputs
        packed_targets    : [total_N, output_channels] per-token averaged ground truth
        tokens_per_sample : List[int]                  token count per sample
        token_lists       : List[List[QuadNode]]       per-sample leaves (affine/dense loss)
        targets           : [B, H, W, output_channels] dense ground truth (affine/dense loss)
    """

    def __init__(self, tokenizer: QuadtreeTokenizer):
        self.tokenizer = tokenizer
        # Per-sample cache keyed by dataset index. The quadtree build and target
        # averaging are deterministic, so the first epoch fills this and every
        # later epoch is a lookup. Only persists with num_workers=0 (workers get
        # their own copy that is discarded after each batch).
        self._cache: Dict[int, tuple] = {}

    def __call__(self, samples: List[Dict]) -> Dict:
        all_tokens = []
        all_targets = []
        tokens_per_sample = []
        token_lists = []
        dense_targets = []

        for s in samples:
            input = s["input"]   # [H, W, C]
            target = s["target"]  # [H, W, output_channels]

            cached = self._cache.get(s["index"])
            if cached is None:
                token_array, leaves = self.tokenizer.tokenize(input)
                token_target = _per_token_targets(target, leaves)
                self._cache[s["index"]] = (token_array, leaves, token_target)
            else:
                token_array, leaves, token_target = cached

            N = len(leaves)
            all_tokens.append(torch.from_numpy(token_array))
            all_targets.append(torch.from_numpy(token_target))
            tokens_per_sample.append(N)
            token_lists.append(leaves)
            dense_targets.append(torch.from_numpy(np.asarray(target, dtype=np.float32)))

        return {
            "packed_tokens": torch.cat(all_tokens,  dim=0),
            "packed_targets": torch.cat(all_targets, dim=0),
            "tokens_per_sample": tokens_per_sample,
            # Dense leaves + targets enable the affine per-pixel loss; the legacy
            # constant path keeps using packed_targets above.
            "token_lists": token_lists,
            "targets": torch.stack(dense_targets, dim=0),
        }


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
        token_lists       : List[List[QuadNode]]       per-sample leaves (affine/dense loss)
        targets           : [B, H, W, output_channels] dense ground truth (affine/dense loss)
    """

    def __init__(self, scorer, min_depth: int, max_depth: int, offset: float = 0.0):
        # Freeze the scorer: it only builds meshes here, it is never trained.
        self.scorer = scorer.eval().requires_grad_(False)
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.offset = offset
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
                token_array = nodes_to_token_array(leaves, H, W, C)
                token_target = _per_token_targets(
                    np.asarray(s["target"], dtype=np.float32), leaves)
                self._cache[s["index"]] = (token_array, leaves, token_target)

        all_tokens, all_targets, tokens_per_sample, token_lists, dense_targets = [], [], [], [], []
        for s in samples:
            token_array, leaves, token_target = self._cache[s["index"]]
            all_tokens.append(torch.from_numpy(token_array))
            all_targets.append(torch.from_numpy(token_target))
            tokens_per_sample.append(len(leaves))
            token_lists.append(leaves)
            dense_targets.append(torch.from_numpy(np.asarray(s["target"], dtype=np.float32)))

        return {
            "packed_tokens": torch.cat(all_tokens, dim=0),
            "packed_targets": torch.cat(all_targets, dim=0),
            "tokens_per_sample": tokens_per_sample,
            "token_lists": token_lists,
            "targets": torch.stack(dense_targets, dim=0),
        }


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
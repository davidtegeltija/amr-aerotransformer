"""
========================================================================
Adaptive-mesh aerodynamic flow predictor

Full pipeline:
grid -> AMR tokens build from quadtree -> transformer
========================================================================

The model turns a dense geometry grid into adaptive-mesh tokens and predicts a
per-token flow field with a transformer. Where the mesh comes from depends on
``refinement_mode``:

--- "deterministic" -----------------------------------------------------
The mesh is built outside the model, in the DataLoader worker, by a physics-
based AMR criterion. The model forward is just the transformer.

    grid_input [B, H, W, C]
      ↓ QuadtreeTokenizer + build_adaptive_mesh   (in DeterministicCollateFn, CPU worker)
    packed_tokens [total_N, C+3]                  (already batched/packed)
      ↓ AeroTransformer                           (model.forward(packed_tokens, tokens_per_sample))
    token predictions [total_N, output_channels]

--- "learned" -----------------------------------------------------------
A RefinementNet CNN scores the grid; a deterministic running-depth quadtree
turns that score into the mesh, inline in the forward pass.

    grid_input [B, H, W, C]
      ↓ RefinementNet (CNN scorer)
    depth_map d_pred [B, 1, H, W]                 (predicted target depth, not a sign logit)
      ↓ build_depth_guided_mesh                   (running-depth quadtree: split iff max(d_pred)>d+offset)
    leaves
      ↓ inline token packing
    packed_tokens [total_N, C+3]
      ↓ AeroTransformer                           (model.forward(grids))
    token predictions [total_N, output_channels]

In learned mode the scorer is trained separately by supervised regression to
the variance oracle (see src/amr/oracle_depth.py and train_scorer_supervised);
this forward is used for inference and the end-to-end sanity metric. There is no
Gumbel sampling, no soft token count, and no straight-through leaf weights — the
mesh build is deterministic and detached from the autograd graph.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

import torch
import torch.nn as nn

from src.amr.refinement_criteria import RefinementCriteria
from src.amr.learned_adaptive_mesh import build_depth_guided_mesh
from src.amr.quadtree_tokenizer import nodes_to_token_array
from src.model.refinement_net import RefinementNet
from src.model.transformer import AeroTransformer


class AdaptiveMeshAeroModel(nn.Module):
    """
    End-to-end steady aerodynamic flow-field predictor.

    A RefinementNet CNN emits a per-pixel importance map on GPU; a per-sample
    score-guided quadtree (CPU, Gumbel-Softmax straight-through) turns that
    map into leaf QuadNodes, which are packed inline into the same [N, C+3]
    token layout the transformer expects.

    Parameters
    ----------
    input_channels      : number of physical input channels
    output_channels     : number of predicted quantities (e.g. 3 for u, v, p)
    d_model             : transformer hidden dimension
    n_layers            : number of transformer encoder layers
    n_heads             : number of attention heads
    d_ff                : feedforward dimension
    dropout             : dropout probability
    min_depth           : quadtree minimum depth
    max_depth           : quadtree maximum depth
    refinement_mode     : "learned" or "deterministic"
    refinement_criteria : optional custom RefinementCriterion (refinement_mode == 'deterministic')

    Configs express these bounds as ``min_patch_size`` / ``max_patch_size`` (pixels),
    converted to the integer ``min_depth`` / ``max_depth`` at config load by
    ``patch_sizes_to_depth_bounds``; this constructor takes the integer depths.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int = 3,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
        min_depth: int = 2,
        max_depth: int = 6,
        refinement_mode: Literal["learned", "deterministic"] = "deterministic",
        refinement_criteria: Optional[RefinementCriteria] = None,
        affine_output: bool = False,
        continuous_output: bool = False,
    ):
        super().__init__()
        if refinement_mode not in ("learned", "deterministic"):
            raise ValueError(
                f"refinement_mode must be 'learned' or 'deterministic', got {refinement_mode!r}"
            )

        if refinement_mode == "deterministic" and refinement_criteria is None:
            raise ValueError(
                "refinement_mode='deterministic' requires a non-None refinement_criteria."
            )

        # affine_output and the implicit-head continuous_output are mutually
        # exclusive output representations.
        if affine_output and continuous_output:
            raise ValueError(
                "affine_output and continuous_output are mutually exclusive. Enable at most one."
            )

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.refinement_mode = refinement_mode
        self.refinement_criteria = refinement_criteria
        self.affine_output = affine_output

        # --- CNN scorer (drives score-guided subdivision) ---
        # Only instantiated in learned mode; deterministic mode has no scorer.
        if refinement_mode == "learned":
            self.scorer = RefinementNet(input_channels=input_channels)
            # Global budget offset added to the running depth in the subdivision
            # test (positive -> coarser mesh, negative -> finer). Solve per sample
            # at eval if an exact token count is required; default 0.
            self.offset = 0.0
            # Per-sample mesh cache keyed by dataset index. During learned-mesh
            # training the scorer is frozen, so the mesh (token array + leaves) is
            # constant across epochs: the first epoch fills this, every later epoch
            # (and validation) is a lookup that skips the scorer and the CPU tree
            # build entirely. Only used when forward() is given sample indices.
            self._mesh_cache: Dict[int, tuple] = {}
        else:
            self.scorer = None

        # --- Transformer solver ---
        token_dim = input_channels + 3  # C + (x_center, y_center, size)
        self.transformer = AeroTransformer(
            token_dim=token_dim,
            output_channels=output_channels,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            affine_output=affine_output,
        )

    # ------------------------------------------------------------------
    # Batched forward (training/eval)
    # ------------------------------------------------------------------

    def forward(self, *args, **kwargs):
        """
        Dispatch by refinement_mode:
          deterministic: forward(packed_tokens, tokens_per_sample) - tokens pre-built by DeterministicCollateFn
          learned:       forward(grids)-  grids [B, H, W, C], tokenization happens inside
        """
        if self.refinement_mode == "deterministic":
            return self._forward_deterministic(*args, **kwargs)
        else:
            return self._forward_learned(*args, **kwargs)

    def _forward_deterministic(self, packed_tokens, tokens_per_sample):
        """
        Tokens were already built in the collate function.
        Forward pass is just the transformer.

        Args:
            packed_tokens :    [total_N, C+3] concatenated tokens of all samples
            tokens_per_sample: List[int] per-sample token counts

        Returns:
            Dict with keys:
                token_preds:       [total_N, output_channels], or
                    [total_N, output_channels, 3] = (value, gx, gy) when
                    affine_output (decoded into a per-cell ramp by the dense loss).
                score_map:         None  (no scorer in this mode)
                soft_N:            None  (no budget loss in this mode)
                tokens_per_sample: List[int] (len B)
                token_lists:       None. In affine mode the dense loss reconstructs
                    from the collate's ``token_lists`` (DeterministicCollateFn),
                    so the leaves are not threaded through this forward.
        """
        preds = self.transformer(packed_tokens, tokens_per_sample)

        return {
            "token_preds": preds,
            "score_map": None,
            "soft_N": None,
            "tokens_per_sample": tokens_per_sample,
            "token_lists": None,
        }

    def _forward_learned(self, grids, indices=None):
        """
        Scorer → depth-guided tree → transformer in one forward pass.

        The scorer emits a predicted depth map ``d_pred``; the deterministic
        running-depth builder turns it into leaves (no Gumbel, no temperature).
        The mesh build is detached — the scorer is trained separately by
        supervised regression to the variance oracle, so no gradient needs to
        flow back through the discrete tree here.

        When ``indices`` is given (the dataset index of each sample) and the
        scorer is frozen, the per-sample mesh never changes across epochs, so the
        token array + leaves are cached in ``self._mesh_cache``: cached samples
        skip the scorer and the CPU tree build entirely. Pass ``indices=None`` to
        always recompute (e.g. while the scorer is still being trained).

        Args:
            grids: [B, H, W, C] float32 input geometry, channel-last.
            indices: Optional[List[int]] dataset index per sample, for mesh caching.

        Returns:
            Dict with keys:
                token_preds:       [total_N, output_channels] from the transformer,
                    or [total_N, output_channels, 3] = (value, gx, gy) when
                    affine_output.
                score_map:         [B, 1, H, W] predicted depth map d_pred, or
                    ``None`` for samples served entirely from the mesh cache.
                tokens_per_sample: List[int] (len B), tokens per sample
                token_lists:       List[List[QuadNode]] (len B)
        """
        B, H, W, C = grids.shape
        device = grids.device

        # keys[b] is the sample's dataset index, or None to force recompute.
        # A sample needs the scorer + tree build if it has no index or is a miss.
        keys = indices if indices is not None else [None] * B
        todo = [b for b in range(B) if keys[b] is None or keys[b] not in self._mesh_cache]

        # 1. Predicted depth map d_pred (GPU), only for the uncached samples. The
        #    scorer takes channel-last grids and permutes internally.
        score_map = None
        computed: Dict[int, tuple] = {}   # batch-position -> mesh, this call only
        if todo:
            score_map = self.scorer(grids[todo])              # [len(todo), 1, H, W]
            depth_np = score_map.squeeze(1).detach().cpu().numpy()
            grids_np = grids[todo].detach().cpu().numpy()
            # 2. Build trees deterministically (CPU, per-sample); cache index-keyed
            #    meshes. Detached: the discrete build carries no gradient by design.
            for j, b in enumerate(todo):
                leaves = build_depth_guided_mesh(
                    data=grids_np[j],
                    depth_map=depth_np[j],
                    max_depth=self.max_depth,
                    min_depth=self.min_depth,
                    offset=self.offset,
                )
                mesh = (torch.from_numpy(nodes_to_token_array(leaves, H, W, C)), leaves)
                computed[b] = mesh
                if keys[b] is not None:
                    self._mesh_cache[keys[b]] = mesh

        all_tokens: List[torch.Tensor] = []
        tokens_per_sample: List[int] = []
        token_lists: List[List] = []
        for b in range(B):
            token_array, leaves = computed[b] if b in computed else self._mesh_cache[keys[b]]
            all_tokens.append(token_array)
            tokens_per_sample.append(len(leaves))
            token_lists.append(leaves)

        packed = torch.cat(all_tokens, dim=0).to(device)
        preds = self.transformer(packed, tokens_per_sample)

        return {
            "token_preds": preds,
            "score_map": score_map,
            "tokens_per_sample": tokens_per_sample,
            "token_lists": token_lists,
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

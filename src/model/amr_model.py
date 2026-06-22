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
from src.model.refinement_net import RefinementNet
from src.model.transformer import AeroTransformer


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

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
    min_cell_size       : minimum cell size in pixels
    refinement_mode     : "learned" or "deterministic"
    refinement_criteria : optional custom RefinementCriterion (refinement_mode == 'deterministic')
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
        min_cell_size: int = 4,
        refinement_mode: Literal["learned", "deterministic"] = "deterministic",
        refinement_criteria: Optional[RefinementCriteria] = None,
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

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.min_cell_size = min_cell_size
        self.refinement_mode = refinement_mode
        self.refinement_criteria = refinement_criteria

        # --- CNN scorer (drives score-guided subdivision) ---
        # Only instantiated in learned mode; deterministic mode has no scorer.
        if refinement_mode == "learned":
            self.scorer = RefinementNet(input_channels=input_channels)
            # Global budget offset added to the running depth in the subdivision
            # test (positive -> coarser mesh, negative -> finer). Solve per sample
            # at eval if an exact token count is required; default 0.
            self.offset = 0.0
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
                token_preds:       [total_N, output_channels]
                score_map:         None  (no scorer in this mode)
                soft_N:            None  (no budget loss in this mode)
                tokens_per_sample: List[int] (len B)
                token_lists:       None  (targets are pre-averaged by DeterministicCollateFn)
        """
        preds = self.transformer(packed_tokens, tokens_per_sample)

        return {
            "token_preds": preds,
            "score_map": None,
            "soft_N": None,
            "tokens_per_sample": tokens_per_sample,
            "token_lists": None,
        }

    def _forward_learned(self, grids):
        """
        Scorer → depth-guided tree → transformer in one forward pass.

        The scorer emits a predicted depth map ``d_pred``; the deterministic
        running-depth builder turns it into leaves (no Gumbel, no temperature).
        The mesh build is detached — the scorer is trained separately by
        supervised regression to the variance oracle, so no gradient needs to
        flow back through the discrete tree here.

        Args:
            grids: [B, H, W, C] float32 input geometry, channel-last.

        Returns:
            Dict with keys:
                token_preds:       [total_N, output_channels] from the transformer
                score_map:         [B, 1, H, W] predicted depth map d_pred (attached)
                tokens_per_sample: List[int] (len B), tokens per sample
                token_lists:       List[List[QuadNode]] (len B)
        """
        B, H, W, C = grids.shape
        device = grids.device

        # 1. Predicted depth map d_pred (GPU).
        geom = grids.permute(0, 3, 1, 2).contiguous()
        score_map = self.scorer(geom)                    # [B, 1, H, W]

        # 2. Build trees deterministically (CPU, per-sample). Detached: the
        #    discrete build carries no gradient by design.
        depth_np = score_map.squeeze(1).detach().cpu().numpy()
        grids_np = grids.detach().cpu().numpy()

        all_tokens: List[torch.Tensor] = []
        tokens_per_sample: List[int] = []
        token_lists: List[List] = []

        for b in range(B):
            leaves = build_depth_guided_mesh(
                data=grids_np[b],
                depth_map=depth_np[b],
                max_depth=self.max_depth,
                min_depth=self.min_depth,
                min_cell_size=self.min_cell_size,
                offset=self.offset,
            )
            all_tokens.append(self._pack_tokens(leaves, H, W, C))
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

    def predict_depth(self, grids: torch.Tensor) -> torch.Tensor:
        """Run only the scorer and return the predicted depth map ``d_pred``.

        This is the forward used by the decoupled supervised scorer training —
        it never touches the transformer or builds a quadtree.

        Args:
            grids: ``[B, H, W, C]`` float32 input geometry, channel-last.

        Returns:
            ``[B, 1, H, W]`` predicted depth map.
        """
        geom = grids.permute(0, 3, 1, 2).contiguous()
        return self.scorer(geom)

    def _pack_tokens(self, leaves, H, W, C):
        """Extract the token-packing loop into a reusable method."""
        N = len(leaves)
        tokens = torch.zeros(N, C + 3, dtype=torch.float32)
        for i, leaf in enumerate(leaves):
            r0, c0, r1, c1 = leaf.bbox
            tokens[i, :C]     = torch.from_numpy(leaf.features)
            tokens[i, C]      = (c0 + c1) / 2.0 / W
            tokens[i, C + 1]  = (r0 + r1) / 2.0 / H
            tokens[i, C + 2]  = max((c1 - c0) / W, (r1 - r0) / H)
        return tokens


    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

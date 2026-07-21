"""
========================================================================
Adaptive-mesh aerodynamic flow predictor

Full pipeline:
grid -> AMR tokens build from quadtree -> transformer
========================================================================

The model turns a dense geometry grid into adaptive-mesh tokens and predicts a
per-token flow field with a transformer. There are two ways to get the adaptive-mesh:

--- "deterministic" -----------------------------------------------------
The mesh is built by a physics-based AMR criterion.

    grid_input [B, H, W, C]
      ↓ build_adaptive_mesh                       (refinement_criteria-based quadtree)
    packed_tokens [total_N, C+3]                  (already batched/packed)
      ↓ AeroTransformer                           (model.forward(packed_tokens, tokens_per_sample))
    token predictions [total_N, output_channels]

--- "learned" -----------------------------------------------------------
A RefinementNet CNN scores the grid and a deterministic running-depth quadtree
turns that score into the mesh.

    grid_input [B, H, W, C]
      ↓ RefinementNet (CNN scorer)
    depth_map d_pred [B, 1, H, W]                 (predicted target depth, not a sign logit)
      ↓ build_depth_guided_mesh                   (depth-based quadtree)
    leaves
      ↓ inline token packing
    packed_tokens [total_N, C+3]
      ↓ AeroTransformer                           (model.forward(grids))
    token predictions [total_N, output_channels]

Both emit the same packed batch dict, so the model — and the training loop — are
identical for both. The scorer is trained separately by supervised regression to
the variance oracle (see src/amr/oracle_depth.py and train_scorer_supervised);
it is not part of this model.
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from src.model.transformer import AeroTransformer


class AdaptiveMeshAeroModel(nn.Module):
    """
    Per-token flow-field predictor: an ``AeroTransformer`` over packed AMR tokens.

    The adaptive mesh is built and tokenized upstream (in the collate function),
    so this module only runs the transformer solver. It knows nothing about the
    scorer or the refinement criteria.

    Parameters
    ----------
    input_channels  : number of physical input channels
    output_channels : number of predicted quantities (e.g. 3 for u, v, p)
    d_model         : transformer hidden dimension
    n_layers        : number of transformer encoder layers
    n_heads         : number of attention heads
    d_ff            : feedforward dimension
    dropout         : dropout probability
    affine_output   : predict (value, gx, gy) per token instead of a constant
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
        affine_output: bool = False,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.affine_output = affine_output

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

    def forward(self, packed_tokens: torch.Tensor, tokens_per_sample: List[int]) -> Dict:
        """Run the transformer over pre-built packed tokens.

        Args:
            packed_tokens: ``[total_N, C+3]`` concatenated tokens of all samples.
            tokens_per_sample: per-sample token counts (``len == B``).

        Returns:
            Dict with keys:
                token_preds:       ``[total_N, output_channels]``, or
                    ``[total_N, output_channels, 3]`` = (value, gx, gy) when
                    ``affine_output`` (decoded into a per-cell ramp by the loss).
                tokens_per_sample: the input ``tokens_per_sample`` (echoed back).
        """
        preds = self.transformer(packed_tokens, tokens_per_sample)
        return {
            "token_preds": preds,
            "tokens_per_sample": tokens_per_sample,
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

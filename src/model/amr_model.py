"""
========================================================================
Full pipeline:
grid -> RefinementNet scorer -> score-guided quadtree -> transformer
========================================================================

    grid_input [B, H, W, C]
      ↓ RefinementNet (CNN scorer)
    score_map [B, 1, H, W]
      ↓ build_learned_adaptive_mesh (Gumbel-ST quadtree)
    leaves + soft_N
      ↓ inline token packing
    packed_tokens [total_N, C+4]
      ↓ AeroTransformer
    token predictions [total_N, output_dim]
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

import torch
import torch.nn as nn

from src.amr.adaptive_mesh import build_adaptive_mesh
from src.amr.refinement_criteria import RefinementCriteria
from src.amr.learned_adaptive_mesh import build_learned_adaptive_mesh
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
    map into leaf QuadNodes, which are packed inline into the same [N, C+4]
    token layout the transformer expects.

    Parameters
    ----------
    input_channels      : C - number of physical input channels
    output_dim          : number of predicted quantities (e.g. 3 for u, v, p)
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
        output_dim: int = 3,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
        min_depth: int = 2,
        max_depth: int = 6,
        min_cell_size: int = 4,
        refinement_mode: Literal["learned", "deterministic"] = "learned",
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
        self.output_dim = output_dim
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.min_cell_size = min_cell_size
        self.refinement_mode = refinement_mode
        self.refinement_criteria = refinement_criteria

        # --- CNN scorer (drives score-guided subdivision) ---
        # Only instantiated in learned mode; deterministic mode has no scorer.
        if refinement_mode == "learned":
            self.scorer = RefinementNet(input_channels=input_channels)
        else:
            self.scorer = None
        self.tau = 1.0  # Gumbel-Softmax temperature; unused in deterministic mode.

        # --- Transformer solver ---
        token_dim = input_channels + 4  # C + (x_c, y_c, s, d_norm)
        self.transformer = AeroTransformer(
            token_dim=token_dim,
            output_dim=output_dim,
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
          deterministic: forward(packed_tokens, seq_lens)  — tokens pre-built by DeterministicCollateFn
          learned:       forward(grids)                    — grids [B, H, W, C], tokenization happens inside
        """
        if self.refinement_mode == "deterministic":
            return self._forward_deterministic(*args, **kwargs)
        else:
            return self._forward_learned(*args, **kwargs)

    def _forward_deterministic(self, packed_tokens, seq_lens):
        """
        Tokens were already built in the collate function.
        Forward pass is just the transformer.

        Args:
            packed_tokens : [total_N, C+4]  - concatenated tokens of all samples
            seq_lens      : List[int]        - per-sample token counts

        Returns:
            Dict with keys:
                token_preds: [total_N, output_dim]
                score_map:   None  (no scorer in this mode)
                soft_N:      None  (no budget loss in this mode)
                seq_lens:    List[int] (len B)
                token_lists: None  (targets are pre-averaged by DeterministicCollateFn)
        """
        preds = self.transformer(packed_tokens, seq_lens)

        return {
            "token_preds": preds,
            "score_map":   None,
            "soft_N":      None,
            "seq_lens":    seq_lens,
            "token_lists": None,
        }

    def _forward_learned(self, grids):
        """
        Scorer → tree → transformer, all in forward for gradient flow.

        Args:
            grids: [B, H, W, C] float32 input geometry, channel-last.

        Returns:
            Dict with keys:
                token_preds: [total_N, output_dim] from the transformer
                score_map:   [B, 1, H, W] raw CNN output (kept attached for L_smooth)
                soft_N:      0-dim differentiable tensor = mean-over-batch soft_N
                seq_lens:    List[int] (len B), tokens per sample
                token_lists: List[List[QuadNode]] (len B)
        """
        B, H, W, C = grids.shape
        device = grids.device

        # 1. Score map (GPU)
        geom = grids.permute(0, 3, 1, 2).contiguous()
        score_map = self.scorer(geom)                    # [B, 1, H, W]

        # 2. Build trees (CPU, per-sample — unavoidable)
        score_np  = score_map.squeeze(1).detach().cpu().numpy()
        score_cpu = score_map.squeeze(1).cpu()
        grids_np  = grids.detach().cpu().numpy()

        all_tokens: List[torch.Tensor] = []
        seq_lens: List[int]            = []
        token_lists: List[List]        = []
        soft_Ns: List[torch.Tensor]    = []

        for b in range(B):
            leaves, soft_N_b = build_learned_adaptive_mesh(
                data=grids_np[b],
                score_map=score_np[b],
                score_tensor=score_cpu[b],
                max_depth=self.max_depth,
                min_depth=self.min_depth,
                min_cell_size=self.min_cell_size,
                training=self.training,
                tau=self.tau,
            )
            tokens = self._pack_tokens(leaves, H, W, C)
            all_tokens.append(tokens)
            seq_lens.append(len(leaves))
            token_lists.append(leaves)
            soft_Ns.append(soft_N_b)

        packed = torch.cat(all_tokens, dim=0).to(device)
        preds = self.transformer(packed, seq_lens)
        soft_N_mean = torch.stack(soft_Ns).mean().to(device)

        return {
            "token_preds": preds,
            "score_map":   score_map,
            "soft_N":      soft_N_mean,
            "seq_lens":    seq_lens,
            "token_lists": token_lists,
        }

    def _pack_tokens(self, leaves, H, W, C):
        """Extract the token-packing loop into a reusable method."""
        N = len(leaves)
        tokens = torch.zeros(N, C + 4, dtype=torch.float32)
        for i, leaf in enumerate(leaves):
            r0, c0, r1, c1 = leaf.bbox
            tokens[i, :C]     = torch.from_numpy(leaf.features)
            tokens[i, C]      = (c0 + c1) / 2.0 / W
            tokens[i, C + 1]  = (r0 + r1) / 2.0 / H
            tokens[i, C + 2]  = max((c1 - c0) / W, (r1 - r0) / H)
            tokens[i, C + 3]  = leaf.depth / self.max_depth
        return tokens

    # ------------------------------------------------------------------
    # Parameter count
    # ------------------------------------------------------------------

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

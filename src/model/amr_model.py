"""
========================================================================
Full pipeline: 
grid -> tokenizer -> transformer -> reconstruction -> predicted field
========================================================================

    grid_input [H, W, C]
      ↓ QuadNode
    adaptive tokens [N, C+4]
      ↓ AdaptiveMeshTransformer
    token predictions [N, output_dim]
      ↓ tokens_to_grid
    predicted flow field [H, W, output_dim]
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.amr.configs import AERODYNAMIC_CRITERIA
from src.amr.physics_metrics import RefinementCriteria
from src.amr.quadtree_tokenizer import QuadNode, QuadtreeTokenizer
from src.model.transformer import AeroTransformer
from src.model.reconstruction import tokens_to_grid, batch_tokens_to_grid


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class AdaptiveMeshAeroModel(nn.Module):
    """
    End-to-end steady aerodynamic flow-field predictor.

    The tokenizer runs on CPU (numpy arrays) since quadtree traversal is
    a sequential recursive algorithm. The transformer runs on GPU/CUDA.

    Parameters
    ----------
    input_channels  : C - number of physical input channels
    output_dim      : number of predicted quantities (e.g. 3 for u, v, p)
    d_model         : transformer hidden dimension
    n_layers        : number of transformer encoder layers
    n_heads         : number of attention heads
    d_ff            : feedforward dimension
    dropout         : dropout probability
    min_depth       : quadtree minimum depth
    max_depth       : quadtree maximum depth
    min_cell_size   : minimum cell size in pixels
    criterion       : optional custom RefinementCriterion
                      (default: aerodynamic composite criterion)
    recon_mode      : "fill" or "interp" - reconstruction mode
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
        criterion: Optional[RefinementCriteria] = None,
        recon_mode: str = "fill",
    ):
        super().__init__()
        self.input_channels = input_channels
        self.output_dim = output_dim
        self.recon_mode = recon_mode

        # --- Tokenizer (not a trainable nn.Module) ---
        self.tokenizer = QuadtreeTokenizer(
            min_depth=min_depth,
            max_depth=max_depth,
            min_cell_size=min_cell_size,
            refinement_criteria=criterion or AERODYNAMIC_CRITERIA,
        )

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
    # Single-sample forward (inference / testing)
    # ------------------------------------------------------------------

    def forward_single(
        self,
        grid: torch.Tensor,
        return_tokens: bool = False,
    ) -> Dict:
        """
        Parameters
        ----------
        grid         : [H, W, C] tensor (on any device)
        return_tokens: if True, also return intermediate token tensors

        Returns
        -------
        Dict with keys:
            "prediction"   : [H, W, output_dim] reconstructed flow field
            "token_preds"  : [N, output_dim]  (if return_tokens)
            "token_list"   : List[QuadtreeToken] (if return_tokens)
            "seq_len"      : N
        """
        H, W, C = grid.shape
        device = grid.device

        # 1. Tokenize (on CPU)
        grid_np = grid.cpu().numpy()
        token_arr_np, token_list = self.tokenizer.tokenize(grid_np)
        token_tensor = torch.from_numpy(token_arr_np).to(device)  # [N, C+4]
        N = token_tensor.shape[0]

        # 2. Transformer prediction
        preds = self.transformer(token_tensor, seq_lens=[N])  # [N, output_dim]

        # 3. Reconstruct
        recon = tokens_to_grid(preds, token_list, H, W, self.output_dim, mode=self.recon_mode)
        recon = recon.to(device)

        out = {"prediction": recon, "seq_len": N}
        if return_tokens:
            out["token_preds"] = preds
            out["token_list"]  = token_list
        return out

    # ------------------------------------------------------------------
    # Batched forward (training) - sequence packing
    # ------------------------------------------------------------------

    def forward(
        self,
        packed_tokens: torch.Tensor,
        seq_lens: List[int],
        grid_shape: Optional[Tuple[int, int]] = None,
        reconstruct: bool = False,
        token_lists: Optional[List[List[QuadNode]]] = None,
    ) -> Dict:
        """
        Batched forward with pre-tokenized, packed inputs.

        Parameters
        ----------
        packed_tokens : [total_N, C+4]  - concatenated tokens of all samples
        seq_lens      : List[int]        - per-sample token counts
        grid_shape    : (H, W)           - required if reconstruct=True
        reconstruct   : if True, reconstruct full grids
        token_lists   : List of token lists per sample (required if reconstruct=True)

        Returns
        -------
        Dict with keys:
            "token_preds"    : [total_N, output_dim]
            "reconstructions": [B, H, W, output_dim]  (if reconstruct=True)
        """
        # Transformer prediction on packed sequence
        preds = self.transformer(packed_tokens, seq_lens)  # [total_N, output_dim]

        out = {"token_preds": preds}

        if reconstruct:
            assert grid_shape is not None and token_lists is not None
            H, W = grid_shape
            grids = batch_tokens_to_grid(
                preds, token_lists, seq_lens, H, W, self.output_dim, mode=self.recon_mode
            )
            out["reconstructions"] = grids.to(packed_tokens.device)

        return out

    # ------------------------------------------------------------------
    # Convenience: full single-sample pipeline from numpy
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, grid_np: np.ndarray) -> np.ndarray:
        """
        Full inference from numpy input to numpy output.

        Args:
            grid_np: [H, W, C] numpy array

        Returns:
            [H, W, output_dim] numpy array
        """
        self.eval()
        device = next(self.parameters()).device
        grid_t = torch.from_numpy(grid_np.astype(np.float32)).to(device)
        result = self.forward_single(grid_t)
        return result["prediction"].cpu().numpy()

    # ------------------------------------------------------------------
    # Parameter count
    # ------------------------------------------------------------------

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

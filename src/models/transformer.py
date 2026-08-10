"""
========================================================================
Shared pre-norm Transformer encoder block.
========================================================================

The one attention/FFN layer both models in this project are built from:

    * ``AMRTransformer`` (src/model/amr_model.py) stacks it over padded
      variable-length AMR token sequences, passing a boolean key-padding mask;
    * ``ViT`` (src/model/vit_model.py) stacks it over fixed-length dense patch
      grids, where every sequence has the same length and no mask is needed.

Keeping the block here — and the model-specific embeddings, heads and batching
in the two model modules — is what lets those two otherwise unrelated
architectures share exactly one implementation of attention.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Pre-norm Transformer encoder layer for a batched [B, N, d] sequence.

    The forward pass is the standard pre-norm formulation:
        x = x + Attn(LayerNorm(x))
        x = x + FFN(LayerNorm(x))

    Attention is computed via torch.nn.functional.scaled_dot_product_attention
    with an optional boolean key-padding mask that keeps real tokens from
    attending to pad slots in a padded batch.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert d_model % n_heads == 0

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Q, K, V projections
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = dropout

        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def _attention_standard(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Multi-head self-attention over a batched [B, N, d] sequence.

        attn_mask is a boolean key-padding mask broadcastable to [B, h, N, N]
        (True = attend); None means full attention.
        """
        qkv = self.qkv(x)  # [..., 3*d]
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape to [..., h, N, hd] for SDPA (Scaled Dot-Product Attention)
        q = q.reshape(*q.shape[:-1], self.n_heads, self.head_dim).transpose(-3, -2)
        k = k.reshape(*k.shape[:-1], self.n_heads, self.head_dim).transpose(-3, -2)
        v = v.reshape(*v.shape[:-1], self.n_heads, self.head_dim).transpose(-3, -2)

        # Boolean [B, 1, 1, N] key-padding mask (True = attend), broadcast over
        # heads and query positions by SDPA; None means full attention.
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )  # [..., h, N, hd]
        out = out.transpose(-3, -2).reshape(x.shape)
        return self.out_proj(out)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply one pre-norm attention + FFN sublayer pair.

        Args:
            x:         Batched [B, N, d] sequences (padded AMR batch or ViT
                       baseline).
            attn_mask: Boolean key-padding mask broadcastable to [B, h, N, N]
                       (True = attend); None means full attention within each
                       sequence.
        """
        x = x + self._attention_standard(self.norm1(x), attn_mask)
        x = x + self.ff(self.norm2(x))

        return x

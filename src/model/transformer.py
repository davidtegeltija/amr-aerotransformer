"""
========================================================================
Transformer-based neural solver for the adaptive mesh CFD pipeline.
========================================================================

Architecture:
    token_embedding: MLP(physical_dim -> d_model) projecting per-token physical features into the latent space. `physical_dim = token_dim - 3`.
    pos_embedding:   Fixed log-spaced Fourier features over the 3 positional meta channels (x_c, y_c, size), followed by an MLP into d_model.
    input_norm:      LayerNorm applied to the sum of token + positional embeddings.
    encoder:         Stack of pre-norm TransformerBlock layers (default: 6 layers, 4 heads, d_model=256, d_ff=1024).
    final_norm:      LayerNorm before the prediction head.
    prediction_head: MLP(d_model -> output_channels) producing per-token predictions.

Batching strategy (sequence packing):
    Instead of padding every sequence to the same length, the token sequences
    of all samples in a batch are concatenated into a single long sequence:

        packed = [sample_0_tokens | sample_1_tokens | ... | sample_B_tokens]

    A block-diagonal additive attention mask (0 / -inf) prevents cross-sample
    attention. This follows the strategy used in NaViT (Dehghani et al., 2023)
    and APT (Choudhury et al., 2025). Attention is computed with
    torch.nn.functional.scaled_dot_product_attention.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _make_block_diagonal_mask(seq_lengths: List[int], device: torch.device) -> torch.Tensor:
    """Build an additive block-diagonal attention mask for a packed sequence.

    Within-sample positions get 0.0 (attend); cross-sample positions get -inf
    (masked out), matching the additive mask convention of
    torch.nn.functional.scaled_dot_product_attention.

    Args:
        seq_lengths: Per-sample token counts in the packed sequence.
        device:      Device on which to allocate the mask.

    Returns:
        Float tensor of shape [total_len, total_len] with values in {0.0, -inf}.
    """
    total = sum(seq_lengths)
    mask = torch.full((total, total), float('-inf'), device=device)
    offset = 0
    for L in seq_lengths:
        mask[offset:offset + L, offset:offset + L] = 0.0
        offset += L
    return mask


# ---------------------------------------------------------------------------
# Packed-sequence Transformer Block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Pre-norm Transformer encoder layer for a packed token sequence.

    The forward pass is the standard pre-norm formulation:
        x = x + Attn(LayerNorm(x))
        x = x + FFN(LayerNorm(x))

    Attention is computed via torch.nn.functional.scaled_dot_product_attention
    with a block-diagonal additive mask that prevents cross-sample attention
    in the packed batch.
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
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Multi-head self-attention over the packed sequence."""
        total, d = x.shape
        qkv = self.qkv(x)  # [total, 3*d]
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape to [h, total, hd] for SDPA
        q = q.reshape(total, self.n_heads, self.head_dim).transpose(0, 1)
        k = k.reshape(total, self.n_heads, self.head_dim).transpose(0, 1)
        v = v.reshape(total, self.n_heads, self.head_dim).transpose(0, 1)

        # torch SDPA broadcasts the mask across heads
        attn_mask_3d = attn_mask.unsqueeze(0)  # [1, total, total]
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask_3d,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )  # [h, total, hd]
        out = out.transpose(0, 1).reshape(total, d)
        return self.out_proj(out)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply one pre-norm attention + FFN sublayer pair."""
        assert attn_mask is not None, "Must provide attn_mask"

        x = x + self._attention_standard(self.norm1(x), attn_mask)
        x = x + self.ff(self.norm2(x))

        return x


# ---------------------------------------------------------------------------
# Main transformer model
# ---------------------------------------------------------------------------

class AeroTransformer(nn.Module):
    """Transformer solver over packed, variable-length token sequences.

    Each token carries `token_dim` features, structured as:
        [physical_channels (token_dim - 3) | positional_meta (3)]
    where the positional meta channels are (x_c, y_c, size). Physical and
    positional features are embedded separately, summed, normalized, then
    refined by a stack of pre-norm Transformer blocks. The final per-token
    embedding is mapped to `output_channels` flow-field predictions.

    Forward pass (packed / training mode):
        tokens:   [total_N, token_dim]  concatenated tokens from all samples
        seq_lens: List[int]             per-sample token counts

    Forward pass (single sample / inference):
        tokens:   [N, token_dim]
        seq_lens: [N]

    Returns:
        Predictions of shape [total_N, output_channels].
    """

    def __init__(
        self,
        token_dim: int,           # C + 3  (physical channels + positional meta)
        output_channels: int = 3,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
        n_fourier: int = 64,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.d_model = d_model
        self.pos_dim = 3

        # --- Token Embedding layer ---
        # Projects raw token features [C] into the latent space [d_model]
        physical_dim = token_dim - self.pos_dim
        self.token_embedding = nn.Sequential(
            nn.LayerNorm(physical_dim),
            nn.Linear(physical_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # --- Positional Embedding layer ---
        # Fixed log-spaced frequencies, not learned
        self.register_buffer("pos_freqs", 2.0 ** torch.linspace(0, 8, n_fourier // 2).unsqueeze(0))  # [1, F/2]
        fourier_dim = self.pos_dim * n_fourier
        # Projects raw token position meta [3] into the latent space [d_model].
        self.pos_embedding = nn.Sequential(
            nn.Linear(fourier_dim, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

        # --- Combine token + positional embeddings ---
        self.input_norm = nn.LayerNorm(d_model)

        # --- Transformer encoder ---
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # --- Prediction head ---
        self.prediction_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_channels),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def embedd_position(self, pos: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        """Encode positional meta channels as concatenated sin/cos Fourier features.

        Args:
            pos:   Positional channels of shape [total_N, 3] (x_c, y_c, size).
            freqs: Fixed log-spaced frequencies of shape [1, F/2].

        Returns:
            Fourier features of shape [total_N, 3*F].
        """
        angles = pos.unsqueeze(-1) * freqs # [total_N, 3, F/2]
        features = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return features.flatten(-2) # [total_N, 3*F]

    def forward(
        self,
        tokens: torch.Tensor,
        seq_lens: List[int],
    ) -> torch.Tensor:
        """Run the encoder over a packed token sequence and produce per-token predictions.

        Args:
            tokens:   Packed token features of shape [total_N, token_dim], with
                the last 3 channels treated as positional meta (x_c, y_c, size).
            seq_lens: Per-sample token counts in the packed sequence.

        Returns:
            Per-token predictions of shape [total_N, output_channels].
        """
        assert tokens.shape[-1] == self.token_dim, (
            f"Expected tokens with {self.token_dim} channels "
            f"(physical + {self.pos_dim} positional meta in the last slots), "
            f"got {tokens.shape[-1]}."
        )

        device = tokens.device

        # Split into physical features and positional meta
        physical_channels = tokens[:, :-self.pos_dim]       # [total_N, 5]
        positional_channels = tokens[:, -self.pos_dim:]     # [total_N, 3]

        # Embed
        tok_emb = self.token_embedding(physical_channels)   # [total_N, d_model]
        pos_emb = self.pos_embedding(self.embedd_position(positional_channels, self.pos_freqs)) # [total_N, d_model]
        x = self.input_norm(tok_emb + pos_emb)   # [total_N, d_model]

        # Prepare attention infrastructure
        attn_mask  = _make_block_diagonal_mask(seq_lens, device)

        # Transformer encoder
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)
        x = self.final_norm(x)

        # Per-token predictions
        return self.prediction_head(x)  # [total_N, output_channels]

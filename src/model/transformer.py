"""
========================================================================
Transformer-based neural solver for the adaptive mesh CFD pipeline.
========================================================================

Architecture
------------
  token_embedding : MLP(C+4  → d_model)          projects raw tokens to latent space
  positional_enc  : learned MLP(4 → d_model)      encodes (x_c, y_c, s, d_norm)
  transformer_enc : nn.TransformerEncoder          6 layers, 4 heads, d_model=256, ff=1024
  prediction_head : MLP(d_model → output_dim)     per-token flow-field prediction

Batching strategy (FlashAttention / sequence packing)
------------------------------------------------------
Instead of padding every sequence to the same length, we *concatenate* the
token sequences of all samples in a batch into a single long sequence:

    packed = [sample_0_tokens | sample_1_tokens | … | sample_B_tokens]

A block-diagonal boolean mask then prevents cross-sample attention.
This is exactly the strategy used in NaViT (Dehghani et al., 2023) and APT
(Choudhury et al., 2025) and is natively supported by FlashAttention-2 via
its `varlen` (variable-length) API.

  - If `flash_attn` is installed: uses flash_attn.flash_attn_varlen_func
    with `cu_seqlens` for zero-overhead masking.
  - Fallback: dense block-diagonal boolean mask passed to
    nn.MultiheadAttention / torch.nn.functional.scaled_dot_product_attention,
    which is still correct but slightly less memory-efficient for very long sequences.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional FlashAttention-2 import
try:
    from flash_attn import flash_attn_varlen_func
    from flash_attn.bert_padding import unpad_input, pad_input
    _FLASH_AVAILABLE = True
except ImportError:
    _FLASH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _make_block_diagonal_mask(seq_lengths: List[int], device: torch.device) -> torch.Tensor:
    """
    Construct a boolean additive attention mask (True = attend, False = mask out)
    for a packed sequence with per-sample lengths `seq_lengths`.

    Returns shape [total_len, total_len] boolean tensor.
    nn.TransformerEncoder expects additive mask where -inf means "do not attend",
    so we return 0 / -inf floats.
    """
    total = sum(seq_lengths)
    # True means "this pair can attend"
    mask = torch.zeros(total, total, dtype=torch.bool, device=device)
    offset = 0
    for L in seq_lengths:
        mask[offset:offset + L, offset:offset + L] = True
        offset += L
    # Convert to additive float mask
    float_mask = torch.zeros(total, total, device=device)
    float_mask[~mask] = float('-inf')
    return float_mask


def _make_cumulative_seqlens(seq_lengths: List[int], device: torch.device) -> torch.Tensor:
    """Cumulative sequence lengths for FlashAttention varlen API (int32)."""
    cu = torch.zeros(len(seq_lengths) + 1, dtype=torch.int32, device=device)
    cu[1:] = torch.tensor(seq_lengths, dtype=torch.int32).cumsum(0)
    return cu


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Simple 2-layer MLP with GELU activation and optional dropout."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TokenEmbedding(nn.Module):
    """
    Projects raw token features [C+4] into the latent space [d_model].

    C+4 = C physical channels + (x_c, y_c, s, d_norm).
    """

    def __init__(self, in_dim: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.proj = MLP(in_dim, d_model * 2, d_model, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x))


class PositionalEncoding(nn.Module):
    # TODO : try without this encoding
    """
    Learned positional encoding from spatial meta-data:
        input  : [N, 4]  → (x_c, y_c, s, d_norm)
        output : [N, d_model]

    We use a small MLP with sinusoidal feature expansion as the first layer
    (Fourier features), following common practice in neural fields.
    """

    def __init__(self, d_model: int, n_fourier: int = 64):
        super().__init__()
        # Fixed Fourier frequencies (not learned) - doubles expressiveness
        self.register_buffer(
            'freqs',
            2.0 ** torch.linspace(0, 8, n_fourier // 2).unsqueeze(0)  # [1, F/2]
        )
        fourier_dim = 4 * n_fourier  # 4 spatial inputs × n_fourier features
        self.mlp = MLP(fourier_dim, d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        """
        pos: [..., 4]  (x_c, y_c, s, d_norm)
        returns: [..., d_model]
        """
        # Fourier expansion: sin + cos for each frequency
        # pos: [..., 4] → [..., 4, 1] × [1, F/2] → [..., 4, F/2]
        angles = pos.unsqueeze(-1) * self.freqs  # [..., 4, F/2]
        features = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # [..., 4, F]
        features = features.flatten(-2)  # [..., 4*F]
        return self.norm(self.mlp(features))


# ---------------------------------------------------------------------------
# Packed-sequence FlashAttention layer
# ---------------------------------------------------------------------------

class PackedFlashTransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer that operates on a packed sequence.
    Uses FlashAttention-2 varlen API when available, otherwise falls back
    to standard PyTorch attention with a block-diagonal mask.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        norm_first: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert d_model % n_heads == 0

        self.norm_first = norm_first
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

    def _attention_flash(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        """FlashAttention-2 varlen path (no padding overhead)."""
        total, d = x.shape
        qkv = self.qkv(x).reshape(total, 3, self.n_heads, self.head_dim)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # [total, h, hd]

        out = flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            dropout_p=self.attn_dropout if self.training else 0.0,
            causal=False,
        )  # [total, h, hd]
        return self.out_proj(out.reshape(total, d))

    def _attention_standard(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Standard PyTorch SDPA path with block-diagonal mask."""
        total, d = x.shape
        qkv = self.qkv(x)  # [total, 3*d]
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape to [h, total, hd] for SDPA
        q = q.reshape(total, self.n_heads, self.head_dim).transpose(0, 1)
        k = k.reshape(total, self.n_heads, self.head_dim).transpose(0, 1)
        v = v.reshape(total, self.n_heads, self.head_dim).transpose(0, 1)

        # torch SDPA broadcasts the mask across heads
        attn_mask_4d = attn_mask.unsqueeze(0)  # [1, total, total]
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask_4d,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )  # [h, total, hd]
        out = out.transpose(0, 1).reshape(total, d)
        return self.out_proj(out)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ) -> torch.Tensor:
        residual = x
        if self.norm_first:
            x = self.norm1(x)

        if _FLASH_AVAILABLE and cu_seqlens is not None:
            attn_out = self._attention_flash(x, cu_seqlens, max_seqlen)
        else:
            assert attn_mask is not None, "Must provide attn_mask when FlashAttention is unavailable"
            attn_out = self._attention_standard(x, attn_mask)

        x = residual + attn_out
        if not self.norm_first:
            x = self.norm1(x)

        residual = x
        if self.norm_first:
            x = self.norm2(x)
        x = residual + self.ff(x)
        if not self.norm_first:
            x = self.norm2(x)

        return x


# ---------------------------------------------------------------------------
# Main transformer model
# ---------------------------------------------------------------------------

class AeroTransformer(nn.Module):
    """
    Transformer solver that operates on packed, variable-length token sequences.

    Forward pass (packed / training mode):
        tokens   : [total_N, token_dim]   - concatenated tokens from all samples
        seq_lens : List[int]              - number of tokens per sample

    Forward pass (single sample / inference):
        tokens   : [N, token_dim]
        seq_lens : [N]

    Returns:
        predictions: [total_N, output_dim]  - token-level flow predictions
    """

    def __init__(
        self,
        token_dim: int,           # C + 4  (physical channels + positional meta)
        output_dim: int = 3,      # e.g. u, v, p
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
        self.pos_dim = 4  # (x_c, y_c, s, d_norm)

        phys_dim = token_dim - self.pos_dim  # C

        # --- Embedding layers ---
        self.token_embedding = TokenEmbedding(token_dim, d_model, dropout=dropout)
        self.pos_encoding    = PositionalEncoding(d_model, n_fourier=n_fourier)

        # Combine token + positional embeddings
        self.input_norm = nn.LayerNorm(d_model)

        # --- Transformer encoder ---
        self.layers = nn.ModuleList([
            PackedFlashTransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # --- Prediction head ---
        self.head = MLP(d_model, d_model, output_dim, dropout=dropout)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        tokens: torch.Tensor,
        seq_lens: List[int],
    ) -> torch.Tensor:
        """
        tokens   : [total_N, token_dim]
        seq_lens : list of per-sample token counts

        Returns predictions: [total_N, output_dim]
        """
        device = tokens.device
        total_N = tokens.shape[0]

        # Split into physical features and positional meta
        pos = tokens[:, -self.pos_dim:]          # [total_N, 4]

        # Embed
        tok_emb = self.token_embedding(tokens)   # [total_N, d_model]
        pos_emb = self.pos_encoding(pos)         # [total_N, d_model]
        x = self.input_norm(tok_emb + pos_emb)   # [total_N, d_model]

        # Prepare attention infrastructure
        if _FLASH_AVAILABLE:
            cu_seqlens = _make_cumulative_seqlens(seq_lens, device)
            max_seqlen = max(seq_lens)
            attn_mask  = None
        else:
            cu_seqlens = None
            max_seqlen = None
            attn_mask  = _make_block_diagonal_mask(seq_lens, device)

        # Transformer encoder
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)

        # Per-token predictions
        return self.head(x)  # [total_N, output_dim]

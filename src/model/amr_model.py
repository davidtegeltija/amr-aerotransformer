"""
========================================================================
Adaptive-mesh aerodynamic flow predictor

Full pipeline:
grid -> AMR tokens build from quadtree -> transformer
========================================================================

A transformer solver over packed, variable-length adaptive-mesh token sequences.
The model turns a dense geometry grid into adaptive-mesh tokens and predicts a
per-token flow field with a transformer. The mesh is built and tokenized upstream, 
in the DataLoader collate functions, so this module only runs the solver.
There are two ways to get the adaptive-mesh:

--- "deterministic" -----------------------------------------------------
The mesh is built by a physics-based AMR criterion.

    grid_input [B, H, W, C]
        ↓ build_adaptive_mesh                       (refinement_criteria-driven quadtree)
    leaves 
        ↓ nodes_to_token_array
    packed_tokens [total_N, C+3]                  (already batched/packed)
        ↓ AdaptiveMeshAeroModel
    token predictions [total_N, output_channels]

--- "learned" -----------------------------------------------------------
A RefinementNet CNN scores the grid and a deterministic running-depth quadtree
turns that score into the mesh.

    grid_input [B, H, W, C]
        ↓ RefinementNet (CNN scorer)
    depth_map d_pred [B, 1, H, W]                 (predicted target depth, not a sign logit)
        ↓ build_depth_guided_mesh                   (depth-based quadtree)
    leaves
        ↓ nodes_to_token_array
    packed_tokens [total_N, C+3]
        ↓ AdaptiveMeshAeroModel
    token predictions [total_N, output_channels]

The scorer is trained separately by supervised regression against the variance
oracle (see src/amr/oracle_depth.py and train_scorer_supervised); it is not part
of this model.

Architecture:
    token_embedding: MLP(physical_dim -> d_model) projecting per-token physical features into the latent space. `physical_dim = token_dim - 3`.
    pos_embedding:   Fixed log-spaced Fourier features over the 3 positional meta channels (x_c, y_c, size), followed by an MLP into d_model.
    input_norm:      LayerNorm applied to the sum of token + positional embeddings.
    encoder:         Stack of pre-norm TransformerBlock layers (src/model/transformer.py, shared with the ViT baseline).
    final_norm:      LayerNorm before the prediction head.
    prediction_head: MLP(d_model -> output_channels) producing per-token predictions.
                     With affine_output=True it instead emits output_channels*3
                     numbers per token, reshaped to [N, C, 3] = (value, gx, gy),
                     a per-token affine (value + 2D gradient) field decoded into a
                     linear ramp across each cell by tokens_to_grid_affine_torch.

Batching strategy (pad to per-batch max around the encoder):
    The model boundary stays packed: all samples' token sequences arrive
    concatenated into a single long sequence

        packed = [sample_0_tokens | sample_1_tokens | ... | sample_B_tokens]

    and the embeddings and prediction head run on this packed [total_N, ...]
    layout. Only around the encoder stack are the per-sample embedding
    sequences padded to [B, N_max, d_model] and attended with a boolean
    key-padding mask (True = real token), computed by torch.nn.functional.
    scaled_dot_product_attention; pad rows are dropped again before the head.
    This replaces the earlier NaViT/APT-style packed layout with a dense
    block-diagonal additive mask, whose [total_N, total_N] float mask cost
    (sum of all lengths, squared) dominated compute and memory and forced the
    slow SDPA math backend on Pascal-era GPUs; a boolean broadcast key-padding
    mask costs B*N_max bytes and computes the same function.
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from src.model.transformer import TransformerBlock


# ---------------------------------------------------------------------------
# Padding mask for token batches
# ---------------------------------------------------------------------------

def _make_key_padding_mask(tokens_per_sample: List[int], n_max: int, device: torch.device) -> torch.Tensor:
    """Build a boolean validity mask for a padded [B, n_max, d] batch.

    True marks real tokens, False marks padding, matching the boolean mask
    convention of torch.nn.functional.scaled_dot_product_attention (True =
    take part in attention).

    Args:
        tokens_per_sample: Per-sample token counts.
        n_max: Padded sequence length (max of tokens_per_sample).
        device: Device on which to allocate the mask.

    Returns:
        Bool tensor of shape [B, n_max]; True = real token, False = pad.
    """
    lengths = torch.tensor(tokens_per_sample, device=device)
    return torch.arange(n_max, device=device)[None, :] < lengths[:, None]


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class AdaptiveMeshAeroModel(nn.Module):
    """Per-token flow-field predictor over packed AMR tokens.

    Each token carries ``input_channels + 3`` features, structured as:
        [physical_channels (input_channels) | positional_meta (3)]
    where the positional meta channels are (x_c, y_c, size). Physical and
    positional features are embedded separately, summed, normalized, then
    refined by a stack of pre-norm Transformer blocks. The final per-token
    embedding is mapped to ``output_channels`` flow-field predictions.

    Forward pass (packed / training mode):
        packed_tokens:     [total_N, input_channels + 3]  concatenated tokens of all samples
        tokens_per_sample: List[int]                      per-sample token counts

    Forward pass (single sample / inference):
        packed_tokens:     [N, input_channels + 3]
        tokens_per_sample: [N]

    Args:
        input_channels: Number of physical input channels per token.
        output_channels: Number of predicted quantities (e.g. 3 for u, v, p).
        d_model: Transformer hidden dimension.
        n_layers: Number of transformer encoder layers.
        n_heads: Number of attention heads (must divide ``d_model``).
        d_ff: Feedforward inner dimension.
        dropout: Dropout probability.
        n_fourier: Number of Fourier features per positional meta channel.
        affine_output: Predict (value, gx, gy) per token instead of a constant.
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
        n_fourier: int = 64,
        affine_output: bool = False,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.affine_output = affine_output
        self.d_model = d_model
        self.pos_dim = 3
        self.token_dim = input_channels + self.pos_dim  # C + (x_center, y_center, size)

        # Every constructor argument, recorded so save_checkpoint can store it and
        # build_model_from_checkpoint can rebuild this exact architecture from the
        # checkpoint alone. Keep in sync with the signature above — the builder
        # raises if an argument is missing here rather than silently defaulting it.
        self.init_kwargs = dict(
            input_channels=input_channels,
            output_channels=output_channels,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            n_fourier=n_fourier,
            affine_output=affine_output,
        )

        # --- Token Embedding layer ---
        # Projects raw token features [C] into the latent space [d_model]
        self.token_embedding = nn.Sequential(
            nn.LayerNorm(input_channels),
            nn.Linear(input_channels, d_model),
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
        # In affine mode the head emits (value, gx, gy) per channel; the gradient
        # components start near 0 under trunc_normal init, so the model begins as
        # ~constant-per-cell and learns the ramps from the dense per-pixel loss.
        head_out = output_channels * 3 if affine_output else output_channels
        self.prediction_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, head_out),
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

    def forward(self, packed_tokens: torch.Tensor, tokens_per_sample: List[int]) -> Dict:
        """Run the encoder over a packed token sequence and produce per-token predictions.

        Args:
            packed_tokens: ``[total_N, C+3]`` concatenated tokens of all samples,
                with the last 3 channels treated as positional meta (x_c, y_c, size).
            tokens_per_sample: per-sample token counts (``len == B``).

        Returns:
            Per-token predictions of shape [total_N, output_channels], or
            [total_N, output_channels, 3] = (value, gx, gy) when affine_output.
            in a dict with keys:
                token_preds:       ``[total_N, output_channels]``, or
                    ``[total_N, output_channels, 3]`` = (value, gx, gy) when
                    ``affine_output`` (decoded into a per-cell ramp by the loss).
                tokens_per_sample: the input ``tokens_per_sample`` (echoed back).
        """
        assert packed_tokens.shape[-1] == self.token_dim, (
            f"Expected tokens with {self.token_dim} channels "
            f"(physical + {self.pos_dim} positional meta in the last slots), "
            f"got {packed_tokens.shape[-1]}."
        )

        device = packed_tokens.device

        # Split into physical features and positional meta
        physical_channels = packed_tokens[:, :-self.pos_dim]       # [total_N, C]
        positional_channels = packed_tokens[:, -self.pos_dim:]     # [total_N, 3]

        # Embed
        tok_emb = self.token_embedding(physical_channels)   # [total_N, d_model]
        pos_emb = self.pos_embedding(self.embedd_position(positional_channels, self.pos_freqs)) # [total_N, d_model]
        x = self.input_norm(tok_emb + pos_emb)   # [total_N, d_model]

        # Pad the per-sample embeddings to [B, N_max, d_model] and attend with a
        # boolean key-padding mask: real tokens never see pad keys, so this is
        # mathematically identical to per-sample (block-diagonal) attention at
        # B*N_max^2 instead of (sum N)^2 cost. Embeddings above and the head
        # below stay on the packed layout, so pad slots never touch them.
        # Split packed [total_N, d_model] into B tensors of shape [N_b, d_model],
        # one per sample (N_b = tokens_per_sample[b]).
        per_sample_embeddings = list(torch.split(x, tokens_per_sample, dim=0))
        x = pad_sequence(per_sample_embeddings, batch_first=True)  # [B, N_max, d_model]
        n_max = x.shape[1]
        valid = _make_key_padding_mask(tokens_per_sample, n_max, device)   # [B, N_max]
        # Equal-length batches (incl. single-sample inference) need no mask at
        # all, which keeps them eligible for the fastest SDPA path.
        has_padding = min(tokens_per_sample) < n_max
        attn_mask = valid[:, None, None, :] if has_padding else None       # [B, 1, 1, N_max]

        # Transformer encoder
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)

        # Drop pad rows -> back to packed [total_N, d_model]; row order is
        # sample 0's tokens, then sample 1's, ..., identical to the input
        # packing, so downstream packed indexing (losses, owner maps) is unchanged.
        x = x[valid] if has_padding else x.reshape(-1, self.d_model)
        x = self.final_norm(x)

        # Per-token predictions
        preds = self.prediction_head(x)                # [total_N, head_out]
        if self.affine_output:
            # [total_N, C, 3] = (value, gx, gy) per channel
            preds = preds.view(-1, self.output_channels, 3)

        return {
            "token_preds": preds,
            "tokens_per_sample": tokens_per_sample,
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

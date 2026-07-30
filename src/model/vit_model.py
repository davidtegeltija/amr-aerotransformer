import torch
import torch.nn as nn

from src.model.transformer import TransformerBlock


def pair(t):
    return t if isinstance(t, (tuple, list)) else (t, t)


def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype=torch.float32):

    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


class ViT(nn.Module):
    """Dense ViT baseline: fixed-size patches over the full [B, C, H, W] grid.

    The counterpart of ``AMRTransformer`` (src/model/amr_model.py) and named to
    match it: both wrap the same ``TransformerBlock`` stack, so the architecture
    keys (``input_channels``, ``output_channels``, ``d_model``, ``n_layers``,
    ``n_heads``, ``d_ff``, ``dropout``) mean the same thing in both and a config
    can be compared key-by-key across the two branches. What differs is the
    tokenization: a uniform patch grid here, adaptive quadtree leaves there —
    hence the three extra arguments below that have no AMR counterpart.

    Args:
        image_size: Grid size (H, W); an int is broadcast to a square.
        patch_size: Patch size (ph, pw); an int is broadcast to a square. Must
            divide ``image_size`` on both axes.
        input_channels: Number of physical input channels per pixel.
        output_channels: Number of predicted quantities (e.g. 3 for u, v, p).
        d_model: Transformer hidden dimension (the patch-embedding width).
        n_layers: Number of transformer encoder layers.
        n_heads: Number of attention heads (must divide ``d_model``).
        d_ff: Feedforward inner dimension. Passed straight to
            ``TransformerBlock``, matching ``AMRTransformer`` — this replaces the
            older ``mlp_ratio`` multiplier, so pass ``d_model * ratio`` directly.
        dropout: Dropout probability.
        pos_embedding: ``'sincos'`` for fixed 2D sin/cos features (requires
            ``d_model`` divisible by 4) or ``'trainable'`` for a learned table.
    """

    def __init__(self,
                 image_size,
                 patch_size,
                 input_channels: int = 3,
                 output_channels: int = 1,
                 d_model: int = 256,
                 n_layers: int = 5,
                 n_heads: int = 8,
                 d_ff: int = 1024,
                 dropout: float = 0.0,
                 pos_embedding: str = 'sincos',
                 ):

        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        patch_dim = input_channels * patch_height * patch_width

        self.ph, self.pw = patch_height, patch_width
        self.nh, self.nw = image_height // patch_height, image_width // patch_width

        # Constructor arguments, recorded for build_model_from_checkpoint (see
        # src/utils/model_utils.py). Keep in sync with the signature above. The
        # sizes are stored already normalized by pair(), which round-trips
        # unchanged through pair() when the model is rebuilt.
        self.init_kwargs = dict(
            image_size=(image_height, image_width),
            patch_size=(patch_height, patch_width),
            input_channels=input_channels,
            output_channels=output_channels,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            pos_embedding=pos_embedding,
        )

        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, d_model),
            nn.LayerNorm(d_model),
        )

        if pos_embedding == 'trainable':
            self.pos_embedding = nn.Parameter((1 / d_model) * torch.rand((1, self.nh * self.nw, d_model), dtype=torch.float))
        elif pos_embedding == 'sincos':
            # Fixed, so a (non-persistent) buffer: moves with .to(device), stays out of checkpoints.
            self.register_buffer("pos_embedding", posemb_sincos_2d(h=self.nh, w=self.nw, dim=d_model), persistent=False)
        else:
            raise KeyError(f"Unknown pos_embedding: {pos_embedding}")

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        self.last_layer = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, self.ph * self.pw * output_channels))
        self.output_channels = output_channels

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        B, C, H, W = img.shape
        # Patchify -> [B, N, patch_dim]
        img = img.reshape(B, C, self.nh, self.ph, self.nw, self.pw).permute(0, 2, 4, 3, 5, 1).reshape(B, self.nh * self.nw, self.ph * self.pw * C)

        fx = self.to_patch_embedding(img)                            # [B, N, D]
        fx = fx + self.pos_embedding                                 # [B, N, D]

        # All sequences share the same length N, so plain batched attention
        # (no mask) replaces the packed layout the AMR transformer needs for
        # its variable-length sequences.
        for block in self.blocks:
            fx = block(fx)                                           # [B, N, D]

        fx = self.last_layer(fx)                                     # [B, N, ph*pw*output_channels]
        return fx.reshape(B, self.nh, self.nw, self.ph, self.pw, self.output_channels) \
                 .permute(0, 5, 1, 3, 2, 4) \
                 .reshape(B, self.output_channels, self.nh * self.ph, self.nw * self.pw)

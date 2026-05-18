import torch
import torch.nn as nn

from src.model.transformer import TransformerBlock, _make_block_diagonal_mask


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

    def __init__(self,
                 image_size,
                 patch_size,
                 fun_dim: int = 3,
                 out_dim: int = 1,
                 n_layers: int = 5,
                 n_hidden: int = 256,
                 n_head: int = 8,
                 mlp_ratio: int = 4,
                 dropout: float = 0.0,
                 pos_embedding: str = 'sincos',
                 ):

        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        patch_dim = fun_dim * patch_height * patch_width

        self.ph, self.pw = patch_height, patch_width
        self.nh, self.nw = image_height // patch_height, image_width // patch_width

        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, n_hidden),
            nn.LayerNorm(n_hidden),
        )

        if pos_embedding == 'trainable':
            self.pos_embedding = nn.Parameter((1 / n_hidden) * torch.rand((1, self.nh * self.nw, n_hidden), dtype=torch.float))
        elif pos_embedding == 'sincos':
            self.pos_embedding = posemb_sincos_2d(h=self.nh, w=self.nw, dim=n_hidden)
        else:
            raise KeyError(f"Unknown pos_embedding: {pos_embedding}")

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=n_hidden,
                n_heads=n_head,
                d_ff=n_hidden * mlp_ratio,
                dropout=dropout,
                norm_first=True,
            )
            for _ in range(n_layers)
        ])

        self.last_layer = nn.Sequential(nn.LayerNorm(n_hidden), nn.Linear(n_hidden, self.ph * self.pw * out_dim))
        self.out_dim = out_dim

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        B, C, H, W = img.shape
        # Patchify -> [B, N, patch_dim]
        img = img.reshape(B, C, self.nh, self.ph, self.nw, self.pw).permute(0, 2, 4, 3, 5, 1).reshape(B, self.nh * self.nw, self.ph * self.pw * C)

        fx = self.to_patch_embedding(img)                            # [B, N, D]
        fx = fx + self.pos_embedding.to(fx.device, dtype=fx.dtype)   # [B, N, D]

        N = self.nh * self.nw
        fx = fx.reshape(B * N, -1)                                   # [B*N, D]
        attn_mask = _make_block_diagonal_mask([N] * B, fx.device)    # [B*N, B*N]

        for block in self.blocks:
            fx = block(fx, attn_mask=attn_mask)                      # [B*N, D]

        fx = fx.reshape(B, N, -1)
        fx = self.last_layer(fx)                                     # [B, N, ph*pw*out_dim]
        return fx.reshape(B, self.nh, self.nw, self.ph, self.pw, self.out_dim) \
                 .permute(0, 5, 1, 3, 2, 4) \
                 .reshape(B, self.out_dim, self.nh * self.ph, self.nw * self.pw)

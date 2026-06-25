"""
========================================================================
RefinementNet — learned per-pixel importance scorer for adaptive meshing.
========================================================================

A small U-Net-style CNN that maps a geometry grid [B, C_in, H, W] to a
per-pixel scalar map [B, 1, H, W] with an unbounded (raw) head. The map is
interpreted as a predicted target depth ``d_pred``: the depth-guided builder
subdivides a depth-``d`` cell iff ``max(d_pred over the cell) > d + offset``
(running-depth comparison). The scorer is trained by supervised regression of
``d_pred`` to the variance-oracle depth target (src/amr/oracle_depth.py); there is
no Gumbel sampling and no sign threshold.

The head is deliberately NOT squashed by a sigmoid: a sigmoid followed by
clamp + log made gradients vanish once the scorer saturated, which made
mesh collapse irreversible. Keeping it unbounded also lets the same output 
represent depths outside the [0, 1] range directly.

Architecture:

    enc1 → enc2 → enc3 → bot → up2 (+ enc2 skip) → up1 (+ enc1 skip) → head

Encoder halves spatial resolution at enc2 and enc3; the decoder restores
it with nearest-neighbor upsampling and concatenative skip connections.
GroupNorm(8, *) + ReLU throughout; raw 1x1 conv head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RefinementNet(nn.Module):
    def __init__(self, input_channels: int = 3):
        super().__init__()
        self.input_channels = input_channels

        self.enc1_conv = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.enc1_norm = nn.GroupNorm(8, 32)

        self.enc2_conv = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.enc2_norm = nn.GroupNorm(8, 64)

        self.enc3_conv = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.enc3_norm = nn.GroupNorm(8, 128)

        self.bot_conv = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bot_norm = nn.GroupNorm(8, 128)

        # up2: upsample bot (128) + concat enc2 (64) = 192 in-channels
        self.up2_conv = nn.Conv2d(192, 64, kernel_size=3, padding=1)
        self.up2_norm = nn.GroupNorm(8, 64)

        # up1: upsample up2 (64) + concat enc1 (32) = 96 in-channels
        self.up1_conv = nn.Conv2d(96, 32, kernel_size=3, padding=1)
        self.up1_norm = nn.GroupNorm(8, 32)

        self.head = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, H, W, C_in] float32 geometry grid (channel-last, the layout the
                rest of the pipeline holds grids in). Permuted to channel-first
                internally for the convolutions.
        Returns:
            score_map: [B, 1, H, W] raw (unbounded) scalar map, interpreted as
                the predicted target depth d_pred.
        """
        x = x.permute(0, 3, 1, 2).contiguous()                # [B, H, W, C] -> [B, C, H, W]
        e1 = F.relu(self.enc1_norm(self.enc1_conv(x)))        # [B, 32, H, W]
        e2 = F.relu(self.enc2_norm(self.enc2_conv(e1)))       # [B, 64, H/2, W/2]
        e3 = F.relu(self.enc3_norm(self.enc3_conv(e2)))       # [B, 128, H/4, W/4]

        b = F.relu(self.bot_norm(self.bot_conv(e3)))          # [B, 128, H/4, W/4]

        u2 = F.interpolate(b, scale_factor=2, mode="nearest") # [B, 128, H/2, W/2]
        u2 = torch.cat([u2, e2], dim=1)                       # [B, 192, H/2, W/2]
        u2 = F.relu(self.up2_norm(self.up2_conv(u2)))         # [B, 64, H/2, W/2]

        u1 = F.interpolate(u2, scale_factor=2, mode="nearest")# [B, 64, H, W]
        u1 = torch.cat([u1, e1], dim=1)                       # [B, 96, H, W]
        u1 = F.relu(self.up1_norm(self.up1_conv(u1)))         # [B, 32, H, W]

        score_map = self.head(u1)                             # [B, 1, H, W] logits
        return score_map


if __name__ == "__main__":
    import torch
    model = RefinementNet(input_channels=3)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"RefinementNet params: {n_params:,}")

    x = torch.randn(2, 256, 128, 3)   # channel-last [B, H, W, C]
    y = model(x)
    assert y.shape == (2, 1, 256, 128), f"shape mismatch: {y.shape}"
    assert torch.isfinite(y).all(), "non-finite logits"
    print(f"Forward OK. Logit range: [{y.min():.4f}, {y.max():.4f}]")

    # Gradient sanity
    loss = y.mean()
    loss.backward()
    grad_norm = sum((p.grad ** 2).sum().item() for p in model.parameters()) ** 0.5
    assert grad_norm > 0, "no gradient flow"
    print(f"Gradient OK. Grad norm: {grad_norm:.4f}")

from __future__ import annotations

"""
Conditional U-Net for DDPM with channel concatenation conditioning.

Instead of using FiLM modulation or cross-attention, this model conditions
by simply concatenating the exemplar images along the channel dimension.

Interface mirrors conditional_model.py but conditioning is via channel concat:
    cond: exemplar images tensor of shape (B, K, C, H, W) or (B, C, H, W)
          These are averaged over K and concatenated to the noisy input.
"""

import math
from typing import Iterable, Sequence, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Time embedding
# -------------------------

def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        torch.arange(half, device=timesteps.device, dtype=torch.float32)
        * -(math.log(10000.0) / (half - 1))
    )
    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


# -------------------------
# Building blocks
# -------------------------

class ResBlock(nn.Module):
    """Standard ResBlock with time conditioning only (no FiLM)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels),
        )

        self.norm2 = nn.GroupNorm(32, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.norm2(h)
        h = self.conv2(self.dropout(F.silu(h)))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        y = self.norm(x).view(b, c, h * w).transpose(1, 2)  # (b, hw, c)
        y, _ = self.attn(y, y, y)
        y = y.transpose(1, 2).view(b, c, h, w)
        return x + y


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


# -------------------------
# Conditional UNet with Channel Concatenation
# -------------------------

class ConcatConditionalUNet(nn.Module):
    """
    UNet with channel concatenation conditioning.
    
    The conditioning image(s) are averaged (if multiple) and concatenated
    to the noisy input along the channel dimension.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        channel_mults: Sequence[int] = (1, 2, 4),
        num_res_blocks: int = 2,
        dropout: float = 0.1,
        attn_resolutions: Iterable[int] = (16,),
        num_heads: int = 4,
        image_size: int = 32,
        time_scale: float = 1000.0,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.time_scale = time_scale
        self.in_channels = in_channels
        time_dim = base_channels * 4

        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Input conv: takes noisy image + conditioning image (2x channels)
        self.in_conv = nn.Conv2d(in_channels * 2, base_channels, 3, padding=1)

        downs = []
        skip_channels = []
        in_ch = base_channels
        curr_res = image_size
        for mult_idx, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                block = ResBlock(in_ch, out_ch, time_dim, dropout)
                downs.append(block)
                in_ch = out_ch
                skip_channels.append(in_ch)
                if curr_res in attn_resolutions:
                    downs.append(SelfAttention(in_ch, num_heads))
            if mult_idx != len(channel_mults) - 1:
                downs.append(Downsample(in_ch))
                curr_res //= 2
        self.downs = nn.ModuleList(downs)
        self.skip_channels = skip_channels

        self.mid = nn.ModuleList(
            [
                ResBlock(in_ch, in_ch, time_dim, dropout),
                SelfAttention(in_ch, num_heads),
                ResBlock(in_ch, in_ch, time_dim, dropout),
            ]
        )

        ups = []
        skip_stack = list(skip_channels)
        for layer in reversed(downs):
            if isinstance(layer, Downsample):
                ups.append(Upsample(in_ch))
                curr_res *= 2
            elif isinstance(layer, SelfAttention):
                ups.append(SelfAttention(in_ch, num_heads))
            elif isinstance(layer, ResBlock):
                if not skip_stack:
                    raise RuntimeError("Skip channel stack empty when building up path.")
                skip_ch = skip_stack.pop()
                ups.append(ResBlock(in_ch + skip_ch, skip_ch, time_dim, dropout))
                in_ch = skip_ch
            else:
                raise TypeError(f"Unexpected layer type in downs: {type(layer)}")
        self.ups = nn.ModuleList(ups)

        self.out_norm = nn.GroupNorm(32, in_ch)
        self.out_conv = nn.Conv2d(in_ch, in_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with channel concatenation conditioning.
        
        Args:
            x: noisy images (B, C, H, W)
            timesteps: diffusion timesteps (B,)
            cond: exemplar images (B, K, C, H, W) or (B, C, H, W) or (K, C, H, W)
                  If None, zeros are used as neutral conditioning.
        """
        B = x.shape[0]
        
        if cond is None:
            # Use zeros as neutral conditioning
            cond_img = torch.zeros_like(x)
        else:
            # Handle different conditioning shapes
            if cond.dim() == 4:
                # (B, C, H, W) or (K, C, H, W)
                if cond.shape[0] == B:
                    # (B, C, H, W) - one cond per sample
                    cond_img = cond
                else:
                    # (K, C, H, W) - K exemplars, broadcast to batch
                    cond_img = cond.mean(dim=0, keepdim=True).expand(B, -1, -1, -1)
            elif cond.dim() == 5:
                # (B, K, C, H, W) or (1, K, C, H, W)
                if cond.shape[0] == B:
                    # (B, K, C, H, W) - average over K exemplars
                    cond_img = cond.mean(dim=1)
                else:
                    # (1, K, C, H, W) - average and broadcast
                    cond_img = cond.mean(dim=1).expand(B, -1, -1, -1)
            else:
                raise ValueError(f"Unexpected cond shape: {cond.shape}")

        # Concatenate noisy image and conditioning image along channel dimension
        x_concat = torch.cat([x, cond_img], dim=1)  # (B, 2*C, H, W)

        t_emb = self.time_mlp(sinusoidal_embedding(timesteps * self.time_scale, self.time_mlp[0].in_features))

        hs = []
        h = self.in_conv(x_concat)
        for layer in self.downs:
            if isinstance(layer, ResBlock):
                h = layer(h, t_emb)
                hs.append(h)
            elif isinstance(layer, SelfAttention):
                h = layer(h)
            else:
                h = layer(h)

        for layer in self.mid:
            if isinstance(layer, ResBlock):
                h = layer(h, t_emb)
            else:
                h = layer(h)

        for layer in self.ups:
            if isinstance(layer, ResBlock):
                skip = hs.pop()
                h = torch.cat([h, skip], dim=1)
                h = layer(h, t_emb)
            elif isinstance(layer, SelfAttention):
                h = layer(h)
            else:
                h = layer(h)

        h = F.silu(self.out_norm(h))
        return self.out_conv(h)


# -------------------------
# DDPM wrapper
# -------------------------

class DDPM(nn.Module):
    """
    Continuous-time DDPM with v-prediction parameterization.
    Uses Kingma VDM parameterization with log-SNR schedule.
    """

    def __init__(
        self,
        model: ConcatConditionalUNet,
        log_snr_max: float = 5.0,
        log_snr_min: float = -15.0,
    ) -> None:
        super().__init__()
        self.model = model
        self.log_snr_max = log_snr_max  # log-SNR at t=0 (high SNR, clean image)
        self.log_snr_min = log_snr_min  # log-SNR at t=1 (low SNR, pure noise)

    def _log_snr(self, t: torch.Tensor) -> torch.Tensor:
        """Linear interpolation of log-SNR from max to min."""
        return self.log_snr_max + t * (self.log_snr_min - self.log_snr_max)

    def _alpha_sigma(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute α(t) and σ(t) from log-SNR using variance-preserving constraint.
        α² + σ² = 1, SNR = α²/σ²
        """
        log_snr = self._log_snr(t)
        alpha_sq = torch.sigmoid(log_snr)
        sigma_sq = torch.sigmoid(-log_snr)
        return torch.sqrt(alpha_sq), torch.sqrt(sigma_sq)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x0)
        alpha, sigma = self._alpha_sigma(t)
        x_t = alpha[:, None, None, None] * x0 + sigma[:, None, None, None] * noise
        return x_t, alpha, sigma

    def p_losses(self, x0: torch.Tensor, t: torch.Tensor | None = None, noise: torch.Tensor | None = None, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        if t is None:
            t = torch.rand(x0.shape[0], device=x0.device)
        if noise is None:
            noise = torch.randn_like(x0)
        x_t, alpha, sigma = self.q_sample(x0, t, noise)
        v_target = alpha[:, None, None, None] * noise - sigma[:, None, None, None] * x0
        v_pred = self.model(x_t, t, cond)
        return F.mse_loss(v_pred, v_target)

    @torch.no_grad()
    def _predict_eps_x0(self, x_t: torch.Tensor, t: torch.Tensor, v_pred: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        alpha, sigma = self._alpha_sigma(t)
        alpha = alpha[:, None, None, None]
        sigma = sigma[:, None, None, None]
        x0_pred = alpha * x_t - sigma * v_pred
        eps_pred = alpha * v_pred + sigma * x_t
        return alpha, sigma, x0_pred, eps_pred

    @torch.no_grad()
    def sample(self, batch_size: int, shape: Tuple[int, int, int], device: torch.device, steps: int | None = None, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        steps = steps or 50
        img = torch.randn(batch_size, *shape, device=device)
        times = torch.linspace(1.0, 0.0, steps + 1, device=device)
        for i in range(steps):
            t_cur = times[i].repeat(batch_size)
            t_prev = times[i + 1].repeat(batch_size)

            v_pred = self.model(img, t_cur, cond)
            alpha_cur, sigma_cur, x0_pred, eps_pred = self._predict_eps_x0(img, t_cur, v_pred)

            alpha_prev, sigma_prev = self._alpha_sigma(t_prev)
            alpha_prev = alpha_prev[:, None, None, None]
            sigma_prev = sigma_prev[:, None, None, None]
            if i == steps - 1:
                img = x0_pred
            else:
                img = alpha_prev * x0_pred + sigma_prev * eps_pred
        return img

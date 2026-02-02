from __future__ import annotations

"""
Class-Conditional U-Net for DDPM with Classifier-Free Guidance (CFG).

Uses learned class embeddings with FiLM modulation, reusing building blocks
from conditional_model.py. Supports CFG training with random class dropout.
"""

import math
from typing import Iterable, Optional, Sequence, Tuple

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

class ResBlockCond(nn.Module):
    """ResBlock with FiLM conditioning from class embedding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        cond_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels),
        )

        # FiLM from class embedding
        self.cond_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, out_channels * 2),  # gamma and beta
        )

        self.norm2 = nn.GroupNorm(32, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_mlp(t_emb)[:, :, None, None]

        # FiLM modulation
        gamma, beta = self.cond_mlp(cond).chunk(2, dim=1)
        h = self.norm2(h)
        h = h * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]
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
# Class-Conditional UNet
# -------------------------

class ClassCondUNet(nn.Module):
    """
    UNet with learned class embeddings for classifier-free guidance.
    Uses FiLM conditioning in ResBlocks.
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
        num_classes: int = 100,
        cond_dim: int = 256,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.time_scale = time_scale
        self.num_classes = num_classes
        time_dim = base_channels * 4
        self.cond_dim = cond_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Class embedding: num_classes + 1 for null class (CFG)
        self.class_embed = nn.Embedding(num_classes + 1, cond_dim)
        self.null_class_id = num_classes  # Use last ID as null class

        self.in_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        downs = []
        skip_channels = []
        in_ch = base_channels
        curr_res = image_size
        for mult_idx, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                block = ResBlockCond(in_ch, out_ch, time_dim, cond_dim, dropout)
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
                ResBlockCond(in_ch, in_ch, time_dim, cond_dim, dropout),
                SelfAttention(in_ch, num_heads),
                ResBlockCond(in_ch, in_ch, time_dim, cond_dim, dropout),
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
            elif isinstance(layer, ResBlockCond):
                if not skip_stack:
                    raise RuntimeError("Skip channel stack empty when building up path.")
                skip_ch = skip_stack.pop()
                ups.append(ResBlockCond(in_ch + skip_ch, skip_ch, time_dim, cond_dim, dropout))
                in_ch = skip_ch
            else:
                raise TypeError(f"Unexpected layer type in downs: {type(layer)}")
        self.ups = nn.ModuleList(ups)

        self.out_norm = nn.GroupNorm(32, in_ch)
        self.out_conv = nn.Conv2d(in_ch, in_channels, 3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        class_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) noisy images
            timesteps: (B,) diffusion timesteps in [0, 1]
            class_ids: (B,) class labels. If None, uses null class (unconditional)
        """
        if class_ids is None:
            class_ids = torch.full((x.shape[0],), self.null_class_id, device=x.device, dtype=torch.long)
        
        cond_vec = self.class_embed(class_ids)  # (B, cond_dim)

        t_emb = self.time_mlp(sinusoidal_embedding(timesteps * self.time_scale, self.time_mlp[0].in_features))

        hs = []
        h = self.in_conv(x)
        for layer in self.downs:
            if isinstance(layer, ResBlockCond):
                h = layer(h, t_emb, cond_vec)
                hs.append(h)
            elif isinstance(layer, SelfAttention):
                h = layer(h)
            else:
                h = layer(h)

        for layer in self.mid:
            if isinstance(layer, ResBlockCond):
                h = layer(h, t_emb, cond_vec)
            else:
                h = layer(h)

        for layer in self.ups:
            if isinstance(layer, ResBlockCond):
                skip = hs.pop()
                h = torch.cat([h, skip], dim=1)
                h = layer(h, t_emb, cond_vec)
            elif isinstance(layer, SelfAttention):
                h = layer(h)
            else:
                h = layer(h)

        h = F.silu(self.out_norm(h))
        return self.out_conv(h)


# -------------------------
# DDPM wrapper with CFG support
# -------------------------

class DDPM(nn.Module):
    """
    Continuous-time DDPM with v-prediction and classifier-free guidance.
    """

    def __init__(
        self,
        model: ClassCondUNet,
        beta_start: float,
        beta_end: float,
        train_steps: int = 1000,
        cfg_drop_prob: float = 0.1,
    ) -> None:
        super().__init__()
        self.model = model
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.train_steps = train_steps
        self.cfg_drop_prob = cfg_drop_prob

    @property
    def num_timesteps(self) -> int:
        return self.train_steps

    def _alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        T = float(self.train_steps)
        return torch.exp(-self.beta_start * T * t - 0.5 * (self.beta_end - self.beta_start) * T * t * t)

    def _alpha_sigma(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        alpha_bar = self._alpha_bar(t)
        alpha = torch.sqrt(alpha_bar)
        sigma = torch.sqrt(1.0 - alpha_bar)
        return alpha, sigma

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x0)
        alpha, sigma = self._alpha_sigma(t)
        x_t = alpha[:, None, None, None] * x0 + sigma[:, None, None, None] * noise
        return x_t, alpha, sigma

    def p_losses(
        self,
        x0: torch.Tensor,
        class_ids: torch.Tensor,
        t: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute training loss with random class dropout for CFG."""
        if t is None:
            t = torch.rand(x0.shape[0], device=x0.device)
        if noise is None:
            noise = torch.randn_like(x0)
        
        # Random class dropout for CFG training
        if self.training and self.cfg_drop_prob > 0:
            drop_mask = torch.rand(x0.shape[0], device=x0.device) < self.cfg_drop_prob
            class_ids = class_ids.clone()
            class_ids[drop_mask] = self.model.null_class_id
        
        x_t, alpha, sigma = self.q_sample(x0, t, noise)
        v_target = alpha[:, None, None, None] * noise - sigma[:, None, None, None] * x0
        v_pred = self.model(x_t, t, class_ids)
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
    def sample(
        self,
        class_ids: torch.Tensor,
        shape: Tuple[int, int, int],
        device: torch.device,
        steps: int | None = None,
        cfg_scale: float = 1.0,
    ) -> torch.Tensor:
        """Sample with classifier-free guidance."""
        steps = steps or 50
        batch_size = class_ids.shape[0]
        img = torch.randn(batch_size, *shape, device=device)
        times = torch.linspace(1.0, 0.0, steps + 1, device=device)
        
        for i in range(steps):
            t_cur = times[i].repeat(batch_size)
            t_prev = times[i + 1].repeat(batch_size)

            if cfg_scale != 1.0:
                # CFG: compute both conditional and unconditional predictions
                v_cond = self.model(img, t_cur, class_ids)
                v_uncond = self.model(img, t_cur, None)  # Uses null class
                v_pred = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                v_pred = self.model(img, t_cur, class_ids)

            alpha_cur, sigma_cur, x0_pred, eps_pred = self._predict_eps_x0(img, t_cur, v_pred)

            alpha_prev, sigma_prev = self._alpha_sigma(t_prev)
            alpha_prev = alpha_prev[:, None, None, None]
            sigma_prev = sigma_prev[:, None, None, None]
            img = alpha_prev * x0_pred + sigma_prev * eps_pred
        return img

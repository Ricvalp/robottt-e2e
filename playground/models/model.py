from __future__ import annotations

import math
from typing import Iterable, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Create sinusoidal timestep embeddings (same as in DDPM/Transformer)."""
    half = dim // 2
    freqs = torch.exp(
        torch.arange(half, device=timesteps.device, dtype=torch.float32)
        * -(math.log(10000.0) / (half - 1))
    )
    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:  # zero pad if dim is odd
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock(nn.Module):
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
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
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


class UNet(nn.Module):
    """
    Minimal U-Net for DDPM, loosely following Ho et al. 2020 and OpenAI guided-diffusion.
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
        time_dim = base_channels * 4

        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.in_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Down path
        downs = []
        skip_channels = []  # channels for skip connections (ResBlocks only)
        in_ch = base_channels
        curr_res = image_size
        for mult_idx, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                block = ResBlock(in_ch, out_ch, time_dim, dropout)
                downs.append(block)
                in_ch = out_ch
                skip_channels.append(in_ch)  # ResBlock contributes a skip
                if curr_res in attn_resolutions:
                    attn = SelfAttention(in_ch, num_heads)
                    downs.append(attn)  # attention, no skip
            if mult_idx != len(channel_mults) - 1:  # no downsample on final stage
                down = Downsample(in_ch)
                downs.append(down)  # downsample, no skip
                curr_res //= 2
        self.downs = nn.ModuleList(downs)
        self.skip_channels = skip_channels

        # Bottleneck
        self.mid = nn.ModuleList(
            [
                ResBlock(in_ch, in_ch, time_dim, dropout),
                SelfAttention(in_ch, num_heads),
                ResBlock(in_ch, in_ch, time_dim, dropout),
            ]
        )

        # Up path mirrors downs (pop skip_channels for ResBlocks only)
        ups = []
        skip_stack = list(skip_channels)  # copy to pop from end
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

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input images in [-1, 1], shape (B, C, H, W)
            timesteps: integer timesteps in [0, T)
        Returns:
            predicted noise of shape (B, C, H, W)
        """
        t_emb = self.time_mlp(sinusoidal_embedding(timesteps * self.time_scale, self.time_mlp[0].in_features))

        hs = []
        h = self.in_conv(x)
        for layer in self.downs:
            if isinstance(layer, ResBlock):
                h = layer(h, t_emb)
                hs.append(h)  # store skip only for ResBlocks
            elif isinstance(layer, SelfAttention):
                h = layer(h)
            else:  # Downsample
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
            else:
                h = layer(h)

        h = F.silu(self.out_norm(h))
        return self.out_conv(h)


class DDPM(nn.Module):
    """
    Continuous-time DDPM with v-prediction parameterization.
    """

    def __init__(
        self,
        model: UNet,
        beta_start: float,
        beta_end: float,
        train_steps: int = 1000,
    ) -> None:
        super().__init__()
        self.model = model
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.train_steps = train_steps  # used for time scaling only

    @property
    def num_timesteps(self) -> int:
        return self.train_steps

    def _alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        """
        Continuous alpha_bar(t) for linear beta schedule on t in [0, 1],
        scaled to match a discrete schedule of `train_steps` steps.
        beta(s) = beta_start + (beta_end - beta_start) * s / train_steps for s in [0, train_steps]
        Integral 0->t*train_steps of beta(s) ds = beta_start * T * t + 0.5 * (beta_end - beta_start) * T * t^2
        """
        T = float(self.train_steps)
        return torch.exp(-self.beta_start * T * t - 0.5 * (self.beta_end - self.beta_start) * T * t * t)

    def _alpha_sigma(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        alpha_bar = self._alpha_bar(t)
        alpha = torch.sqrt(alpha_bar)
        sigma = torch.sqrt(1.0 - alpha_bar)
        return alpha, sigma

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns x_t, alpha, sigma for given continuous times t in [0,1].
        """
        if noise is None:
            noise = torch.randn_like(x0)
        alpha, sigma = self._alpha_sigma(t)
        x_t = alpha[:, None, None, None] * x0 + sigma[:, None, None, None] * noise
        return x_t, alpha, sigma

    def p_losses(self, x0: torch.Tensor, t: torch.Tensor | None = None, noise: torch.Tensor | None = None) -> torch.Tensor:
        """
        v-prediction loss with continuous timesteps sampled uniformly in [0,1].
        """
        if t is None:
            t = torch.rand(x0.shape[0], device=x0.device)
        if noise is None:
            noise = torch.randn_like(x0)

        x_t, alpha, sigma = self.q_sample(x0, t, noise)
        v_target = alpha[:, None, None, None] * noise - sigma[:, None, None, None] * x0
        v_pred = self.model(x_t, t)
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
    def sample(self, batch_size: int, shape: Tuple[int, int, int], device: torch.device, steps: int | None = None) -> torch.Tensor:
        steps = steps or 50
        img = torch.randn(batch_size, *shape, device=device)
        # Create a decreasing time schedule from 1 -> 0
        times = torch.linspace(1.0, 0.0, steps + 1, device=device)
        for i in range(steps):
            t_cur = times[i].repeat(batch_size)
            t_prev = times[i + 1].repeat(batch_size)

            v_pred = self.model(img, t_cur)
            alpha_cur, sigma_cur, x0_pred, eps_pred = self._predict_eps_x0(img, t_cur, v_pred)

            alpha_prev, sigma_prev = self._alpha_sigma(t_prev)
            alpha_prev = alpha_prev[:, None, None, None]
            sigma_prev = sigma_prev[:, None, None, None]

            # DDIM-style deterministic update (eta=0)
            img = alpha_prev * x0_pred + sigma_prev * eps_pred
        return img

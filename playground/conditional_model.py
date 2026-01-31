from __future__ import annotations

"""
Conditional U-Net for DDPM with:
- FiLM modulation in ResBlocks driven by exemplar images
- Optional cross-attention that lets spatial features attend to exemplar tokens

Interface mirrors playground/model.py but adds `cond` argument to forward:
    cond: exemplar images tensor of shape (B, K, C, H, W) or (B, C, H, W)
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

class ResBlockCond(nn.Module):
    """ResBlock with FiLM conditioning."""

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

        # FiLM from conditioning vector
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


class CrossAttention(nn.Module):
    """Cross-attend UNet features (queries) to exemplar tokens (keys/values)."""

    def __init__(self, channels: int, cond_dim: int, num_heads: int) -> None:
        super().__init__()
        self.norm_x = nn.GroupNorm(32, channels)
        self.q_proj = nn.Linear(channels, channels, bias=False)
        self.k_proj = nn.Linear(cond_dim, channels, bias=False)
        self.v_proj = nn.Linear(cond_dim, channels, bias=False)
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        assert channels % num_heads == 0, "channels must be divisible by num_heads"

        self.out = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor, cond_tokens: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        cond_tokens: (B, S, cond_dim)  exemplar tokens
        """
        b, c, h, w = x.shape
        x_norm = self.norm_x(x).view(b, c, h * w).transpose(1, 2)  # (B, HW, C)

        q = self.q_proj(x_norm)  # (B, HW, C)
        k = self.k_proj(cond_tokens)  # (B, S, C)
        v = self.v_proj(cond_tokens)  # (B, S, C)

        def reshape_heads(t):
            return t.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = map(reshape_heads, (q, k, v))  # (B, heads, seq, dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, heads, HW, S)
        attn = attn_scores.softmax(dim=-1)
        out = torch.matmul(attn, v)  # (B, heads, HW, dim)

        out = out.transpose(1, 2).contiguous().view(b, h * w, c)
        out = self.out(out)
        out = out.transpose(1, 2).view(b, c, h, w)
        return x + out


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
# Conditioning encoder
# -------------------------

class ExemplarEncoder(nn.Module):
    """Tiny CNN encoder to turn exemplar images into a single conditioning vector."""

    def __init__(self, in_channels: int, hidden: int, cond_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, stride=2),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(hidden, cond_dim)

    def forward(self, exemplars: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        exemplars: (B, K, C, H, W) or (B, C, H, W)
        Returns:
            cond_vec: (B, cond_dim)
            tokens: (B, K, cond_dim)  per-exemplar tokens for cross-attn
        """
        if exemplars.dim() == 4:  # (B, C, H, W) -> add K=1
            exemplars = exemplars.unsqueeze(1)
        b, k, c, h, w = exemplars.shape
        flat = exemplars.view(b * k, c, h, w)
        feats = self.net(flat).view(b * k, -1)  # (B*K, hidden)
        tokens = self.proj(feats).view(b, k, -1)  # (B, K, cond_dim)
        cond_vec = tokens.mean(dim=1)
        return cond_vec, tokens


# -------------------------
# Conditional UNet
# -------------------------

class ConditionalUNet(nn.Module):
    """
    UNet with FiLM conditioning and optional cross-attention to exemplar tokens.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        channel_mults: Sequence[int] = (1, 2, 4),
        num_res_blocks: int = 2,
        dropout: float = 0.1,
        attn_resolutions: Iterable[int] = (16,),
        cross_attn_resolutions: Iterable[int] = (32,),  # apply cross-attn at highest res by default
        num_heads: int = 4,
        image_size: int = 32,
        time_scale: float = 1000.0,
        cond_dim: int = 256,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.time_scale = time_scale
        time_dim = base_channels * 4
        self.cond_dim = cond_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.cond_encoder = ExemplarEncoder(in_channels, hidden=base_channels, cond_dim=cond_dim)

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
                if curr_res in cross_attn_resolutions:
                    downs.append(CrossAttention(in_ch, cond_dim, num_heads))
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
            elif isinstance(layer, CrossAttention):
                ups.append(CrossAttention(in_ch, cond_dim, num_heads))
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

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        cond: exemplar images (B, K, C, H, W) or (B, C, H, W); if None, zeros are used.
        """
        if cond is None:
            # use zeros as neutral conditioning
            cond_vec = x.new_zeros(x.shape[0], self.cond_dim)
            cond_tokens = cond_vec.unsqueeze(1)
        else:
            cond_vec, cond_tokens = self.cond_encoder(cond)
            # If conditioning batch size differs from x batch, broadcast the mean embedding
            if cond_vec.shape[0] != x.shape[0]:
                cond_vec = cond_vec.mean(dim=0, keepdim=True).repeat(x.shape[0], 1)
                cond_tokens = cond_tokens.mean(dim=1, keepdim=True).repeat(x.shape[0], 1, 1)

        t_emb = self.time_mlp(sinusoidal_embedding(timesteps * self.time_scale, self.time_mlp[0].in_features))

        hs = []
        h = self.in_conv(x)
        for layer in self.downs:
            if isinstance(layer, ResBlockCond):
                h = layer(h, t_emb, cond_vec)
                hs.append(h)
            elif isinstance(layer, SelfAttention):
                h = layer(h)
            elif isinstance(layer, CrossAttention):
                h = layer(h, cond_tokens)
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
            elif isinstance(layer, CrossAttention):
                h = layer(h, cond_tokens)
            else:
                h = layer(h)

        h = F.silu(self.out_norm(h))
        return self.out_conv(h)


# -------------------------
# DDPM wrapper (unchanged except for cond threading)
# -------------------------

class DDPM(nn.Module):
    """
    Continuous-time DDPM with v-prediction parameterization.
    """

    def __init__(
        self,
        model: ConditionalUNet,
        beta_start: float,
        beta_end: float,
        train_steps: int = 1000,
    ) -> None:
        super().__init__()
        self.model = model
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.train_steps = train_steps

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
            img = alpha_prev * x0_pred + sigma_prev * eps_pred
        return img

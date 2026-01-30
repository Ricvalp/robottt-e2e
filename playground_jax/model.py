from __future__ import annotations

import jax.numpy as jnp
import jax
from flax import linen as nn
import math
from typing import Sequence, Iterable


def sinusoidal_embedding(timesteps: jnp.ndarray, dim: int) -> jnp.ndarray:
    half = dim // 2
    freqs = jnp.exp(-math.log(10000.0) * jnp.arange(half) / (half - 1))
    args = timesteps[:, None] * freqs[None]
    emb = jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)
    if dim % 2 == 1:
        emb = jnp.pad(emb, ((0, 0), (0, 1)))
    return emb


class ResBlock(nn.Module):
    in_channels: int
    out_channels: int
    time_dim: int
    dropout: float

    @nn.compact
    def __call__(self, x, t_emb):
        h = x
        h = nn.GroupNorm(num_groups=32)(h)
        h = nn.silu(h)
        h = nn.Conv(self.out_channels, kernel_size=(3, 3), padding="SAME")(h)

        t_proj = nn.Dense(self.out_channels)(nn.silu(t_emb))
        h = h + t_proj[:, None, None, :]

        h = nn.GroupNorm(num_groups=32)(h)
        h = nn.silu(h)
        h = nn.Dropout(self.dropout)(h, deterministic=False)
        h = nn.Conv(self.out_channels, kernel_size=(3, 3), padding="SAME")(h)

        skip = x
        if self.in_channels != self.out_channels:
            skip = nn.Conv(self.out_channels, kernel_size=(1, 1))(skip)
        return h + skip


class SelfAttention(nn.Module):
    channels: int
    num_heads: int

    @nn.compact
    def __call__(self, x):
        b, h, w, c = x.shape
        y = nn.GroupNorm(num_groups=32)(x)
        y = y.reshape((b, h * w, c))
        y = nn.SelfAttention(num_heads=self.num_heads, qkv_features=c)(y)
        y = y.reshape((b, h, w, c))
        return x + y


class UNet(nn.Module):
    in_channels: int = 1
    base_channels: int = 64
    channel_mults: Sequence[int] = (1, 2, 4)
    num_res_blocks: int = 2
    dropout: float = 0.0
    attn_resolutions: Iterable[int] = (16,)
    num_heads: int = 4
    image_size: int = 32
    time_scale: float = 1000.0

    @nn.compact
    def __call__(self, x, timesteps):
        time_dim = self.base_channels * 4
        t_emb = sinusoidal_embedding(timesteps * self.time_scale, self.base_channels)
        t_emb = nn.Dense(time_dim)(t_emb)
        t_emb = nn.silu(t_emb)
        t_emb = nn.Dense(time_dim)(t_emb)

        hs = []
        h = nn.Conv(self.base_channels, kernel_size=(3, 3), padding="SAME")(x)
        in_ch = self.base_channels
        res = self.image_size
        # Down
        for mult in self.channel_mults:
            out_ch = self.base_channels * mult
            for _ in range(self.num_res_blocks):
                h = ResBlock(in_ch, out_ch, time_dim, self.dropout)(h, t_emb)
                in_ch = out_ch
                if res in self.attn_resolutions:
                    h = SelfAttention(in_ch, self.num_heads)(h)
                hs.append(h)
            if mult != self.channel_mults[-1]:
                h = nn.Conv(in_ch, kernel_size=(3, 3), strides=(2, 2), padding="SAME")(h)
                res //= 2
                hs.append(h)

        # Mid
        h = ResBlock(in_ch, in_ch, time_dim, self.dropout)(h, t_emb)
        h = SelfAttention(in_ch, self.num_heads)(h)
        h = ResBlock(in_ch, in_ch, time_dim, self.dropout)(h, t_emb)

        # Up
        for mult in reversed(self.channel_mults):
            out_ch = self.base_channels * mult
            for _ in range(self.num_res_blocks):
                skip = hs.pop()
                h = jnp.concatenate([h, skip], axis=-1)
                h = ResBlock(h.shape[-1], out_ch, time_dim, self.dropout)(h, t_emb)
                in_ch = out_ch
                if res in self.attn_resolutions:
                    h = SelfAttention(in_ch, self.num_heads)(h)
            if mult != self.channel_mults[0]:
                h = nn.ConvTranspose(in_ch, kernel_size=(4, 4), strides=(2, 2), padding="SAME")(h)
                res *= 2

        h = nn.GroupNorm(num_groups=32)(h)
        h = nn.silu(h)
        h = nn.Conv(self.in_channels, kernel_size=(3, 3), padding="SAME")(h)
        return h


class DDPM(nn.Module):
    beta_start: float
    beta_end: float
    train_steps: int
    model_def: nn.Module

    def setup(self):
        self.model = self.model_def

    def _alpha_sigma(self, t):
        T = float(self.train_steps)
        alpha_bar = jnp.exp(-self.beta_start * T * t - 0.5 * (self.beta_end - self.beta_start) * T * t * t)
        alpha = jnp.sqrt(alpha_bar)
        sigma = jnp.sqrt(1.0 - alpha_bar)
        return alpha, sigma

    def q_sample(self, x0, t, noise):
        alpha, sigma = self._alpha_sigma(t)
        return alpha[:, None, None, None] * x0 + sigma[:, None, None, None] * noise

    def loss(self, params, x0, t, rng):
        noise = jax.random.normal(rng, x0.shape)
        alpha, sigma = self._alpha_sigma(t)
        x_t = alpha[:, None, None, None] * x0 + sigma[:, None, None, None] * noise
        v_target = alpha[:, None, None, None] * noise - sigma[:, None, None, None] * x0
        v_pred = self.model.apply({"params": params}, x_t, t)
        return jnp.mean((v_pred - v_target) ** 2)

"""
Conditional U-Net for DDPM in JAX/Flax with:
- FiLM modulation in ResBlocks driven by exemplar images
- Optional cross-attention that lets spatial features attend to exemplar tokens
"""
from __future__ import annotations

import math
from typing import Sequence, Tuple, Optional

import jax
import jax.numpy as jnp
import flax.linen as nn


def sinusoidal_embedding(timesteps: jnp.ndarray, dim: int) -> jnp.ndarray:
    """Create sinusoidal timestep embeddings."""
    half = dim // 2
    freqs = jnp.exp(
        jnp.arange(half, dtype=jnp.float32)
        * -(math.log(10000.0) / (half - 1))
    )
    args = timesteps[:, None].astype(jnp.float32) * freqs[None, :]
    emb = jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=1)
    if dim % 2 == 1:
        emb = jnp.pad(emb, ((0, 0), (0, 1)))
    return emb


class ResBlockCond(nn.Module):
    """Residual block with time conditioning and FiLM conditioning."""
    out_channels: int
    time_emb_dim: int
    cond_dim: int
    dropout: float = 0.1

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        t_emb: jnp.ndarray,
        cond: jnp.ndarray,
        train: bool = True,
    ) -> jnp.ndarray:
        in_channels = x.shape[-1]
        
        # First conv
        h = nn.GroupNorm(num_groups=32)(x)
        h = nn.silu(h)
        h = nn.Conv(self.out_channels, kernel_size=(3, 3), padding='SAME')(h)
        
        # Time embedding injection
        time_out = nn.Dense(self.out_channels)(nn.silu(t_emb))
        h = h + time_out[:, None, None, :]
        
        # FiLM modulation from conditioning
        cond_out = nn.Dense(self.out_channels * 2)(nn.silu(cond))
        gamma, beta = jnp.split(cond_out, 2, axis=-1)
        
        h = nn.GroupNorm(num_groups=32)(h)
        h = h * (1 + gamma[:, None, None, :]) + beta[:, None, None, :]
        h = nn.silu(h)
        h = nn.Dropout(rate=self.dropout, deterministic=not train)(h)
        h = nn.Conv(self.out_channels, kernel_size=(3, 3), padding='SAME')(h)
        
        # Skip connection
        if in_channels != self.out_channels:
            x = nn.Conv(self.out_channels, kernel_size=(1, 1))(x)
        
        return h + x


class SelfAttention(nn.Module):
    """Multi-head self-attention with GroupNorm."""
    num_heads: int = 4

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        b, h, w, c = x.shape
        
        y = nn.GroupNorm(num_groups=32)(x)
        y = y.reshape(b, h * w, c)
        y = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=c,
            out_features=c,
        )(y, y)
        y = y.reshape(b, h, w, c)
        
        return x + y


class CrossAttention(nn.Module):
    """Cross-attend UNet features (queries) to exemplar tokens (keys/values)."""
    channels: int
    cond_dim: int
    num_heads: int = 4

    @nn.compact
    def __call__(self, x: jnp.ndarray, cond_tokens: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x: (B, H, W, C) - UNet features
            cond_tokens: (B, S, cond_dim) - exemplar tokens
        """
        b, h, w, c = x.shape
        
        # Normalize and reshape to sequence
        x_norm = nn.GroupNorm(num_groups=32)(x)
        x_seq = x_norm.reshape(b, h * w, c)
        
        # Project queries from x, keys/values from conditioning
        q = nn.Dense(self.channels, use_bias=False)(x_seq)
        k = nn.Dense(self.channels, use_bias=False)(cond_tokens)
        v = nn.Dense(self.channels, use_bias=False)(cond_tokens)
        
        # Multi-head attention
        head_dim = self.channels // self.num_heads
        
        def reshape_heads(t):
            return t.reshape(b, -1, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        
        q = reshape_heads(q)  # (B, heads, HW, dim)
        k = reshape_heads(k)  # (B, heads, S, dim)
        v = reshape_heads(v)  # (B, heads, S, dim)
        
        attn_scores = jnp.einsum('bhqd,bhkd->bhqk', q, k) / math.sqrt(head_dim)
        attn = jax.nn.softmax(attn_scores, axis=-1)
        out = jnp.einsum('bhqk,bhkd->bhqd', attn, v)
        
        # Reshape back
        out = out.transpose(0, 2, 1, 3).reshape(b, h * w, self.channels)
        out = nn.Dense(self.channels)(out)
        out = out.reshape(b, h, w, c)
        
        return x + out


class Downsample(nn.Module):
    """Spatial downsampling with strided conv."""
    channels: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return nn.Conv(self.channels, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(x)


class Upsample(nn.Module):
    """Spatial upsampling with nearest neighbor + conv."""
    channels: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        b, h, w, c = x.shape
        x = jax.image.resize(x, shape=(b, h * 2, w * 2, c), method='nearest')
        return nn.Conv(self.channels, kernel_size=(3, 3), padding='SAME')(x)


class ExemplarEncoder(nn.Module):
    """CNN encoder to turn exemplar images into conditioning vectors."""
    hidden: int
    cond_dim: int

    @nn.compact
    def __call__(self, exemplars: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Args:
            exemplars: (B, K, H, W, C) or (B, H, W, C)
        Returns:
            cond_vec: (B, cond_dim) - mean conditioning vector
            tokens: (B, K, cond_dim) - per-exemplar tokens for cross-attention
        """
        # Handle single exemplar case
        if exemplars.ndim == 4:
            exemplars = exemplars[:, None, :, :, :]  # Add K=1 dimension
        
        b, k, h, w, c = exemplars.shape
        
        # Flatten batch and K dims for processing
        flat = exemplars.reshape(b * k, h, w, c)
        
        # CNN encoder
        x = nn.Conv(self.hidden, kernel_size=(3, 3), padding='SAME')(flat)
        x = nn.silu(x)
        x = nn.Conv(self.hidden, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(x)
        x = nn.silu(x)
        x = nn.Conv(self.hidden, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(x)
        x = nn.silu(x)
        
        # Global average pooling
        x = jnp.mean(x, axis=(1, 2))  # (B*K, hidden)
        
        # Project to cond_dim
        tokens = nn.Dense(self.cond_dim)(x)  # (B*K, cond_dim)
        tokens = tokens.reshape(b, k, -1)  # (B, K, cond_dim)
        
        # Mean over exemplars for FiLM conditioning
        cond_vec = jnp.mean(tokens, axis=1)  # (B, cond_dim)
        
        return cond_vec, tokens


class ConditionalUNet(nn.Module):
    """
    UNet with FiLM conditioning and optional cross-attention to exemplar tokens.
    Uses NHWC format.
    """
    in_channels: int = 1
    base_channels: int = 64
    channel_mults: Sequence[int] = (1, 2, 4)
    num_res_blocks: int = 2
    dropout: float = 0.1
    attn_resolutions: Sequence[int] = (16,)
    cross_attn_resolutions: Sequence[int] = (32,)
    num_heads: int = 4
    image_size: int = 32
    time_scale: float = 1000.0
    cond_dim: int = 256

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        timesteps: jnp.ndarray,
        cond: Optional[jnp.ndarray] = None,
        train: bool = True,
    ) -> jnp.ndarray:
        """
        Args:
            x: Input images (B, H, W, C)
            timesteps: Timesteps (B,)
            cond: Exemplar images (B, K, H, W, C) or (B, H, W, C), or None
            train: Training mode
        Returns:
            Predicted v (B, H, W, C)
        """
        time_dim = self.base_channels * 4
        
        # Conditioning
        if cond is None:
            cond_vec = jnp.zeros((x.shape[0], self.cond_dim))
            cond_tokens = cond_vec[:, None, :]
        else:
            cond_vec, cond_tokens = ExemplarEncoder(
                hidden=self.base_channels,
                cond_dim=self.cond_dim,
            )(cond)
            # Handle batch size mismatch
            if cond_vec.shape[0] != x.shape[0]:
                cond_vec = jnp.broadcast_to(
                    jnp.mean(cond_vec, axis=0, keepdims=True),
                    (x.shape[0], self.cond_dim)
                )
                cond_tokens = jnp.broadcast_to(
                    jnp.mean(cond_tokens, axis=1, keepdims=True),
                    (x.shape[0], 1, self.cond_dim)
                )
        
        # Time embedding
        t_emb = sinusoidal_embedding(timesteps * self.time_scale, self.base_channels)
        t_emb = nn.Dense(time_dim)(t_emb)
        t_emb = nn.silu(t_emb)
        t_emb = nn.Dense(time_dim)(t_emb)
        
        # Input conv
        h = nn.Conv(self.base_channels, kernel_size=(3, 3), padding='SAME')(x)
        
        # Down path
        hs = []
        in_ch = self.base_channels
        curr_res = self.image_size
        
        for mult_idx, mult in enumerate(self.channel_mults):
            out_ch = self.base_channels * mult
            for _ in range(self.num_res_blocks):
                h = ResBlockCond(out_ch, time_dim, self.cond_dim, self.dropout)(h, t_emb, cond_vec, train)
                hs.append(h)
                in_ch = out_ch
                if curr_res in self.attn_resolutions:
                    h = SelfAttention(self.num_heads)(h)
                if curr_res in self.cross_attn_resolutions:
                    h = CrossAttention(in_ch, self.cond_dim, self.num_heads)(h, cond_tokens)
            if mult_idx != len(self.channel_mults) - 1:
                h = Downsample(in_ch)(h)
                curr_res //= 2
        
        # Middle
        h = ResBlockCond(in_ch, time_dim, self.cond_dim, self.dropout)(h, t_emb, cond_vec, train)
        h = SelfAttention(self.num_heads)(h)
        h = ResBlockCond(in_ch, time_dim, self.cond_dim, self.dropout)(h, t_emb, cond_vec, train)
        
        # Up path
        for mult_idx, mult in enumerate(reversed(self.channel_mults)):
            out_ch = self.base_channels * mult
            for _ in range(self.num_res_blocks):
                skip = hs.pop()
                h = jnp.concatenate([h, skip], axis=-1)
                h = ResBlockCond(out_ch, time_dim, self.cond_dim, self.dropout)(h, t_emb, cond_vec, train)
                in_ch = out_ch
                if curr_res in self.attn_resolutions:
                    h = SelfAttention(self.num_heads)(h)
                if curr_res in self.cross_attn_resolutions:
                    h = CrossAttention(in_ch, self.cond_dim, self.num_heads)(h, cond_tokens)
            if mult_idx != len(self.channel_mults) - 1:
                h = Upsample(in_ch)(h)
                curr_res *= 2
        
        # Output
        h = nn.GroupNorm(num_groups=32)(h)
        h = nn.silu(h)
        h = nn.Conv(self.in_channels, kernel_size=(3, 3), padding='SAME')(h)
        
        return h


class DDPM:
    """
    Continuous-time DDPM with v-prediction parameterization for conditional model.
    """
    
    def __init__(
        self,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        train_steps: int = 1000,
    ):
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.train_steps = train_steps

    @property
    def num_timesteps(self) -> int:
        return self.train_steps

    def alpha_bar(self, t: jnp.ndarray) -> jnp.ndarray:
        T = float(self.train_steps)
        return jnp.exp(-self.beta_start * T * t - 0.5 * (self.beta_end - self.beta_start) * T * t * t)

    def alpha_sigma(self, t: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        alpha_bar = self.alpha_bar(t)
        alpha = jnp.sqrt(alpha_bar)
        sigma = jnp.sqrt(1.0 - alpha_bar)
        return alpha, sigma

    def q_sample(
        self,
        x0: jnp.ndarray,
        t: jnp.ndarray,
        noise: Optional[jnp.ndarray] = None,
        key: Optional[jax.random.PRNGKey] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        if noise is None:
            if key is None:
                raise ValueError("Either noise or key must be provided")
            noise = jax.random.normal(key, x0.shape)
        
        alpha, sigma = self.alpha_sigma(t)
        alpha_bc = alpha[:, None, None, None]
        sigma_bc = sigma[:, None, None, None]
        x_t = alpha_bc * x0 + sigma_bc * noise
        return x_t, alpha, sigma

    def p_losses(
        self,
        model_apply_fn,
        params,
        x0: jnp.ndarray,
        t: jnp.ndarray,
        key: jax.random.PRNGKey,
        cond: Optional[jnp.ndarray] = None,
        train: bool = True,
    ) -> jnp.ndarray:
        noise_key, dropout_key = jax.random.split(key)
        noise = jax.random.normal(noise_key, x0.shape)
        
        x_t, alpha, sigma = self.q_sample(x0, t, noise=noise)
        
        alpha_bc = alpha[:, None, None, None]
        sigma_bc = sigma[:, None, None, None]
        v_target = alpha_bc * noise - sigma_bc * x0
        
        v_pred = model_apply_fn(params, x_t, t, cond=cond, train=train, rngs={'dropout': dropout_key})
        
        return jnp.mean((v_pred - v_target) ** 2)

    def predict_x0_eps(
        self,
        x_t: jnp.ndarray,
        t: jnp.ndarray,
        v_pred: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        alpha, sigma = self.alpha_sigma(t)
        alpha = alpha[:, None, None, None]
        sigma = sigma[:, None, None, None]
        x0_pred = alpha * x_t - sigma * v_pred
        eps_pred = alpha * v_pred + sigma * x_t
        return x0_pred, eps_pred

    def sample(
        self,
        model_apply_fn,
        params,
        key: jax.random.PRNGKey,
        batch_size: int,
        shape: Tuple[int, int, int],
        steps: int = 50,
        cond: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        h, w, c = shape
        img = jax.random.normal(key, (batch_size, h, w, c))
        
        times = jnp.linspace(1.0, 0.0, steps + 1)
        
        def step_fn(i, img):
            t_cur = jnp.full((batch_size,), times[i])
            t_prev = jnp.full((batch_size,), times[i + 1])
            
            v_pred = model_apply_fn(params, img, t_cur, cond=cond, train=False)
            x0_pred, eps_pred = self.predict_x0_eps(img, t_cur, v_pred)
            
            alpha_prev, sigma_prev = self.alpha_sigma(t_prev)
            alpha_prev = alpha_prev[:, None, None, None]
            sigma_prev = sigma_prev[:, None, None, None]
            
            img = alpha_prev * x0_pred + sigma_prev * eps_pred
            return img
        
        img = jax.lax.fori_loop(0, steps, step_fn, img)
        return img

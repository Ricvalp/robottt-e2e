"""
U-Net and DDPM implementation in JAX/Flax.

Minimal U-Net for DDPM with v-prediction parameterization,
following Ho et al. 2020 and OpenAI guided-diffusion.
"""
from __future__ import annotations

import math
from typing import Sequence, Tuple, Optional

import jax
import jax.numpy as jnp
import flax.linen as nn


def sinusoidal_embedding(timesteps: jnp.ndarray, dim: int) -> jnp.ndarray:
    """Create sinusoidal timestep embeddings (same as in DDPM/Transformer)."""
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


class ResBlock(nn.Module):
    """Residual block with time conditioning."""
    out_channels: int
    time_emb_dim: int
    dropout: float = 0.1

    @nn.compact
    def __call__(self, x: jnp.ndarray, t_emb: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        in_channels = x.shape[-1]  # NHWC format
        
        # First conv
        h = nn.GroupNorm(num_groups=32)(x)
        h = nn.silu(h)
        h = nn.Conv(self.out_channels, kernel_size=(3, 3), padding='SAME')(h)
        
        # Time embedding injection
        time_out = nn.Dense(self.out_channels)(nn.silu(t_emb))
        h = h + time_out[:, None, None, :]
        
        # Second conv
        h = nn.GroupNorm(num_groups=32)(h)
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
        
        # Normalize
        y = nn.GroupNorm(num_groups=32)(x)
        
        # Reshape to sequence
        y = y.reshape(b, h * w, c)
        
        # Multi-head attention
        y = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=c,
            out_features=c,
        )(y, y)
        
        # Reshape back
        y = y.reshape(b, h, w, c)
        
        return x + y


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


class UNet(nn.Module):
    """
    Minimal U-Net for DDPM, loosely following Ho et al. 2020 and OpenAI guided-diffusion.
    
    Uses NHWC format (standard for JAX/Flax).
    """
    in_channels: int = 1
    base_channels: int = 64
    channel_mults: Sequence[int] = (1, 2, 4)
    num_res_blocks: int = 2
    dropout: float = 0.1
    attn_resolutions: Sequence[int] = (16,)
    num_heads: int = 4
    image_size: int = 32
    time_scale: float = 1000.0

    def setup(self):
        time_dim = self.base_channels * 4
        
        # Time MLP
        self.time_mlp = nn.Sequential([
            nn.Dense(time_dim),
            nn.silu,
            nn.Dense(time_dim),
        ])

    @nn.compact
    def __call__(self, x: jnp.ndarray, timesteps: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        """
        Args:
            x: Input images in [-1, 1], shape (B, H, W, C) - NHWC format
            timesteps: Float timesteps in [0, 1], shape (B,)
            train: Whether in training mode (for dropout)
        Returns:
            Predicted v of shape (B, H, W, C)
        """
        time_dim = self.base_channels * 4
        
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
                h = ResBlock(out_ch, time_dim, self.dropout)(h, t_emb, train)
                hs.append(h)
                in_ch = out_ch
                if curr_res in self.attn_resolutions:
                    h = SelfAttention(self.num_heads)(h)
            if mult_idx != len(self.channel_mults) - 1:
                h = Downsample(in_ch)(h)
                curr_res //= 2
        
        # Middle
        h = ResBlock(in_ch, time_dim, self.dropout)(h, t_emb, train)
        h = SelfAttention(self.num_heads)(h)
        h = ResBlock(in_ch, time_dim, self.dropout)(h, t_emb, train)
        
        # Up path
        for mult_idx, mult in enumerate(reversed(self.channel_mults)):
            out_ch = self.base_channels * mult
            for _ in range(self.num_res_blocks):
                skip = hs.pop()
                h = jnp.concatenate([h, skip], axis=-1)
                h = ResBlock(out_ch, time_dim, self.dropout)(h, t_emb, train)
                in_ch = out_ch
                if curr_res in self.attn_resolutions:
                    h = SelfAttention(self.num_heads)(h)
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
    Continuous-time DDPM with v-prediction parameterization.
    
    This is a stateless wrapper that provides diffusion utilities.
    The actual model parameters are stored separately.
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
        """
        Continuous alpha_bar(t) for linear beta schedule on t in [0, 1].
        """
        T = float(self.train_steps)
        return jnp.exp(-self.beta_start * T * t - 0.5 * (self.beta_end - self.beta_start) * T * t * t)

    def alpha_sigma(self, t: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Returns (alpha, sigma) for given timesteps."""
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
        """
        Forward diffusion: add noise to x0 at time t.
        Returns (x_t, alpha, sigma).
        """
        if noise is None:
            if key is None:
                raise ValueError("Either noise or key must be provided")
            noise = jax.random.normal(key, x0.shape)
        
        alpha, sigma = self.alpha_sigma(t)
        # Broadcast for NHWC
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
        train: bool = True,
    ) -> jnp.ndarray:
        """
        Compute v-prediction loss.
        
        Args:
            model_apply_fn: Function to apply model (e.g., model.apply)
            params: Model parameters
            x0: Clean images
            t: Timesteps
            key: PRNG key for noise
            train: Training mode flag
        Returns:
            MSE loss
        """
        noise_key, dropout_key = jax.random.split(key)
        noise = jax.random.normal(noise_key, x0.shape)
        
        x_t, alpha, sigma = self.q_sample(x0, t, noise=noise)
        
        # v-target
        alpha_bc = alpha[:, None, None, None]
        sigma_bc = sigma[:, None, None, None]
        v_target = alpha_bc * noise - sigma_bc * x0
        
        # Model prediction
        v_pred = model_apply_fn(params, x_t, t, train=train, rngs={'dropout': dropout_key})
        
        return jnp.mean((v_pred - v_target) ** 2)

    def predict_x0_eps(
        self,
        x_t: jnp.ndarray,
        t: jnp.ndarray,
        v_pred: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Predict x0 and epsilon from v-prediction."""
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
        shape: Tuple[int, int, int],  # (H, W, C)
        steps: int = 50,
    ) -> jnp.ndarray:
        """
        DDIM-style deterministic sampling.
        
        Args:
            model_apply_fn: Function to apply model
            params: Model parameters
            key: PRNG key
            batch_size: Number of samples
            shape: Image shape (H, W, C)
            steps: Number of sampling steps
        Returns:
            Generated samples
        """
        h, w, c = shape
        img = jax.random.normal(key, (batch_size, h, w, c))
        
        times = jnp.linspace(1.0, 0.0, steps + 1)
        
        def step_fn(i, img):
            t_cur = jnp.full((batch_size,), times[i])
            t_prev = jnp.full((batch_size,), times[i + 1])
            
            v_pred = model_apply_fn(params, img, t_cur, train=False)
            x0_pred, eps_pred = self.predict_x0_eps(img, t_cur, v_pred)
            
            alpha_prev, sigma_prev = self.alpha_sigma(t_prev)
            alpha_prev = alpha_prev[:, None, None, None]
            sigma_prev = sigma_prev[:, None, None, None]
            
            img = alpha_prev * x0_pred + sigma_prev * eps_pred
            return img
        
        img = jax.lax.fori_loop(0, steps, step_fn, img)
        return img

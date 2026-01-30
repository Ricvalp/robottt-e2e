from __future__ import annotations
import jax.numpy as jnp
from flax import linen as nn


class BasicBlock(nn.Module):
    in_channels: int
    out_channels: int
    stride: int = 1
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x, training: bool):
        residual = x
        x = nn.Conv(self.out_channels, (3, 3), strides=(self.stride, self.stride), padding="SAME", use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
        x = nn.Dropout(self.dropout, deterministic=not training)(x)
        x = nn.Conv(self.out_channels, (3, 3), strides=(1, 1), padding="SAME", use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        if self.in_channels != self.out_channels or self.stride != 1:
            residual = nn.Conv(self.out_channels, (1, 1), strides=(self.stride, self.stride), use_bias=False)(residual)
            residual = nn.BatchNorm(use_running_average=not training)(residual)
        x = x + residual
        x = nn.relu(x)
        return x


class SmallResNet(nn.Module):
    width: int = 64
    num_blocks: int = 3
    num_classes: int = 10
    dropout: float = 0.1

    @nn.compact
    def __call__(self, x, training: bool = False):
        x = nn.Conv(self.width, (3, 3), padding="SAME", use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)

        in_ch = self.width
        for i in range(self.num_blocks):
            stride = 2 if i > 0 else 1
            x = BasicBlock(in_ch, self.width * (2 ** i), stride=stride, dropout=self.dropout)(x, training)
            in_ch = self.width * (2 ** i)
        x = jnp.mean(x, axis=(1, 2))
        logits = nn.Dense(self.num_classes)(x)
        return logits, x

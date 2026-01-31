"""
Small ResNet classifier for MNIST in JAX/Flax.
Used for evaluation metrics (counting generated samples by class).
"""
from __future__ import annotations

import jax.numpy as jnp
import flax.linen as nn


class BasicBlock(nn.Module):
    """ResNet basic block with optional stride and dropout."""
    out_channels: int
    stride: int = 1
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        in_channels = x.shape[-1]
        
        # First conv
        out = nn.Conv(
            self.out_channels,
            kernel_size=(3, 3),
            strides=(self.stride, self.stride),
            padding='SAME',
            use_bias=False,
        )(x)
        out = nn.BatchNorm(use_running_average=not train)(out)
        out = nn.relu(out)
        out = nn.Dropout(rate=self.dropout, deterministic=not train)(out)
        
        # Second conv
        out = nn.Conv(
            self.out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='SAME',
            use_bias=False,
        )(out)
        out = nn.BatchNorm(use_running_average=not train)(out)
        
        # Shortcut
        if in_channels != self.out_channels or self.stride != 1:
            shortcut = nn.Conv(
                self.out_channels,
                kernel_size=(1, 1),
                strides=(self.stride, self.stride),
                use_bias=False,
            )(x)
            shortcut = nn.BatchNorm(use_running_average=not train)(shortcut)
        else:
            shortcut = x
        
        out = out + shortcut
        out = nn.relu(out)
        return out


class SmallResNet(nn.Module):
    """
    Small ResNet for MNIST classification.
    
    Attributes:
        width: Base channel width
        num_blocks: Number of residual blocks
        num_classes: Number of output classes
        dropout: Dropout rate
    """
    width: int = 64
    num_blocks: int = 3
    num_classes: int = 10
    dropout: float = 0.1

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        """
        Args:
            x: Input images (B, H, W, C), expected in [0, 1] range
            train: Training mode flag
        Returns:
            Logits (B, num_classes)
        """
        # Stem
        x = nn.Conv(self.width, kernel_size=(3, 3), padding='SAME', use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        
        # Residual blocks
        in_ch = self.width
        for i in range(self.num_blocks):
            stride = 2 if i > 0 else 1
            out_ch = self.width * (2 ** i)
            x = BasicBlock(out_ch, stride=stride, dropout=self.dropout)(x, train)
            in_ch = out_ch
        
        # Global average pooling
        x = jnp.mean(x, axis=(1, 2))  # (B, C)
        
        # Classification head
        x = nn.Dense(self.num_classes)(x)
        
        return x

    def get_features(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        """Extract features before the final classification layer."""
        # Stem
        x = nn.Conv(self.width, kernel_size=(3, 3), padding='SAME', use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        
        # Residual blocks
        in_ch = self.width
        for i in range(self.num_blocks):
            stride = 2 if i > 0 else 1
            out_ch = self.width * (2 ** i)
            x = BasicBlock(out_ch, stride=stride, dropout=self.dropout)(x, train)
            in_ch = out_ch
        
        # Global average pooling only
        x = jnp.mean(x, axis=(1, 2))
        
        return x

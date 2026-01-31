#!/usr/bin/env python3
"""
Single-image DDPM training on MNIST using JAX/Flax.
Trains on a single image to test memorization capability.
"""
from __future__ import annotations

import random
from pathlib import Path
from functools import partial

from absl import app
from ml_collections import ConfigDict, config_flags
import numpy as np
import jax
import jax.numpy as jnp
from flax.training import train_state
import optax
import tensorflow_datasets as tfds
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import UNet, DDPM


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def get_single_image(cfg: ConfigDict) -> jnp.ndarray:
    """Get the first image from MNIST."""
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    
    ds = tfds.load("mnist", split="train")
    example = next(iter(ds))
    
    image = tf.image.resize(example['image'], (cfg.data.image_size, cfg.data.image_size))
    image = tf.cast(image, tf.float32) / 127.5 - 1.0
    
    return jnp.array(image.numpy())


def create_train_state(key: jax.random.PRNGKey, cfg: ConfigDict) -> train_state.TrainState:
    """Create initial training state."""
    model = UNet(
        in_channels=cfg.model.in_channels,
        base_channels=cfg.model.base_channels,
        channel_mults=tuple(cfg.model.channel_mults),
        num_res_blocks=cfg.model.num_res_blocks,
        dropout=cfg.model.dropout,
        attn_resolutions=tuple(cfg.model.attn_resolutions),
        num_heads=cfg.model.num_heads,
        image_size=cfg.data.image_size,
        time_scale=cfg.diffusion.train_steps,
    )
    
    dummy_x = jnp.ones((1, cfg.data.image_size, cfg.data.image_size, cfg.model.in_channels))
    dummy_t = jnp.ones((1,))
    
    params = model.init(key, dummy_x, dummy_t, train=False)
    
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"Model parameters: {n_params/1e6:.2f}M ({n_params} params)")
    
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.training.grad_clip),
        optax.adamw(cfg.training.lr, weight_decay=cfg.training.weight_decay),
    )
    
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
    )


@partial(jax.jit, static_argnums=(3,))
def train_step(state: train_state.TrainState, batch: jnp.ndarray, key: jax.random.PRNGKey, ddpm: DDPM):
    """Single training step."""
    t_key, loss_key = jax.random.split(key)
    t = jax.random.uniform(t_key, (batch.shape[0],))
    
    def loss_fn(params):
        return ddpm.p_losses(state.apply_fn, params, batch, t, loss_key, train=True)
    
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    
    return state, loss


def save_samples(
    state: train_state.TrainState,
    ddpm: DDPM,
    cfg: ConfigDict,
    key: jax.random.PRNGKey,
    epoch: int,
) -> jnp.ndarray:
    """Generate and save samples."""
    samples = ddpm.sample(
        state.apply_fn,
        state.params,
        key,
        batch_size=cfg.sample.num_images,
        shape=(cfg.data.image_size, cfg.data.image_size, cfg.model.in_channels),
        steps=cfg.sample.steps,
    )
    
    samples = jnp.clip(samples, -1, 1)
    samples = (samples + 1) / 2
    
    out_dir = Path(cfg.sample.dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    n = int(np.sqrt(cfg.sample.num_images))
    grid = samples.reshape(n, n, cfg.data.image_size, cfg.data.image_size, -1)
    grid = grid.transpose(0, 2, 1, 3, 4).reshape(n * cfg.data.image_size, n * cfg.data.image_size, -1)
    
    plt.imsave(out_dir / f"epoch_{epoch:03d}.png", np.array(grid.squeeze()), cmap='gray')
    
    return samples


def train(cfg: ConfigDict) -> None:
    set_seed(cfg.run.seed)
    key = jax.random.PRNGKey(cfg.run.seed)
    
    # Get single image
    single_image = get_single_image(cfg)
    print(f"Training on single image with shape: {single_image.shape}")
    
    # Create batch by repeating the single image
    batch = jnp.tile(single_image[None, ...], (cfg.data.batch_size, 1, 1, 1))
    
    # Create model
    key, init_key = jax.random.split(key)
    state = create_train_state(init_key, cfg)
    
    ddpm = DDPM(
        beta_start=cfg.diffusion.beta_start,
        beta_end=cfg.diffusion.beta_end,
        train_steps=cfg.diffusion.train_steps,
    )
    
    # Wandb
    wandb_run = None
    if getattr(cfg, "wandb", None) and cfg.wandb.use:
        wandb_run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.run_name,
            config=cfg.to_dict(),
        )
    
    # Training loop
    global_step = 0
    loss_ema = None
    
    for epoch in range(1, cfg.training.epochs + 1):
        pbar = tqdm(range(100), desc=f"Epoch {epoch}", leave=False)  # 100 steps per epoch
        
        for _ in pbar:
            key, step_key = jax.random.split(key)
            state, loss = train_step(state, batch, step_key, ddpm)
            
            loss_val = float(loss)
            loss_ema = loss_val if loss_ema is None else 0.99 * loss_ema + 0.01 * loss_val
            
            global_step += 1
            
            if global_step % cfg.training.log_every == 0:
                pbar.set_postfix(loss=loss_val, ema=loss_ema)
                print(f"[epoch {epoch:03d} step {global_step:06d}] loss={loss_val:.4f} ema={loss_ema:.4f}")
                if wandb_run is not None:
                    wandb.log({"train/loss": loss_val, "train/loss_ema": loss_ema}, step=global_step)
        
        if epoch % cfg.training.sample_every_epochs == 0:
            key, sample_key = jax.random.split(key)
            samples = save_samples(state, ddpm, cfg, sample_key, epoch)
            if wandb_run is not None:
                img = np.array((samples[0].squeeze() + 1) / 2 * 255).astype(np.uint8)
                wandb.log({"samples": wandb.Image(img)}, step=global_step)
        
        if epoch % cfg.training.checkpoint_every_epochs == 0:
            ckpt_dir = Path(cfg.checkpoint.dir).resolve()  # Must be absolute for Orbax
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            
            import orbax.checkpoint as ocp
            checkpointer = ocp.PyTreeCheckpointer()
            checkpointer.save(
                ckpt_dir / f"ddpm_single_epoch_{epoch:03d}",
                state.params,
            )
    
    if wandb_run is not None:
        wandb_run.finish()


# Entry point
_CONFIG = config_flags.DEFINE_config_file(
    "config",
    default="playground_jax/configs/train_unet_mnist.py",
    help_string="Path to a ml_collections config file.",
)


def main(argv=None):
    del argv
    cfg = _CONFIG.value
    train(cfg)


if __name__ == "__main__":
    app.run(main)

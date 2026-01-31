#!/usr/bin/env python3
"""
MAML training for diffusion models on MNIST using JAX/Flax.

Second-order MAML: inner updates are differentiable; outer loss sees them.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple, Dict, Any
from functools import partial

from absl import app
from ml_collections import ConfigDict, config_flags
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.core import freeze, unfreeze
from flax.training import train_state
import optax
import tensorflow_datasets as tfds
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import UNet, DDPM


# -------------------------
# Utilities
# -------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def build_datasets(cfg: ConfigDict):
    """Build MNIST dataset split by digit."""
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    
    def preprocess(example):
        image = tf.image.resize(example['image'], (cfg.data.image_size, cfg.data.image_size))
        image = tf.cast(image, tf.float32) / 127.5 - 1.0
        return {'image': image, 'label': example['label']}
    
    ds = tfds.load("mnist", split="train")
    ds = ds.map(preprocess)
    
    # Convert to numpy for easier indexing by digit
    all_data = list(tfds.as_numpy(ds))
    
    rng = np.random.default_rng(cfg.run.seed)
    holdout = int(cfg.data.holdout_per_class)
    
    train_indices_by_digit = {}
    val_indices_by_digit = {}
    
    if cfg.data.use_full_dataset:
        for d in range(10):
            idxs = [i for i, item in enumerate(all_data) if int(item['label']) == d]
            rng.shuffle(idxs)
            val_indices_by_digit[d] = idxs[:holdout]
            train_indices_by_digit[d] = idxs[holdout:]
        train_digits = list(range(10))
        eval_digits = list(range(10))
    else:
        leave_out = int(cfg.data.leave_out_digit)
        for d in range(10):
            idxs = [i for i, item in enumerate(all_data) if int(item['label']) == d]
            rng.shuffle(idxs)
            if d == leave_out:
                val_indices_by_digit[d] = idxs[:int(cfg.data.leaveout_eval_holdout)]
                train_indices_by_digit[d] = []
            else:
                val_indices_by_digit[d] = idxs[:holdout]
                train_indices_by_digit[d] = idxs[holdout:]
        train_digits = [d for d in range(10) if d != leave_out]
        eval_digits = [leave_out]
    
    return all_data, train_indices_by_digit, val_indices_by_digit, train_digits, eval_digits


def sample_digit_batch(
    all_data: List[Dict],
    indices_by_digit: Dict[int, List[int]],
    digit: int,
    batch_size: int,
) -> jnp.ndarray:
    """Sample a batch of images for a specific digit."""
    idx_pool = indices_by_digit[digit]
    replace = len(idx_pool) < batch_size
    idx = np.random.choice(idx_pool, size=batch_size, replace=replace)
    imgs = np.stack([all_data[i]['image'] for i in idx], axis=0)
    return jnp.array(imgs)


def get_param_paths(params: Dict, prefix: str = "") -> List[str]:
    """Get all parameter paths in a nested dict."""
    paths = []
    for k, v in params.items():
        path = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            paths.extend(get_param_paths(v, path))
        else:
            paths.append(path)
    return paths


def select_fast_params(params: Dict, selector: str) -> List[str]:
    """
    Select which parameters are "fast" (inner-loop adapted).
    
    Returns list of parameter paths that should be updated in inner loop.
    
    Flax names layers like: params.ResBlock_0, params.Dense_0, etc.
    We select based on ResBlock indices and layer types.
    """
    all_paths = get_param_paths(unfreeze(params))
    fast_paths = []
    
    # Parse selector to determine which blocks to include
    if selector == "up_time_mlp":
        include_resblock_dense = True
        include_mid = False
        include_head = False
        include_gn = False
        include_convs = False
    elif selector == "up_down_time_mlp":
        include_resblock_dense = True
        include_mid = False
        include_head = False
        include_gn = False
        include_convs = False
    elif selector == "up_down_mid_head":
        include_resblock_dense = True
        include_mid = True
        include_head = True
        include_gn = False
        include_convs = False
    elif selector == "up_down_mid_head_gn":
        include_resblock_dense = True
        include_mid = True
        include_head = True
        include_gn = True
        include_convs = False
    elif selector == "up_down_mid_full":
        include_resblock_dense = True
        include_mid = True
        include_head = True
        include_gn = False
        include_convs = True
    elif selector == "up_down_mid_full_gn":
        include_resblock_dense = True
        include_mid = True
        include_head = True
        include_gn = True
        include_convs = True
    elif selector == "all":
        # Select all parameters as fast
        return all_paths
    else:
        raise ValueError(f"Unknown selector {selector}")
    
    for path in all_paths:
        parts = path.split('.')
        
        # In Flax, params are under 'params' key first
        # e.g., params.ResBlock_0.Dense_0.kernel
        is_resblock = any('ResBlock' in p for p in parts)
        
        # Dense layers inside ResBlocks (time embedding injection)
        if include_resblock_dense and is_resblock and 'Dense' in path:
            fast_paths.append(path)
            continue
        
        # GroupNorm layers
        if include_gn and 'GroupNorm' in path:
            fast_paths.append(path)
            continue
        
        # Convolutions inside ResBlocks
        if include_convs and is_resblock and 'Conv' in path:
            fast_paths.append(path)
            continue
        
        # Head - final output layers (usually last Conv and GroupNorm)
        # In our UNet, these are the last GroupNorm and Conv before output
        if include_head:
            # The output layers are typically at the end, not inside ResBlocks
            if not is_resblock:
                # Include top-level GroupNorm and Conv (output layers)
                if 'GroupNorm' in path or ('Conv' in path and not 'Downsample' in path and not 'Upsample' in path):
                    # Skip the input conv (Conv_0 at params level)
                    if 'Conv_0' in parts[1] if len(parts) > 1 else False:
                        continue  # Skip input conv
                    fast_paths.append(path)
                    continue
    
    return fast_paths


def get_nested_value(d: Dict, path: str):
    """Get value from nested dict using dot-separated path."""
    keys = path.split('.')
    for k in keys:
        d = d[k]
    return d


def set_nested_value(d: Dict, path: str, value):
    """Set value in nested dict using dot-separated path."""
    keys = path.split('.')
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value


def create_fast_mask(params: Dict, fast_paths: List[str]) -> Dict:
    """Create a mask pytree with 1.0 for fast params and 0.0 for slow.
    
    Returns JAX arrays on device for efficient repeated use.
    """
    params = unfreeze(params)
    
    def make_mask(p, path_so_far=""):
        if isinstance(p, dict):
            return {k: make_mask(v, f"{path_so_far}.{k}" if path_so_far else k) for k, v in p.items()}
        else:
            full_path = path_so_far
            is_fast = any(full_path == fp or full_path.endswith(fp) for fp in fast_paths)
            # Use JAX arrays so they're on device
            return jnp.ones_like(p) if is_fast else jnp.zeros_like(p)
    
    return make_mask(params)


def count_params(params: Dict, fast_paths: List[str]) -> Tuple[int, int]:
    """Count total and fast parameters."""
    params = unfreeze(params)
    total = sum(x.size for x in jax.tree_util.tree_leaves(params))
    
    fast = 0
    for path in fast_paths:
        try:
            p = get_nested_value(params, path)
            fast += p.size
        except (KeyError, TypeError):
            pass
    
    return total, fast


def grid_to_uint8(img: jnp.ndarray) -> np.ndarray:
    """Convert image to uint8."""
    return np.array(jnp.clip(img, 0, 1) * 255).astype(np.uint8)


def make_grid(images: jnp.ndarray, nrow: int = 4) -> jnp.ndarray:
    """Create image grid from batch."""
    n = images.shape[0]
    ncol = (n + nrow - 1) // nrow
    
    # Pad if necessary
    pad_n = ncol * nrow - n
    if pad_n > 0:
        images = jnp.concatenate([images, jnp.zeros((pad_n,) + images.shape[1:])], axis=0)
    
    # Reshape to grid
    h, w, c = images.shape[1:]
    images = images.reshape(ncol, nrow, h, w, c)
    images = images.transpose(0, 2, 1, 3, 4)
    images = images.reshape(ncol * h, nrow * w, c)
    
    return images


# -------------------------
# MAML Core
# -------------------------

def alpha_sigma_fn(t: jnp.ndarray, beta_start: float, beta_end: float, train_steps: int):
    """Compute alpha and sigma for given timesteps."""
    T = float(train_steps)
    alpha_bar = jnp.exp(-beta_start * T * t - 0.5 * (beta_end - beta_start) * T * t * t)
    alpha = jnp.sqrt(alpha_bar)
    sigma = jnp.sqrt(1.0 - alpha_bar)
    return alpha, sigma


def compute_loss(
    apply_fn,
    params: Dict,
    x0: jnp.ndarray,
    t: jnp.ndarray,
    noise: jnp.ndarray,
    beta_start: float,
    beta_end: float,
    train_steps: int,
) -> jnp.ndarray:
    """Compute v-prediction loss."""
    alpha, sigma = alpha_sigma_fn(t, beta_start, beta_end, train_steps)
    alpha_bc = alpha[:, None, None, None]
    sigma_bc = sigma[:, None, None, None]
    
    x_t = alpha_bc * x0 + sigma_bc * noise
    v_target = alpha_bc * noise - sigma_bc * x0
    
    v_pred = apply_fn(params, x_t, t, train=False)
    
    return jnp.mean((v_pred - v_target) ** 2)


def inner_update(params: Dict, grads: Dict, fast_mask: Dict, inner_lr: float) -> Dict:
    """Apply inner loop update using mask."""
    return jax.tree_util.tree_map(
        lambda p, g, m: p - inner_lr * g * m,
        params, grads, fast_mask
    )


def maml_step(
    apply_fn,
    params: Dict,
    ddpm: DDPM,
    batch: jnp.ndarray,
    fast_mask: Dict,
    inner_lr: float,
    inner_steps: int,
    key: jax.random.PRNGKey,
) -> jnp.ndarray:
    """
    Second-order MAML step.
    
    Inner loop updates are differentiable for second-order gradients.
    Uses lax.fori_loop for efficiency.
    """
    beta_start = ddpm.beta_start
    beta_end = ddpm.beta_end
    train_steps = ddpm.train_steps
    
    def inner_loss_fn(params, t, noise):
        return compute_loss(apply_fn, params, batch, t, noise, beta_start, beta_end, train_steps)
    
    def inner_step_fn(i, carry):
        params, key = carry
        key, t_key, noise_key = jax.random.split(key, 3)
        t = jax.random.uniform(t_key, (batch.shape[0],))
        noise = jax.random.normal(noise_key, batch.shape)
        
        loss, grads = jax.value_and_grad(inner_loss_fn)(params, t, noise)
        params = inner_update(params, grads, fast_mask, inner_lr)
        
        return (params, key)
    
    # Run inner loop with lax.fori_loop for efficiency
    adapted_params, key = jax.lax.fori_loop(0, inner_steps, inner_step_fn, (params, key))
    
    # Outer loss on adapted params
    key, t_key, noise_key = jax.random.split(key, 3)
    t_out = jax.random.uniform(t_key, (batch.shape[0],))
    noise_out = jax.random.normal(noise_key, batch.shape)
    outer_loss = inner_loss_fn(adapted_params, t_out, noise_out)
    
    return outer_loss


@partial(jax.jit, static_argnums=(0, 5, 6, 7, 8, 9))
def train_step(
    apply_fn,
    params: Dict,
    batch: jnp.ndarray,
    fast_mask: Dict,
    key: jax.random.PRNGKey,
    inner_lr: float,
    inner_steps: int,
    beta_start: float,
    beta_end: float,
    train_steps: int,
) -> Tuple[jnp.ndarray, Dict]:
    """
    JIT-compiled MAML training step.
    
    Returns:
        (outer_loss, grads) - the outer loss and gradients w.r.t. params
    """
    def inner_loss_fn(params, t, noise):
        return compute_loss(apply_fn, params, batch, t, noise, beta_start, beta_end, train_steps)
    
    def maml_loss(params):
        def inner_step_fn(i, carry):
            params, key = carry
            key, t_key, noise_key = jax.random.split(key, 3)
            t = jax.random.uniform(t_key, (batch.shape[0],))
            noise = jax.random.normal(noise_key, batch.shape)
            
            loss, grads = jax.value_and_grad(inner_loss_fn)(params, t, noise)
            params = inner_update(params, grads, fast_mask, inner_lr)
            
            return (params, key)
        
        # Run inner loop
        adapted_params, inner_key = jax.lax.fori_loop(0, inner_steps, inner_step_fn, (params, key))
        
        # Outer loss on adapted params
        _, t_key, noise_key = jax.random.split(inner_key, 3)
        t_out = jax.random.uniform(t_key, (batch.shape[0],))
        noise_out = jax.random.normal(noise_key, batch.shape)
        outer_loss = inner_loss_fn(adapted_params, t_out, noise_out)
        
        return outer_loss
    
    outer_loss, grads = jax.value_and_grad(maml_loss)(params)
    return outer_loss, grads

def adapt_params(
    apply_fn,
    params: Dict,
    ddpm: DDPM,
    batch: jnp.ndarray,
    fast_mask: Dict,
    inner_steps: int,
    inner_lr: float,
    key: jax.random.PRNGKey,
) -> Dict:
    """Adapt fast params on batch."""
    beta_start = ddpm.beta_start
    beta_end = ddpm.beta_end
    train_steps = ddpm.train_steps
    
    def inner_loss_fn(params, t, noise):
        return compute_loss(apply_fn, params, batch, t, noise, beta_start, beta_end, train_steps)
    
    def inner_step_fn(i, carry):
        params, key = carry
        key, t_key, noise_key = jax.random.split(key, 3)
        t = jax.random.uniform(t_key, (batch.shape[0],))
        noise = jax.random.normal(noise_key, batch.shape)
        
        loss, grads = jax.value_and_grad(inner_loss_fn)(params, t, noise)
        params = inner_update(params, grads, fast_mask, inner_lr)
        
        return (params, key)
    
    adapted_params, _ = jax.lax.fori_loop(0, inner_steps, inner_step_fn, (params, key))
    return adapted_params


def sample_with_params(
    apply_fn,
    params: Dict,
    ddpm: DDPM,
    key: jax.random.PRNGKey,
    num_samples: int,
    image_shape: Tuple[int, int, int],
    steps: int,
) -> jnp.ndarray:
    """Sample from model with given params."""
    return ddpm.sample(
        apply_fn,
        params,
        key,
        batch_size=num_samples,
        shape=image_shape,
        steps=steps,
    )


def adapt_and_sample(
    apply_fn,
    params: Dict,
    ddpm: DDPM,
    batch: jnp.ndarray,
    fast_mask: Dict,
    cfg: ConfigDict,
    key: jax.random.PRNGKey,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Adapt fast params on batch, then sample."""
    key, adapt_key, sample_key = jax.random.split(key, 3)
    
    adapted_params = adapt_params(
        apply_fn,
        params,
        ddpm,
        batch,
        fast_mask,
        cfg.eval.inner_steps,
        cfg.eval.inner_lr,
        adapt_key,
    )
    
    samples = sample_with_params(
        apply_fn,
        adapted_params,
        ddpm,
        sample_key,
        cfg.sample.num_images,
        (cfg.data.image_size, cfg.data.image_size, cfg.model.in_channels),
        cfg.sample.steps,
    )
    
    # Normalize to [0, 1]
    samples = jnp.clip(samples, -1, 1)
    samples = (samples + 1) / 2
    batch_vis = jnp.clip(batch, -1, 1)
    batch_vis = (batch_vis + 1) / 2
    
    sample_grid = make_grid(samples, nrow=int(np.sqrt(cfg.sample.num_images)))
    batch_grid = make_grid(batch_vis[:cfg.sample.num_images], nrow=int(np.sqrt(cfg.sample.num_images)))
    
    return sample_grid, batch_grid


# -------------------------
# Training
# -------------------------

def create_train_state(key: jax.random.PRNGKey, cfg: ConfigDict):
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
    
    # Initialize
    dummy_x = jnp.ones((1, cfg.data.image_size, cfg.data.image_size, cfg.model.in_channels))
    dummy_t = jnp.ones((1,))
    
    params = model.init(key, dummy_x, dummy_t, train=False)
    
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"Model parameters: {n_params/1e6:.2f}M ({n_params} params)")
    
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.training.grad_clip),
        optax.adamw(cfg.training.outer_lr, weight_decay=cfg.training.weight_decay),
    )
    
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
    )
    
    return state


def train(cfg: ConfigDict) -> None:
    set_seed(cfg.run.seed)
    key = jax.random.PRNGKey(cfg.run.seed)
    
    # Build datasets
    all_data, train_indices_by_digit, val_indices_by_digit, train_digits, eval_digits = build_datasets(cfg)
    
    # Create model
    key, init_key = jax.random.split(key)
    state = create_train_state(init_key, cfg)
    
    # DDPM utilities
    ddpm = DDPM(
        beta_start=cfg.diffusion.beta_start,
        beta_end=cfg.diffusion.beta_end,
        train_steps=cfg.diffusion.train_steps,
    )
    
    # Select fast params and create mask
    fast_paths = select_fast_params(state.params, cfg.fast_params.selector)
    fast_mask = create_fast_mask(state.params, fast_paths)
    total_params, fast_params_count = count_params(state.params, fast_paths)
    print(f"Total params: {total_params/1e6:.2f}M, fast params: {fast_params_count/1e6:.4f}M")
    print(f"Fast param paths: {len(fast_paths)}")
    
    # Wandb
    wandb_run = None
    run_save_dir = Path(cfg.checkpoint.dir).resolve()  # Must be absolute for Orbax
    if getattr(cfg, "wandb", None) and cfg.wandb.use:
        wandb_run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=None,
            config=cfg.to_dict(),
        )
        wandb.run.name = wandb.run.id
        run_save_dir = Path(cfg.checkpoint.dir).resolve() / wandb.run.id
    run_save_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    global_step = 0
    
    for epoch in range(1, cfg.training.epochs + 1):
        pbar = tqdm(range(cfg.training.steps_per_epoch), desc=f"Epoch {epoch}", leave=False)
        
        for _ in pbar:
            digit = random.choice(train_digits)
            imgs = sample_digit_batch(all_data, train_indices_by_digit, digit, cfg.data.batch_size)
            
            key, step_key = jax.random.split(key)
            
            # JIT-compiled MAML step with gradient
            outer_loss, grads = train_step(
                state.apply_fn,
                state.params,
                imgs,
                fast_mask,
                step_key,
                cfg.training.inner_lr,
                cfg.training.inner_steps,
                ddpm.beta_start,
                ddpm.beta_end,
                ddpm.train_steps,
            )
            state = state.apply_gradients(grads=grads)
            
            global_step += 1
            pbar.set_postfix(loss=float(outer_loss))
            
            if global_step % cfg.training.log_every == 0:
                print(f"[epoch {epoch:03d} step {global_step:06d}] outer_loss={float(outer_loss):.4f}")
                if wandb_run is not None:
                    wandb.log({"train/outer_loss": float(outer_loss)}, step=global_step)
        
        # Sample
        if epoch % cfg.training.sample_every_epochs == 0:
            eval_digit = random.choice(eval_digits)
            val_imgs = sample_digit_batch(all_data, val_indices_by_digit, eval_digit, cfg.data.batch_size)
            
            key, sample_key = jax.random.split(key)
            sample_grid, batch_grid = adapt_and_sample(
                state.apply_fn,
                state.params,
                ddpm,
                val_imgs,
                fast_mask,
                cfg,
                sample_key,
            )
            
            if wandb_run is not None:
                wandb.log({
                    "samples": wandb.Image(grid_to_uint8(sample_grid.squeeze())),
                    "target_samples": wandb.Image(grid_to_uint8(batch_grid.squeeze())),
                    "eval_digit": eval_digit,
                }, step=global_step)
            
            # Also log train digit adaptation
            train_digit = random.choice(train_digits)
            train_imgs = sample_digit_batch(all_data, train_indices_by_digit, train_digit, cfg.data.batch_size)
            
            key, train_sample_key = jax.random.split(key)
            train_grid, train_batch_grid = adapt_and_sample(
                state.apply_fn,
                state.params,
                ddpm,
                train_imgs,
                fast_mask,
                cfg,
                train_sample_key,
            )
            
            if wandb_run is not None:
                wandb.log({
                    "train_adapt/samples": wandb.Image(grid_to_uint8(train_grid.squeeze())),
                    "train_adapt/target_samples": wandb.Image(grid_to_uint8(train_batch_grid.squeeze())),
                    "train_adapt/digit": train_digit,
                }, step=global_step)
        
        # Checkpoint
        if epoch % cfg.training.checkpoint_every_epochs == 0:
            import orbax.checkpoint as ocp
            checkpointer = ocp.PyTreeCheckpointer()
            checkpointer.save(
                run_save_dir / f"maml_epoch_{epoch:03d}",
                {
                    'params': state.params,
                    'epoch': epoch,
                    'global_step': global_step,
                },
            )
    
    if wandb_run is not None:
        wandb_run.finish()


# -------------------------
# Entry point
# -------------------------

_CONFIG = config_flags.DEFINE_config_file(
    "config",
    default="playground_jax/configs/train_maml_mnist.py",
    help_string="Path to a ml_collections config file.",
)


def main(argv=None):
    del argv
    cfg = _CONFIG.value
    train(cfg)


if __name__ == "__main__":
    app.run(main)

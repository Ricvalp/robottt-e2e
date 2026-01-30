#!/usr/bin/env python3
from __future__ import annotations

import random
from pathlib import Path

from absl import app
from ml_collections import config_flags
import numpy as np
import jax
import jax.numpy as jnp
import optax
import wandb
from flax.training import train_state

from configs import get_config
from model import UNet, DDPM
from dataset import load_mnist, split_per_class, sample_digit_batch
from classifier_model import SmallResNet

_CONFIG = config_flags.DEFINE_config_file(
    "maml_config",
    default="playground_jax/configs.py",
    help_string="Path to a ml_collections config file.",
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    jax.random.PRNGKey(seed)


def create_model(cfg):
    model_def = UNet(
        in_channels=1,
        base_channels=cfg.model.base_channels,
        channel_mults=tuple(cfg.model.channel_mults),
        num_res_blocks=cfg.model.num_res_blocks,
        dropout=cfg.model.dropout,
        attn_resolutions=tuple(cfg.model.attn_resolutions),
        num_heads=cfg.model.num_heads,
        image_size=cfg.data.image_size,
        time_scale=cfg.diffusion.train_steps,
    )
    ddpm_def = DDPM(
        beta_start=cfg.diffusion.beta_start,
        beta_end=cfg.diffusion.beta_end,
        train_steps=cfg.diffusion.train_steps,
        model_def=model_def,
    )
    return ddpm_def


def select_fast_params(params, selector):
    # params is a pytree; we will select keys by path
    fast_paths = []

    def traverse(path, _):
        pname = "/".join(path)
        if selector == "up_time_mlp":
            if "ups" in pname and "Dense" in pname:
                fast_paths.append(tuple(path))
        else:
            # fallback: adapt all params
            fast_paths.append(tuple(path))

    jax.tree_util.tree_map_with_path(traverse, params)
    return fast_paths


def get_params_by_paths(params, paths):
    flat = params
    selected = {}
    for p in paths:
        subtree = flat
        for key in p[:-1]:
            subtree = subtree[key]
        selected[p] = subtree[p[-1]]
    return selected


def set_params_by_paths(params, updates):
    params = params.copy()
    for path, val in updates.items():
        subtree = params
        for key in path[:-1]:
            subtree = subtree[key]
        subtree[path[-1]] = val
    return params


def maml_inner(ddpm_def, params, batch, fast_paths, inner_lr, inner_steps, rng):
    def loss_fn(p, bt, key):
        t = jax.random.uniform(key, (bt.shape[0],))
        return ddpm_def.loss(p, bt, t, key)

    key = rng
    for _ in range(inner_steps):
        key, subkey = jax.random.split(key)
        grads = jax.grad(loss_fn)(params, batch, subkey)
        # update only fast params
        updates = {}
        for path in fast_paths:
            g = grads
            for k in path:
                g = g[k]
            updates[path] = g
        # apply updates
        new_params = {}
        for path, g in updates.items():
            subtree = params
            for k in path[:-1]:
                subtree = subtree[k]
            new_params[path] = subtree[path[-1]] - inner_lr * g
        params = set_params_by_paths(params, new_params)
    return params


def outer_loss(ddpm_def, params, batch, rng):
    t = jax.random.uniform(rng, (batch.shape[0],))
    return ddpm_def.loss(params, batch, t, rng)


def main(argv=None):
    del argv
    cfg = _CONFIG.value
    set_seed(cfg.run.seed)

    (train_imgs, train_labels), (test_imgs, test_labels) = load_mnist(cfg.data.image_size, cfg.run.seed)
    if cfg.data.use_full_dataset:
        leave_out = None
    else:
        leave_out = cfg.data.leave_out_digit
    train_indices, val_indices = split_per_class(
        train_imgs, train_labels, cfg.data.holdout_per_class, leave_out_digit=leave_out, leaveout_eval_holdout=cfg.data.leaveout_eval_holdout
    )
    train_digits = list(d for d in range(10) if leave_out is None or d != leave_out)
    eval_digits = list(range(10)) if leave_out is None else [leave_out]

    ddpm_def = create_model(cfg)
    rng = jax.random.PRNGKey(cfg.run.seed)
    dummy_x = jnp.zeros((1, cfg.data.image_size, cfg.data.image_size, 1), dtype=jnp.float32)
    dummy_t = jnp.zeros((1,), dtype=jnp.float32)
    params = ddpm_def.init(rng, dummy_x, dummy_t)["params"]

    fast_paths = select_fast_params(params, cfg.fast_params.selector)

    tx = optax.adamw(cfg.training.outer_lr)
    state = train_state.TrainState.create(apply_fn=None, params=params, tx=tx)

    run_dir = Path(cfg.checkpoint.dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    wandb_run = None
    if cfg.wandb.use:
        wandb_run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            dir=cfg.wandb.dir,
            config=cfg.to_dict(),
        )
        wandb.run.name = wandb.run.id
        run_dir = run_dir / wandb.run.id
        run_dir.mkdir(parents=True, exist_ok=True)

    rng = jax.random.PRNGKey(cfg.run.seed)
    for epoch in range(1, cfg.training.epochs + 1):
        for step in range(cfg.training.steps_per_epoch):
            rng, sub = jax.random.split(rng)
            digit = random.choice(train_digits)
            batch = sample_digit_batch(train_imgs, train_indices, digit, cfg.data.batch_size, np.random.default_rng())
            batch = jnp.asarray(batch)

            # Inner adaptation
            adapted_params = maml_inner(ddpm_def, state.params, batch, fast_paths, cfg.training.inner_lr, cfg.training.inner_steps, sub)
            rng, sub2 = jax.random.split(rng)
            loss = outer_loss(ddpm_def, adapted_params, batch, sub2)
            grads = jax.grad(lambda p: outer_loss(ddpm_def, adapted_params, batch, sub2))(state.params)
            state = state.apply_gradients(grads=grads)

            if wandb_run and ((epoch - 1) * cfg.training.steps_per_epoch + step + 1) % cfg.training.log_every == 0:
                wandb.log({"train/loss": float(loss)}, step=(epoch - 1) * cfg.training.steps_per_epoch + step + 1)

        if epoch % cfg.training.checkpoint_every_epochs == 0:
            ckpt_path = run_dir / f"maml_epoch_{epoch:03d}.npz"
            np.savez(ckpt_path, **state.params)

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    app.run(main)

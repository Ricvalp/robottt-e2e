#!/usr/bin/env python3
"""
MNIST classifier training using JAX/Flax.
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

from classifier_model import SmallResNet


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def create_train_state(key: jax.random.PRNGKey, cfg: ConfigDict) -> train_state.TrainState:
    """Create initial training state."""
    model = SmallResNet(
        width=cfg.model.width,
        num_blocks=cfg.model.num_blocks,
        dropout=cfg.model.dropout,
    )
    
    # Initialize with dummy input
    dummy_x = jnp.ones((1, cfg.data.image_size, cfg.data.image_size, 1))
    
    variables = model.init(key, dummy_x, train=False)
    params = variables['params']
    batch_stats = variables.get('batch_stats', {})
    
    # Optimizer with gradient clipping
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.training.grad_clip),
        optax.adamw(cfg.training.lr, weight_decay=cfg.training.weight_decay),
    )
    
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
    ), batch_stats


@partial(jax.jit, static_argnums=())
def train_step(state: train_state.TrainState, batch_stats, images: jnp.ndarray, labels: jnp.ndarray, key: jax.random.PRNGKey):
    """Single training step."""
    
    def loss_fn(params):
        logits, updates = state.apply_fn(
            {'params': params, 'batch_stats': batch_stats},
            images,
            train=True,
            mutable=['batch_stats'],
            rngs={'dropout': key},
        )
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
        return loss, (logits, updates)
    
    (loss, (logits, updates)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    batch_stats = updates['batch_stats']
    
    preds = jnp.argmax(logits, axis=-1)
    acc = jnp.mean(preds == labels)
    
    return state, batch_stats, loss, acc


@jax.jit
def eval_step(state: train_state.TrainState, batch_stats, images: jnp.ndarray, labels: jnp.ndarray):
    """Evaluation step."""
    logits = state.apply_fn(
        {'params': state.params, 'batch_stats': batch_stats},
        images,
        train=False,
    )
    preds = jnp.argmax(logits, axis=-1)
    correct = jnp.sum(preds == labels)
    return correct


def evaluate(state: train_state.TrainState, batch_stats, test_ds) -> float:
    """Evaluate on test set."""
    correct = 0
    total = 0
    
    for batch in test_ds:
        images = jnp.array(batch['image'].numpy())
        labels = jnp.array(batch['label'].numpy())
        correct += int(eval_step(state, batch_stats, images, labels))
        total += len(labels)
    
    return correct / total


def train(cfg: ConfigDict) -> None:
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    
    set_seed(cfg.run.seed)
    key = jax.random.PRNGKey(cfg.run.seed)
    
    # Build datasets
    def preprocess(example):
        image = tf.image.resize(example['image'], (cfg.data.image_size, cfg.data.image_size))
        image = tf.cast(image, tf.float32) / 255.0
        return {'image': image, 'label': example['label']}
    
    train_ds = tfds.load("mnist", split="train")
    train_ds = train_ds.map(preprocess)
    train_ds = train_ds.shuffle(10000)
    train_ds = train_ds.batch(cfg.data.batch_size, drop_remainder=True)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    
    test_ds = tfds.load("mnist", split="test")
    test_ds = test_ds.map(preprocess)
    test_ds = test_ds.batch(cfg.eval.batch_size)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
    
    # Create model and optimizer
    key, init_key = jax.random.split(key)
    state, batch_stats = create_train_state(init_key, cfg)
    
    # Count parameters
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    print(f"Model parameters: {n_params/1e6:.2f}M ({n_params} params)")
    
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
    
    for epoch in range(1, cfg.training.epochs + 1):
        pbar = tqdm(train_ds, desc=f"Epoch {epoch}", leave=False)
        
        for batch in pbar:
            images = jnp.array(batch['image'].numpy())
            labels = jnp.array(batch['label'].numpy())
            
            key, step_key = jax.random.split(key)
            state, batch_stats, loss, acc = train_step(state, batch_stats, images, labels, step_key)
            
            global_step += 1
            
            if global_step % cfg.training.log_every == 0:
                pbar.set_postfix(loss=float(loss), acc=float(acc))
                if wandb_run is not None:
                    wandb.log({"train/loss": float(loss), "train/acc": float(acc)}, step=global_step)
        
        # Evaluate
        test_acc = evaluate(state, batch_stats, test_ds)
        print(f"[epoch {epoch}] test_acc={test_acc:.4f}")
        if wandb_run is not None:
            wandb.log({"test/acc": test_acc}, step=global_step)
        
        # Checkpoint
        if epoch % cfg.training.checkpoint_every_epochs == 0:
            ckpt_dir = Path(cfg.checkpoint.dir).resolve()  # Must be absolute for Orbax
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            
            import orbax.checkpoint as ocp
            checkpointer = ocp.PyTreeCheckpointer()
            checkpointer.save(
                ckpt_dir / f"classifier_epoch_{epoch:03d}",
                {'params': state.params, 'batch_stats': batch_stats},
            )
    
    if wandb_run is not None:
        wandb_run.finish()


# Entry point
_CONFIG = config_flags.DEFINE_config_file(
    "config",
    default="playground_jax/configs/train_classifier_mnist.py",
    help_string="Path to ml_collections config file.",
)


def main(argv=None):
    del argv
    cfg = _CONFIG.value
    train(cfg)


if __name__ == "__main__":
    app.run(main)

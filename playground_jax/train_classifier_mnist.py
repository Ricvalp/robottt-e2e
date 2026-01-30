#!/usr/bin/env python3
from __future__ import annotations

from absl import app
from ml_collections import config_flags
import jax
import jax.numpy as jnp
import optax
import numpy as np
import wandb
from flax.training import train_state

from configs import get_config
from dataset import load_mnist
from classifier_model import SmallResNet

_CONFIG = config_flags.DEFINE_config_file(
    "classifier_config",
    default="playground_jax/configs.py",
    help_string="Path to ml_collections config file.",
)


def set_seed(seed: int):
    np.random.seed(seed)
    jax.random.PRNGKey(seed)


def create_state(rng, cfg):
    model = SmallResNet(width=cfg.model.width, num_blocks=cfg.model.num_blocks, dropout=cfg.model.dropout)
    dummy = jnp.zeros((1, cfg.data.image_size, cfg.data.image_size, 1))
    params = model.init(rng, dummy, training=True)["params"]
    tx = optax.adamw(cfg.training.lr, weight_decay=cfg.training.weight_decay)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx), model


def compute_metrics(logits, labels):
    pred = logits.argmax(axis=-1)
    acc = (pred == labels).mean()
    return acc


def train_epoch(state, model, imgs, labels, cfg, rng):
    num = imgs.shape[0]
    idx = np.random.permutation(num)
    imgs = imgs[idx]
    labels = labels[idx]
    batch_size = cfg.data.batch_size
    steps = num // batch_size

    @jax.jit
    def train_step(state, x, y, key):
        def loss_fn(params):
            logits, _ = model.apply({"params": params}, x, training=True, rngs={"dropout": key})
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
            return loss, logits
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, logits), grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        acc = compute_metrics(logits, y)
        return state, loss, acc

    losses = []
    accs = []
    for i in range(steps):
        batch = imgs[i * batch_size : (i + 1) * batch_size]
        y = labels[i * batch_size : (i + 1) * batch_size]
        rng, sub = jax.random.split(rng)
        state, loss, acc = train_step(state, batch, y, sub)
        losses.append(loss)
        accs.append(acc)
    return state, float(np.mean(np.array(losses))), float(np.mean(np.array(accs)))


def eval_model(state, model, imgs, labels, cfg):
    batch_size = cfg.eval.batch_size
    steps = imgs.shape[0] // batch_size
    correct = 0
    total = 0
    for i in range(steps):
        x = imgs[i * batch_size : (i + 1) * batch_size]
        y = labels[i * batch_size : (i + 1) * batch_size]
        logits, _ = model.apply({"params": state.params}, x, training=False)
        pred = logits.argmax(axis=-1)
        correct += int((pred == y).sum())
        total += y.shape[0]
    return correct / total


def main(argv=None):
    del argv
    cfg = _CONFIG.value
    set_seed(cfg.run.seed)
    (train_imgs, train_labels), (test_imgs, test_labels) = load_mnist(cfg.data.image_size, cfg.run.seed)
    rng = jax.random.PRNGKey(cfg.run.seed)
    state, model = create_state(rng, cfg)

    wandb_run = None
    if cfg.wandb.use:
        wandb_run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            dir=cfg.wandb.dir,
            config=cfg.to_dict(),
        )
        wandb.run.name = wandb.run.id

    for epoch in range(1, cfg.training.epochs + 1):
        rng, sub = jax.random.split(rng)
        state, loss, acc = train_epoch(state, model, train_imgs, train_labels, cfg, sub)
        test_acc = eval_model(state, model, test_imgs, test_labels, cfg)
        print(f"[epoch {epoch}] loss={loss:.4f} acc={acc:.4f} test_acc={test_acc:.4f}")
        if wandb_run:
            wandb.log({"train/loss": loss, "train/acc": acc, "test/acc": test_acc}, step=epoch)

        if epoch % cfg.training.checkpoint_every_epochs == 0:
            ckpt_dir = Path(cfg.checkpoint.dir)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / f"classifier_epoch_{epoch:03d}.npz"
            np.savez(ckpt_path, **state.params)

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    app.run(main)

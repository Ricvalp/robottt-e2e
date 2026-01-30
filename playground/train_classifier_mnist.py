#!/usr/bin/env python3
from __future__ import annotations

import random
from pathlib import Path
from typing import Tuple

from absl import app
from ml_collections import ConfigDict, config_flags
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import wandb
from tqdm import tqdm

from classifier_model import SmallResNet


# -------------------------
# Utils
# -------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def build_loaders(cfg: ConfigDict) -> Tuple[DataLoader, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.Resize(cfg.data.image_size),
            transforms.ToTensor(),
        ]
    )
    train_ds = datasets.MNIST(cfg.data.root, train=True, download=cfg.data.download, transform=transform)
    test_ds = datasets.MNIST(cfg.data.root, train=False, download=cfg.data.download, transform=transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.eval.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / total


# -------------------------
# Train
# -------------------------

def train(cfg: ConfigDict) -> None:
    device = torch.device(cfg.run.device if torch.cuda.is_available() or cfg.run.device == "cpu" else "cpu")
    set_seed(cfg.run.seed)
    train_loader, test_loader = build_loaders(cfg)

    model = SmallResNet(width=cfg.model.width, num_blocks=cfg.model.num_blocks, dropout=cfg.model.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    criterion = nn.CrossEntropyLoss()

    wandb_run = None
    if getattr(cfg, "wandb", None) and cfg.wandb.use:
        wandb_run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.run_name,
            dir=cfg.wandb.dir,
            config=cfg.to_dict(),
        )

    global_step = 0
    for epoch in range(1, cfg.training.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            if cfg.training.grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
            optimizer.step()

            global_step += 1
            if global_step % cfg.training.log_every == 0:
                preds = logits.argmax(dim=1)
                acc = (preds == y).float().mean().item()
                pbar.set_postfix(loss=loss.item(), acc=acc)
                if wandb_run is not None:
                    wandb.log({"train/loss": loss.item(), "train/acc": acc}, step=global_step)

        test_acc = evaluate(model, test_loader, device)
        print(f"[epoch {epoch}] test_acc={test_acc:.4f}")
        if wandb_run is not None:
            wandb.log({"test/acc": test_acc}, step=global_step)

        if epoch % cfg.training.checkpoint_every_epochs == 0:
            ckpt_dir = Path(cfg.checkpoint.dir)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / f"classifier_epoch_{epoch:03d}.pt"
            torch.save({"epoch": epoch, "model": model.state_dict()}, ckpt_path)

    if wandb_run is not None:
        wandb_run.finish()


# -------------------------
# Entry
# -------------------------

_CONFIG = config_flags.DEFINE_config_file(
    "classifier_config",
    default="playground/configs/train_classifier_mnist.py",
    help_string="Path to ml_collections config file.",
)


def main(argv=None):
    del argv
    cfg = _CONFIG.value
    train(cfg)


if __name__ == "__main__":
    app.run(main)

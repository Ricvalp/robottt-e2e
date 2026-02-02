#!/usr/bin/env python3
"""
ImageNet Classifier Training with DDP for FID computation.

Launch with:
    torchrun --nproc_per_node=N train_classifier_imagenet.py --classifier_config=configs/train_classifier_imagenet.py
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Tuple

from absl import app
from ml_collections import ConfigDict, config_flags
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
import wandb
from tqdm import tqdm

from classifier_model import SmallResNet


# -------------------------
# DDP Utilities
# -------------------------

def setup_ddp():
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    else:
        return 0, 0, 1


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def set_seed(seed: int, rank: int = 0) -> None:
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)
    torch.backends.cudnn.benchmark = True


# -------------------------
# Data
# -------------------------

def build_loaders(cfg: ConfigDict, world_size: int, rank: int) -> Tuple[DataLoader, DataLoader, DistributedSampler]:
    train_transform = transforms.Compose(
        [
            transforms.Resize(cfg.data.image_size),
            transforms.CenterCrop(cfg.data.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize(cfg.data.image_size),
            transforms.CenterCrop(cfg.data.image_size),
            transforms.ToTensor(),
        ]
    )
    train_ds = datasets.ImageFolder(cfg.data.train_dir, transform=train_transform)
    test_ds = datasets.ImageFolder(cfg.data.val_dir, transform=test_transform)

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.data.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.eval.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader, train_sampler


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Evaluating", leave=False) if is_main_process() else loader:
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
    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.run.seed, rank)

    if is_main_process():
        print(f"Training classifier with {world_size} GPU(s)")

    train_loader, test_loader, train_sampler = build_loaders(cfg, world_size, rank)

    model = SmallResNet(
        width=cfg.model.width,
        num_blocks=cfg.model.num_blocks,
        num_classes=cfg.model.num_classes,
        dropout=cfg.model.dropout,
        in_channels=cfg.model.in_channels,
    ).to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])
    
    model_module = model.module if isinstance(model, DDP) else model

    if is_main_process():
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Classifier parameters: {n_params/1e6:.2f}M ({n_params})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.epochs)
    criterion = nn.CrossEntropyLoss()

    wandb_run = None
    if is_main_process() and getattr(cfg, "wandb", None) and cfg.wandb.use:
        wandb_run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.run_name,
            dir=cfg.wandb.dir,
            config=cfg.to_dict(),
        )

    global_step = 0
    best_acc = 0.0
    
    for epoch in range(1, cfg.training.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False) if is_main_process() else train_loader
        
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
            if global_step % cfg.training.log_every == 0 and is_main_process():
                preds = logits.argmax(dim=1)
                acc = (preds == y).float().mean().item()
                if hasattr(pbar, 'set_postfix'):
                    pbar.set_postfix(loss=loss.item(), acc=acc)
                if wandb_run is not None:
                    wandb.log({"train/loss": loss.item(), "train/acc": acc, "lr": scheduler.get_last_lr()[0]}, step=global_step)

        scheduler.step()
        
        if is_main_process():
            test_acc = evaluate(model_module, test_loader, device)
            print(f"[epoch {epoch}] test_acc={test_acc:.4f}")
            if wandb_run is not None:
                wandb.log({"test/acc": test_acc}, step=global_step)

            if test_acc > best_acc:
                best_acc = test_acc
                ckpt_dir = Path(cfg.checkpoint.dir)
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                ckpt_path = ckpt_dir / "imagenet_classifier.pt"
                torch.save({"epoch": epoch, "model": model_module.state_dict(), "test_acc": test_acc}, ckpt_path)
                print(f"  -> Saved best model (acc={test_acc:.4f})")

    if is_main_process():
        print(f"Best test accuracy: {best_acc:.4f}")
    if wandb_run is not None:
        wandb_run.finish()
    
    cleanup_ddp()


# -------------------------
# Entry
# -------------------------

_CONFIG = config_flags.DEFINE_config_file(
    "classifier_config",
    default="playground/configs/train_classifier_imagenet.py",
    help_string="Path to ml_collections config file.",
)


def main(argv=None):
    del argv
    cfg = _CONFIG.value
    train(cfg)


if __name__ == "__main__":
    app.run(main)

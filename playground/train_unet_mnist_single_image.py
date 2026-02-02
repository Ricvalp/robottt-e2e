#!/usr/bin/env python3
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Tuple

from absl import app
from ml_collections import ConfigDict, config_flags
import numpy as np
import torch
from torch import nn, optim
import torch.amp as amp
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
import wandb
from tqdm import tqdm

from model import DDPM, UNet


# -------------------------
# Utilities
# -------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def build_dataloader_single_image(cfg: ConfigDict, train: bool = True) -> Tuple[torch.utils.data.Dataset, DataLoader, int]:
    name = cfg.data.dataset.lower()
    image_size = int(cfg.data.image_size)

    transform_list = [transforms.Resize(image_size), transforms.ToTensor(), transforms.Lambda(lambda x: x * 2 - 1)]

    if name == "mnist":
        dataset = datasets.MNIST(
            root=cfg.data.root,
            train=train,
            download=cfg.data.download,
            transform=transforms.Compose(transform_list),
        )
        channels = 1
        image = dataset.data[0].unsqueeze(0)  # Get the first image
        image = transforms.Resize(image_size)(image)  # Resize
        image = image / 255.0
        image = image * 2 - 1  # Apply transformations
        dataset = torch.utils.data.TensorDataset(image, torch.tensor([0]))
        
    elif name == "cifar10":
        dataset = datasets.CIFAR10(
            root=cfg.data.root,
            train=train,
            download=cfg.data.download,
            transform=transforms.Compose(transform_list),
        )
        channels = 3

    elif name == "cifar100":
        dataset = datasets.CIFAR100(
            root=cfg.data.root,
            train=train,
            download=cfg.data.download,
            transform=transforms.Compose(transform_list),
        )
        channels = 3

    else:
        raise ValueError(f"Unsupported dataset '{cfg.data.dataset}'. Use 'mnist', 'cifar10', or 'cifar100'.")

    loader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=cfg.data.shuffle if train else False,
        num_workers=cfg.data.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    return dataset, loader, channels


def save_samples(ddpm: DDPM, cfg: ConfigDict, epoch: int, device: torch.device) -> None:
    out_dir = Path(cfg.sample.dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        imgs = ddpm.sample(
            batch_size=cfg.sample.num_images,
            shape=(ddpm.model.in_conv.in_channels, cfg.data.image_size, cfg.data.image_size),
            device=device,
            steps=cfg.sample.steps,
        )
    imgs = (imgs.clamp(-1, 1) + 1) / 2  # back to [0,1]
    grid = utils.make_grid(imgs, nrow=int(cfg.sample.num_images ** 0.5))
    utils.save_image(grid, out_dir / f"epoch_{epoch:03d}.png")
    return grid
    
    


# -------------------------
# Training
# -------------------------

def train(cfg: ConfigDict) -> None:
    device = torch.device(cfg.run.device if torch.cuda.is_available() or cfg.run.device == "cpu" else "cpu")
    set_seed(cfg.run.seed)

    train_set, train_loader, channels = build_dataloader_single_image(cfg, train=True)
    if cfg.model.in_channels is None:
        cfg.model.in_channels = channels

    model = UNet(
        in_channels=cfg.model.in_channels,
        base_channels=cfg.model.base_channels,
        channel_mults=tuple(cfg.model.channel_mults),
        num_res_blocks=cfg.model.num_res_blocks,
        dropout=cfg.model.dropout,
        attn_resolutions=tuple(cfg.model.attn_resolutions),
        num_heads=cfg.model.num_heads,
        image_size=cfg.data.image_size,
        time_scale=1000.0,  # Standard time embedding scale
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params/1e6:.2f}M ({n_params} params)")

    ddpm = DDPM(
        model,
        beta_start=cfg.diffusion.beta_start,
        beta_end=cfg.diffusion.beta_end,
        train_steps=cfg.diffusion.train_steps,
    ).to(device)

    optimizer = optim.AdamW(ddpm.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    scaler = amp.GradScaler(
        device=device.type,
        enabled=cfg.training.use_amp and device.type == "cuda",
    )

    wandb_run = None
    if getattr(cfg, "wandb", None) and cfg.wandb.use:
        wandb_run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.run_name,
            dir=cfg.wandb.dir,
            config=cfg.to_dict(),
        )

    start_epoch = 1
    if cfg.checkpoint.resume:
        ckpt = torch.load(cfg.checkpoint.resume, map_location=device)
        ddpm.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from {cfg.checkpoint.resume} at epoch {start_epoch-1}")

    loss_ema = None
    global_step = (start_epoch - 1) * len(train_loader)
    for epoch in range(start_epoch, cfg.training.epochs + 1):
        ddpm.train()
        for step, (imgs, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}", leave=False), start=1):
            imgs = imgs.to(device)
            t = torch.rand(imgs.shape[0], device=device)

            optimizer.zero_grad(set_to_none=True)
            with amp.autocast(device.type, enabled=scaler.is_enabled()):
                loss = ddpm.p_losses(imgs, t)
            scaler.scale(loss).backward()
            if cfg.training.grad_clip is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(ddpm.parameters(), cfg.training.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            loss_val = loss.detach().item()
            loss_ema = loss_val if loss_ema is None else 0.99 * loss_ema + 0.01 * loss_val

            global_step = (epoch - 1) * len(train_loader) + step
            if global_step % cfg.training.log_every == 0:
                print(f"[epoch {epoch:03d} step {global_step:06d}] loss={loss_val:.4f} ema={loss_ema:.4f}")
                if wandb_run is not None:
                    wandb.log({"train/loss": loss_val, "train/loss_ema": loss_ema}, step=global_step)

        if epoch % cfg.training.sample_every_epochs == 0:
            ddpm.eval()
            grid = save_samples(ddpm, cfg, epoch, device)
            if wandb_run is not None:
                wandb.log({"samples": wandb.Image(grid)}, step=global_step)

        if epoch % cfg.training.checkpoint_every_epochs == 0:
            ckpt_dir = Path(cfg.checkpoint.dir)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / f"ddpm_epoch_{epoch:03d}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model": ddpm.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "cfg": cfg.to_dict(),
                },
                ckpt_path,
            )
    if wandb_run is not None:
        wandb_run.finish()


# -------------------------
# Entry point
# -------------------------

_CONFIG = config_flags.DEFINE_config_file(
    "config",
    default="playground/configs/train_unet_mnist.py",
    help_string="Path to a ml_collections config file.",
)


def main(argv=None):
    del argv
    cfg = _CONFIG.value
    train(cfg)


if __name__ == "__main__":
    app.run(main)

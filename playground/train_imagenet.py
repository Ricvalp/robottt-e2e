#!/usr/bin/env python3
"""
ImageNet 64x64 Class-Conditional Diffusion Training with DistributedDataParallel.

Launch with:
    torchrun --nproc_per_node=N train_imagenet.py --config=configs/train_imagenet.py
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Optional

from absl import app
from ml_collections import ConfigDict, config_flags
import numpy as np
import torch
import torch.distributed as dist
from torch import nn, optim
import torch.amp as amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms, utils
import wandb
from tqdm import tqdm

from playground.models.class_conditional_model import ClassCondUNet, DDPM


# -------------------------
# DDP Utilities
# -------------------------

def setup_ddp():
    """Initialize distributed training."""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    else:
        # Single GPU fallback
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

def build_dataloader(cfg: ConfigDict, world_size: int, rank: int) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.Resize(cfg.data.image_size),
            transforms.CenterCrop(cfg.data.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2 - 1),
        ]
    )
    train_ds = datasets.ImageFolder(
        root=cfg.data.train_dir,
        transform=transform,
    )
    
    sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    
    return DataLoader(
        train_ds,
        batch_size=cfg.data.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=True,
    ), sampler


def log_samples(
    ddpm_module: DDPM,
    device: torch.device,
    cfg: ConfigDict,
    wandb_run,
    global_step: int,
):
    """Generate and log samples for a few classes."""
    if wandb_run is None or not is_main_process():
        return
    
    ddpm_module.eval()
    num_per_class = 4
    classes_to_show = [0, 100, 250, 500, 750, 999]  # Sample from different classes
    
    all_samples = []
    with torch.no_grad():
        for cls in classes_to_show:
            class_ids = torch.full((num_per_class,), cls, device=device, dtype=torch.long)
            samples = ddpm_module.sample(
                class_ids,
                shape=(cfg.model.in_channels, cfg.data.image_size, cfg.data.image_size),
                device=device,
                steps=cfg.sample.steps,
                cfg_scale=cfg.sample.cfg_scale,
            )
            all_samples.append(samples)
    
    all_samples = torch.cat(all_samples, dim=0)
    all_samples = (all_samples.clamp(-1, 1) + 1) / 2
    grid = utils.make_grid(all_samples, nrow=num_per_class)
    wandb.log({"samples": wandb.Image(grid)}, step=global_step)
    ddpm_module.train()


def compute_fid_score(
    ddpm_module: DDPM,
    cfg: ConfigDict,
    device: torch.device,
) -> Optional[float]:
    """Compute FID score using InceptionV3 (standard FID) with class-balanced sampling."""
    if not is_main_process():
        return None
    
    try:
        from fid_utils import (
            get_inception_model, compute_fid_inception, load_fid_stats,
            has_fid_stats, save_fid_stats, compute_reference_stats_inception
        )
    except ImportError:
        print("Warning: fid_utils not available, skipping FID computation")
        return None
    
    stats_file = cfg.fid.stats_file
    dataset_key = "imagenet64"
    
    # Auto-compute reference stats if missing (using InceptionV3)
    if not has_fid_stats(stats_file, dataset_key):
        print(f"FID stats for '{dataset_key}' not found, computing from training data...")
        from torch.utils.data import DataLoader
        
        transform = transforms.Compose([
            transforms.Resize(cfg.data.image_size),
            transforms.CenterCrop(cfg.data.image_size),
            transforms.ToTensor(),
        ])
        ref_dataset = datasets.ImageFolder(cfg.data.train_dir, transform=transform)
        # Use 50k samples for standard FID (matches common benchmark protocol)
        subset_size = min(50000, len(ref_dataset))
        indices = torch.randperm(len(ref_dataset))[:subset_size].tolist()
        ref_subset = torch.utils.data.Subset(ref_dataset, indices)
        ref_loader = DataLoader(ref_subset, batch_size=64, shuffle=False, num_workers=4)
        mu, sigma = compute_reference_stats_inception(ref_loader, device)
        save_fid_stats(stats_file, dataset_key, mu, sigma)
    
    # Load reference stats
    reference_stats = load_fid_stats(stats_file, dataset_key)

    print("Computing FID score (class-balanced sampling)...")

    # Generate samples with class-balanced sampling
    # Uses round-robin class assignment for efficiency while maintaining balance
    ddpm_module.eval()
    num_samples = cfg.fid.num_samples
    batch_size = getattr(cfg.fid, 'batch_size', 256)
    num_classes = cfg.model.num_classes
    all_samples = []

    with torch.no_grad():
        for i in tqdm(range(0, num_samples, batch_size), desc="Generating samples for FID"):
            curr_batch = min(batch_size, num_samples - i)
            # Round-robin class assignment for class-balanced sampling
            # Sample i gets class (i % num_classes)
            class_ids = torch.tensor(
                [(i + j) % num_classes for j in range(curr_batch)],
                device=device,
                dtype=torch.long,
            )
            samples = ddpm_module.sample(
                class_ids,
                shape=(cfg.model.in_channels, cfg.data.image_size, cfg.data.image_size),
                device=device,
                steps=cfg.sample.steps,
                cfg_scale=cfg.sample.cfg_scale,
            )
            all_samples.append(samples.cpu())

    all_samples = torch.cat(all_samples, dim=0)
    
    # Compute FID using InceptionV3
    fid_score = compute_fid_inception(all_samples, reference_stats, device)
    ddpm_module.train()
    
    return fid_score


# -------------------------
# Training loop
# -------------------------

def train(cfg: ConfigDict) -> None:
    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.run.seed, rank)

    if is_main_process():
        print(f"Training with {world_size} GPU(s)")

    dataloader, sampler = build_dataloader(cfg, world_size, rank)

    model = ClassCondUNet(
        in_channels=cfg.model.in_channels,
        base_channels=cfg.model.base_channels,
        channel_mults=tuple(cfg.model.channel_mults),
        num_res_blocks=cfg.model.num_res_blocks,
        dropout=cfg.model.dropout,
        attn_resolutions=tuple(cfg.model.attn_resolutions),
        num_heads=cfg.model.num_heads,
        image_size=cfg.data.image_size,
        time_scale=1000.0,  # Standard time embedding scale
        num_classes=cfg.model.num_classes,
        cond_dim=cfg.model.cond_dim,
    ).to(device)

    ddpm = DDPM(
        model,
        log_snr_max=cfg.diffusion.log_snr_max,
        log_snr_min=cfg.diffusion.log_snr_min,
        cfg_drop_prob=cfg.diffusion.cfg_drop_prob,
        min_snr_gamma=cfg.diffusion.min_snr_gamma,
    ).to(device)

    if world_size > 1:
        ddpm = DDP(ddpm, device_ids=[local_rank])

    ddpm_module = ddpm.module if isinstance(ddpm, DDP) else ddpm

    if is_main_process():
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {n_params/1e6:.2f}M ({n_params})")

    optimizer = optim.AdamW(ddpm.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    scaler = amp.GradScaler(enabled=cfg.training.use_amp)

    wandb_run = None
    run_save_dir = Path(cfg.checkpoint.dir)
    if is_main_process() and getattr(cfg, "wandb", None) and cfg.wandb.use:
        wandb_run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=None,
            dir=cfg.wandb.dir,
            config=cfg.to_dict(),
        )
        if wandb.run is not None:
            wandb.run.name = wandb.run.id
            run_save_dir = Path(cfg.checkpoint.dir) / wandb.run.id
    if is_main_process():
        run_save_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0

    for epoch in range(1, cfg.training.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)

        ddpm.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False) if is_main_process() else dataloader
        epoch_loss = 0.0
        num_batches = 0

        for imgs, labels in pbar:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            with amp.autocast(device_type=device.type, enabled=cfg.training.use_amp):
                loss = ddpm_module.p_losses(imgs, labels)

            scaler.scale(loss).backward()
            if cfg.training.grad_clip is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(ddpm.parameters(), cfg.training.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            num_batches += 1
            epoch_loss += loss.item()

            if is_main_process() and hasattr(pbar, 'set_postfix'):
                pbar.set_postfix(loss=loss.item())

            if global_step % cfg.training.log_every == 0 and wandb_run is not None:
                wandb.log({"train/loss": loss.item()}, step=global_step)

        avg_loss = epoch_loss / max(num_batches, 1)
        if is_main_process():
            print(f"[Epoch {epoch}] avg_loss={avg_loss:.4f}")

        # Sample and log
        if epoch % cfg.training.sample_every_epochs == 0:
            log_samples(ddpm_module, device, cfg, wandb_run, global_step)
            if dist.is_initialized():
                dist.barrier()  # Sync all processes after sampling

        # FID evaluation
        if cfg.fid.enabled and epoch % cfg.training.fid_every_epochs == 0:
            fid_score = compute_fid_score(ddpm_module, cfg, device)
            if fid_score is not None:
                print(f"  FID score: {fid_score:.2f}")
                if wandb_run is not None:
                    wandb.log({"eval/fid": fid_score}, step=global_step)
            if dist.is_initialized():
                dist.barrier()  # Sync all processes after FID computation

        # Checkpoint (main process only)
        if is_main_process() and epoch % cfg.training.checkpoint_every_epochs == 0:
            ckpt_path = run_save_dir / f"imagenet_epoch_{epoch:03d}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model": ddpm_module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "cfg": cfg.to_dict(),
                },
                ckpt_path,
            )

    if wandb_run is not None:
        wandb_run.finish()

    cleanup_ddp()

# -------------------------
# Entry point
# -------------------------

_CONFIG = config_flags.DEFINE_config_file(
    "config",
    default="playground/configs/train_imagenet.py",
    help_string="Path to a ml_collections config file.",
)

def main(argv=None):
    del argv
    cfg = _CONFIG.value
    train(cfg)

if __name__ == "__main__":
    app.run(main)

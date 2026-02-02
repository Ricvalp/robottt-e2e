#!/usr/bin/env python3
from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

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

from playground.models.class_conditional_model import ClassCondUNet, DDPM


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


def build_dataloader(cfg: ConfigDict) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.Resize(cfg.data.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2 - 1),
        ]
    )
    train_ds = datasets.CIFAR100(
        root=cfg.data.root,
        train=True,
        download=cfg.data.download,
        transform=transform,
    )
    return DataLoader(
        train_ds,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )


def log_samples(
    ddpm: DDPM,
    device: torch.device,
    cfg: ConfigDict,
    wandb_run,
    global_step: int,
):
    """Generate and log samples for a few classes."""
    if wandb_run is None:
        return
    
    ddpm.eval()
    num_per_class = 4
    classes_to_show = [0, 10, 25, 50, 75, 99]  # Sample from different classes
    
    all_samples = []
    for cls in classes_to_show:
        class_ids = torch.full((num_per_class,), cls, device=device, dtype=torch.long)
        samples = ddpm.sample(
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
    ddpm.train()


def compute_fid_score(
    ddpm: DDPM,
    cfg: ConfigDict,
    device: torch.device,
) -> Optional[float]:
    """Compute FID score using a trained classifier and precomputed reference stats."""
    try:
        from fid_utils import (
            load_classifier_for_fid, load_fid_stats, compute_fid_from_samples,
            has_fid_stats, save_fid_stats, compute_reference_stats
        )
    except ImportError:
        print("Warning: fid_utils not available, skipping FID computation")
        return None
    
    stats_file = cfg.fid.stats_file
    classifier_path = cfg.fid.classifier_checkpoint
    dataset_key = "cifar100"
    
    if not Path(classifier_path).exists():
        print(f"Warning: Classifier checkpoint not found at {classifier_path}, skipping FID")
        return None
    
    # Load classifier
    classifier = load_classifier_for_fid(
        checkpoint_path=classifier_path,
        width=getattr(cfg.fid, 'classifier_width', 64),
        num_blocks=getattr(cfg.fid, 'classifier_num_blocks', 4),
        num_classes=cfg.model.num_classes,
        in_channels=cfg.model.in_channels,
        device=device,
    )
    
    # Auto-compute reference stats if missing
    if not has_fid_stats(stats_file, dataset_key):
        print(f"FID stats for '{dataset_key}' not found, computing from training data...")
        from torchvision import datasets, transforms
        from torch.utils.data import DataLoader
        
        transform = transforms.Compose([
            transforms.Resize(cfg.data.image_size),
            transforms.ToTensor(),
        ])
        ref_dataset = datasets.CIFAR100(
            root=cfg.data.root,
            train=True,
            download=False,
            transform=transform,
        )
        ref_loader = DataLoader(ref_dataset, batch_size=256, shuffle=False, num_workers=4)
        mu, sigma = compute_reference_stats(classifier, ref_loader, device, extractor_type="classifier")
        save_fid_stats(stats_file, dataset_key, mu, sigma)
    
    # Load reference stats
    reference_stats = load_fid_stats(stats_file, dataset_key)
    
    print("Computing FID score...")
    
    # Generate samples - use larger batch size for speed
    ddpm.eval()
    num_samples = cfg.fid.num_samples
    batch_size = getattr(cfg.fid, 'batch_size', 256)
    all_samples = []
    
    num_classes = cfg.model.num_classes
    
    with torch.no_grad():
        for i in tqdm(range(0, num_samples, batch_size), desc="Generating samples for FID"):
            curr_batch = min(batch_size, num_samples - i)
            # Distribute classes evenly across batch
            class_ids = torch.randint(0, num_classes, (curr_batch,), device=device)
            samples = ddpm.sample(
                class_ids,
                shape=(cfg.model.in_channels, cfg.data.image_size, cfg.data.image_size),
                device=device,
                steps=cfg.sample.steps,
                cfg_scale=cfg.sample.cfg_scale,
            )
            all_samples.append(samples.cpu())
    
    all_samples = torch.cat(all_samples, dim=0)
    
    # Compute FID
    fid_score = compute_fid_from_samples(classifier, all_samples, reference_stats, device)
    ddpm.train()
    
    return fid_score


# -------------------------
# Training loop
# -------------------------

def train(cfg: ConfigDict) -> None:
    device = torch.device(cfg.run.device if torch.cuda.is_available() or cfg.run.device == "cpu" else "cpu")
    set_seed(cfg.run.seed)

    dataloader = build_dataloader(cfg)

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
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params/1e6:.2f}M ({n_params})")

    ddpm = DDPM(
        model,
        log_snr_max=cfg.diffusion.log_snr_max,
        log_snr_min=cfg.diffusion.log_snr_min,
        
        cfg_drop_prob=cfg.diffusion.cfg_drop_prob,
    ).to(device)

    optimizer = optim.AdamW(ddpm.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    scaler = amp.GradScaler(device=device.type, enabled=cfg.training.use_amp and device.type == "cuda")

    wandb_run = None
    run_save_dir = Path(cfg.checkpoint.dir)
    if getattr(cfg, "wandb", None) and cfg.wandb.use:
        wandb_run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=None,
            dir=cfg.wandb.dir,
            config=cfg.to_dict(),
        )
        wandb.run.name = wandb.run.id
        run_save_dir = Path(cfg.checkpoint.dir) / wandb.run.id
    run_save_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    data_iter = iter(dataloader)
    
    for epoch in range(1, cfg.training.epochs + 1):
        ddpm.train()
        pbar = tqdm(range(cfg.training.steps_per_epoch), desc=f"Epoch {epoch}", leave=False)
        epoch_loss = 0.0
        
        for _ in pbar:
            # Get next batch, restart iterator if exhausted
            try:
                imgs, labels = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                imgs, labels = next(data_iter)
            
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            with amp.autocast(device_type=device.type, enabled=cfg.training.use_amp and device.type == "cuda"):
                loss = ddpm.p_losses(imgs, labels)
            
            scaler.scale(loss).backward()
            if cfg.training.grad_clip is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(ddpm.parameters(), cfg.training.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
            
            if global_step % cfg.training.log_every == 0:
                if wandb_run is not None:
                    wandb.log({"train/loss": loss.item()}, step=global_step)

        avg_loss = epoch_loss / cfg.training.steps_per_epoch
        print(f"[Epoch {epoch}] avg_loss={avg_loss:.4f}")

        # Sample and log
        if epoch % cfg.training.sample_every_epochs == 0:
            log_samples(ddpm, device, cfg, wandb_run, global_step)

        # FID evaluation
        if cfg.fid.enabled and epoch % cfg.training.fid_every_epochs == 0:
            fid_score = compute_fid_score(ddpm, cfg, device)
            if fid_score is not None:
                print(f"  FID score: {fid_score:.2f}")
                if wandb_run is not None:
                    wandb.log({"eval/fid": fid_score}, step=global_step)

        # Checkpoint
        if epoch % cfg.training.checkpoint_every_epochs == 0:
            ckpt_path = run_save_dir / f"cifar100_epoch_{epoch:03d}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
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
    default="playground/configs/train_cifar100.py",
    help_string="Path to a ml_collections config file.",
)


def main(argv=None):
    del argv
    cfg = _CONFIG.value
    train(cfg)


if __name__ == "__main__":
    app.run(main)

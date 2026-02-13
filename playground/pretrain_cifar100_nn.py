#!/usr/bin/env python3
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

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

from playground.dataset.cifar100_nearest_dataset import CIFAR100NearestContextDataset
from playground.models.conditional_model import ConditionalUNet, DDPM

try:
    from playground.fid_utils import (
        compute_fid_from_samples,
        compute_reference_stats,
        has_fid_stats,
        load_classifier_for_fid,
        load_fid_stats,
        save_fid_stats,
    )
except ImportError:
    from fid_utils import (  # type: ignore
        compute_fid_from_samples,
        compute_reference_stats,
        has_fid_stats,
        load_classifier_for_fid,
        load_fid_stats,
        save_fid_stats,
    )


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


def _opt_path(path_like: str) -> Optional[str]:
    path_str = str(path_like).strip()
    return path_str if path_str else None


def build_nearest_datasets(
    cfg: ConfigDict,
) -> Tuple[CIFAR100NearestContextDataset, CIFAR100NearestContextDataset]:
    k = int(cfg.data.cond_batch_size)

    train_ds = CIFAR100NearestContextDataset(
        k=k,
        split=str(cfg.data.train_split),
        index_path=_opt_path(getattr(cfg.data, "index_path_train", "")),
        meta_path=_opt_path(getattr(cfg.data, "meta_path_train", "")),
        index_dir=str(cfg.data.index_dir),
        cache_dir=_opt_path(getattr(cfg.data, "hf_cache_dir", "")),
        image_size=int(cfg.data.image_size),
        normalize_minus1_1=True,
        random_query=bool(cfg.data.random_query_train),
        seed=int(cfg.run.seed),
        return_metadata=bool(getattr(cfg.data, "return_metadata_train", False)),
        build_index_if_missing=bool(cfg.data.build_index_if_missing),
    )

    eval_ds = CIFAR100NearestContextDataset(
        k=k,
        split=str(cfg.data.eval_split),
        index_path=_opt_path(getattr(cfg.data, "index_path_eval", "")),
        meta_path=_opt_path(getattr(cfg.data, "meta_path_eval", "")),
        index_dir=str(cfg.data.index_dir),
        cache_dir=_opt_path(getattr(cfg.data, "hf_cache_dir", "")),
        image_size=int(cfg.data.image_size),
        normalize_minus1_1=True,
        random_query=bool(cfg.data.random_query_eval),
        seed=int(cfg.run.seed) + 17,
        return_metadata=bool(getattr(cfg.data, "return_metadata_eval", True)),
        build_index_if_missing=bool(cfg.data.build_index_if_missing),
    )

    return train_ds, eval_ds


def build_loader(
    dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    drop_last: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
    )


def next_batch(
    loader: DataLoader,
    it: Optional[Iterator],
):
    if it is None:
        it = iter(loader)
    try:
        batch = next(it)
    except StopIteration:
        it = iter(loader)
        batch = next(it)
    return batch, it


def split_query_and_context(
    batch,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Convert a dataset batch into:
    - query images x: (B, C, H, W)
    - context images cond: (B, K, C, H, W)
    """
    meta: Dict[str, torch.Tensor] = {}
    if isinstance(batch, dict):
        if "images" not in batch:
            raise KeyError("Expected key 'images' in metadata batch.")
        images = batch["images"]
        meta = {k: v for k, v in batch.items() if k != "images"}
    else:
        images = batch

    if not torch.is_tensor(images):
        raise TypeError(f"Expected tensor batch, got {type(images)}")
    if images.ndim != 5:
        raise ValueError(f"Expected images with shape (B, K+1, C, H, W), got {tuple(images.shape)}")

    x = images[:, 0].contiguous().to(device, non_blocking=True)
    cond = images[:, 1:].contiguous().to(device, non_blocking=True)
    return x, cond, meta


def compute_neighbor_label_agreement(meta: Dict[str, torch.Tensor]) -> Optional[float]:
    """Return fraction of neighbors with same label as query if labels are available."""
    if "query_label" not in meta or "neighbor_labels" not in meta:
        return None
    query_label = meta["query_label"]
    neighbor_labels = meta["neighbor_labels"]
    if not torch.is_tensor(query_label) or not torch.is_tensor(neighbor_labels):
        return None
    if query_label.ndim != 1 or neighbor_labels.ndim != 2:
        return None
    same = (neighbor_labels == query_label.unsqueeze(1)).float().mean().item()
    return float(same)


# -------------------------
# Logging / evaluation
# -------------------------


def _grid_nrow(n: int) -> int:
    nrow = int(np.sqrt(max(1, n)))
    if nrow * nrow < n:
        nrow += 1
    return nrow


def log_samples(
    ddpm: DDPM,
    eval_loader: DataLoader,
    eval_iter: Optional[Iterator],
    device: torch.device,
    cfg: ConfigDict,
    wandb_run,
    global_step: int,
) -> Optional[Iterator]:
    if wandb_run is None:
        return eval_iter

    batch, eval_iter = next_batch(eval_loader, eval_iter)
    x, cond, meta = split_query_and_context(batch, device)
    n = min(int(cfg.sample.num_images), x.shape[0])
    if n <= 0:
        return eval_iter

    was_training = ddpm.training
    ddpm.eval()
    with torch.no_grad():
        samples = ddpm.sample(
            batch_size=n,
            shape=(cfg.model.in_channels, cfg.data.image_size, cfg.data.image_size),
            device=device,
            steps=cfg.sample.steps,
            cond=cond[:n],
        )

    gen_vis = (samples.clamp(-1, 1) + 1) / 2
    query_vis = (x[:n].clamp(-1, 1) + 1) / 2
    cond_vis = (cond[:n, 0].clamp(-1, 1) + 1) / 2

    nrow = _grid_nrow(n)
    gen_grid = utils.make_grid(gen_vis, nrow=nrow)
    query_grid = utils.make_grid(query_vis, nrow=nrow)
    cond_grid = utils.make_grid(cond_vis, nrow=nrow)

    logs: Dict[str, object] = {
        "samples/generated": wandb.Image(gen_grid),
        "samples/query_target": wandb.Image(query_grid),
        "samples/conditioning_first": wandb.Image(cond_grid),
    }

    agreement = compute_neighbor_label_agreement(meta)
    if agreement is not None:
        logs["samples/neighbor_label_agreement"] = agreement

    wandb.log(logs, step=global_step)

    if was_training:
        ddpm.train()
    return eval_iter


def log_x0_debug(
    ddpm: DDPM,
    eval_loader: DataLoader,
    eval_iter: Optional[Iterator],
    device: torch.device,
    cfg: ConfigDict,
    wandb_run,
    global_step: int,
) -> Optional[Iterator]:
    if wandb_run is None:
        return eval_iter

    batch, eval_iter = next_batch(eval_loader, eval_iter)
    x, cond, _ = split_query_and_context(batch, device)

    n_imgs = min(max(1, int(getattr(cfg.fid, "x0_log_images", 8))), x.shape[0])
    x0 = x[:n_imgs]
    cond = cond[:n_imgs]

    was_training = ddpm.training
    ddpm.eval()
    with torch.no_grad():
        t = torch.rand(n_imgs, device=device)
        noise = torch.randn_like(x0)
        alpha, sigma = ddpm._alpha_sigma(t)
        alpha = alpha[:, None, None, None]
        sigma = sigma[:, None, None, None]

        x_t = alpha * x0 + sigma * noise
        v_pred = ddpm.model(x_t, t, cond)
        x0_pred = alpha * x_t - sigma * v_pred

        x0_vis = (x0.clamp(-1, 1) + 1) / 2
        xt_vis = (x_t.clamp(-1, 1) + 1) / 2
        x0_pred_vis = (x0_pred.clamp(-1, 1) + 1) / 2

        vis = torch.cat([x0_vis, xt_vis, x0_pred_vis], dim=0)
        grid = utils.make_grid(vis, nrow=n_imgs)

    wandb.log({"debug/x0_xt_x0pred": wandb.Image(grid)}, step=global_step)

    if was_training:
        ddpm.train()
    return eval_iter


def build_reference_loader(cfg: ConfigDict) -> DataLoader:
    """Build reference dataloader in [0, 1] range for FID stats."""
    transform = transforms.Compose(
        [
            transforms.Resize(cfg.data.image_size),
            transforms.ToTensor(),
        ]
    )

    use_train = str(getattr(cfg.fid, "reference_split", "train")) == "train"
    ref_ds = datasets.CIFAR100(
        root=cfg.data.root,
        train=use_train,
        download=cfg.data.download,
        transform=transform,
    )

    max_samples = int(getattr(cfg.fid, "reference_max_samples", 0))
    if max_samples > 0 and len(ref_ds) > max_samples:
        rng = np.random.default_rng(cfg.run.seed)
        idxs = rng.choice(len(ref_ds), size=max_samples, replace=False)
        ref_ds = torch.utils.data.Subset(ref_ds, idxs.tolist())

    return DataLoader(
        ref_ds,
        batch_size=cfg.fid.reference_batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def compute_fid_score(
    ddpm: DDPM,
    cfg: ConfigDict,
    fid_loader: DataLoader,
    device: torch.device,
) -> Optional[float]:
    """Compute classifier-feature FID for nearest-neighbor conditioned generation."""
    classifier_path = Path(cfg.fid.classifier_checkpoint)
    if not classifier_path.exists():
        print(f"Warning: classifier checkpoint not found at {classifier_path}, skipping FID.")
        return None

    classifier = load_classifier_for_fid(
        checkpoint_path=str(classifier_path),
        width=getattr(cfg.fid, "classifier_width", 64),
        num_blocks=getattr(cfg.fid, "classifier_num_blocks", 4),
        num_classes=100,
        in_channels=cfg.model.in_channels,
        device=device,
    )

    stats_file = cfg.fid.stats_file
    dataset_key = cfg.fid.dataset_key
    if not has_fid_stats(stats_file, dataset_key):
        print(f"FID stats for '{dataset_key}' not found, computing reference stats...")
        ref_loader = build_reference_loader(cfg)
        mu, sigma = compute_reference_stats(
            classifier,
            ref_loader,
            device,
            extractor_type="classifier",
        )
        save_fid_stats(stats_file, dataset_key, mu, sigma)

    reference_stats = load_fid_stats(stats_file, dataset_key)

    ddpm.eval()
    num_samples = int(cfg.fid.num_samples)
    all_samples = []

    fid_iter: Optional[Iterator] = None
    generated = 0
    with torch.no_grad():
        pbar = tqdm(total=num_samples, desc="Generating samples for FID", leave=False)
        while generated < num_samples:
            batch, fid_iter = next_batch(fid_loader, fid_iter)
            _, cond, _ = split_query_and_context(batch, device)
            curr_batch = min(cond.shape[0], num_samples - generated)
            if curr_batch <= 0:
                break

            samples = ddpm.sample(
                batch_size=curr_batch,
                shape=(cfg.model.in_channels, cfg.data.image_size, cfg.data.image_size),
                device=device,
                steps=cfg.sample.steps,
                cond=cond[:curr_batch],
            )
            all_samples.append(samples.cpu())
            generated += curr_batch
            pbar.update(curr_batch)
        pbar.close()

    if len(all_samples) == 0:
        print("Warning: no generated samples for FID, skipping.")
        ddpm.train()
        return None

    all_samples = torch.cat(all_samples, dim=0)
    fid_score = compute_fid_from_samples(
        classifier,
        all_samples,
        reference_stats,
        device,
        batch_size=getattr(cfg.fid, "feature_batch_size", 64),
        extractor_type="classifier",
    )
    ddpm.train()
    return fid_score


def save_checkpoint(
    ckpt_path: Path,
    ddpm: DDPM,
    optimizer: optim.Optimizer,
    scaler: amp.GradScaler,
    cfg: ConfigDict,
    epoch: int,
    global_step: int,
    best_fid: Optional[float],
) -> None:
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "checkpoint_type": "conditional_pretrain_nearest_neighbors",
            "epoch": epoch,
            "global_step": global_step,
            "best_fid": best_fid,
            "model": ddpm.state_dict(),
            "unet": ddpm.model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "finetune_hints": {
                "conditioning_source": "faiss_nearest_neighbors",
                "batch_layout": "(B, K+1, C, H, W) with [query, neighbors]",
                "cond_batch_size": int(cfg.data.cond_batch_size),
            },
            "dataset": {
                "train_split": str(cfg.data.train_split),
                "eval_split": str(cfg.data.eval_split),
                "index_dir": str(cfg.data.index_dir),
            },
            "cfg": cfg.to_dict(),
        },
        ckpt_path,
    )


# -------------------------
# Training loop
# -------------------------


def train(cfg: ConfigDict) -> None:
    device = torch.device(
        cfg.run.device if torch.cuda.is_available() or cfg.run.device == "cpu" else "cpu"
    )
    set_seed(cfg.run.seed)

    train_ds, eval_ds = build_nearest_datasets(cfg)
    train_loader = build_loader(
        train_ds,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        shuffle=True,
        drop_last=True,
    )
    eval_loader = build_loader(
        eval_ds,
        batch_size=cfg.sample.eval_batch_size,
        num_workers=cfg.data.num_workers,
        shuffle=True,
        drop_last=False,
    )
    fid_loader = build_loader(
        eval_ds,
        batch_size=cfg.fid.batch_size,
        num_workers=cfg.data.num_workers,
        shuffle=True,
        drop_last=False,
    )

    print(
        f"Train split: {cfg.data.train_split} ({len(train_ds)} samples) | "
        f"Eval split: {cfg.data.eval_split} ({len(eval_ds)} samples) | "
        f"K={cfg.data.cond_batch_size}"
    )

    model = ConditionalUNet(
        in_channels=cfg.model.in_channels,
        base_channels=cfg.model.base_channels,
        channel_mults=tuple(cfg.model.channel_mults),
        num_res_blocks=cfg.model.num_res_blocks,
        dropout=cfg.model.dropout,
        attn_resolutions=tuple(cfg.model.attn_resolutions),
        cross_attn_resolutions=tuple(cfg.model.cross_attn_resolutions),
        num_heads=cfg.model.num_heads,
        image_size=cfg.data.image_size,
        time_scale=1000.0,
        cond_dim=cfg.model.cond_dim,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params / 1e6:.2f}M ({n_params})")

    ddpm = DDPM(
        model,
        log_snr_max=cfg.diffusion.log_snr_max,
        log_snr_min=cfg.diffusion.log_snr_min,
    ).to(device)

    optimizer = optim.AdamW(
        ddpm.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )
    scaler = amp.GradScaler(
        device=device.type,
        enabled=cfg.training.use_amp and device.type == "cuda",
    )

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
        if wandb.run is not None:
            wandb.run.name = wandb.run.id
            run_save_dir = Path(cfg.checkpoint.dir) / wandb.run.id
    run_save_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 1
    global_step = 0
    best_fid: Optional[float] = None

    resume_path = str(getattr(cfg.checkpoint, "resume", "")).strip()
    if resume_path:
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        ddpm.load_state_dict(ckpt["model"])
        if not bool(getattr(cfg.training, "reset_optimizer_on_resume", False)):
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            if "scaler" in ckpt:
                scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        global_step = int(ckpt.get("global_step", 0))
        best_fid = ckpt.get("best_fid", None)
        print(f"Resumed from {resume_path} at epoch {start_epoch}, step {global_step}.")

    train_iter: Optional[Iterator] = None
    eval_iter: Optional[Iterator] = None

    for epoch in range(start_epoch, cfg.training.epochs + 1):
        ddpm.train()
        pbar = tqdm(range(cfg.training.steps_per_epoch), desc=f"Epoch {epoch}", leave=False)
        epoch_loss = 0.0

        for _ in pbar:
            batch, train_iter = next_batch(train_loader, train_iter)
            imgs, cond, meta = split_query_and_context(batch, device)

            optimizer.zero_grad(set_to_none=True)
            with amp.autocast(
                device_type=device.type,
                enabled=cfg.training.use_amp and device.type == "cuda",
            ):
                loss = ddpm.p_losses(imgs, cond=cond)

            scaler.scale(loss).backward()
            if cfg.training.grad_clip is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(ddpm.parameters(), cfg.training.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            epoch_loss += loss.item()

            postfix = {"loss": f"{loss.item():.4f}"}
            n_unique = None
            if "query_label" in meta and torch.is_tensor(meta["query_label"]):
                n_unique = int(torch.unique(meta["query_label"]).numel())
                postfix["uniq_cls"] = n_unique
            pbar.set_postfix(postfix)

            if global_step % cfg.training.log_every == 0 and wandb_run is not None:
                logs = {"train/loss": loss.item()}
                if n_unique is not None:
                    logs["train/unique_query_classes_in_batch"] = n_unique
                wandb.log(logs, step=global_step)

        avg_loss = epoch_loss / cfg.training.steps_per_epoch
        print(f"[Epoch {epoch}] avg_loss={avg_loss:.4f}")
        if wandb_run is not None:
            wandb.log({"train/epoch_loss": avg_loss}, step=global_step)

        if epoch % cfg.training.sample_every_epochs == 0:
            eval_iter = log_samples(
                ddpm,
                eval_loader,
                eval_iter,
                device,
                cfg,
                wandb_run,
                global_step,
            )

        if cfg.fid.enabled and epoch % cfg.training.fid_every_epochs == 0:
            fid_score = compute_fid_score(ddpm, cfg, fid_loader, device)
            if fid_score is not None:
                print(f"  FID score: {fid_score:.2f}")
                if wandb_run is not None:
                    wandb.log({"eval/fid": fid_score}, step=global_step)
                    eval_iter = log_x0_debug(
                        ddpm,
                        eval_loader,
                        eval_iter,
                        device,
                        cfg,
                        wandb_run,
                        global_step,
                    )

                if best_fid is None or fid_score < best_fid:
                    best_fid = fid_score
                    best_path = run_save_dir / "pretrain_nn_best_fid.pt"
                    save_checkpoint(
                        best_path,
                        ddpm,
                        optimizer,
                        scaler,
                        cfg,
                        epoch,
                        global_step,
                        best_fid,
                    )
                    print(f"  Saved new best-FID checkpoint to {best_path}")

        latest_path = run_save_dir / "pretrain_nn_latest.pt"
        save_checkpoint(
            latest_path,
            ddpm,
            optimizer,
            scaler,
            cfg,
            epoch,
            global_step,
            best_fid,
        )

        if epoch % cfg.training.checkpoint_every_epochs == 0:
            ckpt_path = run_save_dir / f"pretrain_nn_epoch_{epoch:03d}.pt"
            save_checkpoint(
                ckpt_path,
                ddpm,
                optimizer,
                scaler,
                cfg,
                epoch,
                global_step,
                best_fid,
            )

    if wandb_run is not None:
        wandb_run.finish()


# -------------------------
# Entry point
# -------------------------


_CONFIG = config_flags.DEFINE_config_file(
    "config",
    default="playground/configs/pretrain_cifar100_nn.py",
    help_string="Path to a ml_collections config file.",
)


def main(argv=None):
    del argv
    cfg = _CONFIG.value
    train(cfg)


if __name__ == "__main__":
    app.run(main)

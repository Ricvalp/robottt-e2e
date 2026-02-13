#!/usr/bin/env python3
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from absl import app
from ml_collections import ConfigDict, config_flags
import numpy as np
import torch
from torch import nn, optim
import torch.amp as amp
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, utils
import wandb
from tqdm import tqdm

from playground.models.conditional_model import ConditionalUNet, DDPM

try:
    from playground.fid_utils import (
        compute_fid_inception,
        compute_fid_from_samples,
        compute_reference_stats,
        compute_reference_stats_inception,
        has_fid_stats,
        load_classifier_for_fid,
        load_fid_stats,
        save_fid_stats,
    )
except ImportError:
    from fid_utils import (  # type: ignore
        compute_fid_inception,
        compute_fid_from_samples,
        compute_reference_stats,
        compute_reference_stats_inception,
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


def build_datasets(
    cfg: ConfigDict,
) -> Tuple[
    datasets.CIFAR100,
    Dict[int, List[int]],
    Dict[int, List[int]],
    List[int],
    List[int],
]:
    """Build CIFAR-100 and class-wise train/val index pools."""
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

    targets = np.array(train_ds.targets)
    holdout = int(cfg.data.holdout_per_class)
    num_classes = 100
    rng = np.random.default_rng(cfg.run.seed)

    train_indices_by_class: Dict[int, List[int]] = {}
    val_indices_by_class: Dict[int, List[int]] = {}

    if cfg.data.use_full_dataset:
        for cls in range(num_classes):
            idxs = np.where(targets == cls)[0].tolist()
            rng.shuffle(idxs)
            val_indices_by_class[cls] = idxs[:holdout]
            train_indices_by_class[cls] = idxs[holdout:]
        train_classes = list(range(num_classes))
        eval_classes = list(range(num_classes))
    else:
        leave_out = sorted({int(c) for c in cfg.data.leave_out_classes})
        for cls in range(num_classes):
            idxs = np.where(targets == cls)[0].tolist()
            rng.shuffle(idxs)
            if cls in leave_out:
                val_indices_by_class[cls] = idxs[: int(cfg.data.leaveout_eval_holdout)]
                train_indices_by_class[cls] = []
            else:
                val_indices_by_class[cls] = idxs[:holdout]
                train_indices_by_class[cls] = idxs[holdout:]
        train_classes = [c for c in range(num_classes) if c not in leave_out]
        eval_classes = leave_out

    return train_ds, train_indices_by_class, val_indices_by_class, train_classes, eval_classes


def build_cond_index_pool(
    train_indices_by_class: Dict[int, List[int]],
    val_indices_by_class: Dict[int, List[int]],
) -> Dict[int, List[int]]:
    """
    Prefer held-out-per-class indices for conditioning in eval/FID.
    Fall back to train indices when no holdout pool exists.
    """
    cond_pool: Dict[int, List[int]] = {}
    for cls in range(100):
        val_pool = val_indices_by_class.get(cls, [])
        train_pool = train_indices_by_class.get(cls, [])
        cond_pool[cls] = val_pool if len(val_pool) > 0 else train_pool
    return cond_pool


def build_train_loader(
    dataset: datasets.CIFAR100,
    train_indices_by_class: Dict[int, List[int]],
    cfg: ConfigDict,
) -> DataLoader:
    """Build shuffled train loader over all training indices from selected classes."""
    train_indices = []
    for cls in sorted(train_indices_by_class):
        train_indices.extend(train_indices_by_class[cls])
    if len(train_indices) == 0:
        raise ValueError("No training indices available to build DataLoader.")
    subset = Subset(dataset, train_indices)
    return DataLoader(
        subset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )


def sample_conditioning_for_batch(
    dataset: datasets.CIFAR100,
    indices_by_class: Dict[int, List[int]],
    labels: torch.Tensor,
    batch_size_cond: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Build per-sample conditioning exemplars.
    Returns cond of shape (B, K, C, H, W), where each sample i gets K images
    from the same class as labels[i].
    """
    labels_cpu = labels.detach().cpu()
    cond_sets = []
    for cls in labels_cpu.tolist():
        idx_pool = indices_by_class[int(cls)]
        if len(idx_pool) == 0:
            raise ValueError(f"Class {int(cls)} has no available indices.")
        replace = len(idx_pool) < batch_size_cond
        idx = np.random.choice(idx_pool, size=batch_size_cond, replace=replace)
        cond_imgs = torch.stack([dataset[i][0] for i in idx], dim=0)
        cond_sets.append(cond_imgs)
    return torch.stack(cond_sets, dim=0).to(device)


def select_classes(
    train_classes: List[int],
    eval_classes: List[int],
    use_eval_classes: bool,
) -> List[int]:
    if use_eval_classes and len(eval_classes) > 0:
        return eval_classes
    return train_classes


def evenly_spaced_subset(items: List[int], n: int) -> List[int]:
    if len(items) <= n:
        return list(items)
    idxs = np.linspace(0, len(items) - 1, n, dtype=int).tolist()
    return [items[i] for i in idxs]


# -------------------------
# Logging / evaluation
# -------------------------


def log_samples(
    ddpm: DDPM,
    dataset: datasets.CIFAR100,
    cond_indices_by_class: Dict[int, List[int]],
    classes_for_logging: List[int],
    device: torch.device,
    cfg: ConfigDict,
    wandb_run,
    global_step: int,
) -> None:
    if wandb_run is None or len(classes_for_logging) == 0:
        return

    ddpm.eval()
    all_generated = []
    all_cond = []
    num_per_class = int(cfg.sample.num_images_per_class)

    with torch.no_grad():
        for cls in classes_for_logging:
            class_ids = torch.full((num_per_class,), cls, dtype=torch.long)
            cond = sample_conditioning_for_batch(
                dataset,
                cond_indices_by_class,
                class_ids,
                batch_size_cond=cfg.data.cond_batch_size,
                device=device,
            )
            samples = ddpm.sample(
                batch_size=num_per_class,
                shape=(cfg.model.in_channels, cfg.data.image_size, cfg.data.image_size),
                device=device,
                steps=cfg.sample.steps,
                cond=cond,
            )
            all_generated.append(samples)

            # Visualize the first exemplar per generated sample.
            all_cond.append(cond[:, 0])

    gen = torch.cat(all_generated, dim=0)
    cond = torch.cat(all_cond, dim=0)
    gen = (gen.clamp(-1, 1) + 1) / 2
    cond = (cond.clamp(-1, 1) + 1) / 2

    gen_grid = utils.make_grid(gen, nrow=num_per_class)
    cond_grid = utils.make_grid(cond, nrow=num_per_class)

    wandb.log(
        {
            "samples/generated": wandb.Image(gen_grid),
            "samples/conditioning": wandb.Image(cond_grid),
            "samples/classes": ",".join(str(c) for c in classes_for_logging),
        },
        step=global_step,
    )
    ddpm.train()


def log_x0_debug(
    ddpm: DDPM,
    dataset: datasets.CIFAR100,
    cond_indices_by_class: Dict[int, List[int]],
    classes_for_logging: List[int],
    device: torch.device,
    cfg: ConfigDict,
    wandb_run,
    global_step: int,
) -> None:
    """
    Log x0 debugging visuals:
    row 1 = clean target x0, row 2 = noised x_t, row 3 = predicted x0 from model.
    """
    if wandb_run is None or len(classes_for_logging) == 0:
        return

    n_imgs = int(getattr(cfg.fid, "x0_log_images", 8))
    n_imgs = max(1, n_imgs)
    class_ids = torch.tensor(
        [classes_for_logging[i % len(classes_for_logging)] for i in range(n_imgs)],
        dtype=torch.long,
    )

    x0_list = []
    for cls in class_ids.tolist():
        idx_pool = cond_indices_by_class[int(cls)]
        if len(idx_pool) == 0:
            raise ValueError(f"Class {int(cls)} has no conditioning indices.")
        idx = int(np.random.choice(idx_pool))
        x0_list.append(dataset[idx][0])
    x0 = torch.stack(x0_list, dim=0).to(device)

    cond = sample_conditioning_for_batch(
        dataset,
        cond_indices_by_class,
        class_ids,
        batch_size_cond=cfg.data.cond_batch_size,
        device=device,
    )

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
        wandb.log(
            {
                "debug/x0_xt_x0pred": wandb.Image(grid),
                "debug/x0_classes": ",".join(str(c) for c in class_ids.tolist()),
            },
            step=global_step,
        )

    if was_training:
        ddpm.train()


def build_reference_loader(
    cfg: ConfigDict,
    classes_for_fid: List[int],
) -> DataLoader:
    """Build reference dataloader in [0, 1] range for FID stats."""
    transform = transforms.Compose(
        [
            transforms.Resize(cfg.data.image_size),
            transforms.ToTensor(),
        ]
    )
    ref_ds = datasets.CIFAR100(
        root=cfg.data.root,
        train=True,
        download=cfg.data.download,
        transform=transform,
    )
    targets = np.array(ref_ds.targets)
    cls_set = set(classes_for_fid)
    ref_indices = np.where(np.isin(targets, list(cls_set)))[0].tolist()

    max_samples = int(getattr(cfg.fid, "reference_max_samples", 0))
    if max_samples > 0 and len(ref_indices) > max_samples:
        rng = np.random.default_rng(cfg.run.seed)
        ref_indices = rng.choice(ref_indices, size=max_samples, replace=False).tolist()

    subset = Subset(ref_ds, ref_indices)
    return DataLoader(
        subset,
        batch_size=cfg.fid.reference_batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def compute_fid_score(
    ddpm: DDPM,
    cfg: ConfigDict,
    dataset: datasets.CIFAR100,
    cond_indices_by_class: Dict[int, List[int]],
    classes_for_fid: List[int],
    device: torch.device,
) -> Optional[Dict[str, float]]:
    """
    Compute classifier-feature FID and Inception FID from the same generated samples.
    Returns a dict with any available keys:
    - "fid_classifier"
    - "fid_inception"
    """
    if len(classes_for_fid) == 0:
        print("Warning: no classes available for FID, skipping.")
        return None

    stats_file = cfg.fid.stats_file
    dataset_key = cfg.fid.dataset_key
    inception_dataset_key = str(
        getattr(cfg.fid, "inception_dataset_key", f"{dataset_key}_inception")
    )
    ref_loader: Optional[DataLoader] = None

    def get_ref_loader() -> DataLoader:
        nonlocal ref_loader
        if ref_loader is None:
            ref_loader = build_reference_loader(cfg, classes_for_fid)
        return ref_loader

    classifier = None
    classifier_path = Path(cfg.fid.classifier_checkpoint)
    if classifier_path.exists():
        classifier = load_classifier_for_fid(
            checkpoint_path=str(classifier_path),
            width=getattr(cfg.fid, "classifier_width", 64),
            num_blocks=getattr(cfg.fid, "classifier_num_blocks", 4),
            num_classes=100,
            in_channels=cfg.model.in_channels,
            device=device,
        )
        if not has_fid_stats(stats_file, dataset_key):
            print(f"FID stats for '{dataset_key}' not found, computing classifier reference stats...")
            mu, sigma = compute_reference_stats(
                classifier,
                get_ref_loader(),
                device,
                extractor_type="classifier",
            )
            save_fid_stats(stats_file, dataset_key, mu, sigma)
    else:
        print(f"Warning: classifier checkpoint not found at {classifier_path}, skipping classifier FID.")

    if not has_fid_stats(stats_file, inception_dataset_key):
        print(
            f"Inception FID stats for '{inception_dataset_key}' not found, "
            "computing reference stats..."
        )
        mu_i, sigma_i = compute_reference_stats_inception(get_ref_loader(), device)
        save_fid_stats(stats_file, inception_dataset_key, mu_i, sigma_i)

    reference_stats_classifier = load_fid_stats(stats_file, dataset_key) if classifier is not None else None
    reference_stats_inception = load_fid_stats(stats_file, inception_dataset_key)

    ddpm.eval()
    num_samples = int(cfg.fid.num_samples)
    batch_size = int(cfg.fid.batch_size)
    all_samples = []

    n_classes = len(classes_for_fid)
    with torch.no_grad():
        for i in tqdm(range(0, num_samples, batch_size), desc="Generating samples for FID"):
            curr_batch = min(batch_size, num_samples - i)
            # Round-robin assignment gives class-balanced sampling across generated images.
            class_ids = torch.tensor(
                [classes_for_fid[(i + j) % n_classes] for j in range(curr_batch)],
                dtype=torch.long,
            )
            cond = sample_conditioning_for_batch(
                dataset,
                cond_indices_by_class,
                class_ids,
                batch_size_cond=cfg.data.cond_batch_size,
                device=device,
            )
            samples = ddpm.sample(
                batch_size=curr_batch,
                shape=(cfg.model.in_channels, cfg.data.image_size, cfg.data.image_size),
                device=device,
                steps=cfg.sample.steps,
                cond=cond,
            )
            all_samples.append(samples.cpu())

    all_samples = torch.cat(all_samples, dim=0)
    metrics: Dict[str, float] = {}
    if classifier is not None and reference_stats_classifier is not None:
        metrics["fid_classifier"] = compute_fid_from_samples(
            classifier,
            all_samples,
            reference_stats_classifier,
            device,
            batch_size=getattr(cfg.fid, "feature_batch_size", 64),
            extractor_type="classifier",
        )
    metrics["fid_inception"] = compute_fid_inception(
        all_samples,
        reference_stats_inception,
        device,
        batch_size=getattr(
            cfg.fid,
            "inception_feature_batch_size",
            getattr(cfg.fid, "feature_batch_size", 64),
        ),
    )
    ddpm.train()
    return metrics if len(metrics) > 0 else None


def save_checkpoint(
    ckpt_path: Path,
    ddpm: DDPM,
    optimizer: optim.Optimizer,
    scaler: amp.GradScaler,
    cfg: ConfigDict,
    epoch: int,
    global_step: int,
    best_fid: Optional[float],
    train_classes: List[int],
    eval_classes: List[int],
) -> None:
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "checkpoint_type": "conditional_pretrain",
            "epoch": epoch,
            "global_step": global_step,
            "best_fid": best_fid,
            "model": ddpm.state_dict(),
            "unet": ddpm.model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "class_split": {
                "train_classes": train_classes,
                "eval_classes": eval_classes,
            },
            "finetune_hints": {
                "fast_params_selector": getattr(cfg.fast_params, "selector", None),
                "cond_batch_size": int(cfg.data.cond_batch_size),
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

    dataset, train_indices_by_class, val_indices_by_class, train_classes, eval_classes = build_datasets(cfg)
    cond_indices_by_class = build_cond_index_pool(train_indices_by_class, val_indices_by_class)
    train_loader = build_train_loader(dataset, train_indices_by_class, cfg)

    if len(train_classes) == 0:
        raise ValueError("No train classes available. Check leave_out_classes/use_full_dataset.")
    print(
        f"Training classes: {len(train_classes)} | Eval classes: {len(eval_classes)} "
        f"| Train batches/epoch (loader): {len(train_loader)}"
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
    print(f"Model parameters: {n_params/1e6:.2f}M ({n_params})")

    ddpm = DDPM(
        model,
        log_snr_max=cfg.diffusion.log_snr_max,
        log_snr_min=cfg.diffusion.log_snr_min,
        p_uncond=float(getattr(cfg.diffusion, "p_uncond", 0.1)),
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

    classes_for_logging = select_classes(
        train_classes,
        eval_classes,
        bool(getattr(cfg.sample, "use_eval_classes", False)),
    )
    classes_for_logging = evenly_spaced_subset(
        classes_for_logging,
        int(cfg.sample.num_classes_to_show),
    )

    classes_for_fid = select_classes(
        train_classes,
        eval_classes,
        bool(getattr(cfg.fid, "use_eval_classes", False)),
    )

    data_iter = iter(train_loader)
    for epoch in range(start_epoch, cfg.training.epochs + 1):
        ddpm.train()
        pbar = tqdm(range(cfg.training.steps_per_epoch), desc=f"Epoch {epoch}", leave=False)
        epoch_loss = 0.0

        for _ in pbar:
            try:
                imgs, labels = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                imgs, labels = next(data_iter)

            imgs = imgs.to(device)
            cond = sample_conditioning_for_batch(
                dataset,
                train_indices_by_class,
                labels,
                batch_size_cond=cfg.data.cond_batch_size,
                device=device,
            )

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
            n_unique = len(set(labels.tolist()))
            pbar.set_postfix(loss=loss.item(), uniq_cls=n_unique)

            if global_step % cfg.training.log_every == 0 and wandb_run is not None:
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/unique_classes_in_batch": n_unique,
                    },
                    step=global_step,
                )

        avg_loss = epoch_loss / cfg.training.steps_per_epoch
        print(f"[Epoch {epoch}] avg_loss={avg_loss:.4f}")
        if wandb_run is not None:
            wandb.log({"train/epoch_loss": avg_loss}, step=global_step)

        if epoch % cfg.training.sample_every_epochs == 0:
            log_samples(
                ddpm,
                dataset,
                cond_indices_by_class,
                classes_for_logging,
                device,
                cfg,
                wandb_run,
                global_step,
            )

        if cfg.fid.enabled and epoch % cfg.training.fid_every_epochs == 0:
            fid_metrics = compute_fid_score(
                ddpm,
                cfg,
                dataset,
                cond_indices_by_class,
                classes_for_fid,
                device,
            )
            if fid_metrics is not None:
                fid_classifier = fid_metrics.get("fid_classifier", None)
                fid_inception = fid_metrics.get("fid_inception", None)
                if fid_classifier is not None:
                    print(f"  Classifier FID: {fid_classifier:.2f}")
                if fid_inception is not None:
                    print(f"  Inception FID: {fid_inception:.2f}")
                if wandb_run is not None:
                    wandb_logs: Dict[str, float] = {}
                    if fid_classifier is not None:
                        wandb_logs["eval/fid_classifier"] = fid_classifier
                        # Backward-compatible key.
                        wandb_logs["eval/fid"] = fid_classifier
                    if fid_inception is not None:
                        wandb_logs["eval/fid_inception"] = fid_inception
                    if len(wandb_logs) > 0:
                        wandb.log(wandb_logs, step=global_step)
                    log_x0_debug(
                        ddpm,
                        dataset,
                        cond_indices_by_class,
                        classes_for_fid,
                        device,
                        cfg,
                        wandb_run,
                        global_step,
                    )

                # Keep checkpoint selection stable: prefer classifier FID when available.
                fid_for_model_selection = (
                    fid_classifier if fid_classifier is not None else fid_inception
                )
                if (
                    fid_for_model_selection is not None
                    and (best_fid is None or fid_for_model_selection < best_fid)
                ):
                    best_fid = fid_for_model_selection
                    best_path = run_save_dir / "pretrain_best_fid.pt"
                    save_checkpoint(
                        best_path,
                        ddpm,
                        optimizer,
                        scaler,
                        cfg,
                        epoch,
                        global_step,
                        best_fid,
                        train_classes,
                        eval_classes,
                    )
                    print(f"  Saved new best-FID checkpoint to {best_path}")

        latest_path = run_save_dir / "pretrain_latest.pt"
        save_checkpoint(
            latest_path,
            ddpm,
            optimizer,
            scaler,
            cfg,
            epoch,
            global_step,
            best_fid,
            train_classes,
            eval_classes,
        )

        if epoch % cfg.training.checkpoint_every_epochs == 0:
            ckpt_path = run_save_dir / f"pretrain_epoch_{epoch:03d}.pt"
            save_checkpoint(
                ckpt_path,
                ddpm,
                optimizer,
                scaler,
                cfg,
                epoch,
                global_step,
                best_fid,
                train_classes,
                eval_classes,
            )

    if wandb_run is not None:
        wandb_run.finish()


# -------------------------
# Entry point
# -------------------------


_CONFIG = config_flags.DEFINE_config_file(
    "config",
    default="playground/configs/pretrain_cifar100.py",
    help_string="Path to a ml_collections config file.",
)


def main(argv=None):
    del argv
    cfg = _CONFIG.value
    train(cfg)


if __name__ == "__main__":
    app.run(main)

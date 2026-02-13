#!/usr/bin/env python3
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

from absl import app
from ml_collections import ConfigDict, config_flags
import numpy as np
import torch
import torch.amp as amp
import torch.distributed as dist
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms, utils
import wandb
from tqdm import tqdm

from playground.dataset.imagenet_nearest_dataset import ImageNetNearestContextDataset
from playground.models.conditional_model import ConditionalUNet, DDPM

try:
    from playground.fid_utils import (
        compute_fid_inception,
        compute_reference_stats_inception,
        has_fid_stats,
        load_fid_stats,
        save_fid_stats,
    )
except ImportError:
    from fid_utils import (  # type: ignore
        compute_fid_inception,
        compute_reference_stats_inception,
        has_fid_stats,
        load_fid_stats,
        save_fid_stats,
    )


# -------------------------
# Distributed helpers
# -------------------------


def setup_distributed(cfg: ConfigDict) -> Tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if not distributed:
        return False, 0, 1, 0

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    backend = str(getattr(getattr(cfg, "ddp", ConfigDict()), "backend", "nccl"))
    if backend == "nccl" and not torch.cuda.is_available():
        backend = "gloo"

    dist.init_process_group(backend=backend, init_method="env://")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return True, rank, world_size, local_rank


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def maybe_barrier(distributed: bool) -> None:
    if distributed and dist.is_available() and dist.is_initialized():
        dist.barrier()


def reduce_mean(t: torch.Tensor, distributed: bool, world_size: int) -> torch.Tensor:
    if not distributed:
        return t
    rt = t.detach().clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


def is_main_process(rank: int) -> bool:
    return rank == 0


def unwrap_ddpm(ddpm_model: DDPM | DDP) -> DDPM:
    return ddpm_model.module if isinstance(ddpm_model, DDP) else ddpm_model


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


def build_train_dataset(cfg: ConfigDict, rank: int) -> ImageNetNearestContextDataset:
    return ImageNetNearestContextDataset(
        imagenet_dir=str(cfg.data.train_dir),
        embeddings_path=str(cfg.data.train_embeddings),
        index_path=str(cfg.data.train_faiss_index),
        index_meta_path=str(getattr(cfg.data, "train_faiss_meta", "")).strip() or None,
        k=int(cfg.data.cond_batch_size),
        image_size=int(cfg.data.image_size),
        normalize_minus1_1=True,
        random_query=bool(cfg.data.random_query_train),
        seed=int(cfg.run.seed) + rank * 100_003,
        return_metadata=bool(getattr(cfg.data, "return_metadata_train", False)),
        faiss_nprobe=int(getattr(cfg.data, "faiss_nprobe_train", 64)),
        random_horizontal_flip_prob=float(getattr(cfg.data, "train_random_flip_prob", 0.0)),
    )


def build_eval_dataset(cfg: ConfigDict) -> ImageNetNearestContextDataset:
    return ImageNetNearestContextDataset(
        imagenet_dir=str(cfg.data.eval_dir),
        embeddings_path=str(cfg.data.eval_embeddings),
        index_path=str(cfg.data.eval_faiss_index),
        index_meta_path=str(getattr(cfg.data, "eval_faiss_meta", "")).strip() or None,
        k=int(cfg.data.cond_batch_size),
        image_size=int(cfg.data.image_size),
        normalize_minus1_1=True,
        random_query=bool(cfg.data.random_query_eval),
        seed=int(cfg.run.seed) + 999_983,
        return_metadata=bool(getattr(cfg.data, "return_metadata_eval", True)),
        faiss_nprobe=int(getattr(cfg.data, "faiss_nprobe_eval", 64)),
        random_horizontal_flip_prob=0.0,
    )


def build_loader(
    dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    drop_last: bool,
    sampler=None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
    )


def next_batch(loader: DataLoader, it: Optional[Iterator]):
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
        raise ValueError(f"Expected images (B, K+1, C, H, W), got {tuple(images.shape)}")

    x = images[:, 0].contiguous().to(device, non_blocking=True)
    cond = images[:, 1:].contiguous().to(device, non_blocking=True)
    return x, cond, meta


def compute_neighbor_label_agreement(meta: Dict[str, torch.Tensor]) -> Optional[float]:
    if "query_label" not in meta or "neighbor_labels" not in meta:
        return None
    q = meta["query_label"]
    n = meta["neighbor_labels"]
    if not torch.is_tensor(q) or not torch.is_tensor(n):
        return None
    if q.ndim != 1 or n.ndim != 2:
        return None
    return float((n == q.unsqueeze(1)).float().mean().item())


def _grid_nrow(n: int) -> int:
    nrow = int(np.sqrt(max(1, n)))
    if nrow * nrow < n:
        nrow += 1
    return nrow


# -------------------------
# Logging / evaluation
# -------------------------


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
    logs: Dict[str, object] = {
        "samples/generated": wandb.Image(utils.make_grid(gen_vis, nrow=nrow)),
        "samples/query_target": wandb.Image(utils.make_grid(query_vis, nrow=nrow)),
        "samples/conditioning_first": wandb.Image(utils.make_grid(cond_vis, nrow=nrow)),
    }
    agreement = compute_neighbor_label_agreement(meta)
    if agreement is not None:
        logs["samples/neighbor_label_agreement"] = agreement

    wandb_run.log(logs, step=global_step)

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

    wandb_run.log({"debug/x0_xt_x0pred": wandb.Image(grid)}, step=global_step)

    if was_training:
        ddpm.train()
    return eval_iter


def build_reference_loader(cfg: ConfigDict) -> DataLoader:
    ref_dir = str(getattr(cfg.fid, "reference_dir", "")).strip() or str(cfg.data.eval_dir)
    transform = transforms.Compose(
        [
            transforms.Resize(cfg.data.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(cfg.data.image_size),
            transforms.ToTensor(),
        ]
    )
    ref_ds = datasets.ImageFolder(root=ref_dir, transform=transform)

    max_samples = int(getattr(cfg.fid, "reference_max_samples", 0))
    if max_samples > 0 and len(ref_ds) > max_samples:
        rng = np.random.default_rng(cfg.run.seed)
        idxs = rng.choice(len(ref_ds), size=max_samples, replace=False)
        ref_ds = torch.utils.data.Subset(ref_ds, idxs.tolist())

    return DataLoader(
        ref_ds,
        batch_size=int(cfg.fid.reference_batch_size),
        shuffle=False,
        num_workers=int(cfg.data.num_workers),
        pin_memory=torch.cuda.is_available(),
    )


def compute_fid_score(
    ddpm: DDPM,
    cfg: ConfigDict,
    fid_loader: DataLoader,
    device: torch.device,
) -> Optional[float]:
    """Compute Inception-FID for nearest-neighbor-conditioned generation."""
    stats_file = str(cfg.fid.stats_file)
    dataset_key = str(cfg.fid.dataset_key)

    if not has_fid_stats(stats_file, dataset_key):
        print(f"Inception FID stats for '{dataset_key}' not found, computing reference stats...")
        ref_loader = build_reference_loader(cfg)
        mu, sigma = compute_reference_stats_inception(ref_loader, device)
        save_fid_stats(stats_file, dataset_key, mu, sigma)

    reference_stats = load_fid_stats(stats_file, dataset_key)

    was_training = ddpm.training
    ddpm.eval()

    num_samples = int(cfg.fid.num_samples)
    all_samples = []
    generated = 0
    fid_iter: Optional[Iterator] = None

    with torch.no_grad():
        pbar = tqdm(total=num_samples, desc="Generating samples for Inception FID", leave=False)
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
        if was_training:
            ddpm.train()
        return None

    all_samples = torch.cat(all_samples, dim=0)
    fid_score = compute_fid_inception(
        all_samples,
        reference_stats,
        device,
        batch_size=int(getattr(cfg.fid, "feature_batch_size", 64)),
    )

    if was_training:
        ddpm.train()
    return float(fid_score)


def save_checkpoint(
    ckpt_path: Path,
    ddpm: DDPM,
    optimizer: optim.Optimizer,
    scaler: amp.GradScaler,
    cfg: ConfigDict,
    epoch: int,
    global_step: int,
    best_fid: Optional[float],
    rank: int,
    world_size: int,
) -> None:
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "checkpoint_type": "conditional_pretrain_imagenet_nearest_neighbors",
            "epoch": epoch,
            "global_step": global_step,
            "best_fid": best_fid,
            "model": ddpm.state_dict(),
            "unet": ddpm.model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "distributed": {
                "rank": int(rank),
                "world_size": int(world_size),
            },
            "finetune_hints": {
                "conditioning_source": "faiss_nearest_neighbors",
                "batch_layout": "(B, K+1, C, H, W) with [query, neighbors]",
                "cond_batch_size": int(cfg.data.cond_batch_size),
            },
            "dataset": {
                "train_dir": str(cfg.data.train_dir),
                "eval_dir": str(cfg.data.eval_dir),
                "train_embeddings": str(cfg.data.train_embeddings),
                "eval_embeddings": str(cfg.data.eval_embeddings),
                "train_faiss_index": str(cfg.data.train_faiss_index),
                "eval_faiss_index": str(cfg.data.eval_faiss_index),
            },
            "cfg": cfg.to_dict(),
        },
        ckpt_path,
    )


# -------------------------
# Training loop
# -------------------------


def train(cfg: ConfigDict) -> None:
    distributed, rank, world_size, local_rank = setup_distributed(cfg)
    main = is_main_process(rank)

    if torch.cuda.is_available() and str(cfg.run.device).startswith("cuda"):
        device = torch.device(f"cuda:{local_rank}" if distributed else "cuda")
    else:
        device = torch.device("cpu")

    set_seed(int(cfg.run.seed) + rank)

    train_ds = build_train_dataset(cfg, rank=rank)
    train_sampler = (
        DistributedSampler(
            train_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )
        if distributed
        else None
    )

    if int(cfg.data.num_workers) > 0 and main:
        print(
            "Note: ImageNet NN dataset performs FAISS search in __getitem__. "
            "Large num_workers can duplicate index memory."
        )

    train_loader = build_loader(
        train_ds,
        batch_size=int(cfg.data.batch_size),
        num_workers=int(cfg.data.num_workers),
        shuffle=(train_sampler is None),
        drop_last=True,
        sampler=train_sampler,
    )

    eval_loader = None
    fid_loader = None
    if main:
        eval_ds = build_eval_dataset(cfg)
        eval_loader = build_loader(
            eval_ds,
            batch_size=int(cfg.sample.eval_batch_size),
            num_workers=int(cfg.data.num_workers),
            shuffle=True,
            drop_last=False,
        )
        fid_loader = build_loader(
            eval_ds,
            batch_size=int(cfg.fid.batch_size),
            num_workers=int(cfg.data.num_workers),
            shuffle=True,
            drop_last=False,
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

    raw_ddpm = DDPM(
        model,
        log_snr_max=cfg.diffusion.log_snr_max,
        log_snr_min=cfg.diffusion.log_snr_min,
    ).to(device)

    if main:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {n_params / 1e6:.2f}M ({n_params})")
        print(
            f"Train images: {len(train_ds)} | "
            f"Eval images: {len(eval_loader.dataset) if eval_loader is not None else 0} | "
            f"K={cfg.data.cond_batch_size} | world_size={world_size}"
        )

    start_epoch = 1
    global_step = 0
    best_fid: Optional[float] = None
    resume_ckpt = None

    resume_path = str(getattr(cfg.checkpoint, "resume", "")).strip()
    if resume_path:
        resume_ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        raw_ddpm.load_state_dict(resume_ckpt["model"])
        start_epoch = int(resume_ckpt.get("epoch", 0)) + 1
        global_step = int(resume_ckpt.get("global_step", 0))
        best_fid = resume_ckpt.get("best_fid", None)
        if main:
            print(f"Resumed model from {resume_path} at epoch {start_epoch}, step {global_step}.")

    ddpm_train: DDPM | DDP
    if distributed:
        ddpm_train = DDP(
            raw_ddpm,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            find_unused_parameters=bool(getattr(getattr(cfg, "ddp", ConfigDict()), "find_unused_parameters", False)),
        )
    else:
        ddpm_train = raw_ddpm

    optimizer = optim.AdamW(
        ddpm_train.parameters(),
        lr=float(cfg.training.lr),
        weight_decay=float(cfg.training.weight_decay),
    )
    scaler = amp.GradScaler(
        device=device.type,
        enabled=bool(cfg.training.use_amp and device.type == "cuda"),
    )

    if resume_ckpt is not None and not bool(getattr(cfg.training, "reset_optimizer_on_resume", False)):
        if "optimizer" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer"])
        if "scaler" in resume_ckpt:
            scaler.load_state_dict(resume_ckpt["scaler"])

    wandb_run = None
    run_save_dir = Path(cfg.checkpoint.dir)
    if main and getattr(cfg, "wandb", None) and cfg.wandb.use:
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
    if main:
        run_save_dir.mkdir(parents=True, exist_ok=True)

    train_iter: Optional[Iterator] = None
    eval_iter: Optional[Iterator] = None

    for epoch in range(start_epoch, int(cfg.training.epochs) + 1):
        if distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        ddpm_train.train()
        pbar = tqdm(
            range(int(cfg.training.steps_per_epoch)),
            desc=f"Epoch {epoch}",
            leave=False,
            disable=not main,
        )
        epoch_loss = 0.0

        for _ in pbar:
            batch, train_iter = next_batch(train_loader, train_iter)
            imgs, cond, meta = split_query_and_context(batch, device)

            optimizer.zero_grad(set_to_none=True)
            with amp.autocast(device_type=device.type, enabled=bool(cfg.training.use_amp and device.type == "cuda")):
                loss = ddpm_train(imgs, cond=cond)

            scaler.scale(loss).backward()
            if cfg.training.grad_clip is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(ddpm_train.parameters(), float(cfg.training.grad_clip))
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            loss_mean = reduce_mean(loss.detach(), distributed=distributed, world_size=world_size)
            loss_item = float(loss_mean.item())
            epoch_loss += loss_item

            if main:
                postfix = {"loss": f"{loss_item:.4f}"}
                if "query_label" in meta and torch.is_tensor(meta["query_label"]):
                    postfix["uniq_cls"] = int(torch.unique(meta["query_label"]).numel())
                pbar.set_postfix(postfix)

                if global_step % int(cfg.training.log_every) == 0 and wandb_run is not None:
                    logs: Dict[str, float] = {"train/loss": loss_item}
                    if "query_label" in meta and torch.is_tensor(meta["query_label"]):
                        logs["train/unique_query_classes_in_batch"] = float(
                            torch.unique(meta["query_label"]).numel()
                        )
                    wandb_run.log(logs, step=global_step)

        if main:
            avg_loss = epoch_loss / int(cfg.training.steps_per_epoch)
            print(f"[Epoch {epoch}] avg_loss={avg_loss:.4f}")
            if wandb_run is not None:
                wandb_run.log({"train/epoch_loss": avg_loss}, step=global_step)

            ddpm_eval = unwrap_ddpm(ddpm_train)

            if epoch % int(cfg.training.sample_every_epochs) == 0 and eval_loader is not None:
                eval_iter = log_samples(
                    ddpm_eval,
                    eval_loader,
                    eval_iter,
                    device,
                    cfg,
                    wandb_run,
                    global_step,
                )

            if (
                bool(cfg.fid.enabled)
                and epoch % int(cfg.training.fid_every_epochs) == 0
                and fid_loader is not None
            ):
                fid_score = compute_fid_score(ddpm_eval, cfg, fid_loader, device)
                if fid_score is not None:
                    print(f"  Inception FID: {fid_score:.2f}")
                    if wandb_run is not None:
                        wandb_run.log(
                            {
                                "eval/fid_inception": fid_score,
                                # backward-compatible key
                                "eval/fid": fid_score,
                            },
                            step=global_step,
                        )
                        if eval_loader is not None:
                            eval_iter = log_x0_debug(
                                ddpm_eval,
                                eval_loader,
                                eval_iter,
                                device,
                                cfg,
                                wandb_run,
                                global_step,
                            )

                    if best_fid is None or fid_score < best_fid:
                        best_fid = fid_score
                        best_path = run_save_dir / "pretrain_imagenet_nn_best_fid.pt"
                        save_checkpoint(
                            best_path,
                            ddpm_eval,
                            optimizer,
                            scaler,
                            cfg,
                            epoch,
                            global_step,
                            best_fid,
                            rank,
                            world_size,
                        )
                        print(f"  Saved new best-FID checkpoint to {best_path}")

            ddpm_eval = unwrap_ddpm(ddpm_train)
            latest_path = run_save_dir / "pretrain_imagenet_nn_latest.pt"
            save_checkpoint(
                latest_path,
                ddpm_eval,
                optimizer,
                scaler,
                cfg,
                epoch,
                global_step,
                best_fid,
                rank,
                world_size,
            )

            if epoch % int(cfg.training.checkpoint_every_epochs) == 0:
                ckpt_path = run_save_dir / f"pretrain_imagenet_nn_epoch_{epoch:03d}.pt"
                save_checkpoint(
                    ckpt_path,
                    ddpm_eval,
                    optimizer,
                    scaler,
                    cfg,
                    epoch,
                    global_step,
                    best_fid,
                    rank,
                    world_size,
                )

        maybe_barrier(distributed)

    if main and wandb_run is not None:
        wandb_run.finish()

    cleanup_distributed()


# -------------------------
# Entry point
# -------------------------


_CONFIG = config_flags.DEFINE_config_file(
    "config",
    default="playground/configs/pretrain_imagenet_nn.py",
    help_string="Path to a ml_collections config file.",
)


def main(argv=None):
    del argv
    cfg = _CONFIG.value
    train(cfg)


if __name__ == "__main__":
    app.run(main)

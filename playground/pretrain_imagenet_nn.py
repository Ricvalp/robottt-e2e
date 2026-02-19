#!/usr/bin/env python3
from __future__ import annotations

import os
import random
import re
import shutil
import subprocess
import sys
import importlib.util
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple
from datetime import timedelta

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

    dist.init_process_group(backend=backend, init_method="env://", timeout=timedelta(minutes=60))
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
    # Show first 4 conditioning exemplars per query as an (N rows x 4 cols) grid.
    n_cond_show = 4
    cond_show = cond[:n, : min(n_cond_show, cond.shape[1])]
    if cond_show.shape[1] < n_cond_show:
        pad = -torch.ones(
            n,
            n_cond_show - cond_show.shape[1],
            cond_show.shape[2],
            cond_show.shape[3],
            cond_show.shape[4],
            device=cond_show.device,
            dtype=cond_show.dtype,
        )
        cond_show = torch.cat([cond_show, pad], dim=1)
    cond_vis = (cond_show.clamp(-1, 1) + 1) / 2
    cond_vis = cond_vis.reshape(n * n_cond_show, cond_vis.shape[2], cond_vis.shape[3], cond_vis.shape[4])

    nrow = _grid_nrow(n)
    logs: Dict[str, object] = {
        "samples/generated": wandb.Image(utils.make_grid(gen_vis, nrow=nrow)),
        "samples/query_target": wandb.Image(utils.make_grid(query_vis, nrow=nrow)),
        "samples/conditioning_first4": wandb.Image(utils.make_grid(cond_vis, nrow=n_cond_show)),
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


def _device_arg_for_pytorch_fid(device: torch.device) -> str:
    if device.type == "cuda":
        idx = 0 if device.index is None else int(device.index)
        return f"cuda:{idx}"
    return "cpu"


def _save_generated_samples(samples: torch.Tensor, out_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vis = ((samples.clamp(-1, 1) + 1) / 2).cpu()
    for i, img in enumerate(vis):
        utils.save_image(img, out_dir / f"{i:06d}.png")


def _run_pytorch_fid_command(cmd: list[str]) -> Tuple[bool, str]:
    try:
        res = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        return False, str(exc)
    text = (res.stdout or "") + ("\n" + res.stderr if res.stderr else "")
    return res.returncode == 0, text


def _pytorch_fid_available() -> bool:
    return importlib.util.find_spec("pytorch_fid") is not None


def _collect_image_files_for_fid(root_dir: str) -> list[str]:
    root = Path(root_dir)
    if not root.exists():
        return []
    allowed_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in allowed_ext:
            files.append(str(p))
    files.sort()
    return files


def _build_pytorch_fid_stats_from_files(
    files: list[str],
    stats_path: Path,
    cfg: ConfigDict,
    device: torch.device,
) -> bool:
    if len(files) == 0:
        print("Warning: no reference images found to build pytorch-fid stats.")
        return False
    try:
        from pytorch_fid.fid_score import InceptionV3, calculate_activation_statistics
    except Exception as exc:
        print(f"Warning: failed to import pytorch_fid API: {exc}")
        return False

    dims = int(getattr(cfg.fid, "pytorch_fid_dims", 2048))
    batch_size = int(getattr(cfg.fid, "pytorch_fid_batch_size", 64))
    num_workers = int(getattr(cfg.fid, "pytorch_fid_num_workers", 0))
    if dims not in InceptionV3.BLOCK_INDEX_BY_DIM:
        print(f"Warning: unsupported pytorch-fid dims={dims}.")
        return False

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)
    mu, sigma = calculate_activation_statistics(
        files,
        model,
        batch_size=batch_size,
        dims=dims,
        device=device,
        num_workers=num_workers,
    )
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(stats_path), mu=mu, sigma=sigma)
    return True


def _ensure_pytorch_fid_reference_stats(cfg: ConfigDict, device: torch.device) -> Optional[Path]:
    stats_path_raw = str(getattr(cfg.fid, "pytorch_fid_stats_file", "")).strip()
    if stats_path_raw == "":
        return None
    stats_path = Path(stats_path_raw)
    if stats_path.exists():
        try:
            with np.load(str(stats_path)) as data:
                if "mu" in data and "sigma" in data:
                    return stats_path
            print(f"Warning: existing pytorch-fid stats file is invalid, rebuilding: {stats_path}")
        except Exception:
            print(f"Warning: failed to read existing pytorch-fid stats file, rebuilding: {stats_path}")
        try:
            stats_path.unlink()
        except Exception:
            pass

    ref_dir = str(getattr(cfg.fid, "reference_dir", "")).strip() or str(cfg.data.eval_dir)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "pytorch_fid",
        "--save-stats",
        "--device",
        _device_arg_for_pytorch_fid(device),
        "--batch-size",
        str(int(getattr(cfg.fid, "pytorch_fid_batch_size", 64))),
        "--num-workers",
        str(int(getattr(cfg.fid, "pytorch_fid_num_workers", 0))),
        "--dims",
        str(int(getattr(cfg.fid, "pytorch_fid_dims", 2048))),
        ref_dir,
        str(stats_path),
    ]
    ok, output = _run_pytorch_fid_command(cmd)
    if not ok:
        print(
            "Warning: failed to create pytorch-fid reference stats. "
            f"Command: {' '.join(cmd)}"
        )
        if output.strip():
            print(output.strip())
        print(
            "Falling back to in-process pytorch-fid stats computation "
            "(manual file discovery, case-insensitive extensions)."
        )
        files = _collect_image_files_for_fid(ref_dir)
        ok_api = _build_pytorch_fid_stats_from_files(files, stats_path, cfg, device)
        if ok_api:
            return stats_path
        return None
    return stats_path


def _compute_pytorch_fid(
    generated_dir: Path,
    reference_stats: Path,
    cfg: ConfigDict,
    device: torch.device,
) -> Optional[float]:
    cmd = [
        sys.executable,
        "-m",
        "pytorch_fid",
        "--device",
        _device_arg_for_pytorch_fid(device),
        "--batch-size",
        str(int(getattr(cfg.fid, "pytorch_fid_batch_size", 64))),
        "--num-workers",
        str(int(getattr(cfg.fid, "pytorch_fid_num_workers", 0))),
        "--dims",
        str(int(getattr(cfg.fid, "pytorch_fid_dims", 2048))),
        str(generated_dir),
        str(reference_stats),
    ]
    ok, output = _run_pytorch_fid_command(cmd)
    if not ok:
        print(f"Warning: pytorch-fid failed. Command: {' '.join(cmd)}")
        if output.strip():
            print(output.strip())
        return None

    match = re.search(r"FID:\s*([0-9eE+\-.]+)", output)
    if match is None:
        print("Warning: could not parse FID value from pytorch-fid output.")
        if output.strip():
            print(output.strip())
        return None
    return float(match.group(1))


def compute_fid_score(
    ddpm: DDPM,
    cfg: ConfigDict,
    fid_loader: DataLoader,
    device: torch.device,
    run_save_dir: Path,
    epoch: int,
    global_step: int,
) -> Optional[float]:
    """Compute FID with pytorch-fid for nearest-neighbor-conditioned generation."""
    backend = str(getattr(cfg.fid, "backend", "pytorch_fid")).lower()
    keep_generated = bool(getattr(cfg.fid, "keep_generated_samples", False))
    if backend != "pytorch_fid":
        print(f"Warning: cfg.fid.backend='{backend}' is unsupported here; using pytorch_fid.")

    was_training = ddpm.training
    ddpm.eval()

    num_samples = int(cfg.fid.num_samples)
    all_samples = []
    generated = 0
    fid_iter: Optional[Iterator] = None

    with torch.no_grad():
        pbar = tqdm(total=num_samples, desc="Generating samples for pytorch-FID", leave=False)
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

    generated_root = Path(
        str(getattr(cfg.fid, "generated_samples_dir", "")).strip()
        or str(run_save_dir / "fid_generated_samples")
    )
    generated_dir = generated_root / f"epoch_{int(epoch):04d}_step_{int(global_step):08d}"
    _save_generated_samples(all_samples, generated_dir)

    if not _pytorch_fid_available():
        print("Warning: package 'pytorch-fid' is not installed. Install with: pip install pytorch-fid")
        print(f"Generated samples saved at: {generated_dir}")
        if was_training:
            ddpm.train()
        return None

    reference_stats_path = _ensure_pytorch_fid_reference_stats(cfg, device)
    if reference_stats_path is None:
        ref_dir = str(getattr(cfg.fid, "reference_dir", "")).strip() or str(cfg.data.eval_dir)
        print("Warning: could not prepare pytorch-fid reference stats.")
        print(f"Generated samples saved at: {generated_dir}")
        print(
            "Manual command: "
            f"{sys.executable} -m pytorch_fid --device {_device_arg_for_pytorch_fid(device)} "
            f"{generated_dir} {ref_dir}"
        )
        if was_training:
            ddpm.train()
        return None

    fid_score = _compute_pytorch_fid(generated_dir, reference_stats_path, cfg, device)
    if fid_score is None:
        ref_dir = str(getattr(cfg.fid, "reference_dir", "")).strip() or str(cfg.data.eval_dir)
        print("Warning: could not compute pytorch-fid in-process.")
        print(f"Generated samples saved at: {generated_dir}")
        print(
            "Manual command: "
            f"{sys.executable} -m pytorch_fid --device {_device_arg_for_pytorch_fid(device)} "
            f"{generated_dir} {ref_dir}"
        )
        if was_training:
            ddpm.train()
        return None

    if (not keep_generated) and generated_dir.exists():
        shutil.rmtree(generated_dir, ignore_errors=True)
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
        p_uncond=float(getattr(cfg.diffusion, "p_uncond", 0.1)),
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
    use_amp_cuda = bool(cfg.training.use_amp and device.type == "cuda")
    amp_dtype = torch.bfloat16 if use_amp_cuda else torch.float32
    # bf16 does not need gradient scaling; keep GradScaler disabled in this mode.
    scaler = amp.GradScaler(
        device=device.type,
        enabled=False,
    )

    if resume_ckpt is not None and not bool(getattr(cfg.training, "reset_optimizer_on_resume", False)):
        if "optimizer" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer"])
        if "scaler" in resume_ckpt:
            scaler.load_state_dict(resume_ckpt["scaler"])

    base_lr = float(cfg.training.lr)
    warmup_steps_on_reset = int(getattr(cfg.training, "reset_optimizer_warmup_steps", 0))
    warmup_start_factor = float(getattr(cfg.training, "reset_optimizer_warmup_start_factor", 0.1))
    warmup_start_factor = max(0.0, min(1.0, warmup_start_factor))
    warmup_active = bool(
        resume_ckpt is not None
        and bool(getattr(cfg.training, "reset_optimizer_on_resume", False))
        and warmup_steps_on_reset > 0
    )
    warmup_step = 0
    if warmup_active:
        start_lr = base_lr * warmup_start_factor
        for pg in optimizer.param_groups:
            pg["lr"] = start_lr
        if main:
            print(
                f"Enabled LR warmup after optimizer reset: "
                f"{warmup_steps_on_reset} steps, start_factor={warmup_start_factor:.4f}."
            )

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
            with amp.autocast(
                device_type=device.type,
                enabled=use_amp_cuda,
                dtype=amp_dtype,
            ):
                loss = ddpm_train(imgs, cond=cond)

            scaler.scale(loss).backward()
            if cfg.training.grad_clip is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(ddpm_train.parameters(), float(cfg.training.grad_clip))
            scaler.step(optimizer)
            scaler.update()

            global_step += 1

            if warmup_active:
                warmup_step += 1
                progress = min(1.0, warmup_step / max(1, warmup_steps_on_reset))
                lr_scale = warmup_start_factor + (1.0 - warmup_start_factor) * progress
                curr_lr = base_lr * lr_scale
                for pg in optimizer.param_groups:
                    pg["lr"] = curr_lr
                if warmup_step >= warmup_steps_on_reset:
                    warmup_active = False
            else:
                curr_lr = float(optimizer.param_groups[0]["lr"])

            loss_mean = reduce_mean(loss.detach(), distributed=distributed, world_size=world_size)
            loss_item = float(loss_mean.item())
            epoch_loss += loss_item

            if main:
                postfix = {"loss": f"{loss_item:.4f}", "lr": f"{curr_lr:.2e}"}
                if "query_label" in meta and torch.is_tensor(meta["query_label"]):
                    postfix["uniq_cls"] = int(torch.unique(meta["query_label"]).numel())
                pbar.set_postfix(postfix)

                if global_step % int(cfg.training.log_every) == 0 and wandb_run is not None:
                    logs: Dict[str, float] = {"train/loss": loss_item, "train/lr": curr_lr}
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
                fid_score = compute_fid_score(
                    ddpm_eval,
                    cfg,
                    fid_loader,
                    device,
                    run_save_dir=run_save_dir,
                    epoch=epoch,
                    global_step=global_step,
                )
                if fid_score is not None:
                    print(f"  pytorch-FID: {fid_score:.2f}")
                    if wandb_run is not None:
                        wandb_run.log(
                            {
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

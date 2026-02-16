#!/usr/bin/env python3
"""
MAML finetuning on ImageNet nearest-neighbor conditioning tasks.

Task construction:
- each dataset sample is (K+1, C, H, W) = [query, nn_1..nn_K]
- inner-loop support set = K neighbors
- conditioning for both inner/outer losses = same K neighbors
- outer loss query = single query image

Multi-GPU support is implemented with torch.distributed + per-rank data
partitioning (DistributedSampler) + gradient all-reduce across ranks.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from absl import app
from ml_collections import ConfigDict, config_flags
import numpy as np
import torch
import torch.amp as amp
import torch.distributed as dist
from torch import nn, optim
from torch.func import functional_call
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import utils
import wandb
from tqdm import tqdm

from playground.dataset.imagenet_nearest_dataset import ImageNetNearestContextDataset
from playground.models.conditional_model import ConditionalUNet, DDPM, ResBlockCond


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


def all_reduce_gradients(parameters, distributed: bool, world_size: int) -> None:
    if not distributed:
        return
    for p in parameters:
        if p.grad is None:
            continue
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        p.grad /= world_size


def is_main_process(rank: int) -> bool:
    return rank == 0


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

    query = images[:, 0].contiguous().to(device, non_blocking=True)
    support = images[:, 1:].contiguous().to(device, non_blocking=True)
    return query, support, meta


def _grid_nrow(n: int) -> int:
    nrow = int(np.sqrt(max(1, n)))
    if nrow * nrow < n:
        nrow += 1
    return nrow


def _task_neighbor_agreement(meta: Dict[str, torch.Tensor], task_idx: int) -> Optional[float]:
    if "query_label" not in meta or "neighbor_labels" not in meta:
        return None
    q = meta["query_label"]
    n = meta["neighbor_labels"]
    if not torch.is_tensor(q) or not torch.is_tensor(n):
        return None
    if q.ndim != 1 or n.ndim != 2:
        return None
    if task_idx < 0 or task_idx >= q.shape[0]:
        return None
    return float((n[task_idx] == q[task_idx]).float().mean().item())


# -------------------------
# Fast parameter selection
# -------------------------


def select_fast_params(model: ConditionalUNet, selector: str) -> Tuple[List[str], List[nn.Parameter]]:
    fast_params: List[nn.Parameter] = []
    fast_names: List[str] = []

    def add_mlp(name: str, module: ResBlockCond) -> None:
        for pname, p in module.time_mlp.named_parameters():
            fast_names.append(f"{name}.time_mlp.{pname}")
            fast_params.append(p)
        for pname, p in module.cond_mlp.named_parameters():
            fast_names.append(f"{name}.cond_mlp.{pname}")
            fast_params.append(p)

    def add_gn(name: str, gn: nn.GroupNorm) -> None:
        for pname, p in gn.named_parameters():
            fast_names.append(f"{name}.{pname}")
            fast_params.append(p)

    def add_convs(name: str, module: ResBlockCond) -> None:
        for pname, p in module.conv1.named_parameters():
            fast_names.append(f"{name}.conv1.{pname}")
            fast_params.append(p)
        for pname, p in module.conv2.named_parameters():
            fast_names.append(f"{name}.conv2.{pname}")
            fast_params.append(p)
        if not isinstance(module.skip, nn.Identity):
            for pname, p in module.skip.named_parameters():
                fast_names.append(f"{name}.skip.{pname}")
                fast_params.append(p)

    if selector == "up_time_mlp":
        targets = ["ups"]
        include_mid = False
        include_head = False
        include_gn = False
        include_convs = False
    elif selector == "up_down_time_mlp":
        targets = ["ups", "downs"]
        include_mid = False
        include_head = False
        include_gn = False
        include_convs = False
    elif selector == "up_down_mid_head":
        targets = ["ups", "downs"]
        include_mid = True
        include_head = True
        include_gn = False
        include_convs = False
    elif selector == "up_down_mid_head_gn":
        targets = ["ups", "downs"]
        include_mid = True
        include_head = True
        include_gn = True
        include_convs = False
    elif selector == "up_down_mid_full":
        targets = ["ups", "downs"]
        include_mid = True
        include_head = True
        include_gn = False
        include_convs = True
    elif selector == "up_down_mid_full_gn":
        targets = ["ups", "downs"]
        include_mid = True
        include_head = True
        include_gn = True
        include_convs = True
    else:
        raise ValueError(f"Unknown selector {selector}")

    for name, module in model.named_modules():
        if isinstance(module, ResBlockCond):
            is_up = name.startswith("ups")
            is_down = name.startswith("downs")
            is_mid = name.startswith("mid")
            if (is_up and "ups" in targets) or (is_down and "downs" in targets) or (include_mid and is_mid):
                add_mlp(name, module)
                if include_gn:
                    add_gn(f"{name}.norm1", module.norm1)
                    add_gn(f"{name}.norm2", module.norm2)
                if include_convs:
                    add_convs(name, module)

    if include_head:
        fast_names.extend(["out_norm.weight", "out_norm.bias", "out_conv.weight", "out_conv.bias"])
        fast_params.extend([model.out_norm.weight, model.out_norm.bias, model.out_conv.weight, model.out_conv.bias])

    return fast_names, fast_params


def build_param_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {name: p for name, p in model.named_parameters()}


def count_params(all_params, fast_names: List[str], base_params: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    total = sum(p.numel() for p in all_params)
    fast = sum(base_params[n].numel() for n in fast_names)
    return total, fast


def _load_state_dict_with_report(
    module: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    strict: bool,
    label: str,
) -> bool:
    module_keys = set(module.state_dict().keys())
    provided_keys = set(state_dict.keys())
    if len(module_keys.intersection(provided_keys)) == 0:
        print(f"{label}: no matching parameter keys, skipping this load path.")
        return False

    try:
        load_result = module.load_state_dict(state_dict, strict=strict)
    except RuntimeError as exc:
        print(f"Failed loading {label}: {exc}")
        return False

    if hasattr(load_result, "missing_keys") and hasattr(load_result, "unexpected_keys"):
        if len(load_result.missing_keys) > 0:
            print(f"{label}: missing keys ({len(load_result.missing_keys)}), e.g. {load_result.missing_keys[:5]}")
        if len(load_result.unexpected_keys) > 0:
            print(
                f"{label}: unexpected keys ({len(load_result.unexpected_keys)}), "
                f"e.g. {load_result.unexpected_keys[:5]}"
            )
    return True


def load_pretrained_if_available(ddpm: DDPM, cfg: ConfigDict, device: torch.device) -> bool:
    pretrained_cfg = getattr(cfg, "pretrained", None)
    if pretrained_cfg is None or not bool(getattr(pretrained_cfg, "use", False)):
        return False

    ckpt_path = Path(str(getattr(pretrained_cfg, "checkpoint", "")).strip())
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {ckpt_path}")

    strict = bool(getattr(pretrained_cfg, "strict", False))
    print(f"Loading pretrained checkpoint from {ckpt_path} (strict={strict})")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    loaded = False
    if isinstance(ckpt, dict):
        if "unet" in ckpt:
            loaded = _load_state_dict_with_report(ddpm.model, ckpt["unet"], strict, "pretrained/unet")
        if not loaded and "model" in ckpt:
            state = ckpt["model"]
            loaded = _load_state_dict_with_report(ddpm, state, strict, "pretrained/model_as_ddpm")
            if not loaded:
                loaded = _load_state_dict_with_report(ddpm.model, state, strict, "pretrained/model_as_unet")
        if not loaded and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            loaded = _load_state_dict_with_report(ddpm.model, ckpt, strict, "pretrained/raw_unet")
            if not loaded:
                loaded = _load_state_dict_with_report(ddpm, ckpt, strict, "pretrained/raw_ddpm")
    else:
        print(f"Unsupported checkpoint format type: {type(ckpt)}")

    if not loaded:
        raise RuntimeError(f"Could not load pretrained weights from {ckpt_path}")
    print("Pretrained weights loaded successfully.")
    return True


# -------------------------
# Functional MAML helpers
# -------------------------


def compute_loss_functional(
    params: Dict[str, torch.Tensor],
    model: ConditionalUNet,
    x0: torch.Tensor,
    t: torch.Tensor,
    cond: torch.Tensor,
    log_snr_max: float,
    log_snr_min: float,
) -> torch.Tensor:
    log_snr = log_snr_max + t * (log_snr_min - log_snr_max)
    alpha = torch.sqrt(torch.sigmoid(log_snr))
    sigma = torch.sqrt(torch.sigmoid(-log_snr))

    noise = torch.randn_like(x0)
    x_t = alpha[:, None, None, None] * x0 + sigma[:, None, None, None] * noise
    v_target = alpha[:, None, None, None] * noise - sigma[:, None, None, None] * x0
    v_pred = functional_call(model, params, (x_t, t, cond))
    return torch.mean((v_pred - v_target) ** 2)


def adapt_fast_params_for_task(
    params: Dict[str, torch.Tensor],
    fast_names: List[str],
    model: ConditionalUNet,
    support_imgs: torch.Tensor,
    inner_lr: float,
    inner_steps: int,
    log_snr_max: float,
    log_snr_min: float,
    create_graph: bool,
) -> Dict[str, torch.Tensor]:
    adapted = params
    if len(fast_names) == 0 or inner_steps <= 0:
        return adapted

    cond = support_imgs.unsqueeze(0)
    for _ in range(inner_steps):
        t = torch.rand(support_imgs.shape[0], device=support_imgs.device)
        loss = compute_loss_functional(
            adapted,
            model,
            support_imgs,
            t,
            cond,
            log_snr_max,
            log_snr_min,
        )
        grads = torch.autograd.grad(loss, [adapted[n] for n in fast_names], create_graph=create_graph)
        updated = adapted.copy()
        for name, g in zip(fast_names, grads):
            updated[name] = adapted[name] - inner_lr * g
        adapted = updated

    return adapted


def single_task_maml_loss(
    params: Dict[str, torch.Tensor],
    fast_names: List[str],
    model: ConditionalUNet,
    query_img: torch.Tensor,
    support_imgs: torch.Tensor,
    inner_lr: float,
    inner_steps: int,
    log_snr_max: float,
    log_snr_min: float,
) -> torch.Tensor:
    adapted = adapt_fast_params_for_task(
        params,
        fast_names,
        model,
        support_imgs,
        inner_lr,
        inner_steps,
        log_snr_max,
        log_snr_min,
        create_graph=True,
    )

    query = query_img.unsqueeze(0)
    cond = support_imgs.unsqueeze(0)
    t_out = torch.rand(query.shape[0], device=query.device)
    return compute_loss_functional(adapted, model, query, t_out, cond, log_snr_max, log_snr_min)


def multi_task_maml_step(
    ddpm: DDPM,
    query_all: torch.Tensor,
    support_all: torch.Tensor,
    fast_names: List[str],
    base_params: Dict[str, torch.Tensor],
    inner_lr: float,
    inner_steps: int,
    device: torch.device,
) -> torch.Tensor:
    num_tasks = int(query_all.shape[0])
    total_loss = torch.tensor(0.0, device=device)
    for i in range(num_tasks):
        task_loss = single_task_maml_loss(
            base_params,
            fast_names,
            ddpm.model,
            query_all[i],
            support_all[i],
            inner_lr,
            inner_steps,
            ddpm.log_snr_max,
            ddpm.log_snr_min,
        )
        total_loss = total_loss + task_loss
    return total_loss / max(1, num_tasks)


# -------------------------
# Adaptation logging
# -------------------------


def _sample_with_params(
    ddpm: DDPM,
    params_detached: Dict[str, torch.Tensor],
    cond_imgs: torch.Tensor,
    cfg: ConfigDict,
    device: torch.device,
) -> torch.Tensor:
    img = torch.randn(
        int(cfg.sample.num_images),
        ddpm.model.in_conv.in_channels,
        int(cfg.data.image_size),
        int(cfg.data.image_size),
        device=device,
    )
    cond_for_gen = cond_imgs.unsqueeze(0)
    times = torch.linspace(1.0, 0.0, int(cfg.sample.steps) + 1, device=device)

    for i in range(int(cfg.sample.steps)):
        t_cur = times[i].repeat(img.shape[0])
        t_prev = times[i + 1].repeat(img.shape[0])
        v_pred = functional_call(ddpm.model, params_detached, (img, t_cur, cond_for_gen))

        log_snr_cur = ddpm.log_snr_max + t_cur * (ddpm.log_snr_min - ddpm.log_snr_max)
        alpha_cur = torch.sqrt(torch.sigmoid(log_snr_cur))[:, None, None, None]
        sigma_cur = torch.sqrt(torch.sigmoid(-log_snr_cur))[:, None, None, None]

        log_snr_prev = ddpm.log_snr_max + t_prev * (ddpm.log_snr_min - ddpm.log_snr_max)
        alpha_prev = torch.sqrt(torch.sigmoid(log_snr_prev))[:, None, None, None]
        sigma_prev = torch.sqrt(torch.sigmoid(-log_snr_prev))[:, None, None, None]

        x0_pred = alpha_cur * img - sigma_cur * v_pred
        eps_pred = alpha_cur * v_pred + sigma_cur * img
        img = x0_pred if i == int(cfg.sample.steps) - 1 else alpha_prev * x0_pred + sigma_prev * eps_pred

    return img


def adapt_and_sample(
    ddpm: DDPM,
    fast_names: List[str],
    base_params: Dict[str, torch.Tensor],
    query_img: torch.Tensor,
    support_imgs: torch.Tensor,
    cfg: ConfigDict,
    inner_steps: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    adapted = adapt_fast_params_for_task(
        base_params,
        fast_names,
        ddpm.model,
        support_imgs,
        float(cfg.eval.inner_lr),
        inner_steps,
        ddpm.log_snr_max,
        ddpm.log_snr_min,
        create_graph=False,
    )
    params_detached = {k: v.detach() for k, v in adapted.items()}

    with torch.no_grad():
        gen = _sample_with_params(ddpm, params_detached, support_imgs, cfg, query_img.device)

    gen_vis = (gen.clamp(-1, 1) + 1) / 2
    query_vis = (query_img.unsqueeze(0).clamp(-1, 1) + 1) / 2
    cond_vis = (support_imgs[: min(4, support_imgs.shape[0])].clamp(-1, 1) + 1) / 2

    gen_grid = utils.make_grid(gen_vis, nrow=_grid_nrow(gen_vis.shape[0]))
    query_grid = utils.make_grid(query_vis, nrow=1)
    cond_grid = utils.make_grid(cond_vis, nrow=min(4, cond_vis.shape[0]))
    return gen_grid, query_grid, cond_grid


def adapt_and_log_denoise(
    ddpm: DDPM,
    fast_names: List[str],
    base_params: Dict[str, torch.Tensor],
    query_img: torch.Tensor,
    support_imgs: torch.Tensor,
    cfg: ConfigDict,
    wandb_run,
    global_step: int,
    inner_steps: int,
    key_name: str,
) -> None:
    if wandb_run is None:
        return

    adapted = adapt_fast_params_for_task(
        base_params,
        fast_names,
        ddpm.model,
        support_imgs,
        float(cfg.eval.inner_lr),
        inner_steps,
        ddpm.log_snr_max,
        ddpm.log_snr_min,
        create_graph=False,
    )
    params_detached = {k: v.detach() for k, v in adapted.items()}

    with torch.no_grad():
        x0 = query_img.unsqueeze(0)
        t = torch.rand(1, device=x0.device)
        noise = torch.randn_like(x0)
        log_snr = ddpm.log_snr_max + t * (ddpm.log_snr_min - ddpm.log_snr_max)
        alpha = torch.sqrt(torch.sigmoid(log_snr))[:, None, None, None]
        sigma = torch.sqrt(torch.sigmoid(-log_snr))[:, None, None, None]
        x_t = alpha * x0 + sigma * noise
        v_pred = functional_call(ddpm.model, params_detached, (x_t, t, support_imgs.unsqueeze(0)))
        x0_pred = alpha * x_t - sigma * v_pred

        vis = torch.cat(
            [
                (x0.clamp(-1, 1) + 1) / 2,
                (x_t.clamp(-1, 1) + 1) / 2,
                (x0_pred.clamp(-1, 1) + 1) / 2,
            ],
            dim=0,
        )
        grid = utils.make_grid(vis, nrow=1)
    wandb_run.log({key_name: wandb.Image(grid)}, step=global_step)


def log_adaptation_samples(
    ddpm: DDPM,
    train_loader: DataLoader,
    eval_loader: Optional[DataLoader],
    train_iter: Optional[Iterator],
    eval_iter: Optional[Iterator],
    fast_names: List[str],
    base_params: Dict[str, torch.Tensor],
    cfg: ConfigDict,
    device: torch.device,
    wandb_run,
    global_step: int,
    inner_steps: int,
    key_prefix: str = "",
) -> Tuple[Optional[Iterator], Optional[Iterator]]:
    if wandb_run is None:
        return train_iter, eval_iter

    def k(name: str) -> str:
        return f"{key_prefix}{name}" if key_prefix else name

    was_training = ddpm.training
    ddpm.eval()

    if eval_loader is not None:
        batch, eval_iter = next_batch(eval_loader, eval_iter)
        query_all, support_all, meta = split_query_and_context(batch, device)
        task_idx = random.randrange(query_all.shape[0])

        gen_grid, query_grid, cond_grid = adapt_and_sample(
            ddpm,
            fast_names,
            base_params,
            query_all[task_idx],
            support_all[task_idx],
            cfg,
            inner_steps=inner_steps,
        )
        logs: Dict[str, object] = {
            k("samples/generated"): wandb.Image(gen_grid),
            k("samples/query_target"): wandb.Image(query_grid),
            k("samples/conditioning_first4"): wandb.Image(cond_grid),
        }
        agreement = _task_neighbor_agreement(meta, task_idx)
        if agreement is not None:
            logs[k("samples/neighbor_label_agreement")] = agreement
        wandb_run.log(logs, step=global_step)

        adapt_and_log_denoise(
            ddpm,
            fast_names,
            base_params,
            query_all[task_idx],
            support_all[task_idx],
            cfg,
            wandb_run,
            global_step,
            inner_steps=inner_steps,
            key_name=k("debug/x0_xt_x0pred"),
        )

    batch, train_iter = next_batch(train_loader, train_iter)
    query_all, support_all, meta_train = split_query_and_context(batch, device)
    task_idx = random.randrange(query_all.shape[0])
    train_grid, train_query_grid, train_cond_grid = adapt_and_sample(
        ddpm,
        fast_names,
        base_params,
        query_all[task_idx],
        support_all[task_idx],
        cfg,
        inner_steps=inner_steps,
    )
    train_logs: Dict[str, object] = {
        k("train_adapt/generated"): wandb.Image(train_grid),
        k("train_adapt/query_target"): wandb.Image(train_query_grid),
        k("train_adapt/conditioning_first4"): wandb.Image(train_cond_grid),
    }
    agreement_train = _task_neighbor_agreement(meta_train, task_idx)
    if agreement_train is not None:
        train_logs[k("train_adapt/neighbor_label_agreement")] = agreement_train
    wandb_run.log(train_logs, step=global_step)

    if was_training:
        ddpm.train()
    return train_iter, eval_iter


# -------------------------
# Checkpoint helpers
# -------------------------


def save_checkpoint(
    ckpt_path: Path,
    ddpm: DDPM,
    optimizer: optim.Optimizer,
    scaler: amp.GradScaler,
    cfg: ConfigDict,
    epoch: int,
    global_step: int,
    rank: int,
    world_size: int,
) -> None:
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "checkpoint_type": "conditional_maml_imagenet_nearest_neighbors",
            "epoch": epoch,
            "global_step": global_step,
            "model": ddpm.state_dict(),
            "unet": ddpm.model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "distributed": {
                "rank": int(rank),
                "world_size": int(world_size),
            },
            "maml": {
                "num_tasks": int(cfg.training.num_tasks),
                "inner_steps": int(cfg.training.inner_steps),
                "inner_lr": float(cfg.training.inner_lr),
                "fast_params_selector": str(cfg.fast_params.selector),
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
        batch_size=int(cfg.training.num_tasks),
        num_workers=int(cfg.data.num_workers),
        shuffle=(train_sampler is None),
        drop_last=True,
        sampler=train_sampler,
    )

    eval_loader = None
    if main:
        eval_ds = build_eval_dataset(cfg)
        eval_loader = build_loader(
            eval_ds,
            batch_size=int(cfg.sample.eval_batch_size),
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

    ddpm = DDPM(
        model,
        log_snr_max=cfg.diffusion.log_snr_max,
        log_snr_min=cfg.diffusion.log_snr_min,
        p_uncond=float(getattr(cfg.diffusion, "p_uncond", 0.1)),
    ).to(device)

    if main:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {n_params / 1e6:.2f}M ({n_params})")
        print(
            f"Train tasks/step per-rank: {cfg.training.num_tasks} | "
            f"K={cfg.data.cond_batch_size} | world_size={world_size}"
        )

    pretrained_loaded = load_pretrained_if_available(ddpm, cfg, device)

    fast_names, _ = select_fast_params(model, cfg.fast_params.selector)
    base_params = build_param_dict(ddpm.model)
    total_params, fast_params_count = count_params(ddpm.parameters(), fast_names, base_params)
    if main:
        print(f"Total params: {total_params / 1e6:.2f}M, fast params: {fast_params_count / 1e6:.4f}M")

    optimizer = optim.AdamW(
        ddpm.parameters(),
        lr=float(cfg.training.outer_lr),
        weight_decay=float(cfg.training.weight_decay),
    )
    scaler = amp.GradScaler(
        device=device.type,
        enabled=bool(cfg.training.use_amp and device.type == "cuda"),
    )

    start_epoch = 1
    global_step = 0
    resume_ckpt = None
    resume_path = str(getattr(cfg.checkpoint, "resume", "")).strip()
    if resume_path:
        resume_ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        ddpm.load_state_dict(resume_ckpt["model"])
        start_epoch = int(resume_ckpt.get("epoch", 0)) + 1
        global_step = int(resume_ckpt.get("global_step", 0))
        if main:
            print(f"Resumed model from {resume_path} at epoch {start_epoch}, step {global_step}.")

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
    train_log_iter: Optional[Iterator] = None
    eval_log_iter: Optional[Iterator] = None

    if pretrained_loaded and main and wandb_run is not None:
        print("Logging pretrained sanity-check samples before meta-training...")
        train_log_iter, eval_log_iter = log_adaptation_samples(
            ddpm,
            train_loader,
            eval_loader,
            train_log_iter,
            eval_log_iter,
            fast_names,
            base_params,
            cfg,
            device,
            wandb_run,
            global_step=global_step,
            inner_steps=0,
            key_prefix="pretrained_check/",
        )

    for epoch in range(start_epoch, int(cfg.training.epochs) + 1):
        if distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        ddpm.train()
        pbar = tqdm(
            range(int(cfg.training.steps_per_epoch)),
            desc=f"Epoch {epoch}",
            leave=False,
            disable=not main,
        )
        epoch_loss = 0.0

        for _ in pbar:
            batch, train_iter = next_batch(train_loader, train_iter)
            query_all, support_all, meta = split_query_and_context(batch, device)

            optimizer.zero_grad(set_to_none=True)

            outer_loss = multi_task_maml_step(
                ddpm,
                query_all,
                support_all,
                fast_names=fast_names,
                base_params=base_params,
                inner_lr=float(cfg.training.inner_lr),
                inner_steps=int(cfg.training.inner_steps),
                device=device,
            )

            scaler.scale(outer_loss).backward()
            scaler.unscale_(optimizer)
            all_reduce_gradients(ddpm.parameters(), distributed=distributed, world_size=world_size)
            if cfg.training.grad_clip is not None:
                nn.utils.clip_grad_norm_(ddpm.parameters(), float(cfg.training.grad_clip))
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            loss_mean = reduce_mean(outer_loss.detach(), distributed=distributed, world_size=world_size)
            loss_item = float(loss_mean.item())
            epoch_loss += loss_item

            if main:
                postfix = {"loss": f"{loss_item:.4f}"}
                if "query_label" in meta and torch.is_tensor(meta["query_label"]):
                    postfix["uniq_cls"] = int(torch.unique(meta["query_label"]).numel())
                pbar.set_postfix(postfix)

                if global_step % int(cfg.training.log_every) == 0 and wandb_run is not None:
                    logs: Dict[str, float] = {"train/outer_loss": loss_item}
                    if "query_label" in meta and torch.is_tensor(meta["query_label"]):
                        logs["train/unique_query_classes_in_batch"] = float(
                            torch.unique(meta["query_label"]).numel()
                        )
                    wandb_run.log(logs, step=global_step)

        if main:
            avg_loss = epoch_loss / int(cfg.training.steps_per_epoch)
            print(f"[Epoch {epoch}] avg_outer_loss={avg_loss:.4f}")
            if wandb_run is not None:
                wandb_run.log({"train/epoch_outer_loss": avg_loss}, step=global_step)

            if epoch % int(cfg.training.sample_every_epochs) == 0:
                train_log_iter, eval_log_iter = log_adaptation_samples(
                    ddpm,
                    train_loader,
                    eval_loader,
                    train_log_iter,
                    eval_log_iter,
                    fast_names,
                    base_params,
                    cfg,
                    device,
                    wandb_run,
                    global_step=global_step,
                    inner_steps=int(cfg.eval.inner_steps),
                )

            latest_path = run_save_dir / "cond_maml_imagenet_nn_latest.pt"
            save_checkpoint(
                latest_path,
                ddpm,
                optimizer,
                scaler,
                cfg,
                epoch,
                global_step,
                rank,
                world_size,
            )
            if epoch % int(cfg.training.checkpoint_every_epochs) == 0:
                ckpt_path = run_save_dir / f"cond_maml_imagenet_nn_epoch_{epoch:03d}.pt"
                save_checkpoint(
                    ckpt_path,
                    ddpm,
                    optimizer,
                    scaler,
                    cfg,
                    epoch,
                    global_step,
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
    default="playground/configs/train_cond_maml_imagenet_nn.py",
    help_string="Path to a ml_collections config file.",
)


def main(argv=None):
    del argv
    cfg = _CONFIG.value
    train(cfg)


if __name__ == "__main__":
    app.run(main)

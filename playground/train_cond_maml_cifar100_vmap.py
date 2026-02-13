#!/usr/bin/env python3
"""
Task-Parallel MAML Training with vmap.

This script implements MAML with task parallelism using torch.func.vmap to reduce
gradient variance by averaging outer gradients across multiple tasks per meta-step.

Key differences from train_cond_maml_cifar100.py:
- Samples N tasks (classes) per outer step instead of 1
- Uses vmap to parallelize inner loop computation across tasks
- Data shape: (N, B, C, H, W) for images, (N, K, C, H, W) for conditioning
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Tuple

from absl import app
from ml_collections import ConfigDict, config_flags
import numpy as np
import torch
from torch import nn, optim
import torch.amp as amp
from torch.func import functional_call
from torchvision import datasets, transforms, utils
import wandb
from tqdm import tqdm

from playground.models.conditional_model import ConditionalUNet, DDPM, ResBlockCond


# -------------------------
# Logging helpers
# -------------------------


def adapt_and_log_denoise(
    ddpm: DDPM,
    imgs: torch.Tensor,
    cond_imgs: torch.Tensor,
    fast_names: List[str],
    base_params: Dict[str, torch.Tensor],
    cfg: ConfigDict,
    device: torch.device,
    wandb_run,
    global_step: int,
    inner_steps: int,
    max_imgs: int = 8,
):
    """Adapt to task and log noised images with predicted x0 for debugging."""
    if wandb_run is None:
        return
    
    params = base_params

    def p_losses_with_params(params_dict, x0, t, cond):
        noise = torch.randn_like(x0)
        log_snr = ddpm.log_snr_max + t * (ddpm.log_snr_min - ddpm.log_snr_max)
        alpha = torch.sqrt(torch.sigmoid(log_snr))
        sigma = torch.sqrt(torch.sigmoid(-log_snr))
        x_t = alpha[:, None, None, None] * x0 + sigma[:, None, None, None] * noise
        v_target = alpha[:, None, None, None] * noise - sigma[:, None, None, None] * x0
        v_pred = functional_call(ddpm.model, params_dict, (x_t, t, cond))
        return torch.mean((v_pred - v_target) ** 2)

    # Adapt parameters
    for _ in range(inner_steps):
        t = torch.rand(imgs.shape[0], device=device)
        loss = p_losses_with_params(params, imgs, t, cond_imgs.unsqueeze(0))
        grads = torch.autograd.grad(loss, [params[n] for n in fast_names], create_graph=False)
        params = params.copy()
        for name, g in zip(fast_names, grads):
            params[name] = params[name] - cfg.eval.inner_lr * g
    
    params_detached = {k: v.detach() for k, v in params.items()}
    
    # Log denoising with adapted params
    with torch.no_grad():
        x0 = imgs[:max_imgs].to(device)
        cond = cond_imgs.unsqueeze(0)  # Use all K exemplars
        t = torch.rand(x0.shape[0], device=device)
        noise = torch.randn_like(x0)
        log_snr = ddpm.log_snr_max + t * (ddpm.log_snr_min - ddpm.log_snr_max)
        alpha = torch.sqrt(torch.sigmoid(log_snr))
        sigma = torch.sqrt(torch.sigmoid(-log_snr))
        x_t = alpha[:, None, None, None] * x0 + sigma[:, None, None, None] * noise
        v_pred = functional_call(ddpm.model, params_detached, (x_t, t, cond))
        x0_pred = alpha[:, None, None, None] * x_t - sigma[:, None, None, None] * v_pred
        vis = torch.cat([(x_t.clamp(-1, 1) + 1) / 2, (x0_pred.clamp(-1, 1) + 1) / 2], dim=0)
        grid = utils.make_grid(vis, nrow=max_imgs)
        wandb_run.log({"debug/x_t_and_x0": wandb.Image(grid)}, step=global_step)


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


def build_datasets(cfg: ConfigDict):
    """Build CIFAR-100 dataset with class-based splits."""
    transform = transforms.Compose(
        [
            transforms.Resize(cfg.data.image_size),
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
    rng = np.random.default_rng(cfg.run.seed)
    holdout = int(cfg.data.holdout_per_class)
    train_indices_by_class = {}
    val_indices_by_class = {}

    targets = np.array(train_ds.targets)
    num_classes = 100

    if cfg.data.use_full_dataset:
        for c in range(num_classes):
            idxs = np.where(targets == c)[0].tolist()
            rng.shuffle(idxs)
            val_indices_by_class[c] = idxs[:holdout]
            train_indices_by_class[c] = idxs[holdout:]
        train_classes = list(range(num_classes))
        eval_classes = list(range(num_classes))
    else:
        leave_out = list(cfg.data.leave_out_classes)
        for c in range(num_classes):
            idxs = np.where(targets == c)[0].tolist()
            rng.shuffle(idxs)
            if c in leave_out:
                val_indices_by_class[c] = idxs[: int(cfg.data.leaveout_eval_holdout)]
                train_indices_by_class[c] = []
            else:
                val_indices_by_class[c] = idxs[:holdout]
                train_indices_by_class[c] = idxs[holdout:]
        train_classes = [c for c in range(num_classes) if c not in leave_out]
        eval_classes = leave_out

    return train_ds, train_indices_by_class, val_indices_by_class, train_classes, eval_classes


def sample_class_batch(
    dataset,
    indices_by_class,
    cls: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Sample a batch of images from a single class."""
    idx_pool = indices_by_class[cls]
    replace = len(idx_pool) < batch_size
    idx = np.random.choice(idx_pool, size=batch_size, replace=replace)
    imgs = torch.stack([dataset[i][0] for i in idx], dim=0).to(device)
    return imgs


def sample_multi_task_batches(
    dataset,
    indices_by_class,
    classes: List[int],
    num_tasks: int,
    batch_size_x: int,
    batch_size_cond: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """
    Sample batches for N tasks (different classes).
    
    Returns:
        imgs_x: (N, B, C, H, W) - images for inner loop / outer loss
        imgs_cond: (N, K, C, H, W) - conditioning exemplars
        sampled_classes: list of N class indices that were sampled
    """
    sampled_classes = random.sample(classes, num_tasks)
    
    all_x = []
    all_cond = []
    
    for cls in sampled_classes:
        idx_pool = indices_by_class[cls]
        total = batch_size_x + batch_size_cond
        replace = len(idx_pool) < total
        idx = np.random.choice(idx_pool, size=total, replace=replace)
        np.random.shuffle(idx)
        
        x_idx = idx[:batch_size_x]
        cond_idx = idx[batch_size_x:]
        
        imgs_x = torch.stack([dataset[i][0] for i in x_idx], dim=0)
        imgs_cond = torch.stack([dataset[i][0] for i in cond_idx], dim=0)
        
        all_x.append(imgs_x)
        all_cond.append(imgs_cond)
    
    # Stack along task dimension: (N, B, C, H, W)
    imgs_x = torch.stack(all_x, dim=0).to(device)
    imgs_cond = torch.stack(all_cond, dim=0).to(device)
    
    return imgs_x, imgs_cond, sampled_classes


def select_fast_params(model: ConditionalUNet, selector: str) -> Tuple[List[str], List[nn.Parameter]]:
    """
    Select which parameters are 'fast' (updated in inner loop).
    """
    fast_params: List[nn.Parameter] = []
    fast_names: List[str] = []

    def add_mlp(name, module):
        for pname, p in module.time_mlp.named_parameters():
            full = f"{name}.time_mlp.{pname}"
            fast_names.append(full)
            fast_params.append(p)
        for pname, p in module.cond_mlp.named_parameters():
            full = f"{name}.cond_mlp.{pname}"
            fast_names.append(full)
            fast_params.append(p)

    def add_gn(name, gn: nn.GroupNorm):
        for pname, p in gn.named_parameters():
            full = f"{name}.{pname}"
            fast_names.append(full)
            fast_params.append(p)

    def add_convs(name, module: ResBlockCond):
        for pname, p in module.conv1.named_parameters():
            full = f"{name}.conv1.{pname}"
            fast_names.append(full)
            fast_params.append(p)
        for pname, p in module.conv2.named_parameters():
            full = f"{name}.conv2.{pname}"
            fast_names.append(full)
            fast_params.append(p)
        if not isinstance(module.skip, nn.Identity):
            for pname, p in module.skip.named_parameters():
                full = f"{name}.skip.{pname}"
                fast_names.append(full)
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
        fast_names.extend(
            ["out_norm.weight", "out_norm.bias", "out_conv.weight", "out_conv.bias"]
        )
        fast_params.extend(
            [model.out_norm.weight, model.out_norm.bias, model.out_conv.weight, model.out_conv.bias]
        )

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
    """Load state dict and print missing/unexpected keys; return success flag."""
    module_keys = set(module.state_dict().keys())
    provided_keys = set(state_dict.keys())
    key_overlap = module_keys.intersection(provided_keys)
    if len(key_overlap) == 0:
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
            print(f"{label}: unexpected keys ({len(load_result.unexpected_keys)}), e.g. {load_result.unexpected_keys[:5]}")
    return True


def load_pretrained_if_available(ddpm: DDPM, cfg: ConfigDict, device: torch.device) -> bool:
    """
    Optionally load pretrained weights before MAML meta-training.

    Supports checkpoints that contain:
    - {"unet": ...} from pretrain_cifar100.py
    - {"model": ...} either DDPM state or UNet state
    - raw state_dict
    """
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
            # First try as full DDPM state, then as UNet-only state.
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
# Functional helpers for vmap
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
    """
    Compute diffusion loss given parameters (functional style for vmap/grad).
    
    Args:
        params: parameter dict for functional_call
        model: the ConditionalUNet (structure only, params from dict)
        x0: clean images (B, C, H, W)
        t: timesteps (B,)
        cond: conditioning images (K, C, H, W)
        log_snr_max, log_snr_min: diffusion schedule params
    
    Returns:
        scalar loss
    """
    # Compute alpha, sigma from log-SNR
    log_snr = log_snr_max + t * (log_snr_min - log_snr_max)
    alpha_sq = torch.sigmoid(log_snr)
    sigma_sq = torch.sigmoid(-log_snr)
    alpha = torch.sqrt(alpha_sq)
    sigma = torch.sqrt(sigma_sq)
    
    # Sample noise and create noisy images
    noise = torch.randn_like(x0)
    x_t = alpha[:, None, None, None] * x0 + sigma[:, None, None, None] * noise
    
    # v-target
    v_target = alpha[:, None, None, None] * noise - sigma[:, None, None, None] * x0
    
    # Forward pass through model
    v_pred = functional_call(model, params, (x_t, t, cond))
    
    return torch.mean((v_pred - v_target) ** 2)


def inner_loop_step(
    params: Dict[str, torch.Tensor],
    fast_names: List[str],
    model: ConditionalUNet,
    imgs: torch.Tensor,
    cond: torch.Tensor,
    inner_lr: float,
    log_snr_max: float,
    log_snr_min: float,
    rng_key: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Perform one inner loop gradient step on fast params.
    
    Args:
        params: current parameter dict
        fast_names: names of fast parameters to update
        model: model structure
        imgs: (B, C, H, W) images for this task
        cond: (K, C, H, W) conditioning images
        inner_lr: inner loop learning rate
        log_snr_max, log_snr_min: diffusion params
        rng_key: random key for timestep sampling (for reproducibility)
    
    Returns:
        updated params dict
    """
    # Sample random timesteps
    B = imgs.shape[0]
    device = imgs.device
    
    # Use rng_key to generate deterministic timesteps for vmap compatibility
    # In practice, we'll just use torch.rand since vmap handles RNG correctly
    t = torch.rand(B, device=device)
    
    # Compute loss
    loss = compute_loss_functional(
        params, model, imgs, t, cond, log_snr_max, log_snr_min
    )
    
    # Compute gradients for fast params only
    fast_params = [params[n] for n in fast_names]
    grads = torch.autograd.grad(loss, fast_params, create_graph=True)
    
    # Update fast params
    new_params = params.copy()
    for name, g in zip(fast_names, grads):
        new_params[name] = params[name] - inner_lr * g
    
    return new_params


def single_task_maml_loss(
    params: Dict[str, torch.Tensor],
    fast_names: List[str],
    model: ConditionalUNet,
    imgs: torch.Tensor,
    cond: torch.Tensor,
    inner_lr: float,
    inner_steps: int,
    log_snr_max: float,
    log_snr_min: float,
) -> torch.Tensor:
    """
    Compute outer loss for a single task after inner loop adaptation.
    
    Args:
        params: base parameter dict
        fast_names: names of fast parameters
        model: model structure
        imgs: (B, C, H, W) images for this task
        cond: (K, C, H, W) conditioning images
        inner_lr: inner loop learning rate
        inner_steps: number of inner loop steps
        log_snr_max, log_snr_min: diffusion params
    
    Returns:
        scalar outer loss
    """
    adapted_params = params
    
    for step in range(inner_steps):
        rng_key = torch.tensor([step], device=imgs.device)  # placeholder
        adapted_params = inner_loop_step(
            adapted_params, fast_names, model, imgs, cond,
            inner_lr, log_snr_max, log_snr_min, rng_key
        )
    
    # Compute outer loss with adapted params
    B = imgs.shape[0]
    t_out = torch.rand(B, device=imgs.device)
    outer_loss = compute_loss_functional(
        adapted_params, model, imgs, t_out, cond, log_snr_max, log_snr_min
    )
    
    return outer_loss


def multi_task_maml_step(
    ddpm: DDPM,
    imgs_all: torch.Tensor,
    cond_all: torch.Tensor,
    fast_names: List[str],
    base_params: Dict[str, torch.Tensor],
    inner_lr: float,
    inner_steps: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute averaged outer loss over N tasks.
    
    Uses a sequential loop to process tasks and average their losses.
    This is more reliable than vmap for MAML's nested autograd.
    
    Args:
        ddpm: DDPM model wrapper
        imgs_all: (N, B, C, H, W) images for N tasks
        cond_all: (N, K, C, H, W) conditioning images for N tasks
        fast_names: names of fast parameters
        base_params: base parameter dict
        inner_lr: inner loop learning rate
        inner_steps: number of inner loop steps
        device: torch device
    
    Returns:
        scalar averaged outer loss
    """
    N = imgs_all.shape[0]
    total_loss = torch.tensor(0.0, device=device)
    
    for i in range(N):
        task_loss = single_task_maml_loss(
            base_params,
            fast_names,
            ddpm.model,
            imgs_all[i],  # (B, C, H, W)
            cond_all[i].unsqueeze(0),  # (K, C, H, W)
            inner_lr,
            inner_steps,
            ddpm.log_snr_max,
            ddpm.log_snr_min,
        )
        total_loss = total_loss + task_loss
    
    return total_loss / N


# -------------------------
# Adaptation and sampling (unchanged from original)
# -------------------------


def adapt_and_sample(
    ddpm: DDPM,
    fast_names: List[str],
    base_params: Dict[str, torch.Tensor],
    imgs: torch.Tensor,
    cond_imgs: torch.Tensor,
    cfg: ConfigDict,
    device: torch.device,
    inner_steps: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Adapt to a task and sample images."""
    params = base_params

    def p_losses_with_params(params_dict, x0, t, cond):
        noise = torch.randn_like(x0)
        log_snr = ddpm.log_snr_max + t * (ddpm.log_snr_min - ddpm.log_snr_max)
        alpha = torch.sqrt(torch.sigmoid(log_snr))
        sigma = torch.sqrt(torch.sigmoid(-log_snr))
        x_t = alpha[:, None, None, None] * x0 + sigma[:, None, None, None] * noise
        v_target = alpha[:, None, None, None] * noise - sigma[:, None, None, None] * x0
        v_pred = functional_call(ddpm.model, params_dict, (x_t, t, cond))
        return torch.mean((v_pred - v_target) ** 2)

    for _ in range(inner_steps):
        t = torch.rand(imgs.shape[0], device=device)
        loss = p_losses_with_params(params, imgs, t, cond_imgs.unsqueeze(0))
        grads = torch.autograd.grad(loss, [params[n] for n in fast_names], create_graph=False)
        params = params.copy()
        for name, g in zip(fast_names, grads):
            params[name] = params[name] - cfg.eval.inner_lr * g

    # Detach adapted params
    params_detached = {k: v.detach() for k, v in params.items()}

    # Sampling
    with torch.no_grad():
        img = torch.randn(
            cfg.sample.num_images,
            ddpm.model.in_conv.in_channels,
            cfg.data.image_size,
            cfg.data.image_size,
            device=device,
        )
        cond_for_gen = cond_imgs.unsqueeze(0)
        times = torch.linspace(1.0, 0.0, cfg.sample.steps + 1, device=device)
        for i in range(cfg.sample.steps):
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
            if i == cfg.sample.steps - 1:
                img = x0_pred  # Final step: use clean prediction
            else:
                img = alpha_prev * x0_pred + sigma_prev * eps_pred

    samples = (img.clamp(-1, 1) + 1) / 2
    target = (imgs.clamp(-1, 1) + 1) / 2
    grid = utils.make_grid(samples, nrow=int(cfg.sample.num_images**0.5))
    target_grid = utils.make_grid(target[: cfg.sample.num_images], nrow=int(cfg.sample.num_images**0.5))
    return grid, target_grid


def log_adaptation_samples(
    ddpm: DDPM,
    dataset,
    train_indices_by_class,
    val_indices_by_class,
    train_classes: List[int],
    eval_classes: List[int],
    fast_names: List[str],
    base_params: Dict[str, torch.Tensor],
    cfg: ConfigDict,
    device: torch.device,
    wandb_run,
    global_step: int,
    inner_steps: int,
    key_prefix: str = "",
) -> None:
    """
    Log adaptation samples exactly as done during training evaluation.
    `key_prefix` can be used for special one-off logs (e.g., pretrained sanity check).
    """
    if wandb_run is None:
        return

    eval_pool = eval_classes if len(eval_classes) > 0 else train_classes
    train_pool = train_classes if len(train_classes) > 0 else eval_classes
    if len(eval_pool) == 0:
        print("Warning: no classes available for adaptation logging.")
        return

    def k(name: str) -> str:
        return f"{key_prefix}{name}" if key_prefix else name

    ddpm.eval()

    cls = random.choice(eval_pool)
    val_imgs = sample_class_batch(dataset, val_indices_by_class, cls, cfg.data.batch_size, device)
    val_cond = sample_class_batch(dataset, val_indices_by_class, cls, cfg.data.cond_batch_size, device)
    grid, target_grid = adapt_and_sample(
        ddpm, fast_names, base_params, val_imgs, val_cond, cfg, device, inner_steps=inner_steps
    )
    wandb.log(
        {
            k("samples"): wandb.Image(grid),
            k("target_samples"): wandb.Image(target_grid),
            k("eval_class"): cls,
        },
        step=global_step,
    )
    adapt_and_log_denoise(
        ddpm,
        val_imgs,
        val_cond,
        fast_names,
        base_params,
        cfg,
        device,
        wandb_run,
        global_step,
        inner_steps=inner_steps,
    )

    if len(train_pool) == 0:
        return
    train_cls = random.choice(train_pool)
    tr_imgs = sample_class_batch(dataset, train_indices_by_class, train_cls, cfg.data.batch_size, device)
    tr_cond = sample_class_batch(dataset, train_indices_by_class, train_cls, cfg.data.cond_batch_size, device)
    train_grid, train_target_grid = adapt_and_sample(
        ddpm, fast_names, base_params, tr_imgs, tr_cond, cfg, device, inner_steps=inner_steps
    )
    wandb.log(
        {
            k("train_adapt/samples"): wandb.Image(train_grid),
            k("train_adapt/target_samples"): wandb.Image(train_target_grid),
            k("train_adapt/class"): train_cls,
        },
        step=global_step,
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
    pretrained_loaded = load_pretrained_if_available(ddpm, cfg, device)

    fast_names, fast_params = select_fast_params(model, cfg.fast_params.selector)
    base_params = build_param_dict(ddpm.model)

    optimizer = optim.AdamW(
        ddpm.parameters(), lr=cfg.training.outer_lr, weight_decay=cfg.training.weight_decay
    )
    total_params, fast_params_count = count_params(ddpm.parameters(), fast_names, base_params)
    print(f"Total params: {total_params / 1e6:.2f}M, fast params: {fast_params_count / 1e6:.4f}M")
    print(f"Tasks per outer step: {cfg.training.num_tasks}")
    
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
    if pretrained_loaded and wandb_run is not None:
        print("Logging pretrained sanity-check samples before meta-training...")
        log_adaptation_samples(
            ddpm,
            dataset,
            train_indices_by_class,
            val_indices_by_class,
            train_classes,
            eval_classes,
            fast_names,
            base_params,
            cfg,
            device,
            wandb_run,
            global_step=global_step,
            inner_steps=0,
            key_prefix="pretrained_check/",
        )

    for epoch in range(1, cfg.training.epochs + 1):
        ddpm.train()
        pbar = tqdm(range(cfg.training.steps_per_epoch), desc=f"Epoch {epoch}", leave=False)
        
        for _ in pbar:
            # Sample N tasks (different classes)
            imgs_all, cond_all, sampled_classes = sample_multi_task_batches(
                dataset,
                train_indices_by_class,
                train_classes,
                num_tasks=cfg.training.num_tasks,
                batch_size_x=cfg.data.batch_size,
                batch_size_cond=cfg.data.cond_batch_size,
                device=device,
            )

            optimizer.zero_grad(set_to_none=True)
            
            # Compute averaged outer loss over N tasks
            outer_loss = multi_task_maml_step(
                ddpm,
                imgs_all,
                cond_all,
                fast_names=fast_names,
                base_params=base_params,
                inner_lr=cfg.training.inner_lr,
                inner_steps=cfg.training.inner_steps,
                device=device,
            )
            
            scaler.scale(outer_loss).backward()
            if cfg.training.grad_clip is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(ddpm.parameters(), cfg.training.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            pbar.set_postfix(loss=outer_loss.item())
            if global_step % cfg.training.log_every == 0:
                if wandb_run is not None:
                    wandb.log({"train/outer_loss": outer_loss.item()}, step=global_step)

        # Evaluation sampling
        if epoch % cfg.training.sample_every_epochs == 0:
            log_adaptation_samples(
                ddpm,
                dataset,
                train_indices_by_class,
                val_indices_by_class,
                train_classes,
                eval_classes,
                fast_names,
                base_params,
                cfg,
                device,
                wandb_run,
                global_step=global_step,
                inner_steps=cfg.eval.inner_steps,
            )

        # Checkpointing
        if epoch % cfg.training.checkpoint_every_epochs == 0:
            ckpt_path = run_save_dir / f"cond_maml_vmap_epoch_{epoch:03d}.pt"
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
    default="playground/configs/train_cond_maml_cifar100_vmap.py",
    help_string="Path to a ml_collections config file.",
)


def main(argv=None):
    del argv
    cfg = _CONFIG.value
    train(cfg)


if __name__ == "__main__":
    app.run(main)

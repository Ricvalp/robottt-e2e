#!/usr/bin/env python3
from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, List, Tuple, Dict

from absl import app
from ml_collections import ConfigDict, config_flags
import numpy as np
import torch
from torch import nn, optim
import torch.amp as amp
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
import torch.func as func
import torch.nn.functional as F
from classifier_model import SmallResNet
import json
import matplotlib.pyplot as plt
import os

from model import UNet, DDPM, ResBlock


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
    transform = transforms.Compose(
        [
            transforms.Resize(cfg.data.image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2 - 1),
        ]
    )
    train_ds = datasets.MNIST(
        root=cfg.data.root,
        train=True,
        download=cfg.data.download,
        transform=transform,
    )
    rng = np.random.default_rng(cfg.run.seed)
    holdout = int(cfg.data.holdout_per_class)
    train_indices_by_digit = {}
    val_indices_by_digit = {}

    if cfg.data.use_full_dataset:
        for d in range(10):
            idxs = [i for i, t in enumerate(train_ds.targets) if int(t) == d]
            rng.shuffle(idxs)
            val_indices_by_digit[d] = idxs[:holdout]
            train_indices_by_digit[d] = idxs[holdout:]
        train_digits = list(range(10))
        eval_digits = list(range(10))
    else:
        leave_out = int(cfg.data.leave_out_digit)
        for d in range(10):
            idxs = [i for i, t in enumerate(train_ds.targets) if int(t) == d]
            rng.shuffle(idxs)
            if d == leave_out:
                val_indices_by_digit[d] = idxs[: int(cfg.data.leaveout_eval_holdout)]
                train_indices_by_digit[d] = []
            else:
                val_indices_by_digit[d] = idxs[:holdout]
                train_indices_by_digit[d] = idxs[holdout:]
        train_digits = [d for d in range(10) if d != leave_out]
        eval_digits = [leave_out]

    return train_ds, train_indices_by_digit, val_indices_by_digit, train_digits, eval_digits


def sample_digit_batch(dataset, indices_by_digit, digit: int, batch_size: int, device: torch.device) -> torch.Tensor:
    idx_pool = indices_by_digit[digit]
    replace = len(idx_pool) < batch_size
    idx = np.random.choice(idx_pool, size=batch_size, replace=replace)
    imgs = [dataset[i][0] for i in idx]
    imgs = torch.stack(imgs, dim=0).to(device)
    return imgs


def select_fast_params(model: UNet, selector: str) -> Tuple[List[str], List[nn.Parameter]]:
    """
    selector options:
      - 'up_time_mlp': time MLPs in up ResBlocks
      - 'up_down_time_mlp': time MLPs in up + down ResBlocks
      - 'up_down_mid_head': time MLPs in up/down/mid ResBlocks plus final norm/conv
      - 'up_down_mid_head_gn': previous + GroupNorm weights/bias in up/down/mid ResBlocks
      - 'up_down_mid_full': time MLPs + convs (+skip) in up/down/mid ResBlocks, plus head
      - 'up_down_mid_full_gn': previous + GroupNorm in those blocks
    """
    fast_params: List[nn.Parameter] = []
    fast_names: List[str] = []

    def add_mlp(name, module):
        for pname, p in module.time_mlp.named_parameters():
            full = f"{name}.time_mlp.{pname}"
            fast_names.append(full)
            fast_params.append(p)

    def add_gn(name, gn: nn.GroupNorm):
        for pname, p in gn.named_parameters():
            full = f"{name}.{pname}"
            fast_names.append(full)
            fast_params.append(p)

    def add_convs(name, module: ResBlock):
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
        if isinstance(module, ResBlock):
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
        fast_names.append("out_norm.weight")
        fast_params.append(model.out_norm.weight)
        fast_names.append("out_norm.bias")
        fast_params.append(model.out_norm.bias)
        fast_names.append("out_conv.weight")
        fast_params.append(model.out_conv.weight)
        fast_names.append("out_conv.bias")
        fast_params.append(model.out_conv.bias)

    return fast_names, fast_params


def clone_state(params: Iterable[nn.Parameter]) -> List[torch.Tensor]:
    return [p.detach().clone() for p in params]


def load_state(params: Iterable[nn.Parameter], state: List[torch.Tensor]) -> None:
    for p, s in zip(params, state):
        p.data.copy_(s)


def build_param_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {name: p for name, p in model.named_parameters()}


def count_params(all_params: Iterable[torch.Tensor], fast_names: List[str], base_params: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    total = sum(p.numel() for p in all_params)
    fast = sum(base_params[n].numel() for n in fast_names)
    return total, fast


def grid_to_uint8(grid: torch.Tensor) -> torch.Tensor:
    """Convert CHW grid in [0,1] float to uint8 CHW on CPU to avoid autoscaling/fading in logs."""
    return (grid.clamp(0, 1) * 255).round().byte().cpu()


def sample_with_params(
    ddpm: DDPM,
    params_dict: Dict[str, torch.Tensor],
    num_samples: int,
    cfg: ConfigDict,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    imgs: List[torch.Tensor] = []
    steps = cfg.sample.steps
    generated = 0
    while generated < num_samples:
        bs = min(batch_size, num_samples - generated)
        img = torch.randn(bs, ddpm.model.in_conv.in_channels, cfg.data.image_size, cfg.data.image_size, device=device)
        times = torch.linspace(1.0, 0.0, steps + 1, device=device)
        for i in range(steps):
            t_cur = times[i].repeat(img.shape[0])
            t_prev = times[i + 1].repeat(img.shape[0])
            v_pred = func.functional_call(ddpm.model, params_dict, (img, t_cur))
            alpha_cur, sigma_cur = ddpm._alpha_sigma(t_cur)
            alpha_cur = alpha_cur[:, None, None, None]
            sigma_cur = sigma_cur[:, None, None, None]
            alpha_prev, sigma_prev = ddpm._alpha_sigma(t_prev)
            alpha_prev = alpha_prev[:, None, None, None]
            sigma_prev = sigma_prev[:, None, None, None]
            x0_pred = alpha_cur * img - sigma_cur * v_pred
            eps_pred = alpha_cur * v_pred + sigma_cur * img
            img = alpha_prev * x0_pred + sigma_prev * eps_pred
        imgs.append(img.detach())
        generated += bs
    return torch.cat(imgs, dim=0)


@torch.no_grad()
def get_classifier_features(classifier: nn.Module, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Assumes SmallResNet structure
    with torch.no_grad():
        h = classifier.stem(x)
        h = classifier.blocks(h)
        h = classifier.pool(h).flatten(1)
        logits = classifier.fc(h)
    return logits, h


@torch.no_grad()
def classify_and_fid(
    ddpm: DDPM,
    params_dict: Dict[str, torch.Tensor],
    classifier: nn.Module,
    cfg: ConfigDict,
    device: torch.device,
    num_samples: int,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    classifier.eval()
    samples = sample_with_params(
        ddpm,
        params_dict=params_dict,
        num_samples=num_samples,
        cfg=cfg,
        device=device,
        batch_size=batch_size,
    )
    samples = (samples.clamp(-1, 1) + 1) / 2
    counts = torch.zeros(10, device=device)
    feats = []
    for i in range(0, samples.shape[0], batch_size):
        batch = samples[i : i + batch_size]
        logits, h = get_classifier_features(classifier, batch)
        preds = logits.argmax(dim=1)
        counts += torch.bincount(preds, minlength=10).to(device)
        feats.append(h)
    feats = torch.cat(feats, dim=0)
    return counts.cpu(), feats.cpu()


# Backward-compat convenience wrapper (used in logging block)
@torch.no_grad()
def count_generated_with_classifier(
    ddpm: DDPM,
    params_dict: Dict[str, torch.Tensor],
    classifier: nn.Module,
    cfg: ConfigDict,
    device: torch.device,
) -> torch.Tensor:
    counts, _ = classify_and_fid(
        ddpm,
        params_dict=params_dict,
        classifier=classifier,
        cfg=cfg,
        device=device,
        num_samples=cfg.counting.num_samples,
        batch_size=cfg.counting.batch_size,
    )
    return counts


def compute_fid(mu1: torch.Tensor, sigma1: torch.Tensor, mu2: torch.Tensor, sigma2: torch.Tensor) -> float:
    mu1_np = mu1.numpy()
    mu2_np = mu2.numpy()
    sigma1_np = sigma1.numpy()
    sigma2_np = sigma2.numpy()
    diff = mu1_np - mu2_np
    cov_prod = sigma1_np @ sigma2_np
    cov_prod = (cov_prod + cov_prod.T) * 0.5  # symmetrize
    eigvals, eigvecs = np.linalg.eigh(cov_prod)
    sqrt_eigvals = np.sqrt(np.clip(eigvals, 0, None))
    covmean = eigvecs @ np.diag(sqrt_eigvals) @ eigvecs.T
    fid = diff.dot(diff) + np.trace(sigma1_np + sigma2_np - 2 * covmean)
    return float(max(fid, 0.0))


def compute_dataset_stats(classifier: nn.Module, loader: DataLoader, device: torch.device, stats_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    classifier.eval()
    feats = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            _, h = get_classifier_features(classifier, x)
            feats.append(h.cpu())
    feats = torch.cat(feats, dim=0)
    mu = feats.mean(dim=0)
    sigma = torch.from_numpy(np.cov(feats.numpy(), rowvar=False))
    Path(stats_path).parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump({"mu": mu.tolist(), "sigma": sigma.tolist()}, f)
    return mu, sigma


def load_or_compute_stats(classifier: nn.Module, cfg: ConfigDict, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    stats_path = cfg.fid.stats_path
    if os.path.exists(stats_path):
        with open(stats_path, "r") as f:
            data = json.load(f)
        mu = torch.tensor(data["mu"], dtype=torch.float32)
        sigma = torch.tensor(data["sigma"], dtype=torch.float32)
        return mu, sigma
    else:
        transform = transforms.Compose(
            [transforms.Resize(cfg.data.image_size), transforms.ToTensor()]
        )
        test_ds = datasets.MNIST(cfg.data.root, train=False, download=True, transform=transform)
        test_loader = DataLoader(
            test_ds,
            batch_size=cfg.counting.batch_size,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        return compute_dataset_stats(classifier, test_loader, device, stats_path)


def adapt_params(
    ddpm: DDPM,
    fast_names: List[str],
    base_params: Dict[str, torch.Tensor],
    batch: torch.Tensor,
    inner_steps: int,
    inner_lr: float,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    params = base_params
    def p_losses_with_params(params_dict, x0, t):
        noise = torch.randn_like(x0)
        alpha, sigma = ddpm._alpha_sigma(t)
        x_t = alpha[:, None, None, None] * x0 + sigma[:, None, None, None] * noise
        v_target = alpha[:, None, None, None] * noise - sigma[:, None, None, None] * x0
        v_pred = func.functional_call(ddpm.model, params_dict, (x_t, t))
        return torch.mean((v_pred - v_target) ** 2)

    for _ in range(inner_steps):
        t = torch.rand(batch.shape[0], device=device)
        loss = p_losses_with_params(params, batch, t)
        grads = torch.autograd.grad(loss, [params[n] for n in fast_names], create_graph=False)
        params = params.copy()
        for name, g in zip(fast_names, grads):
            params[name] = params[name] - inner_lr * g
    return params


def log_denoise(
    ddpm: DDPM,
    batch: torch.Tensor,
    fast_names: List[str],
    base_params: Dict[str, torch.Tensor],
    cfg: ConfigDict,
    device: torch.device,
    wandb_run,
    global_step: int,
    max_imgs: int = 8,
):
    """Adapt fast params on the batch copy, then log x_t and x0_pred from the adapted model."""
    if wandb_run is None:
        return

    # Adapt params functionally (no in-place)
    params = base_params

    def p_losses_with_params(params_dict, x0, t):
        noise = torch.randn_like(x0)
        alpha, sigma = ddpm._alpha_sigma(t)
        x_t = alpha[:, None, None, None] * x0 + sigma[:, None, None, None] * noise
        v_target = alpha[:, None, None, None] * noise - sigma[:, None, None, None] * x0
        v_pred = func.functional_call(ddpm.model, params_dict, (x_t, t))
        return torch.mean((v_pred - v_target) ** 2), x_t, v_pred, alpha, sigma

    b = batch[:max_imgs].to(device)
    for _ in range(cfg.eval.inner_steps):
        t = torch.rand(b.shape[0], device=device)
        loss, _, _, _, _ = p_losses_with_params(params, b, t)
        grads = torch.autograd.grad(loss, [params[n] for n in fast_names], create_graph=False)
        params = params.copy()
        for name, g in zip(fast_names, grads):
            params[name] = params[name] - cfg.eval.inner_lr * g

    with torch.no_grad():
        t = torch.rand(b.shape[0], device=device)
        _, x_t, v_pred, alpha, sigma = p_losses_with_params(params, b, t)
        x0_pred = alpha[:, None, None, None] * x_t - sigma[:, None, None, None] * v_pred
        x_t_vis = (x_t.clamp(-1, 1) + 1) / 2
        x0_vis = (x0_pred.clamp(-1, 1) + 1) / 2
        combined = torch.cat([x_t_vis, x0_vis], dim=0)
        grid = utils.make_grid(combined, nrow=max_imgs, padding=2)
        wandb_run.log({"debug/x_t_and_x0": wandb.Image(grid_to_uint8(grid))}, step=global_step)


def maml_step(
    ddpm: DDPM,
    batch: torch.Tensor,
    fast_names: List[str],
    base_params: Dict[str, torch.Tensor],
    inner_lr: float,
    inner_steps: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Full second-order MAML: inner updates are differentiable; outer loss sees them.
    """
    params = base_params
    def p_losses_with_params(params_dict, x0, t):
        if t is None:
            t = torch.rand(x0.shape[0], device=device)
        if isinstance(t, torch.Tensor) and t.dim() == 0:
            t = t.repeat(x0.shape[0])
        noise = torch.randn_like(x0)
        alpha, sigma = ddpm._alpha_sigma(t)
        x_t = alpha[:, None, None, None] * x0 + sigma[:, None, None, None] * noise
        v_target = alpha[:, None, None, None] * noise - sigma[:, None, None, None] * x0
        v_pred = func.functional_call(ddpm.model, params_dict, (x_t, t))
        return torch.mean((v_pred - v_target) ** 2)

    for _ in range(inner_steps):
        t = torch.rand(batch.shape[0], device=device)
        loss = p_losses_with_params(params, batch, t)
        grads = torch.autograd.grad(loss, [params[n] for n in fast_names], create_graph=True)
        params = params.copy()
        for name, g in zip(fast_names, grads):
            params[name] = params[name] - inner_lr * g

    t_out = torch.rand(batch.shape[0], device=device)
    outer_loss = p_losses_with_params(params, batch, t_out)
    return outer_loss


def adapt_and_sample(
    ddpm: DDPM,
    fast_names: List[str],
    base_params: Dict[str, torch.Tensor],
    batch: torch.Tensor,
    cfg: ConfigDict,
    device: torch.device,
    epoch: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Adapt fast params (differentiable copies) on a single-image batch, then sample."""
    params = adapt_params(
        ddpm,
        fast_names=fast_names,
        base_params=base_params,
        batch=batch,
        inner_steps=cfg.eval.inner_steps,
        inner_lr=cfg.eval.inner_lr,
        device=device,
    )

    ddpm.eval()
    with torch.no_grad():
        samples = sample_with_params(
            ddpm,
            params_dict=params,
            num_samples=cfg.sample.num_images,
            cfg=cfg,
            device=device,
            batch_size=cfg.sample.num_images,
        )
        samples = (samples.clamp(-1, 1) + 1) / 2
        grid = utils.make_grid(samples, nrow=int(cfg.sample.num_images ** 0.5))
        batch_grid = utils.make_grid((batch + 1) / 2, nrow=int(batch.shape[0] ** 0.5))
        grid = grid_to_uint8(grid)
        batch_grid = grid_to_uint8(batch_grid)
    return grid, batch_grid


def train(cfg: ConfigDict) -> None:
    device = torch.device(cfg.run.device if torch.cuda.is_available() or cfg.run.device == "cpu" else "cpu")
    set_seed(cfg.run.seed)

    dataset, train_indices_by_digit, val_indices_by_digit, train_digits, eval_digits = build_datasets(cfg)

    model = UNet(
        in_channels=cfg.model.in_channels,
        base_channels=cfg.model.base_channels,
        channel_mults=tuple(cfg.model.channel_mults),
        num_res_blocks=cfg.model.num_res_blocks,
        dropout=cfg.model.dropout,
        attn_resolutions=tuple(cfg.model.attn_resolutions),
        num_heads=cfg.model.num_heads,
        image_size=cfg.data.image_size,
        time_scale=cfg.diffusion.train_steps,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params/1e6:.2f}M ({n_params} params)")

    ddpm = DDPM(
        model,
        beta_start=cfg.diffusion.beta_start,
        beta_end=cfg.diffusion.beta_end,
        train_steps=cfg.diffusion.train_steps,
    ).to(device)

    fast_names, fast_params = select_fast_params(model, cfg.fast_params.selector)
    base_params = build_param_dict(ddpm.model)

    # Outer optimizer updates all params (meta-learns fast init as well)
    optimizer = optim.AdamW(ddpm.parameters(), lr=cfg.training.outer_lr, weight_decay=cfg.training.weight_decay)
    total_params, fast_params_count = count_params(ddpm.parameters(), fast_names, base_params)
    print(f"Total params: {total_params/1e6:.2f}M, fast params: {fast_params_count/1e6:.4f}M")
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
        # set run name to run id
        wandb.run.name = wandb.run.id
        run_save_dir = Path(cfg.checkpoint.dir) / wandb.run.id
    run_save_dir.mkdir(parents=True, exist_ok=True)

    classifier = None
    mu_ref = sigma_ref = None
    if (cfg.counting.use or cfg.fid.use) and cfg.counting.classifier_ckpt:
        classifier = SmallResNet().to(device)
        ckpt = torch.load(cfg.counting.classifier_ckpt, map_location=device)
        classifier.load_state_dict(ckpt["model"])
        classifier.eval()
        if cfg.fid.use:
            mu_ref, sigma_ref = load_or_compute_stats(classifier, cfg, device)

    global_step = 0
    for epoch in range(1, cfg.training.epochs + 1):
        ddpm.train()
        pbar = tqdm(range(cfg.training.steps_per_epoch), desc=f"Epoch {epoch}", leave=False)
        for _ in pbar:
            digit = random.choice(train_digits)
            imgs = sample_digit_batch(dataset, train_indices_by_digit, digit, cfg.data.batch_size, device)

            optimizer.zero_grad(set_to_none=True)
            outer_loss = maml_step(
                ddpm,
                imgs,
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
                print(f"[epoch {epoch:03d} step {global_step:06d}] outer_loss={outer_loss.item():.4f}")
                if wandb_run is not None:
                    wandb.log({"train/outer_loss": outer_loss.item()}, step=global_step)
                    log_denoise(ddpm, imgs, fast_names, base_params, cfg, device, wandb_run, global_step)

        if epoch % cfg.training.sample_every_epochs == 0:
            ddpm.eval()
            # log on a random eval digit (held-out slices per class or leave-out digit)
            eval_digit = random.choice(eval_digits)
            val_imgs = sample_digit_batch(dataset, val_indices_by_digit, eval_digit, cfg.data.batch_size, device)
            grid, batch_grid = adapt_and_sample(ddpm, fast_names, base_params, val_imgs, cfg, device, epoch)
            if wandb_run is not None:
                wandb.log(
                    {
                        "samples": wandb.Image(grid),
                        "target_samples": wandb.Image(batch_grid),
                        "eval_digit": eval_digit,
                    },
                    step=global_step,
                )
            # Also log adapted samples on a training digit/batch
            train_digit = random.choice(train_digits)
            train_imgs = sample_digit_batch(dataset, train_indices_by_digit, train_digit, cfg.data.batch_size, device)
            train_grid, train_batch_grid = adapt_and_sample(ddpm, fast_names, base_params, train_imgs, cfg, device, epoch)
            if wandb_run is not None:
                wandb.log(
                    {
                        "train_adapt/samples": wandb.Image(train_grid),
                        "train_adapt/target_samples": wandb.Image(train_batch_grid),
                        "train_adapt/digit": train_digit,
                    },
                    step=global_step,
                )
            # Class-count histogram using classifier on adapted params
            if classifier is not None and (cfg.counting.use or cfg.fid.use):
                adapted_params = adapt_params(
                    ddpm,
                    fast_names=fast_names,
                    base_params=base_params,
                    batch=val_imgs,
                    inner_steps=cfg.eval.inner_steps,
                    inner_lr=cfg.eval.inner_lr,
                    device=device,
                )
                num_samples = max(cfg.counting.num_samples if cfg.counting.use else 0, cfg.fid.num_samples if cfg.fid.use else 0)
                counts, feats = classify_and_fid(
                    ddpm,
                    params_dict=adapted_params,
                    classifier=classifier,
                    cfg=cfg,
                    device=device,
                    num_samples=num_samples,
                    batch_size=cfg.counting.batch_size,
                )
                if cfg.counting.use and wandb_run is not None:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.bar(range(10), counts.numpy())
                    ax.set_xlabel("Digit")
                    ax.set_ylabel("Count")
                    ax.set_title(f"Counts after adapting on digit {eval_digit}")
                    fig.tight_layout()
                    wandb_run.log(
                        {
                            "counting/histogram": wandb.Image(fig),
                            "counting/total": int(counts.sum().item()),
                            "counting/adapt_digit": eval_digit,
                        },
                        step=global_step,
                    )
                if cfg.fid.use and mu_ref is not None and sigma_ref is not None and wandb_run is not None:
                    mu_gen = feats.mean(dim=0)
                    sigma_gen = torch.from_numpy(np.cov(feats.numpy(), rowvar=False))
                    fid_val = compute_fid(mu_gen, sigma_gen, mu_ref, sigma_ref)
                    wandb_run.log({"fid": fid_val}, step=global_step)

        if epoch % cfg.training.checkpoint_every_epochs == 0:
            ckpt_path = run_save_dir / f"maml_epoch_{epoch:03d}.pt"
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
    default="playground/configs/train_maml_mnist.py",
    help_string="Path to a ml_collections config file.",
)


def main(argv=None):
    del argv
    cfg = _CONFIG.value
    train(cfg)


if __name__ == "__main__":
    app.run(main)

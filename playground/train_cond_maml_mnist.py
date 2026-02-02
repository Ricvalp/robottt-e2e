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
import wandb
from tqdm import tqdm
import torch.func as func

from conditional_model import ConditionalUNet, DDPM, ResBlockCond


# -------------------------
# Logging helpers
# -------------------------


def log_denoise(ddpm: DDPM, imgs: torch.Tensor, cond_imgs: torch.Tensor, device: torch.device, wandb_run, global_step: int, max_imgs: int = 8):
    if wandb_run is None:
        return
    with torch.no_grad():
        x0 = imgs[:max_imgs].to(device)
        cond = cond_imgs[:max_imgs].to(device)
        t = torch.rand(x0.shape[0], device=device)
        noise = torch.randn_like(x0)
        alpha, sigma = ddpm._alpha_sigma(t)
        x_t = alpha[:, None, None, None] * x0 + sigma[:, None, None, None] * noise
        v_pred = ddpm.model(x_t, t, cond)
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


def sample_digit_pair_batches(dataset, indices_by_digit, digit: int, batch_size_x: int, batch_size_cond: int, device: torch.device):
    """
    Returns two batches of the same digit but (ideally) different instances:
    - imgs_x: used for denoising loss / inner-loop
    - imgs_cond: used as conditioning exemplars
    """
    idx_pool = indices_by_digit[digit]
    total = batch_size_x + batch_size_cond
    replace = len(idx_pool) < total
    idx = np.random.choice(idx_pool, size=total, replace=replace)
    np.random.shuffle(idx)
    x_idx = idx[:batch_size_x]
    cond_idx = idx[batch_size_x:]
    imgs_x = torch.stack([dataset[i][0] for i in x_idx], dim=0).to(device)
    imgs_cond = torch.stack([dataset[i][0] for i in cond_idx], dim=0).to(device)
    return imgs_x, imgs_cond


def select_fast_params(model: ConditionalUNet, selector: str) -> Tuple[List[str], List[nn.Parameter]]:
    """
    selector options mirror the unconditional version, applied to ResBlockCond.
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


def count_params(all_params: Iterable[torch.Tensor], fast_names: List[str], base_params: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    total = sum(p.numel() for p in all_params)
    fast = sum(base_params[n].numel() for n in fast_names)
    return total, fast


# -------------------------
# MAML core
# -------------------------

def maml_step(
    ddpm: DDPM,
    imgs: torch.Tensor,
    cond_imgs: torch.Tensor,
    fast_names: List[str],
    base_params: Dict[str, torch.Tensor],
    inner_lr: float,
    inner_steps: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Second-order MAML with conditioning.
    """
    params = base_params

    def p_losses_with_params(params_dict, x0, t, cond):
        noise = torch.randn_like(x0)
        alpha, sigma = ddpm._alpha_sigma(t)
        x_t = alpha[:, None, None, None] * x0 + sigma[:, None, None, None] * noise
        v_target = alpha[:, None, None, None] * noise - sigma[:, None, None, None] * x0
        v_pred = func.functional_call(ddpm.model, params_dict, (x_t, t, cond))
        return torch.mean((v_pred - v_target) ** 2)

    for _ in range(inner_steps):
        t = torch.rand(imgs.shape[0], device=device)
        loss = p_losses_with_params(params, imgs, t, cond_imgs)
        grads = torch.autograd.grad(loss, [params[n] for n in fast_names], create_graph=True)
        params = params.copy()
        for name, g in zip(fast_names, grads):
            params[name] = params[name] - inner_lr * g

    t_out = torch.rand(imgs.shape[0], device=device)
    outer_loss = p_losses_with_params(params, imgs, t_out, cond_imgs)
    return outer_loss


def adapt_and_sample(
    ddpm: DDPM,
    fast_names: List[str],
    base_params: Dict[str, torch.Tensor],
    imgs: torch.Tensor,
    cond_imgs: torch.Tensor,
    cfg: ConfigDict,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    params = base_params

    def p_losses_with_params(params_dict, x0, t, cond):
        noise = torch.randn_like(x0)
        alpha, sigma = ddpm._alpha_sigma(t)
        x_t = alpha[:, None, None, None] * x0 + sigma[:, None, None, None] * noise
        v_target = alpha[:, None, None, None] * noise - sigma[:, None, None, None] * x0
        v_pred = func.functional_call(ddpm.model, params_dict, (x_t, t, cond))
        return torch.mean((v_pred - v_target) ** 2)

    for _ in range(cfg.eval.inner_steps):
        t = torch.rand(imgs.shape[0], device=device)
        loss = p_losses_with_params(params, imgs, t, cond_imgs)
        grads = torch.autograd.grad(loss, [params[n] for n in fast_names], create_graph=False)
        params = params.copy()
        for name, g in zip(fast_names, grads):
            params[name] = params[name] - cfg.eval.inner_lr * g

    # Detach adapted params so sampling doesn't hold the inner-loop graph
    params_detached = {k: v.detach() for k, v in params.items()}

    # sampling with adapted params and conditioning cond_imgs
    with torch.no_grad():
        img = torch.randn(cfg.sample.num_images, ddpm.model.in_conv.in_channels, cfg.data.image_size, cfg.data.image_size, device=device)
        cond_for_gen = cond_imgs[:1].repeat(img.shape[0], 1, 1, 1)  # reuse one exemplar to match batch
        times = torch.linspace(1.0, 0.0, cfg.sample.steps + 1, device=device)
        for i in range(cfg.sample.steps):
            t_cur = times[i].repeat(img.shape[0])
            t_prev = times[i + 1].repeat(img.shape[0])
            v_pred = func.functional_call(ddpm.model, params_detached, (img, t_cur, cond_for_gen))
            alpha_cur, sigma_cur = ddpm._alpha_sigma(t_cur)
            alpha_cur = alpha_cur[:, None, None, None]
            sigma_cur = sigma_cur[:, None, None, None]
            alpha_prev, sigma_prev = ddpm._alpha_sigma(t_prev)
            alpha_prev = alpha_prev[:, None, None, None]
            sigma_prev = sigma_prev[:, None, None, None]
            x0_pred = alpha_cur * img - sigma_cur * v_pred
            eps_pred = alpha_cur * v_pred + sigma_cur * img
            img = alpha_prev * x0_pred + sigma_prev * eps_pred

    samples = (img.clamp(-1, 1) + 1) / 2
    target = (imgs.clamp(-1, 1) + 1) / 2
    grid = utils.make_grid(samples, nrow=int(cfg.sample.num_images ** 0.5))
    target_grid = utils.make_grid(target[: cfg.sample.num_images], nrow=int(cfg.sample.num_images ** 0.5))
    return grid, target_grid


# -------------------------
# Training loop
# -------------------------

def train(cfg: ConfigDict) -> None:
    device = torch.device(cfg.run.device if torch.cuda.is_available() or cfg.run.device == "cpu" else "cpu")
    set_seed(cfg.run.seed)

    dataset, train_indices_by_digit, val_indices_by_digit, train_digits, eval_digits = build_datasets(cfg)

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
        time_scale=1000.0,  # Standard time embedding scale
        cond_dim=cfg.model.cond_dim,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params/1e6:.2f}M ({n_params})")

    ddpm = DDPM(
        model,
        log_snr_max=cfg.diffusion.log_snr_max,
        log_snr_min=cfg.diffusion.log_snr_min,
        
    ).to(device)

    fast_names, fast_params = select_fast_params(model, cfg.fast_params.selector)
    base_params = build_param_dict(ddpm.model)

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
        wandb.run.name = wandb.run.id
        run_save_dir = Path(cfg.checkpoint.dir) / wandb.run.id
    run_save_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    for epoch in range(1, cfg.training.epochs + 1):
        ddpm.train()
        pbar = tqdm(range(cfg.training.steps_per_epoch), desc=f"Epoch {epoch}", leave=False)
        for _ in pbar:
            digit = random.choice(train_digits)
            imgs, cond_imgs = sample_digit_pair_batches(
                dataset,
                train_indices_by_digit,
                digit,
                cfg.data.batch_size,
                cfg.data.cond_batch_size,
                device,
            )

            optimizer.zero_grad(set_to_none=True)
            outer_loss = maml_step(
                ddpm,
                imgs,
                cond_imgs,
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
                    log_denoise(ddpm, imgs, cond_imgs, device, wandb_run, global_step)

        if epoch % cfg.training.sample_every_epochs == 0:
            ddpm.eval()
            digit = random.choice(eval_digits)
            val_imgs, val_cond = sample_digit_pair_batches(
                dataset, val_indices_by_digit, digit, cfg.data.batch_size, cfg.data.cond_batch_size, device
            )
            grid, target_grid = adapt_and_sample(ddpm, fast_names, base_params, val_imgs, val_cond, cfg, device)
            if wandb_run is not None:
                wandb.log({"samples": wandb.Image(grid), "target_samples": wandb.Image(target_grid), "eval_digit": digit}, step=global_step)

            # also log adaptation on a training digit
            train_digit = random.choice(train_digits)
            tr_imgs, tr_cond = sample_digit_pair_batches(
                dataset, train_indices_by_digit, train_digit, cfg.data.batch_size, cfg.data.cond_batch_size, device
            )
            train_grid, train_target_grid = adapt_and_sample(ddpm, fast_names, base_params, tr_imgs, tr_cond, cfg, device)
            if wandb_run is not None:
                wandb.log(
                    {
                        "train_adapt/samples": wandb.Image(train_grid),
                        "train_adapt/target_samples": wandb.Image(train_target_grid),
                        "train_adapt/digit": train_digit,
                    },
                    step=global_step,
                )

        if epoch % cfg.training.checkpoint_every_epochs == 0:
            ckpt_path = run_save_dir / f"cond_maml_epoch_{epoch:03d}.pt"
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
    default="playground/configs/train_cond_maml_mnist.py",
    help_string="Path to a ml_collections config file.",
)


def main(argv=None):
    del argv
    cfg = _CONFIG.value
    train(cfg)


if __name__ == "__main__":
    app.run(main)

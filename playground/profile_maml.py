#!/usr/bin/env python3
"""
Profile a short MAML diffusion run (4 outer steps) and export a chrome trace for Perfetto.
Uses the same config and code path as playground/train_maml_mnist.py but omits all logging.
"""
from __future__ import annotations

from pathlib import Path
import random

from absl import app
from ml_collections import config_flags
import numpy as np
import torch
from torch import nn, optim
import torch.amp as amp
from torch.profiler import profile, ProfilerActivity

import train_maml_mnist as tm
from model import UNet, DDPM


_CONFIG = config_flags.DEFINE_config_file(
    "maml_config",
    default="playground/configs/train_maml_mnist.py",
    help_string="Path to a ml_collections config file.",
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def main(argv=None):
    del argv
    cfg = _CONFIG.value
    # Disable wandb/logging side effects
    if hasattr(cfg, "wandb"):
        cfg.wandb.use = False

    device = torch.device(cfg.run.device if torch.cuda.is_available() or cfg.run.device == "cpu" else "cpu")
    set_seed(cfg.run.seed)

    dataset, train_indices_by_digit, val_indices_by_digit, train_digits, eval_digits = tm.build_datasets(cfg)

    model = UNet(
        in_channels=cfg.model.in_channels,
        base_channels=cfg.model.base_channels,
        channel_mults=tuple(cfg.model.channel_mults),
        num_res_blocks=cfg.model.num_res_blocks,
        dropout=cfg.model.dropout,
        attn_resolutions=tuple(cfg.model.attn_resolutions),
        num_heads=cfg.model.num_heads,
        image_size=cfg.data.image_size,
        time_scale=1000.0,  # Standard time embedding scale
    ).to(device)

    ddpm = DDPM(
        model,
        beta_start=cfg.diffusion.beta_start,
        beta_end=cfg.diffusion.beta_end,
        train_steps=cfg.diffusion.train_steps,
    ).to(device)


    fast_names, _ = tm.select_fast_params(model, cfg.fast_params.selector)
    optimizer = optim.AdamW(ddpm.parameters(), lr=cfg.training.outer_lr, weight_decay=cfg.training.weight_decay)
    scaler = amp.GradScaler(device=device.type, enabled=cfg.training.use_amp and device.type == "cuda")

    trace_dir = Path("playground/profiling/trace")
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = trace_dir / "trace.json"

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    steps_to_profile = 4

    with profile(
        activities=activities,
        record_shapes=False,
        with_stack=False,
        profile_memory=False,
    ) as prof:
        for step in range(steps_to_profile):
            digit = random.choice(train_digits)
            imgs = tm.sample_digit_batch(dataset, train_indices_by_digit, digit, cfg.data.batch_size, device)

            base_params = tm.build_param_dict(ddpm.model)

            optimizer.zero_grad(set_to_none=True)
            outer_loss = tm.maml_step(
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

            prof.step()

    prof.export_chrome_trace(str(trace_path))
    print(f"Saved trace to {trace_path}")


if __name__ == "__main__":
    app.run(main)

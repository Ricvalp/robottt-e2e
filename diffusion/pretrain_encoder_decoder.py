#!/usr/bin/env python3
"""
Train a DiT-based diffusion policy on QuickDraw episodes using the existing dataset pipeline.
"""

from __future__ import annotations

from typing import Any
from pathlib import Path

import matplotlib
import torch
from ml_collections import ConfigDict, config_flags
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset.episode_builder import EpisodeBuilderSimilar

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import wandb
from dataset.diffusion import ContextQueryInContextDiffusionCollator
from dataset.loader import QuickDrawEpisodes
from diffusion import DiTEncDecDiffusionPolicy, DiTEncDecDiffusionPolicyConfig
from diffusion.sampling import sample_quickdraw_tokens_encoder_decoder
from lstm.utils import WarmupCosineScheduler


def load_config(_CONFIG_FILE: str) -> ConfigDict:
    cfg = _CONFIG_FILE.value
    return cfg


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _log_qualitative_samples(
    policy: DiTEncDecDiffusionPolicy,
    context: dict,
    split: str,
    cfg: dict,
    step: int,
    device: torch.device,
) -> None:
    """Sample sketches and push them to WandB for visual inspection."""

    if (not cfg.wandb.use) or cfg.wandb.project is None or cfg.eval.samples <= 0:
        return

    prev_mode = policy.training
    policy.eval()

    generator = torch.Generator(device=device)
    generator.manual_seed(cfg.eval.seed + step)

    samples = sample_quickdraw_tokens_encoder_decoder(
        policy=policy,
        max_tokens=cfg.data.max_query_len,
        demos=context,
        generator=generator,
    )

    def _plot_tokens(
        ax, tokens: torch.Tensor, title: str, coordinate_mode: str
    ) -> None:
        """Render `(N, 3)` tokens on the provided axis."""
        array = tokens.detach().cpu().numpy()
        coords = (
            array[:, :2].cumsum(axis=0) if coordinate_mode == "delta" else array[:, :2]
        )
        pen_state = array[:, 2]
        for token_idx in range(1, coords.shape[0]):
            start = coords[token_idx - 1]
            end = coords[token_idx]
            active = pen_state[token_idx] >= 0.5
            ax.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                color="black" if active else "tab:red",
                linewidth=1.5,
                linestyle="-" if active else "--",
            )
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.axis("off")

    def _split_context_prompts(ctx_tokens: torch.Tensor) -> list[torch.Tensor]:
        """Split the concatenated context tokens into individual prompt sketches."""
        sketches = []
        current = []
        for token in ctx_tokens:
            if token[5] > 0.5:
                break
            if token[4] > 0.5:
                if current:
                    sketches.append(torch.stack(current))
                    current = []
                continue
            current.append(token[[0, 1, 2]])
        if current:
            sketches.append(torch.stack(current))
        return sketches[: cfg.data.K]

    images = []
    batch_size = len(samples)
    for idx in range(batch_size):
        ctx_tokens = context["context"][idx]
        ctx_mask = context["context_mask"][idx]
        valid_ctx = ctx_tokens[ctx_mask].detach().cpu()
        prompts = _split_context_prompts(valid_ctx)
        sample_tokens = samples[idx]

        total_plots = len(prompts) + 1
        cols = min(total_plots, 3)
        rows = (total_plots + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), dpi=150)
        axes = list(axes.flat) if hasattr(axes, "flat") else [axes]

        for prompt_idx, prompt_tokens in enumerate(prompts):
            _plot_tokens(
                axes[prompt_idx],
                prompt_tokens,
                f"Context {prompt_idx + 1}",
                cfg.data.coordinate_mode,
            )

        _plot_tokens(
            axes[len(prompts)],
            sample_tokens,
            "Sample",
            cfg.data.coordinate_mode,
        )

        for ax in axes[total_plots:]:
            ax.axis("off")

        fig.tight_layout()
        images.append(wandb.Image(fig, caption=f"step {step + 1} sample {idx}"))
        plt.close(fig)

    if images:
        wandb.log({f"samples/sketches/{split}": images}, step=step + 1)

    if prev_mode:
        policy.train()


def _state_dict_or_none(obj: Any) -> Any:
    if obj is None:
        return None
    return obj.state_dict()


def _load_state_dict_if_present(obj: Any, state_dict: Any) -> None:
    if obj is None or state_dict is None:
        return
    obj.load_state_dict(state_dict)


def _save_checkpoint(
    ckpt_path: Path,
    *,
    epoch: int,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    cfg: ConfigDict,
) -> None:
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "step": int(step),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": _state_dict_or_none(scheduler),
            "scaler": _state_dict_or_none(scaler),
            "config": cfg.to_dict(),
        },
        ckpt_path,
    )


def _load_checkpoint(
    ckpt_path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    device: torch.device,
) -> tuple[int, int]:
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    _load_state_dict_if_present(scheduler, checkpoint.get("scheduler"))
    _load_state_dict_if_present(scaler, checkpoint.get("scaler"))

    start_epoch = int(checkpoint.get("epoch", 0))
    global_step = int(checkpoint.get("step", 0))
    return start_epoch, global_step


_CONFIG_FILE = config_flags.DEFINE_config_file(
    "config",
    default="configs/diffusion/pretrain_encoder_decoder.py",
)


def main(_) -> None:
    cfg = load_config(_CONFIG_FILE)
    set_seed(cfg.run.seed)
    device = torch.device(cfg.run.device if torch.cuda.is_available() else "cpu")

    dataset = QuickDrawEpisodes(
        root=cfg.data.root,
        split=cfg.data.split,
        K=cfg.data.K,
        backend=cfg.data.backend,
        max_seq_len=cfg.data.max_seq_len,
        seed=cfg.run.seed,
        coordinate_mode=cfg.data.coordinate_mode,
        builder_cls=EpisodeBuilderSimilar,
        index_dir=cfg.data.index_dir,
        ids_dir=cfg.data.ids_dir,
        families_cache_path=cfg.data.families_cache_path,
    )
    collator = ContextQueryInContextDiffusionCollator(
        horizon=cfg.model.horizon, seed=cfg.run.seed
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.loader.batch_size,
        shuffle=True,
        num_workers=cfg.loader.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collator,
    )

    eval_dataset = QuickDrawEpisodes(
        root=cfg.data.root,
        split="val" if not cfg.eval.eval_on_train else cfg.data.split,
        K=cfg.data.K,
        backend=cfg.data.backend,
        max_seq_len=cfg.data.max_seq_len,
        seed=cfg.run.seed,
        coordinate_mode=cfg.data.coordinate_mode,
        builder_cls=EpisodeBuilderSimilar,
        index_dir=cfg.data.index_dir,
        ids_dir=cfg.data.ids_dir,
        families_cache_path=cfg.data.families_cache_path,
    )
    eval_collator = ContextQueryInContextDiffusionCollator(
        horizon=cfg.model.horizon, seed=cfg.run.seed
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=cfg.eval.samples,
        shuffle=True,
        collate_fn=eval_collator,
    )

    noise_scheduler_kwargs = {
        "num_train_timesteps": cfg.model.num_train_timesteps,
        "beta_start": cfg.model.beta_start,
        "beta_end": cfg.model.beta_end,
        "beta_schedule": cfg.model.beta_schedule,
    }

    policy_cfg = DiTEncDecDiffusionPolicyConfig(
        horizon=cfg.model.horizon,
        point_feature_dim=cfg.model.input_dim,
        action_dim=cfg.model.output_dim,
        hidden_dim=cfg.model.hidden_dim,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        mlp_dim=cfg.model.mlp_dim,
        dropout=cfg.model.dropout,
        attention_dropout=cfg.model.attention_dropout,
        num_inference_steps=cfg.eval.num_inference_steps,
        noise_scheduler_kwargs=noise_scheduler_kwargs,
    )
    policy = DiTEncDecDiffusionPolicy(policy_cfg).to(device)
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )

    if cfg.training.warmup_cosine_annealing.use:
        scheduler_cfg = cfg.training.warmup_cosine_annealing

        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_steps=scheduler_cfg.warmup_steps,
            total_steps=scheduler_cfg.T_max,
            max_lr=scheduler_cfg.max_lr,
            min_lr=scheduler_cfg.min_lr,
        )

    elif cfg.training.cosine_annealing.use:
        scheduler_cfg = cfg.training.cosine_annealing

        scheduler = CosineAnnealingLR(
            optimizer, T_max=scheduler_cfg.T_max, eta_min=scheduler_cfg.eta_min
        )

    else:
        scheduler = None

    scaler = None

    base_save_dir = Path(cfg.checkpoint.dir)
    latest_ckpt_path = base_save_dir / cfg.checkpoint.latest_filename

    if cfg.wandb.use and cfg.wandb.project:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            config={
                **vars(cfg),
                "model": policy_cfg,
            },
        )
        wandb.run.name = wandb.run.id
        total_params = sum(p.numel() for p in policy.parameters())
        print(f"Model parameter count: {total_params:,}")
        wandb.log({"model/parameters": total_params}, step=0)
    else:
        total_params = sum(p.numel() for p in policy.parameters())
        print(f"Model parameter count: {total_params:,}")

    if cfg.wandb.use and wandb.run is not None:
        save_dir = base_save_dir / wandb.run.id
    else:
        save_dir = base_save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    resume_path = None
    if cfg.checkpoint.resume_from:
        resume_path = Path(cfg.checkpoint.resume_from)
    elif cfg.checkpoint.auto_resume and latest_ckpt_path.exists():
        resume_path = latest_ckpt_path

    start_epoch = 0
    global_step = 0
    if resume_path is not None:
        start_epoch, global_step = _load_checkpoint(
            resume_path,
            model=policy,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
        )
        print(
            f"Resumed from checkpoint {resume_path} "
            f"(epoch={start_epoch}, step={global_step})"
        )

    eval_iterator = iter(eval_dataloader)

    for epoch in range(start_epoch, cfg.training.epochs):
        policy.train()
        progress = tqdm(
            dataloader, desc=f"Epoch {epoch+1}/{cfg.training.epochs}", leave=False
        )
        for batch in progress:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = policy.compute_loss(batch)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            global_step += 1
            progress.set_postfix({"mse": metrics["mse"]})

            if (
                cfg.wandb.use
                and cfg.logging.loss_log_every > 0
                and global_step % cfg.logging.loss_log_every == 0
            ):
                wandb.log(
                    {
                        "train/loss": metrics["mse"],
                        "train/lr": optimizer.param_groups[0]["lr"],
                    },
                    step=global_step,
                )

            if (
                cfg.checkpoint.save_latest_every_steps is not None
                and cfg.checkpoint.save_latest_every_steps > 0
                and global_step % cfg.checkpoint.save_latest_every_steps == 0
            ):
                _save_checkpoint(
                    latest_ckpt_path,
                    epoch=epoch,
                    step=global_step,
                    model=policy,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    cfg=cfg,
                )
            
        try:
            eval_batch = next(eval_iterator)
        except StopIteration:
            eval_iterator = iter(eval_dataloader)
            eval_batch = next(eval_iterator)

        _log_qualitative_samples(
            policy=policy,
            context=eval_batch,
            cfg=cfg,
            step=global_step,
            device=device,
            split="eval",
        )

        _log_qualitative_samples(
            policy=policy,
            context={key: batch[key][: cfg.eval.samples] for key in batch.keys()},
            cfg=cfg,
            step=global_step,
            device=device,
            split="train",
        )

        _save_checkpoint(
            latest_ckpt_path,
            epoch=epoch + 1,
            step=global_step,
            model=policy,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            cfg=cfg,
        )

        if (
            cfg.checkpoint.save_interval is not None
            and (epoch + 1) % max(1, cfg.checkpoint.save_interval) == 0
        ):
            checkpoint_path = save_dir / f"policy_epoch_{epoch+1:03d}.pt"
            _save_checkpoint(
                checkpoint_path,
                epoch=epoch + 1,
                step=global_step,
                model=policy,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                cfg=cfg,
            )

    if cfg.wandb.use:
        wandb.finish()


if __name__ == "__main__":
    from absl import app

    app.run(main)

"""Utilities for sampling and qualitative logging of encoder–decoder diffusion policies."""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import torch

# --------------------------------------------------------------------------------------
# Collator for quick evaluation (context-only)
# --------------------------------------------------------------------------------------


class InContextDiffusionCollatorEval:
    """
    Minimal eval collator that extracts only the context portion of a token stream.

    Expects each sample dict to contain `tokens` shaped (T, F) with stop flag in channel 5
    and separators in channel 4. Everything before the first stop is treated as context.
    """

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        points_batch: List[torch.Tensor] = []
        points_lengths: List[int] = []

        for sample in batch:
            tokens = sample["tokens"]
            reset_idx = int((tokens[:, 5] == 1.0).nonzero(as_tuple=True)[0][0])
            context = tokens[:reset_idx].clone()
            points_batch.append(context)
            points_lengths.append(context.shape[0])

        max_len = max(points_lengths)
        batch_size = len(points_batch)
        feature_dim = points_batch[0].shape[-1]

        points = torch.zeros(batch_size, max_len, feature_dim, dtype=torch.float32)
        mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

        for idx, (pts, points_len) in enumerate(zip(points_batch, points_lengths)):
            points[idx, -points_len:] = pts
            mask[idx, -points_len:] = True

        return {"points": points, "mask": mask}


# --------------------------------------------------------------------------------------
# Sampling helpers
# --------------------------------------------------------------------------------------


@torch.no_grad()
def sample_quickdraw_tokens_encoder_decoder(
    policy: torch.nn.Module,
    max_tokens: int,
    demos: Dict[str, torch.Tensor],
    *,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Autoregressively sample `max_tokens` conditioned on full context (no observed query)."""

    device = next(policy.parameters()).device
    feature_dim = policy.cfg.point_feature_dim

    if demos["context"].shape[-1] != feature_dim:
        raise ValueError(
            f"start_token feature dim {demos['context'].shape[-1]} != {feature_dim}."
        )

    demos = {key: v.to(device) for key, v in demos.items()}

    horizon = policy.cfg.horizon
    max_chunks = math.ceil(max_tokens / horizon)
    samples: List[torch.Tensor] = []

    context = demos["context"]
    context_mask = demos["context_mask"]
    # start token with sep bit set (channel 4)
    start_token = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0, 0.0]], device=device)
    history = start_token.repeat(context.shape[0], 1, 1)

    history_mask = torch.ones(history.shape[:2], device=device, dtype=torch.bool)
    history_mask = torch.cat(
        [history_mask, torch.ones(history.shape[0], horizon, device=device, dtype=torch.bool)],
        dim=1,
    )

    for _ in range(max_chunks):
        actions = policy.sample_actions(
            context=context,
            context_mask=context_mask,
            history=history,
            history_mask=history_mask,
            generator=generator,
        )
        samples.append(actions)

        history = torch.cat([history, actions], dim=1)
        history_mask = torch.cat(
            [history_mask, torch.ones(actions.shape[:2], device=device, dtype=torch.bool)],
            dim=1,
        )

    generated = torch.cat(samples, dim=1)
    sketches = clean_sketches_encoder_decoder(generated)
    return sketches


@torch.no_grad()
def sample_quickdraw_tokens_encoder_decoder_from_partial(
    policy: torch.nn.Module,
    max_tokens: int,
    demos: Dict[str, torch.Tensor],
    *,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Sample conditioned on context plus a provided history prefix (from MAML query batches).
    Expects keys: context, context_mask, history, query_mask.
    """

    device = next(policy.parameters()).device
    feature_dim = policy.cfg.point_feature_dim

    if demos["context"].shape[-1] != feature_dim:
        raise ValueError(
            f"context feature dim {demos['context'].shape[-1]} != {feature_dim}."
        )

    demos = {k: v.to(device) for k, v in demos.items()}

    horizon = policy.cfg.horizon
    max_chunks = math.ceil(max_tokens / horizon)
    samples: List[torch.Tensor] = []

    context = demos["context"]
    context_mask = demos["context_mask"]
    history = demos["history"]
    history_mask = demos["query_mask"]  # same polarity as training (True = keep)

    for _ in range(max_chunks):
        actions = policy.sample_actions(
            context=context,
            context_mask=context_mask,
            history=history,
            history_mask=history_mask,
            generator=generator,
        )
        samples.append(actions)

        history = torch.cat([history, actions], dim=1)
        history_mask = torch.cat(
            [history_mask, torch.ones(actions.shape[:2], device=device, dtype=torch.bool)],
            dim=1,
        )

    generated = torch.cat(samples, dim=1)
    sketches = clean_sketches_encoder_decoder(generated)
    return sketches


@torch.no_grad()
def sample_quickdraw_tokens_decoder_only(
    policy: torch.nn.Module,
    max_tokens: int,
    demos: Dict[str, torch.Tensor],
    *,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sampling loop for decoder-only policies (kept for compatibility)."""

    device = next(policy.parameters()).device
    feature_dim = policy.cfg.point_feature_dim

    if demos["context"].shape[-1] != feature_dim:
        raise ValueError(
            f"context feature dim {demos['context'].shape[-1]} != {feature_dim}."
        )

    demos = {key: v.to(device) for key, v in demos.items()}

    horizon = policy.cfg.horizon
    max_chunks = math.ceil(max_tokens / horizon)
    samples: List[torch.Tensor] = []

    context = demos["context"]
    mask = demos["mask"]

    for _ in range(max_chunks):
        actions = policy.sample_actions(
            context=context,
            mask=mask,
            generator=generator,
        )
        samples.append(actions)

        context = torch.cat([context, actions], dim=1)
        mask = torch.cat(
            [mask, torch.ones(actions.shape[:2], device=device, dtype=torch.bool)],
            dim=1,
        )

    generated = torch.cat(samples, dim=1)
    sketches = clean_sketches_decoder_only(generated)
    return sketches


def clean_sketches_decoder_only(generated: torch.Tensor) -> List[torch.Tensor]:
    sketches: List[torch.Tensor] = []
    for sketch in generated:
        end_idx = (sketch[:, 5] >= 0.5).nonzero(as_tuple=True)[0]
        end_idx = end_idx[0] if end_idx.numel() > 0 else sketch.shape[0]
        sketches.append(sketch[:end_idx, :3])
    return sketches


def clean_sketches_encoder_decoder(generated: torch.Tensor) -> List[torch.Tensor]:
    sketches: List[torch.Tensor] = []
    for sketch in generated:
        end_idx = (sketch[:, 5] >= 0.5).nonzero(as_tuple=True)[0]
        end_idx = end_idx[0] if end_idx.numel() > 0 else sketch.shape[0]
        sketches.append(sketch[:end_idx, :3])
    return sketches


# --------------------------------------------------------------------------------------
# Logging helpers (W&B)
# --------------------------------------------------------------------------------------


def _split_context_prompts(ctx_tokens: torch.Tensor, coordinate_mode: str, K: int) -> List[torch.Tensor]:
    """
    Split concatenated context tokens using separator flag (channel 4) until stop (channel 5).
    Returns up to K sketches with (N,3) (x,y,pen) slices.
    """
    sketches: List[torch.Tensor] = []
    current: List[torch.Tensor] = []
    for token in ctx_tokens:
        if token[5] > 0.5:  # stop
            break
        if token[4] > 0.5:  # separator
            if current:
                sketches.append(torch.stack(current))
                current = []
            continue
        current.append(token[[0, 1, 2]])
    if current:
        sketches.append(torch.stack(current))
    return sketches[:K]


def _plot_tokens(ax, tokens: torch.Tensor, title: str, coordinate_mode: str) -> None:
    """Render `(N, 3)` tokens on the provided axis."""
    array = tokens.detach().cpu().numpy()
    coords = array[:, :2].cumsum(axis=0) if coordinate_mode == "delta" else array[:, :2]
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


def log_qualitative_samples(
    policy: torch.nn.Module,
    context: Dict[str, torch.Tensor],
    *,
    split: str,
    cfg,
    step: int,
    device: torch.device,
    use_partial_history: bool = False,
) -> None:
    """
    Sample sketches and push them to WandB for visual inspection.
    Works for both full-context (no observed query) and partial-history (MAML) cases.
    """
    if (not cfg.wandb.use) or cfg.wandb.project is None or cfg.eval.samples <= 0:
        return

    # lazy imports to keep training light if wandb/mpl not installed
    import wandb  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore

    prev_mode = policy.training
    policy.eval()

    generator = torch.Generator(device=device)
    generator.manual_seed(cfg.eval.seed + step)

    sampler = sample_quickdraw_tokens_encoder_decoder_from_partial if use_partial_history else sample_quickdraw_tokens_encoder_decoder

    samples = sampler(
        policy=policy,
        max_tokens=cfg.data.max_query_len,
        demos=context,
        generator=generator,
    )

    images = []
    batch_size = len(samples)
    for idx in range(batch_size):
        ctx_tokens = context["context"][idx]
        ctx_mask = context["context_mask"][idx]
        valid_ctx = ctx_tokens[ctx_mask].detach().cpu()
        prompts = _split_context_prompts(valid_ctx, cfg.data.coordinate_mode, cfg.data.K)
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


__all__ = [
    "InContextDiffusionCollatorEval",
    "sample_quickdraw_tokens_encoder_decoder",
    "sample_quickdraw_tokens_encoder_decoder_from_partial",
    "sample_quickdraw_tokens_decoder_only",
    "clean_sketches_decoder_only",
    "clean_sketches_encoder_decoder",
    "log_qualitative_samples",
]

#!/usr/bin/env python3
"""
maml_train.py

Second-order MAML training loop skeleton for an encoder-decoder diffusion policy (DiT),
using torch.func.functional_call and math attention (SDPA) for grad-of-grad.

Assumptions:
- You already implemented:
  - model.loss_only(batch) -> torch.Tensor scalar loss
  - get_fast_param_names(model, last_frac=0.25, include_ada=True) -> list[str]
  - dataset yields a "task" per sample: K context episodes + 1 query episode
- You will implement:
  - build_query_batch(task, horizon, rng) -> batch dict for model.loss_only
  - build_support_batch_loo(task, holdout_idx, horizon, rng) -> batch dict (LOO) for model.loss_only

The training loop below shows:
- outer batch of tasks
- for each task: inner adaptation on LOO support (1-2 steps) updating fast params only
- outer loss on query using adapted fast params
- second-order gradients (create_graph=True)
- math attention forced during maml step
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple, Optional
from pathlib import Path
from tqdm import tqdm

from absl import app
from ml_collections import ConfigDict, config_flags
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.func import functional_call
import wandb

from dataset import QuickDrawEpisodesMAML, MAMLDiffusionCollator
from diffusion.policies import MAMLDiTEncDecDiffusionPolicy, DiTEncDecDiffusionPolicyConfig
from diffusion.sampling import log_qualitative_samples
from diffusion.utils import ProfilerGuard


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_fast_param_names(
    model: nn.Module,
    last_frac: float = 0.25,
    include_ada: bool = True,
    include_final_norm: bool = True,
) -> List[str]:
    """
    Return names of *parameters* to be updated in the MAML inner loop (fast weights).

    Policy (matches what we discussed + the TTT-E2E paper spirit):
      - Only touch decoder-side "MLP-like" components, not attention.
      - Use only the last `last_frac` decoder blocks (>= 1 block).
      - Always include FFN/MLP weights in those blocks: `decoder_transformer.blocks[i].mlp.*`
      - Optionally include AdaLN modulation MLP in those blocks: `decoder_transformer.blocks[i].ada_ln.*`
      - Optionally include the decoder final AdaLNZero MLP: `decoder_transformer.final_norm.mlp.*`

    Notes:
      - This function returns *names* that must exist in dict(model.named_parameters()).
      - It intentionally excludes any attention params:
          `self_attn.*`, `cross_attn.*`, and all encoder params.
      - It does NOT return buffers.

    Args:
      model: Your DiTEncDecDiffusionPolicy instance.
      last_frac: Fraction of decoder blocks (from the end) to adapt. E.g. 0.25 => last quarter.
      include_ada: Include `ada_ln` parameters in selected blocks.
      include_final_norm: Include `decoder_transformer.final_norm.mlp` parameters.

    Returns:
      Sorted list of parameter names to treat as fast parameters.
    """
    # Basic validation
    if not hasattr(model, "decoder_transformer"):
        raise AttributeError("Model has no attribute 'decoder_transformer'.")
    dec = getattr(model, "decoder_transformer")

    if not hasattr(dec, "blocks"):
        raise AttributeError("decoder_transformer has no attribute 'blocks'.")

    n_blocks = len(dec.blocks)
    if n_blocks <= 0:
        raise ValueError("decoder_transformer.blocks is empty.")

    # Compute how many blocks to include (at least 1)
    if last_frac <= 0:
        L = 1
    else:
        L = max(1, int(round(n_blocks * last_frac)))
        L = min(L, n_blocks)

    start_idx = n_blocks - L
    fast_prefixes: List[str] = []

    # Decoder block FFNs (and optionally ada_ln) in last L blocks
    for i in range(start_idx, n_blocks):
        fast_prefixes.append(f"decoder_transformer.blocks.{i}.mlp.")
        if include_ada:
            fast_prefixes.append(f"decoder_transformer.blocks.{i}.ada_ln.")

    # Decoder final norm modulation MLP (AdaLNZero)
    if include_final_norm and hasattr(dec, "final_norm") and hasattr(dec.final_norm, "mlp"):
        fast_prefixes.append("decoder_transformer.final_norm.mlp.")

    # Filter actual named_parameters by prefix match
    all_param_names = [name for name, _ in model.named_parameters()]
    fast_names = [name for name in all_param_names if any(name.startswith(p) for p in fast_prefixes)]

    # Sanity checks: ensure we didn't accidentally include attention weights
    forbidden_substrings = (
        ".self_attn.",
        ".cross_attn.",
        ".attn_norm.",
        ".cross_norm.",
        ".mlp_norm.",
        "encoder_transformer.",
    )
    bad = [n for n in fast_names if any(s in n for s in forbidden_substrings)]
    if bad:
        raise RuntimeError(
            "get_fast_param_names selected forbidden params (attention/norm/encoder). "
            f"Examples: {bad[:5]}"
        )

    # Another sanity check: ensure names exist and are parameters
    param_dict = dict(model.named_parameters())
    missing = [n for n in fast_names if n not in param_dict]
    if missing:
        raise RuntimeError(f"Some fast param names are not in model.named_parameters(): {missing[:5]}")

    # Optional: ensure non-empty
    if not fast_names:
        raise RuntimeError(
            "No fast parameters were selected. "
            "Check that your decoder block attributes match 'mlp' and 'ada_ln' naming."
        )

    # Return deterministic ordering
    return sorted(fast_names)


def _special_token(
    sep: float = 0.0,
    stop: float = 0.0,
) -> np.ndarray:
    
    token = np.zeros(6)
    token[4] = sep
    token[5] = stop
    return token


def _pad_actions(actions: torch.Tensor, horizon: int) -> torch.Tensor:
    """Pads actions that are shorter than the horizon with end-tokens."""
    pad_len = horizon - actions.shape[0]
    padding = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]).tile((pad_len, 1))

    return torch.cat([actions, padding])


def _prepare_loo_episode(heldout: np.ndarray, kept: np.ndarray, horizon: int, rng: np.random.Generator, device: torch.device):
    
    segments: List[np.ndarray] = [_special_token(sep=1.0)]  # Dummy start
    for sketch in kept:
        segments.append(sketch)
        segments.append(_special_token(sep=1.0))
    context_tokens = np.vstack(segments).astype(dtype=np.float32, copy=False)
    context_tokens = torch.from_numpy(context_tokens)
    
    segments: List[np.ndarray] = [_special_token(sep=1.0)]  # Dummy start
    segments.append(heldout)
    segments.append(_special_token(stop=1.0))
    query_tokens = np.vstack(segments).astype(dtype=np.float32, copy=False)
    query_tokens = torch.from_numpy(query_tokens)
    
    max_start = max(2, query_tokens.shape[0] - 1)          # exclude stop
    max_start = max(2, min(max_start, query_tokens.shape[0] - horizon))  # leave room

    start_idx = torch.randint(
        low=1,
        high=max_start,
        size=(1,),
        generator=rng,
        device=device,
        ).item()

    points = query_tokens[1 : start_idx + 1].clone()
    actions = query_tokens[start_idx + 1 : start_idx + 1 + horizon].clone()
    
    context_mask = torch.ones(context_tokens.shape[0], dtype=torch.bool, device=device)
    query_mask = torch.ones(points.shape[0] + horizon, dtype=torch.bool, device=device)
    

    if actions.shape[0] < horizon:
        actions = _pad_actions(actions, horizon=horizon)
        
        
    # Add batch dim
    points = points.unsqueeze(0)
    actions = actions.unsqueeze(0)
    context_tokens = context_tokens.unsqueeze(0)
    query_mask = query_mask.unsqueeze(0)
    context_mask = context_mask.unsqueeze(0)

    return {
        "history": points.to(device=device, dtype=torch.float32),
        "actions": actions.to(device=device, dtype=torch.float32),
        "context": context_tokens.to(device=device, dtype=torch.float32),
        "query_mask": query_mask,
        "context_mask": context_mask,
    }


def build_support_batch_loo(
    task: Dict[str, Any],
    holdout_idx: int,
    *,
    horizon: int,
    device: torch.device,
    noise: torch.Tensor | None = None,
    timesteps: torch.Tensor | None = None,
    # --- optional knobs ---
    rng: Optional[torch.Generator] = None,
    add_sep_between_context_episodes: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Build a *single* support batch dict for one LOO held-out context episode.

    Expected task format:
      task["context_episodes"]: List[Tensor[T_i, F]]   length K
      task["query_episode"]:   Tensor[T_q, F]          (unused here)

    Returns a dict compatible with DiTEncDecDiffusionPolicy.compute_loss(), with batch size 1.

    Notes on shapes (matching your previous collator conventions):
      - history is allocated with length = (points_len + horizon) and we right-align points
        leaving an extra prefix of length horizon as zeros.
      - query_mask has length = history_len + horizon and marks the last (points_len + horizon)
        positions as True (same as your old collator).
      - context is the concatenation of all context episodes except the held-out one
        (optionally inserting a SEP token between episodes).
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    context_eps: List[np.ndarray] = task["context_episodes"]
    if not (0 <= holdout_idx < len(context_eps)):
        raise IndexError(f"holdout_idx={holdout_idx} out of range for K={len(context_eps)}")

    kept: List[np.ndarray] = []
    for k, ep in enumerate(context_eps):
        if k == holdout_idx:
            continue
        if ep.ndim != 2:
            raise ValueError(f"context_episodes[{k}] must have shape (T,F), got {tuple(ep.shape)}")
        kept.append(ep)

    if len(kept) == 0:
        raise ValueError("LOO resulted in empty context (K=1). Need at least 2 context episodes.")

    feature_dim = kept[0].shape[-1]
    for ep in kept:
        if ep.shape[-1] != feature_dim:
            raise ValueError("All context episodes must share the same feature_dim.")

    heldout = context_eps[holdout_idx]
    if heldout.ndim != 2:
        raise ValueError(f"held-out episode must have shape (T,F), got {tuple(heldout.shape)}")
    if heldout.shape[-1] != feature_dim:
        raise ValueError("held-out episode feature_dim != context feature_dim")

    T = heldout.shape[0]
    if T < 1:
        raise ValueError("held-out episode is empty.")

    if rng is None:
        rng = torch.Generator(device=device)
        rng.manual_seed(torch.seed())  # non-deterministic by default

    batch_out = _prepare_loo_episode(heldout=heldout, kept=kept, horizon=horizon, rng=rng, device=device)
    
    if noise is not None:
        batch_out["noise"] = noise
    if timesteps is not None:
        batch_out["timesteps"] = timesteps

    return batch_out


def _pad_actions_to_horizon(
    actions: torch.Tensor, horizon: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pad (T,F) -> (horizon,F). Also returns action_valid_mask (horizon,) True for real tokens.

    Padding convention: last channel is "stop/end" -> set to 1.0 on padded rows.
    (Matches your previous collator.)
    """
    if actions.ndim != 2:
        raise ValueError(f"actions must be (T,F), got {tuple(actions.shape)}")
    T, F = actions.shape
    if T >= horizon:
        return actions[:horizon], torch.ones((horizon,), dtype=torch.bool, device=actions.device)

    pad_len = horizon - T
    pad = torch.zeros((pad_len, F), dtype=actions.dtype, device=actions.device)
    pad[:, -1] = 1.0  # stop/end flag
    out = torch.cat([actions, pad], dim=0)

    valid = torch.zeros((horizon,), dtype=torch.bool, device=actions.device)
    valid[:T] = True
    return out, valid


def _concat_context_episodes(
    context_episodes: List[np.ndarray],
    *,
    device: torch.device,
    add_sep_between_episodes: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Concatenate episodes into a single context sequence (batch size 1):
      context:      (1, C_len, F)
      context_mask: (1, C_len) bool (all True, no padding here)

    If add_sep_between_episodes: inserts a 1-token SEP between episodes.
    SEP convention: feature index 4 is 'sep' if F >= 5.
    """
    if len(context_episodes) == 0:
        raise ValueError("context_episodes is empty.")
    if any(ep.ndim != 2 for ep in context_episodes):
        raise ValueError("Each context episode must have shape (T,F).")

    F = context_episodes[0].shape[-1]
    for j, ep in enumerate(context_episodes):
        if ep.shape[-1] != F:
            raise ValueError(f"Episode {j} has feature_dim {ep.shape[-1]} != {F}")

    eps = [torch.from_numpy(ep).to(device=device, dtype=torch.float32) for ep in context_episodes]

    if add_sep_between_episodes:
        sep = torch.zeros((1, F), dtype=torch.float32, device=device)
        if F >= 5:
            sep[0, 4] = 1.0
        pieces: List[torch.Tensor] = []
        for j, ep in enumerate(eps):
            pieces.append(ep)
            if j != len(eps) - 1:
                pieces.append(sep)
        ctx_1d = torch.cat(pieces, dim=0)
    else:
        ctx_1d = torch.cat(eps, dim=0)

    context = ctx_1d.unsqueeze(0)  # (1, C_len, F)
    context_mask = torch.ones((1, ctx_1d.shape[0]), dtype=torch.bool, device=device)
    return context, context_mask


def build_query_batch(
    task: Dict[str, List[np.ndarray]],
    *,
    horizon: int,
    device: torch.device,
    noise: torch.Tensor | None = None,
    timesteps: torch.Tensor | None = None,
    # Optional knobs (safe defaults)
    rng: Optional[torch.Generator] = None,
    add_sep_between_context_episodes: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Build the query batch dict (condition on all K context episodes, target is query episode).
    Same shape contract as build_support_batch_loo.

    Expected task format:
      task["context_episodes"]: List[Tensor[T_i, F]]  length K
      task["query_episode"]:   Tensor[T_q, F]

    Output dict (batch size 1):
      {
        "context":      (1, C_len, F),
        "context_mask": (1, C_len) bool,
        "history":      (1, H, F),           # H = observed history length (no extra prefix)
        "actions":      (1, horizon, F),     # padded to horizon
        "query_mask":   (1, H + horizon) bool (history valid + action_valid),
        optional: "noise": (1, horizon, F), "timesteps": (1,)
      }

    Notes:
    - This chooses a random split point start_idx in the query episode:
        history = query[:start_idx+1]
        actions = query[start_idx+1 : start_idx+1+horizon] (padded)
    - If you prefer a deterministic split (e.g., always last horizon tokens),
      change the start_idx logic.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive.")

    context_episodes: List[np.ndarray] = task["context_episodes"]
    query_ep: np.ndarray = task["query_episode"]

    # 1) Build concatenated context sequence
    context, context_mask = _concat_context_episodes(
        context_episodes,
        device=device,
        add_sep_between_episodes=add_sep_between_context_episodes,
    )

    # 2) Split query episode into history + actions
    if query_ep.ndim != 2:
        raise ValueError(f"query_episode must be (T,F), got {tuple(query_ep.shape)}")

    query_ep = torch.from_numpy(query_ep).to(device=device, dtype=torch.float32)

    Tq, F = query_ep.shape
    if Tq < 1:
        raise ValueError("query_episode is empty.")

    if rng is None:
        rng = torch.Generator(device=device)
        rng.manual_seed(torch.seed())

    # choose start_idx in [0, Tq-1] => history has at least 1 token
    start_idx = int(torch.randint(0, Tq, (1,), generator=rng, device=device).item())

    history_1d = query_ep[: start_idx + 1].clone()  # (H, F)
    actions_1d = query_ep[start_idx + 1 : start_idx + 1 + horizon].clone()  # (<=horizon, F)

    actions_pad, action_valid = _pad_actions_to_horizon(actions_1d, horizon)

    history = history_1d.unsqueeze(0)      # (1, H, F)
    actions = actions_pad.unsqueeze(0)     # (1, horizon, F)

    H = history.shape[1]
    query_mask = torch.ones((1, H + horizon), dtype=torch.bool, device=device)
    
    # query_mask = torch.zeros((1, H + horizon), dtype=torch.bool, device=device)
    # query_mask[0, :H] = True
    # query_mask[0, H:] = action_valid  # mask out padded tail if any

    out: Dict[str, torch.Tensor] = {
        "context": context,
        "context_mask": context_mask,
        "history": history,
        "actions": actions,
        "query_mask": query_mask,
    }

    # 3) Optional diffusion variance control
    if noise is not None:
        if noise.ndim == 2:
            noise = noise.unsqueeze(0)
        if noise.shape != actions.shape:
            raise ValueError(f"noise shape {tuple(noise.shape)} must match actions shape {tuple(actions.shape)}")
        out["noise"] = noise.to(device=device, dtype=torch.float32)

    if timesteps is not None:
        if timesteps.ndim == 0:
            timesteps = timesteps.view(1)
        if timesteps.shape != (1,):
            raise ValueError(f"timesteps must have shape (1,) or scalar, got {tuple(timesteps.shape)}")
        out["timesteps"] = timesteps.to(device=device, dtype=torch.long)

    return out


# -------------------------
# MAML core
# -------------------------


@dataclass
class MAMLConfig:
    inner_steps: int = 1
    inner_lr: float = 1e-4
    outer_lr: float = 1e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    last_frac_fast: float = 0.25
    include_ada_fast: bool = True
    num_loo_per_task: int = 2        # how many held-out context eps per inner step (subsample if K large)
    reuse_diffusion_noise: bool = True  # reuse noise/timesteps across inner steps + query (lower variance)
    use_math_attention: bool = True
    device: str = "cuda"


def _clip_grads_in_list(grads: List[torch.Tensor], max_norm: float) -> List[torch.Tensor]:
    if max_norm <= 0:
        return grads
    # Compute global norm
    norms = [g.norm(2) for g in grads if g is not None]
    if not norms:
        return grads
    total_norm = torch.norm(torch.stack(norms), 2)
    if total_norm <= max_norm:
        return grads
    scale = (max_norm / (total_norm + 1e-6))
    return [g * scale if g is not None else None for g in grads]


def maml_task_loss_second_order(
    model: nn.Module,
    task: Dict[str, Any],
    *,
    fast_names: List[str],
    cfg: MAMLConfig,
    horizon: int,
) -> torch.Tensor:
    """
    Compute meta-loss for one task:
      inner: adapt fast params on LOO support loss
      outer: query loss with adapted params
    Returns scalar loss (requires grad).
    """
    device = torch.device(cfg.device)

    # Grab params/buffers (stateless execution)
    params = {k: v for k, v in model.named_parameters()}
    buffers = {k: v for k, v in model.named_buffers()}

    # Shared diffusion randomness for stability (sample lazily from first built batch)
    shared_noise = None
    shared_timesteps = None

    # Inner loop: update only fast params
    adapted_params = params
    K = len(task["context_episodes"])

    # Choose which indices to hold out (subsample for speed)
    if cfg.num_loo_per_task >= K:
        loo_indices = list(range(K))
    else:
        perm = torch.randperm(K, device=device)
        loo_indices = perm[: cfg.num_loo_per_task].tolist()

    for _step in range(cfg.inner_steps):
        support_losses: List[torch.Tensor] = []

        for i in loo_indices:
            # Build batch; if shared_noise/shared_timesteps exist, pass them in.
            # Otherwise build without them, then lazily sample and inject.
            support_batch = build_support_batch_loo(
                task,
                holdout_idx=i,
                horizon=horizon,
                device=device,
                noise=shared_noise if cfg.reuse_diffusion_noise else None,
                timesteps=shared_timesteps if cfg.reuse_diffusion_noise else None,
            )

            if cfg.reuse_diffusion_noise and (shared_noise is None or shared_timesteps is None):
                # Lazily sample from correct shape/dtype/device
                shared_noise = torch.randn_like(support_batch["actions"])
                shared_timesteps = torch.randint(
                    0,
                    model.scheduler.config.num_train_timesteps,
                    (support_batch["actions"].shape[0],),  # batch size (likely 1)
                    device=device,
                    dtype=torch.long,
                )
                # Inject into already-built batch dict (no rebuild)
                support_batch["noise"] = shared_noise
                support_batch["timesteps"] = shared_timesteps

            loss_s = functional_call(model, (adapted_params, buffers), (support_batch,))
            support_losses.append(loss_s)

        support_loss = torch.stack(support_losses).mean()

        fast_tensors = [adapted_params[n] for n in fast_names]
        grads = torch.autograd.grad(
            support_loss,
            fast_tensors,
            create_graph=True,   # <-- full MAML (grad-of-grad)
            retain_graph=True,
            allow_unused=False,
        )
        grads = list(grads)
        grads = _clip_grads_in_list(grads, cfg.max_grad_norm)

        new_params = dict(adapted_params)
        for name, p, g in zip(fast_names, fast_tensors, grads):
            new_params[name] = p - cfg.inner_lr * g
        adapted_params = new_params

    # Outer: query loss evaluated with adapted fast params
    query_batch = build_query_batch(
        task,
        horizon=horizon,
        device=device,
        noise=shared_noise if cfg.reuse_diffusion_noise else None,
        timesteps=shared_timesteps if cfg.reuse_diffusion_noise else None,
    )

    # If query_batch builder didn't inject (e.g., you rely on passing), inject here too.
    if cfg.reuse_diffusion_noise and (shared_noise is not None) and ("noise" not in query_batch):
        query_batch["noise"] = shared_noise
    if cfg.reuse_diffusion_noise and (shared_timesteps is not None) and ("timesteps" not in query_batch):
        query_batch["timesteps"] = shared_timesteps

    query_loss = functional_call(model, (adapted_params, buffers), (query_batch,))
    return query_loss


def maml_step(
    model: nn.Module,
    tasks: List[Dict[str, Any]],
    *,
    fast_names: List[str],
    cfg: MAMLConfig,
    horizon: int,
) -> torch.Tensor:
    """
    Compute mean meta-loss over an outer batch of tasks (keeps computation graph).
    """
    losses: List[torch.Tensor] = []
    for task in tasks:
        loss = maml_task_loss_second_order(
            model, task, fast_names=fast_names, cfg=cfg, horizon=horizon
        )
        losses.append(loss)
    return torch.stack(losses).mean()


# -------------------------
# Config loading
# -------------------------

_CONFIG = config_flags.DEFINE_config_file(
    "config",
    default="configs/train_maml_icil.py",
)

def load_config(config_flag) -> ConfigDict:
    """Load the config pointed to by --config."""
    return config_flag.value


# -----------------------------------------------------------------------------------------------
# #################################### Main training script #####################################
# -----------------------------------------------------------------------------------------------

def main(argv: List[str] | None = None) -> None:
    del argv
    config = load_config(_CONFIG)
    cfg = MAMLConfig(
        inner_steps=config.maml.inner_steps,
        inner_lr=config.maml.inner_lr,
        outer_lr=config.maml.outer_lr,
        weight_decay=config.training.weight_decay,
        max_grad_norm=config.maml.max_grad_norm,
        last_frac_fast=config.maml.last_frac_fast,
        include_ada_fast=config.maml.include_ada_fast,
        num_loo_per_task=config.maml.num_loo_per_task,
        reuse_diffusion_noise=config.maml.reuse_diffusion_noise,
        use_math_attention=config.maml.math_attention,
        device=config.run.device,
    )

    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
    
    set_seed(config.run.seed)
    
    dataset = QuickDrawEpisodesMAML(
        root=config.data.root,
        split=config.data.split,
        K=config.data.K,
        max_seq_len=config.data.max_seq_len,
        max_query_len=config.data.max_query_len,
        max_context_len=config.data.max_context_len,
        backend=config.data.backend,
        coordinate_mode=config.data.coordinate_mode,
        index_dir=config.data.index_dir,
        ids_dir=config.data.ids_dir,
        seed=config.run.seed,
    )
    
    eval_dataset = QuickDrawEpisodesMAML(
        root=config.data.root,
        split="val",
        K=config.data.K,
        max_seq_len=config.data.max_seq_len,
        max_query_len=config.data.max_query_len,
        max_context_len=config.data.max_context_len,
        backend=config.data.backend,
        coordinate_mode=config.data.coordinate_mode,
        index_dir=config.data.index_dir,
        ids_dir=config.data.ids_dir,
        seed=config.run.seed + 1234,
    )
    
    noise_scheduler_kwargs = {
        "num_train_timesteps": config.model.num_train_timesteps,
        "beta_start": config.model.beta_start,
        "beta_end": config.model.beta_end,
        "beta_schedule": config.model.beta_schedule,
    }
    
    policy_cfg = DiTEncDecDiffusionPolicyConfig(
        horizon=config.model.horizon,
        point_feature_dim=config.model.input_dim,
        action_dim=config.model.output_dim,
        hidden_dim=config.model.hidden_dim,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        mlp_dim=config.model.mlp_dim,
        dropout=config.model.dropout,
        attention_dropout=config.model.attention_dropout,
        num_inference_steps=config.eval.num_inference_steps,
        noise_scheduler_kwargs=noise_scheduler_kwargs,
    )
    policy = MAMLDiTEncDecDiffusionPolicy(policy_cfg).to(device)

    policy = policy.to(device)
    policy.train()

    fast_names = get_fast_param_names(
        policy,
        last_frac=cfg.last_frac_fast,
        include_ada=cfg.include_ada_fast,
    )

    # Outer optimizer updates *all* model params (including fast params initializations)
    outer_opt = torch.optim.AdamW(policy.parameters(), lr=cfg.outer_lr, weight_decay=cfg.weight_decay)

    base_save_dir = Path(config.checkpoint.dir)
    if config.wandb.use and config.wandb.project:
        wandb.init(
            project=config.wandb.project,
            entity=getattr(config.wandb, "entity", None),
            config=config.to_dict(),
        )
        wandb.run.name = wandb.run.id
        save_dir = base_save_dir / wandb.run.id
    else:
        save_dir = base_save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    worker_seed = config.run.seed
    g = torch.Generator()
    g.manual_seed(worker_seed)

    def _worker_init_fn(worker_id):
        base = worker_seed + worker_id
        np.random.seed(base)
        random.seed(base)
        torch.manual_seed(base)


    loader = DataLoader(
        dataset,
        batch_size=config.loader.batch_size,
        shuffle=True,
        num_workers=config.loader.num_workers,
        collate_fn=MAMLDiffusionCollator(token_dim=6, coordinate_mode=config.data.coordinate_mode),
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        worker_init_fn=_worker_init_fn,
        generator=g,
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        collate_fn=MAMLDiffusionCollator(token_dim=6, coordinate_mode=config.data.coordinate_mode),
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    
    eval_iter = iter(eval_loader)

    pg = ProfilerGuard(
        use=config.profiling.use,
        start_step=0,
        end_step=3,
        trace_path=config.profiling.trace_dir + "trace.json"
    )

    global_step = 0
    for epoch in range(1, config.training.epochs + 1):
        
        for tasks in tqdm(loader):
            
            # Start profiler
            pg.start(global_step)
            
            global_step += 1
            outer_opt.zero_grad(set_to_none=True)

            # Force math attention for second-order meta-gradients (safer)
            if cfg.use_math_attention and device.type == "cuda":
                ctx = torch.backends.cuda.sdp_kernel(
                    enable_flash=False,
                    enable_mem_efficient=False,
                    enable_math=True,
                )
            else:
                # no-op context manager
                class _NullCtx:
                    def __enter__(self): return None
                    def __exit__(self, exc_type, exc, tb): return False
                ctx = _NullCtx()

            with ctx:
                meta_loss = maml_step(
                    policy,
                    tasks=tasks,
                    fast_names=fast_names,
                    cfg=cfg,
                    horizon=config.model.horizon,
                )

            meta_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            outer_opt.step()
            
            pg.step()
            
            # Stop profiler
            pg.stop(global_step)

            if global_step % config.logging.log_loss_every == 0:
                print(f"[epoch {epoch:03d} step {global_step:06d}] meta_loss={meta_loss.item():.6f}")
                if config.wandb.use and wandb.run is not None:
                    wandb.log({"train/meta_loss": meta_loss.item()}, step=global_step)

            if config.wandb.use and wandb.run is not None and (global_step % config.wandb.samples_log_interval == 0):
                with torch.no_grad():
                    try:
                        eval_tasks = next(eval_iter)
                    except StopIteration:
                        eval_iter = iter(eval_loader)
                        eval_tasks = next(eval_iter)
                    
                    qb = build_query_batch(
                        task=eval_tasks[0],
                        horizon=config.model.horizon,
                        device=device,
                    )
                    log_qualitative_samples(
                        policy,
                        qb,
                        split="train",
                        cfg=config,
                        step=global_step,
                        device=device,
                        use_partial_history=True,
                    )

        if epoch % config.checkpoint.save_interval == 0:
            ckpt_path = save_dir / f"maml_epoch_{epoch:03d}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model": policy.state_dict(),
                    "outer_opt": outer_opt.state_dict(),
                    "fast_names": fast_names,
                    "cfg": config.__dict__,
                    "maml_cfg": cfg.__dict__,
                },
                ckpt_path,
            )
            print(f"Saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    app.run(main)

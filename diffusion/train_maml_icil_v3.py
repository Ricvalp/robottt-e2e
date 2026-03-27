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
import copy
from dataclasses import asdict, dataclass
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
from diffusion.sampling import build_qualitative_sample_images
from diffusion.utils import ProfilerGuard


class TorchMAMLDiffusionCollator:
    """Wrap the dataset collator and convert task episodes to torch tensors once."""

    def __init__(self, token_dim: int, coordinate_mode: str = "delta") -> None:
        self.base_collator = MAMLDiffusionCollator(
            token_dim=token_dim,
            coordinate_mode=coordinate_mode,
        )

    def __call__(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        tasks = self.base_collator(batch)
        return [
            {
                "context_episodes": [
                    torch.from_numpy(episode) for episode in task["context_episodes"]
                ],
                "query_episode": torch.from_numpy(task["query_episode"]),
            }
            for task in tasks
        ]


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


def get_outer_param_names(
    model: nn.Module,
    *,
    train_encoder: bool = False,
    train_decoder: bool = True,
    train_input_projections: bool = True,
    train_output_head: bool = True,
    train_diffusion_conditioning: bool = True,
) -> List[str]:
    """Return names of parameters updated by the outer optimizer."""
    outer_prefixes: List[str] = []
    if train_encoder:
        outer_prefixes.append("encoder_transformer.")
    if train_decoder:
        outer_prefixes.append("decoder_transformer.")
    if train_input_projections:
        outer_prefixes.extend(
            [
                "point_feature_proj.",
                "history_feature_proj.",
                "action_encoder.",
            ]
        )
    if train_output_head:
        outer_prefixes.append("output_head.")
    if train_diffusion_conditioning:
        outer_prefixes.extend(
            [
                "diffusion_proj.",
                "world_time_embedder.",
                "diffusion_time_embedder.",
            ]
        )

    outer_names = [
        name
        for name, _ in model.named_parameters()
        if any(name.startswith(prefix) for prefix in outer_prefixes)
    ]
    if not outer_names:
        raise RuntimeError("No outer-trainable parameters were selected.")
    return sorted(outer_names)


def _set_outer_trainable_params(
    model: nn.Module,
    outer_names: List[str],
) -> None:
    outer_name_set = set(outer_names)
    for name, param in model.named_parameters():
        param.requires_grad_(name in outer_name_set)


def _count_params_by_name(model: nn.Module, names: List[str]) -> int:
    name_set = set(names)
    return sum(param.numel() for name, param in model.named_parameters() if name in name_set)


def _load_checkpoint_for_finetuning(
    ckpt_path: Path,
    *,
    device: torch.device,
) -> Dict[str, Any]:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint at {ckpt_path} is not a dict.")
    if "model" not in checkpoint:
        raise KeyError(f"Checkpoint at {ckpt_path} does not contain a 'model' state dict.")
    checkpoint_cfg = checkpoint.get("config")
    if not isinstance(checkpoint_cfg, dict):
        raise ValueError(
            f"Checkpoint at {ckpt_path} does not contain a valid 'config' dict. "
            "Expected a pretrain checkpoint produced by pretrain_encoder_decoder.py."
        )
    return checkpoint


def _resolve_policy_config(
    config: ConfigDict,
    *,
    pretrained_config: Optional[Dict[str, Any]] = None,
) -> tuple[DiTEncDecDiffusionPolicyConfig, Dict[str, Any]]:
    if pretrained_config is not None:
        model_cfg = pretrained_config.get("model")
        if not isinstance(model_cfg, dict):
            raise ValueError("Pretrained checkpoint config is missing the 'model' section.")
        eval_cfg = pretrained_config.get("eval", {})
        if not isinstance(eval_cfg, dict):
            eval_cfg = {}
        source = "checkpoint"
    else:
        model_cfg = config.model.to_dict()
        eval_cfg = config.eval.to_dict()
        source = "config"

    required_model_keys = (
        "horizon",
        "input_dim",
        "output_dim",
        "hidden_dim",
        "num_layers",
        "num_heads",
        "mlp_dim",
        "dropout",
        "attention_dropout",
        "prediction_type",
        "num_train_timesteps",
        "beta_start",
        "beta_end",
        "beta_schedule",
    )
    missing_keys = [key for key in required_model_keys if key not in model_cfg]
    if missing_keys:
        raise KeyError(
            f"Missing model fields in {source} policy config: {missing_keys}"
        )

    num_inference_steps = int(config.eval.num_inference_steps)
    if num_inference_steps <= 0:
        num_inference_steps = int(eval_cfg.get("num_inference_steps", 50))
    if num_inference_steps <= 0:
        num_inference_steps = 50

    resolved = {
        "source": source,
        "model": dict(model_cfg),
        "num_inference_steps": num_inference_steps,
    }
    policy_cfg = DiTEncDecDiffusionPolicyConfig(
        horizon=int(model_cfg["horizon"]),
        point_feature_dim=int(model_cfg["input_dim"]),
        action_dim=int(model_cfg["output_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        num_layers=int(model_cfg["num_layers"]),
        num_heads=int(model_cfg["num_heads"]),
        mlp_dim=int(model_cfg["mlp_dim"]),
        dropout=float(model_cfg["dropout"]),
        attention_dropout=float(model_cfg["attention_dropout"]),
        prediction_type=str(model_cfg["prediction_type"]),
        num_inference_steps=num_inference_steps,
        noise_scheduler_kwargs={
            "num_train_timesteps": int(model_cfg["num_train_timesteps"]),
            "beta_start": float(model_cfg["beta_start"]),
            "beta_end": float(model_cfg["beta_end"]),
            "beta_schedule": str(model_cfg["beta_schedule"]),
        },
    )
    return policy_cfg, resolved


def _resolve_outer_context_size(
    *,
    configured_size: int,
    data_k: int,
    pretrained_config: Optional[Dict[str, Any]] = None,
) -> int:
    if configured_size > 0:
        resolved_size = configured_size
    elif pretrained_config is not None:
        data_cfg = pretrained_config.get("data")
        if not isinstance(data_cfg, dict) or "K" not in data_cfg:
            raise ValueError(
                "Unable to infer K_pretrain from pretrained checkpoint config. "
                "Set config.maml.outer_context_size explicitly."
            )
        resolved_size = int(data_cfg["K"])
    else:
        resolved_size = data_k

    if resolved_size <= 0:
        raise ValueError("outer_context_size must be positive.")
    if resolved_size > data_k:
        raise ValueError(
            f"outer_context_size={resolved_size} exceeds resolved data.K={data_k}."
        )
    return resolved_size


def _resolve_data_k(
    config: ConfigDict,
    *,
    pretrained_config: Optional[Dict[str, Any]] = None,
) -> int:
    configured_k = int(config.data.K)
    if configured_k > 0:
        return configured_k
    if pretrained_config is None:
        raise ValueError(
            "data.K=0 requires config.finetune.pretrained_checkpoint so K_maml can "
            "be inferred as K_pretrain + 1."
        )

    data_cfg = pretrained_config.get("data")
    if not isinstance(data_cfg, dict) or "K" not in data_cfg:
        raise ValueError(
            "Unable to infer K_pretrain from pretrained checkpoint config. "
            "Set config.data.K explicitly."
        )
    pretrained_k = int(data_cfg["K"])
    if pretrained_k <= 0:
        raise ValueError(
            f"Invalid K={pretrained_k} found in pretrained checkpoint config."
        )
    return pretrained_k + 1


def _resolve_logging_max_tokens(
    config: ConfigDict,
    *,
    pretrained_config: Optional[Dict[str, Any]] = None,
) -> int:
    if pretrained_config is not None:
        data_cfg = pretrained_config.get("data")
        if isinstance(data_cfg, dict):
            max_query_len = data_cfg.get("max_query_len")
            if max_query_len is not None and int(max_query_len) > 0:
                return int(max_query_len)
    return int(config.data.max_seq_len)


class _NullCtx:
    def __enter__(self): return None
    def __exit__(self, exc_type, exc, tb): return False


def _maml_attention_ctx(cfg: MAMLConfig, device: torch.device):
    if cfg.use_math_attention and device.type == "cuda":
        return torch.backends.cuda.sdp_kernel(
            enable_flash=False,
            enable_mem_efficient=False,
            enable_math=True,
        )
    return _NullCtx()


def _special_token(
    sep: float = 0.0,
    stop: float = 0.0,
) -> np.ndarray:
    
    token = np.zeros(6)
    token[4] = sep
    token[5] = stop
    return token


def _pretrain_special_token(
    sep: float = 0.0,
    reset: float = 0.0,
    stop: float = 0.0,
) -> torch.Tensor:
    token = torch.zeros((1, 7), dtype=torch.float32)
    token[0, 4] = sep
    token[0, 5] = reset
    token[0, 6] = stop
    return token


def _to_pretrain_token_space(sketch_tokens: torch.Tensor) -> torch.Tensor:
    if sketch_tokens.ndim != 2 or sketch_tokens.shape[1] != 6:
        raise ValueError(
            f"Expected sketch tokens with shape (T, 6), got {tuple(sketch_tokens.shape)}."
        )
    expanded = torch.zeros(
        (sketch_tokens.shape[0], 7),
        dtype=sketch_tokens.dtype,
        device=sketch_tokens.device,
    )
    expanded[:, :5] = sketch_tokens[:, :5]
    expanded[:, 6] = sketch_tokens[:, 5]
    return expanded


def _compose_pretrain_style_episode(
    prompt_episodes: List[torch.Tensor],
    query_episode: torch.Tensor,
) -> torch.Tensor:
    segments: List[torch.Tensor] = [
        _pretrain_special_token(sep=1.0).to(
            device=query_episode.device,
            dtype=query_episode.dtype,
        )
    ]
    for sketch in prompt_episodes:
        segments.append(_to_pretrain_token_space(sketch))
        segments.append(
            _pretrain_special_token(sep=1.0).to(
                device=sketch.device,
                dtype=sketch.dtype,
            )
        )
    segments.append(
        _pretrain_special_token(reset=1.0).to(
            device=query_episode.device,
            dtype=query_episode.dtype,
        )
    )
    segments.append(
        _pretrain_special_token(sep=1.0).to(
            device=query_episode.device,
            dtype=query_episode.dtype,
        )
    )
    segments.append(_to_pretrain_token_space(query_episode))
    segments.append(
        _pretrain_special_token(stop=1.0).to(
            device=query_episode.device,
            dtype=query_episode.dtype,
        )
    )
    return torch.cat(segments, dim=0)


def _task_to_device(
    task: Dict[str, Any],
    *,
    device: torch.device,
) -> Dict[str, Any]:
    return {
        "context_episodes": [
            episode.to(device=device, dtype=torch.float32, non_blocking=True)
            for episode in task["context_episodes"]
        ],
        "query_episode": task["query_episode"].to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        ),
    }


def _prepare_pretrain_style_sample(
    episode_tokens: torch.Tensor,
    *,
    horizon: int,
    rng: torch.Generator,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    tokens = episode_tokens.to(device=device, dtype=torch.float32, non_blocking=True)
    reset_idx = (tokens[:, 5] == 1.0).nonzero(as_tuple=True)[0]
    if reset_idx.numel() != 1:
        raise ValueError(
            f"Expected exactly one reset token in pseudo-episode, found {reset_idx.numel()}."
        )
    reset_idx = int(reset_idx.item())
    start_idx = int(
        torch.randint(
            low=reset_idx + 1,
            high=tokens.shape[0],
            size=(1,),
            generator=rng,
            device=device,
        ).item()
    )

    tokens = torch.cat([tokens[:, :5], tokens[:, 6:]], dim=-1)

    context = tokens[:reset_idx].clone()
    points = tokens[reset_idx + 1 : start_idx + 1].clone()
    actions = tokens[start_idx + 1 : start_idx + 1 + horizon].clone()
    if actions.shape[0] < horizon:
        actions = _pad_actions(actions, horizon=horizon)

    query_len = points.shape[0] + actions.shape[0]
    points_len = points.shape[0]
    feature_dim = tokens.shape[-1]

    history = torch.zeros((query_len, feature_dim), dtype=torch.float32, device=device)
    history[-points_len:] = points

    query_mask = torch.zeros((query_len + horizon,), dtype=torch.bool, device=device)
    query_mask[-query_len:] = True
    context_mask = torch.ones((context.shape[0],), dtype=torch.bool, device=device)

    return {
        "history": history,
        "actions": actions.to(device=device, dtype=torch.float32),
        "context": context.to(device=device, dtype=torch.float32),
        "query_mask": query_mask,
        "context_mask": context_mask,
    }


def _collate_pretrain_style_samples(
    samples: List[Dict[str, torch.Tensor]],
    *,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    if not samples:
        raise ValueError("No samples to collate.")

    batch_size = len(samples)
    feature_dim = samples[0]["history"].shape[-1]
    horizon = samples[0]["actions"].shape[0]
    max_history_len = max(sample["history"].shape[0] for sample in samples)
    max_context_len = max(sample["context"].shape[0] for sample in samples)

    history = torch.zeros(
        (batch_size, max_history_len, feature_dim),
        dtype=torch.float32,
        device=device,
    )
    actions = torch.stack([sample["actions"] for sample in samples], dim=0)
    context = torch.zeros(
        (batch_size, max_context_len, feature_dim),
        dtype=torch.float32,
        device=device,
    )
    query_mask = torch.zeros(
        (batch_size, max_history_len + horizon),
        dtype=torch.bool,
        device=device,
    )
    context_mask = torch.zeros(
        (batch_size, max_context_len),
        dtype=torch.bool,
        device=device,
    )

    for idx, sample in enumerate(samples):
        history_len = sample["history"].shape[0]
        context_len = sample["context"].shape[0]
        query_mask_len = sample["query_mask"].shape[0]
        history[idx, -history_len:] = sample["history"]
        context[idx, -context_len:] = sample["context"]
        query_mask[idx, -query_mask_len:] = sample["query_mask"]
        context_mask[idx, -context_len:] = sample["context_mask"]

    return {
        "history": history,
        "actions": actions,
        "context": context,
        "query_mask": query_mask,
        "context_mask": context_mask,
    }


def _attach_diffusion_inputs(
    batch_out: Dict[str, torch.Tensor],
    *,
    noise: Optional[torch.Tensor],
    timesteps: Optional[torch.Tensor],
) -> None:
    if noise is not None:
        if noise.ndim == 2:
            noise = noise.unsqueeze(0)
        if noise.shape[0] == 1 and batch_out["actions"].shape[0] > 1:
            noise = noise.expand(batch_out["actions"].shape[0], -1, -1)
        if noise.shape != batch_out["actions"].shape:
            raise ValueError(
                f"noise shape {tuple(noise.shape)} must match actions shape "
                f"{tuple(batch_out['actions'].shape)}"
            )
        batch_out["noise"] = noise.to(
            device=batch_out["actions"].device,
            dtype=torch.float32,
        )

    if timesteps is not None:
        if timesteps.ndim == 0:
            timesteps = timesteps.view(1)
        if timesteps.shape[0] == 1 and batch_out["actions"].shape[0] > 1:
            timesteps = timesteps.expand(batch_out["actions"].shape[0])
        if timesteps.shape != (batch_out["actions"].shape[0],):
            raise ValueError(
                "timesteps must have shape "
                f"({batch_out['actions'].shape[0]},), got {tuple(timesteps.shape)}"
            )
        batch_out["timesteps"] = timesteps.to(
            device=batch_out["actions"].device,
            dtype=torch.long,
        )


def _prepare_pretrain_style_batch(
    episode_tokens: torch.Tensor,
    *,
    horizon: int,
    rng: torch.Generator,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    sample = _prepare_pretrain_style_sample(
        episode_tokens,
        horizon=horizon,
        rng=rng,
        device=device,
    )
    return _collate_pretrain_style_samples([sample], device=device)


def _pad_actions(actions: torch.Tensor, horizon: int) -> torch.Tensor:
    """Pads actions that are shorter than the horizon with end-tokens."""
    pad_len = horizon - actions.shape[0]
    padding = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
        dtype=actions.dtype,
        device=actions.device,
    ).tile((pad_len, 1))

    return torch.cat([actions, padding])


def _prepare_loo_episode(
    heldout: torch.Tensor,
    kept: List[torch.Tensor],
    horizon: int,
    rng: torch.Generator,
    device: torch.device,
):
    episode_tokens = _compose_pretrain_style_episode(
        prompt_episodes=list(kept),
        query_episode=heldout,
    )
    return _prepare_pretrain_style_sample(
        episode_tokens,
        horizon=horizon,
        rng=rng,
        device=device,
    )


def build_support_batch_loo(
    task: Dict[str, Any],
    holdout_indices: int | List[int],
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
    Build a support batch dict for one or more LOO held-out context episodes.

    Expected task format:
      task["context_episodes"]: List[Tensor[T_i, F]]   length K
      task["query_episode"]:   Tensor[T_q, F]          (unused here)

    Returns a dict compatible with DiTEncDecDiffusionPolicy.compute_loss().
    The held-out sketches are wrapped into the exact same pseudo-episode structure
    used in pretraining before splitting into context/history/actions, then padded
    into a single batch.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    context_eps: List[torch.Tensor] = task["context_episodes"]
    if isinstance(holdout_indices, int):
        holdout_indices = [holdout_indices]
    if not holdout_indices:
        raise ValueError("holdout_indices is empty.")
    for holdout_idx in holdout_indices:
        if not (0 <= holdout_idx < len(context_eps)):
            raise IndexError(
                f"holdout_idx={holdout_idx} out of range for K={len(context_eps)}"
            )

    if rng is None:
        rng = torch.Generator(device=device)
        rng.manual_seed(torch.seed())  # non-deterministic by default

    samples: List[Dict[str, torch.Tensor]] = []
    for holdout_idx in holdout_indices:
        kept: List[torch.Tensor] = []
        for k, ep in enumerate(context_eps):
            if k == holdout_idx:
                continue
            if ep.ndim != 2:
                raise ValueError(
                    f"context_episodes[{k}] must have shape (T,F), got {tuple(ep.shape)}"
                )
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
        if heldout.shape[0] < 1:
            raise ValueError("held-out episode is empty.")

        samples.append(
            _prepare_loo_episode(
                heldout=heldout,
                kept=kept,
                horizon=horizon,
                rng=rng,
                device=device,
            )
        )

    batch_out = _collate_pretrain_style_samples(samples, device=device)
    _attach_diffusion_inputs(
        batch_out,
        noise=noise,
        timesteps=timesteps,
    )

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
    context_episodes: List[torch.Tensor],
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

    eps = [ep.to(device=device, dtype=torch.float32, non_blocking=True) for ep in context_episodes]

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


def _sample_context_subset(
    context_episodes: List[torch.Tensor],
    *,
    num_context_episodes: Optional[int],
    rng: torch.Generator,
    device: torch.device,
) -> List[torch.Tensor]:
    if num_context_episodes is None or num_context_episodes >= len(context_episodes):
        return context_episodes
    if num_context_episodes <= 0:
        raise ValueError("num_context_episodes must be positive when provided.")
    if num_context_episodes > len(context_episodes):
        raise ValueError(
            f"Requested {num_context_episodes} context episodes but task only has "
            f"{len(context_episodes)}."
        )

    keep_indices = torch.randperm(
        len(context_episodes), generator=rng, device=device
    )[:num_context_episodes]
    keep_indices = sorted(int(idx) for idx in keep_indices.tolist())
    return [context_episodes[idx] for idx in keep_indices]


def build_query_batch(
    task: Dict[str, List[torch.Tensor]],
    *,
    horizon: int,
    device: torch.device,
    noise: torch.Tensor | None = None,
    timesteps: torch.Tensor | None = None,
    # Optional knobs (safe defaults)
    rng: Optional[torch.Generator] = None,
    add_sep_between_context_episodes: bool = True,
    num_context_episodes: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    Build the query batch dict for the query episode using the exact same
    pseudo-episode structure as pretraining.

    Expected task format:
      task["context_episodes"]: List[Tensor[T_i, F]]  length K
      task["query_episode"]:   Tensor[T_q, F]
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive.")

    context_episodes: List[torch.Tensor] = task["context_episodes"]
    query_ep: torch.Tensor = task["query_episode"]

    if rng is None:
        rng = torch.Generator(device=device)
        rng.manual_seed(torch.seed())

    selected_context_episodes = _sample_context_subset(
        context_episodes,
        num_context_episodes=num_context_episodes,
        rng=rng,
        device=device,
    )
    episode_tokens = _compose_pretrain_style_episode(
        prompt_episodes=selected_context_episodes,
        query_episode=query_ep,
    )
    out = _prepare_pretrain_style_batch(
        episode_tokens,
        horizon=horizon,
        rng=rng,
        device=device,
    )

    _attach_diffusion_inputs(
        out,
        noise=noise,
        timesteps=timesteps,
    )

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
    include_final_norm_fast: bool = True
    num_loo_per_task: int = 2        # how many held-out context eps per inner step (subsample if K large)
    outer_context_size: int = 0
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


def _sample_loo_indices(
    K: int,
    *,
    num_loo_per_task: int,
    device: torch.device,
    rng: Optional[torch.Generator] = None,
) -> List[int]:
    if num_loo_per_task >= K:
        return list(range(K))
    perm = torch.randperm(K, generator=rng, device=device)
    return perm[:num_loo_per_task].tolist()


def _adapt_fast_params_for_task(
    model: nn.Module,
    task: Dict[str, Any],
    *,
    fast_names: List[str],
    cfg: MAMLConfig,
    horizon: int,
    create_graph: bool,
    base_params: Optional[Dict[str, torch.Tensor]] = None,
    buffers: Optional[Dict[str, torch.Tensor]] = None,
    rng: Optional[torch.Generator] = None,
) -> tuple[Dict[str, torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    device = torch.device(cfg.device)
    params = base_params if base_params is not None else {k: v for k, v in model.named_parameters()}
    buffers = buffers if buffers is not None else {k: v for k, v in model.named_buffers()}

    shared_noise = None
    shared_timesteps = None
    adapted_params = params
    K = len(task["context_episodes"])
    loo_indices = _sample_loo_indices(
        K,
        num_loo_per_task=cfg.num_loo_per_task,
        device=device,
        rng=rng,
    )

    for _ in range(cfg.inner_steps):
        support_batch = build_support_batch_loo(
            task,
            holdout_indices=loo_indices,
            horizon=horizon,
            device=device,
            rng=rng,
        )

        if cfg.reuse_diffusion_noise and (shared_noise is None or shared_timesteps is None):
            shared_noise = torch.randn(
                (1, *support_batch["actions"].shape[1:]),
                device=device,
                dtype=support_batch["actions"].dtype,
            )
            shared_timesteps = torch.randint(
                0,
                model.scheduler.config.num_train_timesteps,
                (1,),
                device=device,
                dtype=torch.long,
            )

        if cfg.reuse_diffusion_noise:
            _attach_diffusion_inputs(
                support_batch,
                noise=shared_noise,
                timesteps=shared_timesteps,
            )

        support_loss = functional_call(model, (adapted_params, buffers), (support_batch,))
        fast_tensors = [adapted_params[n] for n in fast_names]
        grads = torch.autograd.grad(
            support_loss,
            fast_tensors,
            create_graph=create_graph,
            retain_graph=create_graph,
            allow_unused=False,
        )
        grads = _clip_grads_in_list(list(grads), cfg.max_grad_norm)

        new_params = dict(adapted_params)
        for name, p, g in zip(fast_names, fast_tensors, grads):
            new_params[name] = p - cfg.inner_lr * g
        adapted_params = new_params

    return adapted_params, shared_noise, shared_timesteps


def _prepare_task_for_meta_step(
    task: Dict[str, Any],
    *,
    cfg: MAMLConfig,
    horizon: int,
    device: torch.device,
    num_train_timesteps: int,
    rng: Optional[torch.Generator] = None,
) -> Dict[str, Any]:
    task_on_device = _task_to_device(task, device=device)
    num_context_episodes = (
        cfg.outer_context_size if cfg.outer_context_size > 0 else None
    )
    K = len(task_on_device["context_episodes"])
    loo_indices = _sample_loo_indices(
        K,
        num_loo_per_task=cfg.num_loo_per_task,
        device=device,
        rng=rng,
    )

    shared_noise = None
    shared_timesteps = None
    if cfg.reuse_diffusion_noise:
        feature_dim = int(task_on_device["query_episode"].shape[-1])
        shared_noise = torch.randn(
            (1, horizon, feature_dim),
            device=device,
            dtype=torch.float32,
        )
        shared_timesteps = torch.randint(
            0,
            num_train_timesteps,
            (1,),
            device=device,
            dtype=torch.long,
        )

    support_batches: List[Dict[str, torch.Tensor]] = []
    for _ in range(cfg.inner_steps):
        support_batches.append(
            build_support_batch_loo(
                task_on_device,
                holdout_indices=loo_indices,
                horizon=horizon,
                device=device,
                noise=shared_noise if cfg.reuse_diffusion_noise else None,
                timesteps=shared_timesteps if cfg.reuse_diffusion_noise else None,
                rng=rng,
            )
        )

    query_batch = build_query_batch(
        task_on_device,
        horizon=horizon,
        device=device,
        noise=shared_noise if cfg.reuse_diffusion_noise else None,
        timesteps=shared_timesteps if cfg.reuse_diffusion_noise else None,
        rng=rng,
        num_context_episodes=num_context_episodes,
    )

    return {
        "support_batches": support_batches,
        "query_batch": query_batch,
    }


def _prepare_outer_batch_for_meta_step(
    tasks: List[Dict[str, Any]],
    *,
    cfg: MAMLConfig,
    horizon: int,
    device: torch.device,
    num_train_timesteps: int,
) -> List[Dict[str, Any]]:
    return [
        _prepare_task_for_meta_step(
            task,
            cfg=cfg,
            horizon=horizon,
            device=device,
            num_train_timesteps=num_train_timesteps,
        )
        for task in tasks
    ]


def _adapt_fast_params_for_prepared_task(
    model: nn.Module,
    prepared_task: Dict[str, Any],
    *,
    fast_names: List[str],
    cfg: MAMLConfig,
    create_graph: bool,
    base_params: Optional[Dict[str, torch.Tensor]] = None,
    buffers: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
    adapted_params = (
        base_params if base_params is not None else {k: v for k, v in model.named_parameters()}
    )
    buffers = buffers if buffers is not None else {k: v for k, v in model.named_buffers()}

    for support_batch in prepared_task["support_batches"]:
        support_loss = functional_call(model, (adapted_params, buffers), (support_batch,))
        fast_tensors = [adapted_params[n] for n in fast_names]
        grads = torch.autograd.grad(
            support_loss,
            fast_tensors,
            create_graph=create_graph,
            retain_graph=create_graph,
            allow_unused=False,
        )
        grads = _clip_grads_in_list(list(grads), cfg.max_grad_norm)

        new_params = dict(adapted_params)
        for name, p, g in zip(fast_names, fast_tensors, grads):
            new_params[name] = p - cfg.inner_lr * g
        adapted_params = new_params

    return adapted_params


def _copy_fast_params_into_model(
    target_model: nn.Module,
    *,
    adapted_params: Dict[str, torch.Tensor],
    fast_names: List[str],
) -> None:
    fast_name_set = set(fast_names)
    with torch.no_grad():
        for name, param in target_model.named_parameters():
            if name in fast_name_set:
                param.copy_(adapted_params[name].detach())


def maml_task_loss_second_order(
    model: nn.Module,
    prepared_task: Dict[str, Any],
    *,
    fast_names: List[str],
    cfg: MAMLConfig,
    horizon: int,
    base_params: Optional[Dict[str, torch.Tensor]] = None,
    buffers: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.Tensor:
    """
    Compute meta-loss for one task:
      inner: adapt fast params on LOO support loss
      outer: query loss with adapted params
    Returns scalar loss (requires grad).
    """
    buffers = buffers if buffers is not None else {k: v for k, v in model.named_buffers()}
    adapted_params = _adapt_fast_params_for_prepared_task(
        model,
        prepared_task,
        fast_names=fast_names,
        cfg=cfg,
        create_graph=True,
        base_params=base_params,
        buffers=buffers,
    )
    query_loss = functional_call(model, (adapted_params, buffers), (prepared_task["query_batch"],))
    return query_loss


def _log_maml_eval_samples(
    policy: nn.Module,
    eval_tasks: List[Dict[str, Any]],
    *,
    fast_names: List[str],
    cfg: MAMLConfig,
    config: ConfigDict,
    policy_cfg: DiTEncDecDiffusionPolicyConfig,
    device: torch.device,
    step: int,
    max_tokens: int,
) -> None:
    if not eval_tasks or not config.wandb.use or wandb.run is None or config.eval.samples <= 0:
        return

    num_log_tasks = min(len(eval_tasks), int(config.eval.samples))
    adapted_policy = copy.deepcopy(policy).to(device)
    adapted_policy.eval()
    for param in adapted_policy.parameters():
        param.requires_grad_(False)

    images = []
    for task_idx, task in enumerate(eval_tasks[:num_log_tasks]):
        task_on_device = _task_to_device(task, device=device)
        task_seed = int(config.eval.seed) + step * 10_000 + task_idx
        task_rng = torch.Generator(device=device)
        task_rng.manual_seed(task_seed)

        with _maml_attention_ctx(cfg, device), torch.enable_grad():
            adapted_params, _, _ = _adapt_fast_params_for_task(
                policy,
                task_on_device,
                fast_names=fast_names,
                cfg=cfg,
                horizon=policy_cfg.horizon,
                create_graph=False,
                rng=task_rng,
            )

        _copy_fast_params_into_model(
            adapted_policy,
            adapted_params=adapted_params,
            fast_names=fast_names,
        )

        query_batch = build_query_batch(
            task=task_on_device,
            horizon=policy_cfg.horizon,
            device=device,
            rng=task_rng,
            num_context_episodes=cfg.outer_context_size,
        )
        sample_generator = torch.Generator(device=device)
        sample_generator.manual_seed(task_seed)
        images.extend(
            build_qualitative_sample_images(
                adapted_policy,
                query_batch,
                cfg=config,
                step=step,
                device=device,
                use_partial_history=False, # True,
                generator=sample_generator,
                max_tokens=max_tokens,
                num_context_demos=cfg.outer_context_size,
            )
        )

    if images:
        wandb.log({"samples/sketches/eval": images}, step=step + 1)


def _log_initial_eval_samples(
    policy: nn.Module,
    eval_tasks: List[Dict[str, Any]],
    *,
    config: ConfigDict,
    policy_cfg: DiTEncDecDiffusionPolicyConfig,
    device: torch.device,
    step: int,
    max_tokens: int,
    num_context_demos: int,
) -> None:
    if not eval_tasks or not config.wandb.use or wandb.run is None or config.eval.samples <= 0:
        return

    num_log_tasks = min(len(eval_tasks), int(config.eval.samples))
    was_training = policy.training
    policy.eval()

    images = []
    with torch.no_grad():
        for task_idx, task in enumerate(eval_tasks[:num_log_tasks]):
            task_on_device = _task_to_device(task, device=device)
            task_seed = int(config.eval.seed) + step * 10_000 + task_idx
            task_rng = torch.Generator(device=device)
            task_rng.manual_seed(task_seed)

            query_batch = build_query_batch(
                task=task_on_device,
                horizon=policy_cfg.horizon,
                device=device,
                rng=task_rng,
                num_context_episodes=num_context_demos,
            )
            sample_generator = torch.Generator(device=device)
            sample_generator.manual_seed(task_seed)
            images.extend(
                build_qualitative_sample_images(
                    policy,
                    query_batch,
                    cfg=config,
                    step=step,
                    device=device,
                    use_partial_history=False, # True,
                    generator=sample_generator,
                    max_tokens=max_tokens,
                    num_context_demos=num_context_demos,
                )
            )

    if was_training:
        policy.train()

    if images:
        wandb.log({"samples/sketches/eval_no_adaptation": images}, step=step)


def maml_step(
    model: nn.Module,
    prepared_tasks: List[Dict[str, Any]],
    *,
    fast_names: List[str],
    cfg: MAMLConfig,
    horizon: int,
) -> torch.Tensor:
    """
    Compute mean meta-loss over an outer batch of tasks (keeps computation graph).
    """
    base_params = {k: v for k, v in model.named_parameters()}
    buffers = {k: v for k, v in model.named_buffers()}
    losses: List[torch.Tensor] = []
    for prepared_task in prepared_tasks:
        loss = maml_task_loss_second_order(
            model,
            prepared_task,
            fast_names=fast_names,
            cfg=cfg,
            horizon=horizon,
            base_params=base_params,
            buffers=buffers,
        )
        losses.append(loss)
    return torch.stack(losses).mean()


# -------------------------
# Config loading
# -------------------------

_CONFIG = config_flags.DEFINE_config_file(
    "config",
    default="configs/diffusion/train_maml_icil.py",
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
    device = torch.device(
        config.run.device if torch.cuda.is_available() or config.run.device == "cpu" else "cpu"
    )

    pretrained_ckpt_path = None
    pretrained_checkpoint = None
    pretrained_config = None
    pretrained_checkpoint_value = str(config.finetune.pretrained_checkpoint).strip()
    if pretrained_checkpoint_value:
        pretrained_ckpt_path = Path(pretrained_checkpoint_value).expanduser()
        pretrained_checkpoint = _load_checkpoint_for_finetuning(
            pretrained_ckpt_path,
            device=device,
        )
        pretrained_config = pretrained_checkpoint["config"]
        checkpoint_data_cfg = pretrained_config.get("data", {})
        if isinstance(checkpoint_data_cfg, dict):
            checkpoint_coordinate_mode = checkpoint_data_cfg.get("coordinate_mode")
            if (
                checkpoint_coordinate_mode is not None
                and checkpoint_coordinate_mode != config.data.coordinate_mode
            ):
                raise ValueError(
                    "coordinate_mode mismatch between pretrained checkpoint "
                    f"('{checkpoint_coordinate_mode}') and current MAML config "
                    f"('{config.data.coordinate_mode}')."
                )

    policy_cfg, resolved_policy = _resolve_policy_config(
        config,
        pretrained_config=pretrained_config,
    )
    initial_global_step = (
        int(pretrained_checkpoint.get("step", 0))
        if pretrained_checkpoint is not None
        else 0
    )
    resolved_data_k = _resolve_data_k(
        config,
        pretrained_config=pretrained_config,
    )
    resolved_logging_max_tokens = _resolve_logging_max_tokens(
        config,
        pretrained_config=pretrained_config,
    )
    outer_context_size = _resolve_outer_context_size(
        configured_size=int(config.maml.outer_context_size),
        data_k=resolved_data_k,
        pretrained_config=pretrained_config,
    )

    cfg = MAMLConfig(
        inner_steps=config.maml.inner_steps,
        inner_lr=config.maml.inner_lr,
        outer_lr=config.maml.outer_lr,
        weight_decay=config.training.weight_decay,
        max_grad_norm=config.maml.max_grad_norm,
        last_frac_fast=config.maml.last_frac_fast,
        include_ada_fast=config.maml.include_ada_fast,
        include_final_norm_fast=config.maml.include_final_norm_fast,
        num_loo_per_task=config.maml.num_loo_per_task,
        outer_context_size=outer_context_size,
        reuse_diffusion_noise=config.maml.reuse_diffusion_noise,
        use_math_attention=config.maml.math_attention,
        device=config.run.device,
    )

    set_seed(config.run.seed)
    
    dataset = QuickDrawEpisodesMAML(
        root=config.data.root,
        split=config.data.split,
        K=resolved_data_k,
        max_seq_len=config.data.max_seq_len,
        backend=config.data.backend,
        coordinate_mode=config.data.coordinate_mode,
        index_dir=config.data.index_dir,
        ids_dir=config.data.ids_dir,
        seed=config.run.seed,
        families_cache_path=config.data.families_cache_path,
    )
    
    eval_dataset = QuickDrawEpisodesMAML(
        root=config.data.root,
        split="val",
        K=resolved_data_k,
        max_seq_len=config.data.max_seq_len,
        backend=config.data.backend,
        coordinate_mode=config.data.coordinate_mode,
        index_dir=config.data.index_dir,
        ids_dir=config.data.ids_dir,
        seed=config.run.seed + 1234,
        families_cache_path=config.data.families_cache_path,
    )
    
    policy = MAMLDiTEncDecDiffusionPolicy(policy_cfg).to(device)
    if pretrained_checkpoint is not None:
        load_result = policy.load_state_dict(
            pretrained_checkpoint["model"],
            strict=bool(config.finetune.strict_load),
        )
        if not bool(config.finetune.strict_load):
            print(
                f"Loaded pretrained checkpoint {pretrained_ckpt_path} "
                f"with missing_keys={load_result.missing_keys} "
                f"unexpected_keys={load_result.unexpected_keys}"
            )
        else:
            print(f"Loaded pretrained checkpoint {pretrained_ckpt_path}")

    policy = policy.to(device)
    policy.train()

    fast_names = get_fast_param_names(
        policy,
        last_frac=cfg.last_frac_fast,
        include_ada=cfg.include_ada_fast,
        include_final_norm=cfg.include_final_norm_fast,
    )
    outer_names = get_outer_param_names(
        policy,
        train_encoder=bool(config.outer.train_encoder),
        train_decoder=bool(config.outer.train_decoder),
        train_input_projections=bool(config.outer.train_input_projections),
        train_output_head=bool(config.outer.train_output_head),
        train_diffusion_conditioning=bool(config.outer.train_diffusion_conditioning),
    )
    missing_fast_outer = sorted(set(fast_names) - set(outer_names))
    if missing_fast_outer:
        raise ValueError(
            "Fast parameters must also be outer-trainable. "
            f"Missing outer assignments for: {missing_fast_outer[:5]}"
        )
    _set_outer_trainable_params(policy, outer_names)

    outer_params = [param for param in policy.parameters() if param.requires_grad]
    outer_opt = torch.optim.AdamW(
        outer_params,
        lr=cfg.outer_lr,
        weight_decay=cfg.weight_decay,
    )

    print(
        "Resolved MAML finetuning setup: "
        f"policy_source={resolved_policy['source']}, "
        f"data.K={resolved_data_k}, "
        f"prediction_type={policy_cfg.prediction_type}, "
        f"outer_context_size={cfg.outer_context_size}, "
        f"fast_param_tensors={len(fast_names)}, "
        f"outer_param_tensors={len(outer_names)}, "
        f"outer_param_count={_count_params_by_name(policy, outer_names):,}, "
        f"encoder_trainable={bool(config.outer.train_encoder)}"
    )

    base_save_dir = Path(config.checkpoint.dir)
    run_config = config.to_dict()
    run_config["resolved"] = {
        "pretrained_checkpoint": (
            str(pretrained_ckpt_path) if pretrained_ckpt_path is not None else ""
        ),
        "policy": asdict(policy_cfg),
        "data_k": resolved_data_k,
        "logging_max_tokens": resolved_logging_max_tokens,
        "outer_context_size": cfg.outer_context_size,
        "initial_global_step": initial_global_step,
        "fast_param_names": fast_names,
        "outer_param_names": outer_names,
    }
    if config.wandb.use and config.wandb.project:
        wandb.init(
            project=config.wandb.project,
            entity=getattr(config.wandb, "entity", None),
            config=run_config,
        )
        wandb.run.name = wandb.run.id
        save_dir = base_save_dir / wandb.run.id
    else:
        save_dir = base_save_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt_path = save_dir / "latest.pt"

    def _checkpoint_payload() -> Dict[str, Any]:
        return {
            "epoch": epoch,
            "global_step": global_step,
            "model": policy.state_dict(),
            "outer_opt": outer_opt.state_dict(),
            "policy_cfg": asdict(policy_cfg),
            "fast_names": fast_names,
            "outer_names": outer_names,
            "resolved": run_config["resolved"],
            "config": run_config,
            "maml_cfg": asdict(cfg),
        }

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
        collate_fn=TorchMAMLDiffusionCollator(
            token_dim=6,
            coordinate_mode=config.data.coordinate_mode,
        ),
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        worker_init_fn=_worker_init_fn,
        generator=g,
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=max(1, int(config.eval.samples)),
        shuffle=True,
        num_workers=0,
        collate_fn=TorchMAMLDiffusionCollator(
            token_dim=6,
            coordinate_mode=config.data.coordinate_mode,
        ),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    
    eval_iter = iter(eval_loader)

    if config.wandb.use and wandb.run is not None and config.eval.samples > 0:
        try:
            initial_eval_tasks = next(iter(eval_loader))
        except StopIteration:
            initial_eval_tasks = []
        _log_initial_eval_samples(
            policy,
            initial_eval_tasks,
            config=config,
            policy_cfg=policy_cfg,
            device=device,
            step=initial_global_step,
            max_tokens=resolved_logging_max_tokens,
            num_context_demos=cfg.outer_context_size,
        )

    pg = ProfilerGuard(
        use=config.profiling.use,
        start_step=0,
        end_step=3,
        trace_path=os.path.join(
            config.profiling.trace_dir,
            config.profiling.trace_filename,
        ),
    )

    global_step = initial_global_step
    for epoch in range(1, config.training.epochs + 1):
        
        for tasks in tqdm(loader):
            
            # Start profiler
            pg.start(global_step)
            
            global_step += 1
            outer_opt.zero_grad(set_to_none=True)
            prepared_tasks = _prepare_outer_batch_for_meta_step(
                tasks,
                cfg=cfg,
                horizon=policy_cfg.horizon,
                device=device,
                num_train_timesteps=policy.scheduler.config.num_train_timesteps,
            )

            # Force math attention for second-order meta-gradients (safer)
            with _maml_attention_ctx(cfg, device):
                meta_loss = maml_step(
                    policy,
                    prepared_tasks=prepared_tasks,
                    fast_names=fast_names,
                    cfg=cfg,
                    horizon=policy_cfg.horizon,
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

                    _log_maml_eval_samples(
                        policy,
                        eval_tasks,
                        fast_names=fast_names,
                        cfg=cfg,
                        config=config,
                        policy_cfg=policy_cfg,
                        device=device,
                        step=global_step,
                        max_tokens=resolved_logging_max_tokens,
                    )

            if (
                config.checkpoint.save_latest_every_steps > 0
                and global_step % config.checkpoint.save_latest_every_steps == 0
            ):
                torch.save(_checkpoint_payload(), latest_ckpt_path)
                print(f"Saved checkpoint: {latest_ckpt_path}")

        if epoch % config.checkpoint.save_interval == 0:
            ckpt_path = save_dir / f"maml_epoch_{epoch:03d}.pt"
            torch.save(_checkpoint_payload(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    app.run(main)

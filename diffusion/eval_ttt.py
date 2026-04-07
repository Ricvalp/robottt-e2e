#!/usr/bin/env python3
"""Evaluate pretrained encoder-decoder policies with test-time training."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import matplotlib
import numpy as np
import torch
import torch.nn as nn
from ml_collections import ConfigDict, config_flags
from torch.func import functional_call
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset import MAMLDiffusionCollator, QuickDrawEpisodesMAML, RasterizerConfig, rasterize_absolute_points
from diffusion.policies import MAMLDiTEncDecDiffusionPolicy, DiTEncDecDiffusionPolicyConfig
from diffusion.sampling import (
    sample_quickdraw_tokens_encoder_decoder,
    sample_quickdraw_tokens_encoder_decoder_from_partial,
)
from metrics import ResNet18FeatureExtractor, compute_fid

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_config(config_flag: str) -> ConfigDict:
    return config_flag.value


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _next_or_restart_tasks(
    iterator: Iterator[List[Dict[str, Any]]],
    loader: DataLoader,
) -> tuple[List[Dict[str, Any]], Iterator[List[Dict[str, Any]]]]:
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def _plot_tokens(
    ax,
    tokens: torch.Tensor,
    *,
    title: str,
    coordinate_mode: str,
    color: str = "black",
    invert_axis: bool = True,
) -> None:
    array = tokens.detach().cpu().numpy()
    if array.shape[0] == 0:
        ax.set_title(title)
        ax.set_aspect("equal")
        if invert_axis:
            ax.invert_yaxis()
        ax.axis("off")
        return

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
            color=color if active else "tab:red",
            linewidth=1.5,
            linestyle="-" if active else "--",
        )
    ax.set_title(title)
    ax.set_aspect("equal")
    if invert_axis:
        ax.invert_yaxis()
    ax.axis("off")


def _split_context_prompts(ctx_tokens: torch.Tensor, k: int) -> list[torch.Tensor]:
    sketches: List[torch.Tensor] = []
    current: List[torch.Tensor] = []
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
    return sketches[:k]


def plot_image_grid(
    images: list,
    *,
    name: str,
    output_dir: str | Path,
    dpi: int = 150,
) -> None:
    if not images:
        raise ValueError("images must be a non-empty list.")

    num_images = len(images)
    cols = math.ceil(math.sqrt(num_images))
    rows = math.ceil(num_images / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), dpi=dpi)
    axes = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for ax, image in zip(axes, images):
        ax.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        ax.axis("off")

    for ax in axes[num_images:]:
        ax.axis("off")

    fig.tight_layout()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path / name)
    plt.close(fig)


def _raw_data_root(output_dir: str | Path) -> Path:
    return Path(output_dir) / "raw_data"


def _serialize_raw_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _serialize_raw_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize_raw_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_serialize_raw_value(item) for item in value)
    if isinstance(value, Path):
        return str(value)
    return value


def _save_raw_plot_data(
    *,
    output_dir: str | Path,
    plot_type: str,
    name: str,
    payload: Dict[str, Any],
    split: str | None = None,
) -> Path:
    raw_dir = _raw_data_root(output_dir) / plot_type
    if split is not None:
        raw_dir = raw_dir / str(split)
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"{name}.pt"
    torch.save(_serialize_raw_value(payload), raw_path)
    return raw_path


@dataclass
class TTTConfig:
    inner_steps: int = 1
    inner_lr: float = 1e-4
    max_grad_norm: float = 1.0
    last_frac_fast: float = 0.25
    include_decoder_mlp_fast: bool = True
    include_ada_fast: bool = True
    include_final_norm_fast: bool = True
    include_decoder_self_attention_fast: bool = False
    include_decoder_cross_attention_fast: bool = False
    include_encoder_fast: bool = False
    include_input_projections_fast: bool = False
    include_output_head_fast: bool = False
    include_diffusion_conditioning_fast: bool = False
    num_loo_per_task: int = 2
    outer_context_size: int = 0
    reuse_diffusion_noise: bool = True
    use_math_attention: bool = True
    device: str = "cuda"


class _NullCtx:
    def __enter__(self): return None
    def __exit__(self, exc_type, exc, tb): return False


def _maml_attention_ctx(cfg: TTTConfig, device: torch.device):
    if cfg.use_math_attention and device.type == "cuda":
        return torch.backends.cuda.sdp_kernel(
            enable_flash=False,
            enable_mem_efficient=False,
            enable_math=True,
        )
    return _NullCtx()


def _resolve_checkpoint_path(cfg: ConfigDict) -> Path:
    if cfg.checkpoint.path:
        return Path(cfg.checkpoint.path)

    base_dir = Path(cfg.checkpoint.dir)
    if cfg.checkpoint.run_name:
        run_dir = base_dir / cfg.checkpoint.run_name
        if cfg.checkpoint.epoch > 0:
            return run_dir / f"policy_epoch_{cfg.checkpoint.epoch:03d}.pt"

        checkpoint_files = sorted(run_dir.glob("policy_epoch_*.pt"))
        if not checkpoint_files:
            raise FileNotFoundError(f"No epoch checkpoints found in {run_dir}.")
        return checkpoint_files[-1]

    checkpoint_path = base_dir / cfg.checkpoint.latest_filename
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "No explicit checkpoint path was provided and the latest checkpoint "
            f"was not found at {checkpoint_path}."
        )
    return checkpoint_path


def _resolve_output_dir(cfg: ConfigDict, checkpoint_path: Path) -> Path:
    checkpoint_path = checkpoint_path.expanduser().resolve()
    checkpoint_root = Path(cfg.checkpoint.dir).expanduser().resolve()
    if checkpoint_path.parent == checkpoint_root:
        raise ValueError(
            "Cannot infer a W&B run id from a checkpoint stored directly under "
            f"{checkpoint_root}. Use a checkpoint inside a run-id directory."
        )

    run_id = checkpoint_path.parent.name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(cfg.logging.output_parent_dir).expanduser() / run_id / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _policy_cfg_from_checkpoint(checkpoint: Dict[str, Any]) -> DiTEncDecDiffusionPolicyConfig:
    saved_cfg = checkpoint["config"]
    model_cfg = saved_cfg["model"]
    eval_cfg = saved_cfg["eval"]
    noise_scheduler_kwargs = {
        "num_train_timesteps": model_cfg["num_train_timesteps"],
        "beta_start": model_cfg["beta_start"],
        "beta_end": model_cfg["beta_end"],
        "beta_schedule": model_cfg["beta_schedule"],
    }
    return DiTEncDecDiffusionPolicyConfig(
        horizon=model_cfg["horizon"],
        point_feature_dim=model_cfg["input_dim"],
        action_dim=model_cfg["output_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        num_layers=model_cfg["num_layers"],
        num_heads=model_cfg["num_heads"],
        mlp_dim=model_cfg["mlp_dim"],
        dropout=model_cfg.get("dropout", 0.0),
        attention_dropout=model_cfg.get("attention_dropout", 0.0),
        activation=model_cfg.get("activation", "gelu"),
        layer_norm_eps=model_cfg.get("layer_norm_eps", 1e-5),
        scalar_embedding_hidden_dim=model_cfg.get("scalar_embedding_hidden_dim", 128),
        time_embedding_base=model_cfg.get("time_embedding_base", 10000.0),
        diffusion_embedding_base=model_cfg.get("diffusion_embedding_base", 10000.0),
        objective_type=model_cfg.get("objective_type", "diffusion"),
        regression_loss=model_cfg.get("regression_loss", "mse"),
        regression_fixed_variance=model_cfg.get("regression_fixed_variance", 1.0),
        prediction_type=model_cfg.get("prediction_type", "epsilon"),
        num_inference_steps=eval_cfg.get("num_inference_steps", 50),
        noise_scheduler_kwargs=noise_scheduler_kwargs,
    )


def _pretrain_k_from_checkpoint(checkpoint: Dict[str, Any]) -> int:
    saved_cfg = checkpoint["config"]
    data_cfg = saved_cfg.get("data", {})
    value = int(data_cfg.get("K", 0))
    if value > 0:
        return value
    raise ValueError("Unable to resolve pretraining data.K from checkpoint.")


def _resolved_data_k(
    cfg: ConfigDict,
    checkpoint: Dict[str, Any],
) -> int:
    configured_k = int(cfg.data.K)
    if configured_k > 0:
        return configured_k
    return _pretrain_k_from_checkpoint(checkpoint) + 1


def _resolved_outer_context_size(
    cfg: ConfigDict,
    checkpoint: Dict[str, Any],
    *,
    data_k: int,
) -> int:
    configured_size = int(cfg.ttt.outer_context_size)
    if configured_size > 0:
        resolved_size = configured_size
    else:
        resolved_size = _pretrain_k_from_checkpoint(checkpoint)
    if resolved_size <= 0:
        raise ValueError("outer_context_size must be positive.")
    if resolved_size > data_k:
        raise ValueError(
            f"outer_context_size={resolved_size} exceeds resolved data.K={data_k}."
        )
    return resolved_size


def _ttt_cfg_from_config(
    cfg: ConfigDict,
    checkpoint: Dict[str, Any],
    *,
    device: torch.device,
    data_k: int,
) -> TTTConfig:
    outer_context_size = _resolved_outer_context_size(cfg, checkpoint, data_k=data_k)
    return TTTConfig(
        inner_steps=int(cfg.ttt.inner_steps),
        inner_lr=float(cfg.ttt.inner_lr),
        max_grad_norm=float(cfg.ttt.max_grad_norm),
        last_frac_fast=float(cfg.ttt.last_frac_fast),
        include_decoder_mlp_fast=bool(cfg.ttt.include_decoder_mlp_fast),
        include_ada_fast=bool(cfg.ttt.include_ada_fast),
        include_final_norm_fast=bool(cfg.ttt.include_final_norm_fast),
        include_decoder_self_attention_fast=bool(cfg.ttt.include_decoder_self_attention_fast),
        include_decoder_cross_attention_fast=bool(cfg.ttt.include_decoder_cross_attention_fast),
        include_encoder_fast=bool(cfg.ttt.include_encoder_fast),
        include_input_projections_fast=bool(cfg.ttt.include_input_projections_fast),
        include_output_head_fast=bool(cfg.ttt.include_output_head_fast),
        include_diffusion_conditioning_fast=bool(cfg.ttt.include_diffusion_conditioning_fast),
        num_loo_per_task=int(cfg.ttt.num_loo_per_task),
        outer_context_size=outer_context_size,
        reuse_diffusion_noise=bool(cfg.ttt.reuse_diffusion_noise),
        use_math_attention=bool(cfg.ttt.math_attention),
        device=str(device),
    )


def _fast_names_from_config(
    model: nn.Module,
    cfg: TTTConfig,
) -> List[str]:
    if not hasattr(model, "decoder_transformer"):
        raise AttributeError("Model has no attribute 'decoder_transformer'.")
    decoder = getattr(model, "decoder_transformer")
    if not hasattr(decoder, "blocks"):
        raise AttributeError("decoder_transformer has no attribute 'blocks'.")

    n_blocks = len(decoder.blocks)
    if n_blocks <= 0:
        raise ValueError("decoder_transformer.blocks is empty.")

    if cfg.last_frac_fast <= 0:
        num_blocks = 1
    else:
        num_blocks = max(1, int(round(n_blocks * cfg.last_frac_fast)))
        num_blocks = min(num_blocks, n_blocks)

    start_idx = n_blocks - num_blocks
    fast_prefixes: List[str] = []
    for block_idx in range(start_idx, n_blocks):
        block_prefix = f"decoder_transformer.blocks.{block_idx}."
        if cfg.include_decoder_mlp_fast:
            fast_prefixes.append(f"{block_prefix}mlp.")
        if cfg.include_ada_fast:
            fast_prefixes.append(f"{block_prefix}ada_ln.")
        if cfg.include_decoder_self_attention_fast:
            fast_prefixes.append(f"{block_prefix}self_attn.")
        if cfg.include_decoder_cross_attention_fast:
            fast_prefixes.append(f"{block_prefix}cross_attn.")

    if cfg.include_final_norm_fast and hasattr(decoder, "final_norm") and hasattr(decoder.final_norm, "mlp"):
        fast_prefixes.append("decoder_transformer.final_norm.mlp.")
    if cfg.include_encoder_fast:
        fast_prefixes.append("encoder_transformer.")
    if cfg.include_input_projections_fast:
        fast_prefixes.extend(
            [
                "point_feature_proj.",
                "history_feature_proj.",
                "action_encoder.",
            ]
        )
    if cfg.include_output_head_fast:
        fast_prefixes.append("output_head.")
    if cfg.include_diffusion_conditioning_fast:
        fast_prefixes.extend(
            [
                "diffusion_proj.",
                "world_time_embedder.",
                "diffusion_time_embedder.",
            ]
        )

    if not fast_prefixes:
        raise RuntimeError("No fast parameter families were enabled in cfg.ttt.")

    param_dict = dict(model.named_parameters())
    fast_names = [
        name
        for name in param_dict
        if any(name.startswith(prefix) for prefix in fast_prefixes)
    ]
    if not fast_names:
        raise RuntimeError("No fast parameters were selected from the current model.")
    return sorted(fast_names)


def _resolved_max_tokens(
    cfg: ConfigDict,
    checkpoint: Dict[str, Any],
) -> int:
    if int(cfg.eval.max_tokens) > 0:
        return int(cfg.eval.max_tokens)

    resolved = checkpoint.get("resolved")
    if not isinstance(resolved, dict):
        resolved = checkpoint.get("config", {}).get("resolved", {})
    value = int(resolved.get("logging_max_tokens", 0))
    if value > 0:
        return value

    data_cfg = checkpoint.get("config", {}).get("data", {})
    value = int(data_cfg.get("max_query_len", 0))
    if value > 0:
        return value
    value = int(data_cfg.get("max_seq_len", 0))
    if value > 0:
        return value
    raise ValueError("Unable to resolve max_tokens for sampling from checkpoint.")


def _resolved_max_seq_len(
    cfg: ConfigDict,
    checkpoint: Dict[str, Any],
) -> int:
    if int(cfg.data.max_seq_len) > 0:
        return int(cfg.data.max_seq_len)
    data_cfg = checkpoint.get("config", {}).get("data", {})
    value = int(data_cfg.get("max_seq_len", 0))
    if value > 0:
        return value
    raise ValueError("Unable to resolve max_seq_len from checkpoint.")


def _resolved_coordinate_mode(
    cfg: ConfigDict,
    checkpoint: Dict[str, Any],
) -> str:
    cfg_value = str(cfg.data.coordinate_mode).strip()
    saved_value = str(checkpoint.get("config", {}).get("data", {}).get("coordinate_mode", "")).strip()
    if cfg_value and saved_value and cfg_value != saved_value:
        raise ValueError(
            f"coordinate_mode mismatch between eval config ('{cfg_value}') "
            f"and checkpoint ('{saved_value}')."
        )
    resolved = cfg_value or saved_value
    if not resolved:
        raise ValueError("Unable to resolve coordinate_mode from config/checkpoint.")
    return resolved


def _resolve_inference_steps(cfg: ConfigDict) -> Optional[int]:
    value = int(getattr(cfg.eval, "inference_steps", 0))
    return value if value > 0 else None


def _load_policy_from_checkpoint(
    cfg: ConfigDict,
    device: torch.device,
) -> tuple[MAMLDiTEncDecDiffusionPolicy, Dict[str, Any], Path, DiTEncDecDiffusionPolicyConfig, TTTConfig, List[str], int, int, str]:
    checkpoint_path = _resolve_checkpoint_path(cfg)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint at {checkpoint_path} is not a dict.")
    if "model" not in checkpoint:
        raise KeyError(f"Checkpoint at {checkpoint_path} does not contain 'model'.")
    if "config" not in checkpoint or not isinstance(checkpoint["config"], dict):
        raise KeyError(f"Checkpoint at {checkpoint_path} does not contain a valid 'config'.")

    policy_cfg = _policy_cfg_from_checkpoint(checkpoint)
    resolved_data_k = _resolved_data_k(cfg, checkpoint)
    max_tokens = _resolved_max_tokens(cfg, checkpoint)
    coordinate_mode = _resolved_coordinate_mode(cfg, checkpoint)
    ttt_cfg = _ttt_cfg_from_config(
        cfg,
        checkpoint,
        device=device,
        data_k=resolved_data_k,
    )

    policy = MAMLDiTEncDecDiffusionPolicy(policy_cfg).to(device)
    policy.load_state_dict(checkpoint["model"])
    policy.eval()
    fast_names = _fast_names_from_config(policy, ttt_cfg)

    return (
        policy,
        checkpoint,
        checkpoint_path,
        policy_cfg,
        ttt_cfg,
        fast_names,
        resolved_data_k,
        max_tokens,
        coordinate_mode,
    )


def _build_loader(
    cfg: ConfigDict,
    *,
    split: str,
    resolved_data_k: int,
    max_seq_len: int,
    coordinate_mode: str,
) -> DataLoader:
    dataset = QuickDrawEpisodesMAML(
        root=cfg.data.root,
        split=split,
        K=resolved_data_k,
        max_seq_len=max_seq_len,
        backend=cfg.data.backend,
        coordinate_mode=coordinate_mode,
        index_dir=cfg.data.index_dir,
        ids_dir=cfg.data.ids_dir,
        seed=cfg.run.seed if split == "train" else cfg.run.seed + 1234,
        families_cache_path=cfg.data.families_cache_path,
    )
    return DataLoader(
        dataset,
        batch_size=max(1, int(cfg.eval.samples)),
        shuffle=False,
        num_workers=cfg.loader.num_workers,
        pin_memory=(cfg.run.device == "cuda" and torch.cuda.is_available()),
        drop_last=False,
        collate_fn=MAMLDiffusionCollator(
            token_dim=6,
            coordinate_mode=coordinate_mode,
        ),
    )


def _clip_grads_in_list(grads: List[torch.Tensor], max_norm: float) -> List[torch.Tensor]:
    if max_norm <= 0:
        return grads
    norms = [g.norm(2) for g in grads if g is not None]
    if not norms:
        return grads
    total_norm = torch.norm(torch.stack(norms), 2)
    if total_norm <= max_norm:
        return grads
    scale = max_norm / (total_norm + 1e-6)
    return [g * scale if g is not None else None for g in grads]


def _sample_loo_indices(
    K: int,
    *,
    num_loo_per_task: int,
    device: torch.device,
    rng: Optional[torch.Generator] = None,
) -> List[int]:
    if num_loo_per_task <= 0:
        raise ValueError("num_loo_per_task must be positive.")
    if K <= 0:
        raise ValueError("K must be positive.")
    if num_loo_per_task == K:
        return list(range(K))
    if num_loo_per_task < K:
        perm = torch.randperm(K, generator=rng, device=device)
        return perm[:num_loo_per_task].tolist()
    return torch.randint(
        low=0,
        high=K,
        size=(num_loo_per_task,),
        generator=rng,
        device=device,
    ).tolist()


def _special_token(
    sep: float = 0.0,
    stop: float = 0.0,
) -> np.ndarray:
    token = np.zeros(6, dtype=np.float32)
    token[4] = sep
    token[5] = stop
    return token


def _pretrain_special_token(
    sep: float = 0.0,
    reset: float = 0.0,
    stop: float = 0.0,
) -> np.ndarray:
    token = np.zeros(7, dtype=np.float32)
    token[4] = sep
    token[5] = reset
    token[6] = stop
    return token


def _to_pretrain_token_space(sketch_tokens: np.ndarray) -> np.ndarray:
    if sketch_tokens.ndim != 2 or sketch_tokens.shape[1] != 6:
        raise ValueError(
            f"Expected sketch tokens with shape (T, 6), got {tuple(sketch_tokens.shape)}."
        )
    expanded = np.zeros((sketch_tokens.shape[0], 7), dtype=np.float32)
    expanded[:, :5] = sketch_tokens[:, :5]
    expanded[:, 6] = sketch_tokens[:, 5]
    return expanded


def _compose_pretrain_style_episode(
    prompt_episodes: List[np.ndarray],
    query_episode: np.ndarray,
) -> np.ndarray:
    segments: List[np.ndarray] = [_pretrain_special_token(sep=1.0)]
    for sketch in prompt_episodes:
        segments.append(_to_pretrain_token_space(sketch))
        segments.append(_pretrain_special_token(sep=1.0))
    segments.append(_pretrain_special_token(reset=1.0))
    segments.append(_pretrain_special_token(sep=1.0))
    segments.append(_to_pretrain_token_space(query_episode))
    segments.append(_pretrain_special_token(stop=1.0))
    return np.vstack(segments).astype(dtype=np.float32, copy=False)


def _compose_context_tokens(prompt_episodes: List[np.ndarray]) -> np.ndarray:
    segments: List[np.ndarray] = [_special_token(sep=1.0)]
    for sketch in prompt_episodes:
        segments.append(sketch.astype(np.float32, copy=False))
        segments.append(_special_token(sep=1.0))
    return np.vstack(segments).astype(dtype=np.float32, copy=False)


def _pad_actions(actions: torch.Tensor, horizon: int) -> torch.Tensor:
    pad_len = horizon - actions.shape[0]
    padding = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
        dtype=actions.dtype,
        device=actions.device,
    ).tile((pad_len, 1))
    return torch.cat([actions, padding])


def _prepare_pretrain_style_batch(
    episode_tokens: np.ndarray,
    *,
    horizon: int,
    rng: torch.Generator,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    tokens = torch.from_numpy(episode_tokens).to(device=device, dtype=torch.float32)
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
    context_len = context.shape[0]
    feature_dim = tokens.shape[-1]

    history = torch.zeros((1, query_len, feature_dim), dtype=torch.float32, device=device)
    context_batch = torch.zeros((1, context_len, feature_dim), dtype=torch.float32, device=device)
    actions_batch = actions.unsqueeze(0).to(device=device, dtype=torch.float32)
    query_mask = torch.zeros((1, query_len + horizon), dtype=torch.bool, device=device)
    context_mask = torch.zeros((1, context_len), dtype=torch.bool, device=device)

    history[0, -points_len:] = points
    context_batch[0, -context_len:] = context
    query_mask[0, -query_len:] = True
    context_mask[0, -context_len:] = True

    return {
        "history": history,
        "actions": actions_batch,
        "context": context_batch,
        "query_mask": query_mask,
        "context_mask": context_mask,
    }


def _prepare_loo_episode(
    heldout: np.ndarray,
    kept: np.ndarray,
    horizon: int,
    rng: torch.Generator,
    device: torch.device,
):
    episode_tokens = _compose_pretrain_style_episode(
        prompt_episodes=list(kept),
        query_episode=heldout,
    )
    return _prepare_pretrain_style_batch(
        episode_tokens,
        horizon=horizon,
        rng=rng,
        device=device,
    )


def _collate_pretrain_style_batches(
    examples: List[Dict[str, torch.Tensor]],
    *,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    if not examples:
        raise ValueError("examples must be a non-empty list.")

    batch_size = len(examples)
    max_history_len = max(int(example["history"].shape[1]) for example in examples)
    max_context_len = max(int(example["context"].shape[1]) for example in examples)
    horizon = int(examples[0]["actions"].shape[1])
    feature_dim = int(examples[0]["history"].shape[-1])
    dtype = examples[0]["history"].dtype

    history = torch.zeros(
        (batch_size, max_history_len, feature_dim),
        dtype=dtype,
        device=device,
    )
    actions = torch.zeros(
        (batch_size, horizon, feature_dim),
        dtype=dtype,
        device=device,
    )
    context = torch.zeros(
        (batch_size, max_context_len, feature_dim),
        dtype=dtype,
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

    for batch_idx, example in enumerate(examples):
        history_tokens = example["history"][0]
        context_tokens = example["context"][0]
        action_tokens = example["actions"][0]
        history_len = int(history_tokens.shape[0])
        context_len = int(context_tokens.shape[0])
        query_valid_len = int(example["query_mask"][0].sum().item())

        history[batch_idx, -history_len:] = history_tokens
        actions[batch_idx] = action_tokens
        context[batch_idx, -context_len:] = context_tokens
        query_mask[batch_idx, -query_valid_len:] = True
        context_mask[batch_idx, -context_len:] = True

    return {
        "history": history,
        "actions": actions,
        "context": context,
        "query_mask": query_mask,
        "context_mask": context_mask,
    }


def build_support_batch_loo(
    task: Dict[str, Any],
    holdout_indices: int | List[int],
    *,
    horizon: int,
    device: torch.device,
    noise: torch.Tensor | None = None,
    timesteps: torch.Tensor | None = None,
    rng: Optional[torch.Generator] = None,
) -> Dict[str, torch.Tensor]:
    context_eps: List[np.ndarray] = task["context_episodes"]
    if rng is None:
        rng = torch.Generator(device=device)
        rng.manual_seed(torch.seed())

    if isinstance(holdout_indices, int):
        holdout_indices = [holdout_indices]
    if not holdout_indices:
        raise ValueError("holdout_indices must be non-empty.")

    examples: List[Dict[str, torch.Tensor]] = []
    for holdout_idx in holdout_indices:
        kept: List[np.ndarray] = []
        for idx, episode in enumerate(context_eps):
            if idx != holdout_idx:
                kept.append(episode)

        if not kept:
            raise ValueError("LOO resulted in empty context. Need at least 2 context episodes.")

        examples.append(
            _prepare_loo_episode(
                heldout=context_eps[holdout_idx],
                kept=kept,
                horizon=horizon,
                rng=rng,
                device=device,
            )
        )

    batch_out = _collate_pretrain_style_batches(examples, device=device)
    batch_size = len(examples)

    if noise is not None:
        if noise.ndim == 2:
            noise = noise.unsqueeze(0)
        if noise.shape[0] == 1 and batch_size > 1:
            noise = noise.expand(batch_size, -1, -1)
        if noise.shape != batch_out["actions"].shape:
            raise ValueError(
                f"noise shape {tuple(noise.shape)} must match actions shape "
                f"{tuple(batch_out['actions'].shape)}"
            )
        batch_out["noise"] = noise.to(device=device, dtype=torch.float32)
    if timesteps is not None:
        if timesteps.ndim == 0:
            timesteps = timesteps.view(1)
        if timesteps.shape == (1,) and batch_size > 1:
            timesteps = timesteps.expand(batch_size)
        if timesteps.shape != (batch_size,):
            raise ValueError(
                f"timesteps must have shape ({batch_size},) or scalar/1-vector, "
                f"got {tuple(timesteps.shape)}"
            )
        batch_out["timesteps"] = timesteps.to(device=device, dtype=torch.long)
    return batch_out


def _sample_context_subset(
    context_episodes: List[np.ndarray],
    *,
    num_context_episodes: Optional[int],
    rng: torch.Generator,
    device: torch.device,
) -> List[np.ndarray]:
    if num_context_episodes is None or num_context_episodes >= len(context_episodes):
        return context_episodes
    if num_context_episodes <= 0:
        raise ValueError("num_context_episodes must be positive when provided.")

    keep_indices = torch.randperm(
        len(context_episodes), generator=rng, device=device
    )[:num_context_episodes]
    keep_indices = sorted(int(idx) for idx in keep_indices.tolist())
    return [context_episodes[idx] for idx in keep_indices]


def build_query_batch(
    task: Dict[str, List[np.ndarray]],
    *,
    horizon: int,
    device: torch.device,
    noise: torch.Tensor | None = None,
    timesteps: torch.Tensor | None = None,
    rng: Optional[torch.Generator] = None,
    num_context_episodes: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    context_episodes: List[np.ndarray] = task["context_episodes"]
    query_ep: np.ndarray = task["query_episode"]

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

    if noise is not None:
        if noise.ndim == 2:
            noise = noise.unsqueeze(0)
        if noise.shape != out["actions"].shape:
            raise ValueError(
                f"noise shape {tuple(noise.shape)} must match actions shape "
                f"{tuple(out['actions'].shape)}"
            )
        out["noise"] = noise.to(device=device, dtype=torch.float32)

    if timesteps is not None:
        if timesteps.ndim == 0:
            timesteps = timesteps.view(1)
        if timesteps.shape != (1,):
            raise ValueError(f"timesteps must have shape (1,) or scalar, got {tuple(timesteps.shape)}")
        out["timesteps"] = timesteps.to(device=device, dtype=torch.long)

    return out


def _build_context_only_batch(
    task: Dict[str, Any],
    *,
    device: torch.device,
    rng: torch.Generator,
    num_context_episodes: int,
) -> Dict[str, torch.Tensor]:
    selected_context_episodes = _sample_context_subset(
        task["context_episodes"],
        num_context_episodes=num_context_episodes,
        rng=rng,
        device=device,
    )
    context_tokens = torch.from_numpy(
        _compose_context_tokens(selected_context_episodes)
    ).to(device=device, dtype=torch.float32)
    context = context_tokens.unsqueeze(0)
    context_mask = torch.ones((1, context_tokens.shape[0]), dtype=torch.bool, device=device)
    return {
        "context": context,
        "context_mask": context_mask,
    }


def _query_sketches_from_tasks(tasks: List[Dict[str, Any]]) -> list[torch.Tensor]:
    return [_query_sketch_from_task(task) for task in tasks]


def _build_context_only_batch_for_tasks(
    tasks: List[Dict[str, Any]],
    *,
    device: torch.device,
    rng: torch.Generator,
    num_context_episodes: int,
) -> Dict[str, torch.Tensor]:
    context_sequences: List[torch.Tensor] = []
    context_lengths: List[int] = []

    for task in tasks:
        selected_context_episodes = _sample_context_subset(
            task["context_episodes"],
            num_context_episodes=num_context_episodes,
            rng=rng,
            device=device,
        )
        context_tokens = torch.from_numpy(
            _compose_context_tokens(selected_context_episodes)
        ).to(device=device, dtype=torch.float32)
        context_sequences.append(context_tokens)
        context_lengths.append(int(context_tokens.shape[0]))

    if not context_sequences:
        raise ValueError("No tasks provided to build batched context.")

    batch_size = len(context_sequences)
    max_context_len = max(context_lengths)
    feature_dim = int(context_sequences[0].shape[-1])

    context = torch.zeros(
        (batch_size, max_context_len, feature_dim),
        dtype=torch.float32,
        device=device,
    )
    context_mask = torch.zeros(
        (batch_size, max_context_len),
        dtype=torch.bool,
        device=device,
    )

    for idx, (context_tokens, context_len) in enumerate(zip(context_sequences, context_lengths)):
        context[idx, -context_len:] = context_tokens
        context_mask[idx, -context_len:] = True

    return {
        "context": context,
        "context_mask": context_mask,
    }


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


def _adapt_fast_params_for_task(
    model: nn.Module,
    task: Dict[str, Any],
    *,
    fast_names: List[str],
    cfg: TTTConfig,
    horizon: int,
    create_graph: bool,
    rng: Optional[torch.Generator] = None,
) -> tuple[Dict[str, torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], List[float]]:
    device = torch.device(cfg.device)
    params = {k: v for k, v in model.named_parameters()}
    buffers = {k: v for k, v in model.named_buffers()}

    shared_noise = None
    shared_timesteps = None
    adapted_params = params
    inner_loss_history: List[float] = []
    K = len(task["context_episodes"])

    for _ in range(cfg.inner_steps):
        loo_indices = _sample_loo_indices(
            K,
            num_loo_per_task=cfg.num_loo_per_task,
            device=device,
            rng=rng,
        )
        if cfg.reuse_diffusion_noise and (shared_noise is None or shared_timesteps is None):
            feature_dim = int(task["query_episode"].shape[-1])
            shared_noise = torch.randn(
                (1, horizon, feature_dim),
                device=device,
                dtype=torch.float32,
            )
            shared_timesteps = torch.randint(
                0,
                model.scheduler.config.num_train_timesteps,
                (1,),
                device=device,
                dtype=torch.long,
            )

        support_batch = build_support_batch_loo(
            task,
            holdout_indices=loo_indices,
            horizon=horizon,
            device=device,
            noise=shared_noise if cfg.reuse_diffusion_noise else None,
            timesteps=shared_timesteps if cfg.reuse_diffusion_noise else None,
            rng=rng,
        )

        support_loss = functional_call(model, (adapted_params, buffers), (support_batch,))
        inner_loss_history.append(float(support_loss.detach().item()))
        fast_tensors = [adapted_params[name] for name in fast_names]
        grads = torch.autograd.grad(
            support_loss,
            fast_tensors,
            create_graph=create_graph,
            retain_graph=create_graph,
            allow_unused=False,
        )
        grads = _clip_grads_in_list(list(grads), cfg.max_grad_norm)

        new_params = dict(adapted_params)
        for name, param, grad in zip(fast_names, fast_tensors, grads):
            new_params[name] = param - cfg.inner_lr * grad
        adapted_params = new_params

    return adapted_params, shared_noise, shared_timesteps, inner_loss_history


def _history_sketch_from_query_batch(batch: Dict[str, torch.Tensor], idx: int) -> torch.Tensor:
    history = batch["history"][idx]
    history_len = history.shape[0]
    valid_mask = batch["query_mask"][idx, :history_len]
    if valid_mask.numel() == 0:
        return history.new_zeros((0, 3))
    keep = (history[:, 4] < 0.5) & (history[:, 5] < 0.5)
    return history[valid_mask & keep][:, :3]


def _query_sketch_from_task(task: Dict[str, Any]) -> torch.Tensor:
    return torch.from_numpy(task["query_episode"][:, :3]).to(dtype=torch.float32)


def _prepare_adapted_policy_for_task(
    *,
    base_policy: MAMLDiTEncDecDiffusionPolicy,
    adapted_policy: MAMLDiTEncDecDiffusionPolicy,
    task: Dict[str, Any],
    fast_names: List[str],
    ttt_cfg: TTTConfig,
    horizon: int,
    rng: torch.Generator,
    device: torch.device,
) -> List[float]:
    previous_mode = base_policy.training
    base_policy.train()
    with _maml_attention_ctx(ttt_cfg, device), torch.enable_grad():
        adapted_params, _, _, inner_loss_history = _adapt_fast_params_for_task(
            base_policy,
            task,
            fast_names=fast_names,
            cfg=ttt_cfg,
            horizon=horizon,
            create_graph=False,
            rng=rng,
        )
    if not previous_mode:
        base_policy.eval()

    _copy_fast_params_into_model(
        adapted_policy,
        adapted_params=adapted_params,
        fast_names=fast_names,
    )
    return inner_loss_history


def _save_empty_sketch_samples(
    *,
    policy: MAMLDiTEncDecDiffusionPolicy,
    loader: DataLoader,
    split: str,
    cfg: ConfigDict,
    device: torch.device,
    ttt_cfg: TTTConfig,
    fast_names: List[str],
    max_tokens: int,
    coordinate_mode: str,
) -> None:
    iterator = iter(loader)
    tasks, _ = _next_or_restart_tasks(iterator, loader)
    adapted_policy = copy.deepcopy(policy).to(device)
    adapted_policy.eval()
    for param in adapted_policy.parameters():
        param.requires_grad_(False)

    for idx, task in enumerate(tasks[: int(cfg.eval.samples)]):
        task_seed = int(cfg.eval.seed) + idx
        task_rng = torch.Generator(device=device)
        task_rng.manual_seed(task_seed)
        inner_loss_history = _prepare_adapted_policy_for_task(
            base_policy=policy,
            adapted_policy=adapted_policy,
            task=task,
            fast_names=fast_names,
            ttt_cfg=ttt_cfg,
            horizon=policy.cfg.horizon,
            rng=task_rng,
            device=device,
        )
        context_batch = _build_context_only_batch(
            task,
            device=device,
            rng=task_rng,
            num_context_episodes=ttt_cfg.outer_context_size,
        )
        sample_generator = torch.Generator(device=device)
        sample_generator.manual_seed(task_seed)
        sample_tokens = sample_quickdraw_tokens_encoder_decoder(
            policy=adapted_policy,
            max_tokens=max_tokens,
            demos=context_batch,
            generator=sample_generator,
            inference_steps=_resolve_inference_steps(cfg),
        )[0]

        ctx_tokens = context_batch["context"][0]
        ctx_mask = context_batch["context_mask"][0]
        valid_ctx = ctx_tokens[ctx_mask].detach().cpu()
        prompts = _split_context_prompts(valid_ctx, ttt_cfg.outer_context_size)
        _save_raw_plot_data(
            output_dir=cfg.logging.dir,
            plot_type="empty_sketches",
            split=split,
            name=f"sample_{idx:04d}",
            payload={
                "plot_type": "empty_sketches",
                "split": split,
                "index": idx,
                "coordinate_mode": str(coordinate_mode),
                "prompts": prompts,
                "sample": sample_tokens,
                "inner_loss_history": inner_loss_history,
            },
        )

        if inner_loss_history:
            loss_steps = list(range(1, len(inner_loss_history) + 1))
            _save_raw_plot_data(
                output_dir=cfg.logging.dir,
                plot_type="empty_sketches_inner_loss",
                split=split,
                name=f"sample_{idx:04d}",
                payload={
                    "plot_type": "empty_sketches_inner_loss",
                    "split": split,
                    "index": idx,
                    "steps": loss_steps,
                    "inner_loss_history": inner_loss_history,
                },
            )

            loss_fig, loss_ax = plt.subplots(figsize=(5, 4), dpi=150)
            loss_ax.plot(loss_steps, inner_loss_history, marker="o", linewidth=1.5)
            loss_ax.set_xlabel("Fast Params GD Step")
            loss_ax.set_ylabel("Support Loss")
            loss_ax.set_yscale("log")
            loss_ax.set_title("TTT Inner Loss")
            loss_ax.grid(True, which="both", alpha=0.3)
            loss_fig.tight_layout()
            loss_fig.savefig(
                Path(cfg.logging.dir) / f"{split}_empty_samples_{idx}_inner_loss.png"
            )
            plt.close(loss_fig)

        total_plots = len(prompts) + 1
        cols = min(total_plots, 3)
        rows = (total_plots + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), dpi=150)
        axes = list(axes.flat) if hasattr(axes, "flat") else [axes]

        for prompt_idx, prompt_tokens in enumerate(prompts):
            _plot_tokens(
                axes[prompt_idx],
                prompt_tokens,
                title=f"Context {prompt_idx + 1}",
                coordinate_mode=coordinate_mode,
            )

        _plot_tokens(
            axes[len(prompts)],
            sample_tokens,
            title="Sample",
            coordinate_mode=coordinate_mode,
        )

        for ax in axes[total_plots:]:
            ax.axis("off")

        fig.tight_layout()
        plt.savefig(Path(cfg.logging.dir) / f"{split}_empty_samples_{idx}.png")
        plt.close(fig)


def _save_partial_sketch_samples(
    *,
    policy: MAMLDiTEncDecDiffusionPolicy,
    loader: DataLoader,
    split: str,
    cfg: ConfigDict,
    device: torch.device,
    ttt_cfg: TTTConfig,
    fast_names: List[str],
    max_tokens: int,
    coordinate_mode: str,
) -> None:
    iterator = iter(loader)
    tasks, _ = _next_or_restart_tasks(iterator, loader)
    adapted_policy = copy.deepcopy(policy).to(device)
    adapted_policy.eval()
    for param in adapted_policy.parameters():
        param.requires_grad_(False)

    for idx, task in enumerate(tasks[: int(cfg.eval.samples)]):
        task_seed = int(cfg.eval.seed) + idx
        task_rng = torch.Generator(device=device)
        task_rng.manual_seed(task_seed)
        _prepare_adapted_policy_for_task(
            base_policy=policy,
            adapted_policy=adapted_policy,
            task=task,
            fast_names=fast_names,
            ttt_cfg=ttt_cfg,
            horizon=policy.cfg.horizon,
            rng=task_rng,
            device=device,
        )
        query_batch = build_query_batch(
            task=task,
            horizon=policy.cfg.horizon,
            device=device,
            rng=task_rng,
            num_context_episodes=ttt_cfg.outer_context_size,
        )
        sample_generator = torch.Generator(device=device)
        sample_generator.manual_seed(task_seed)
        sample_tokens = sample_quickdraw_tokens_encoder_decoder_from_partial(
            policy=adapted_policy,
            max_tokens=max_tokens,
            demos=query_batch,
            generator=sample_generator,
            inference_steps=_resolve_inference_steps(cfg),
        )[0]

        ctx_tokens = query_batch["context"][0]
        ctx_mask = query_batch["context_mask"][0]
        valid_ctx = ctx_tokens[ctx_mask].detach().cpu()
        prompts = _split_context_prompts(valid_ctx, ttt_cfg.outer_context_size)
        history_tokens = _history_sketch_from_query_batch(query_batch, 0)
        _save_raw_plot_data(
            output_dir=cfg.logging.dir,
            plot_type="partial_sketches",
            split=split,
            name=f"sample_{idx:04d}",
            payload={
                "plot_type": "partial_sketches",
                "split": split,
                "index": idx,
                "coordinate_mode": str(coordinate_mode),
                "prompts": prompts,
                "history": history_tokens,
                "sample": sample_tokens,
            },
        )

        total_plots = len(prompts) + 1
        cols = min(total_plots, 3)
        rows = (total_plots + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), dpi=150)
        axes = list(axes.flat) if hasattr(axes, "flat") else [axes]

        for prompt_idx, prompt_tokens in enumerate(prompts):
            _plot_tokens(
                axes[prompt_idx],
                prompt_tokens,
                title=f"Context {prompt_idx + 1}",
                coordinate_mode=coordinate_mode,
            )

        _plot_tokens(
            axes[len(prompts)],
            history_tokens,
            title="Sample",
            coordinate_mode=coordinate_mode,
            color="green",
            invert_axis=False,
        )
        _plot_tokens(
            axes[len(prompts)],
            sample_tokens,
            title="Sample",
            coordinate_mode=coordinate_mode,
        )

        for ax in axes[total_plots:]:
            ax.axis("off")

        fig.tight_layout()
        plt.savefig(Path(cfg.logging.dir) / f"{split}_partial_samples_{idx}.png")
        plt.close(fig)


def _save_many_samples(
    *,
    policy: MAMLDiTEncDecDiffusionPolicy,
    loader: DataLoader,
    split: str,
    cfg: ConfigDict,
    device: torch.device,
    ttt_cfg: TTTConfig,
    fast_names: List[str],
    max_tokens: int,
    coordinate_mode: str,
) -> None:
    iterator = iter(loader)
    tasks, _ = _next_or_restart_tasks(iterator, loader)
    if not tasks:
        raise ValueError("Eval loader returned an empty batch.")

    task = tasks[0]
    adapted_policy = copy.deepcopy(policy).to(device)
    adapted_policy.eval()
    for param in adapted_policy.parameters():
        param.requires_grad_(False)

    task_rng = torch.Generator(device=device)
    task_rng.manual_seed(int(cfg.eval.seed))
    _prepare_adapted_policy_for_task(
        base_policy=policy,
        adapted_policy=adapted_policy,
        task=task,
        fast_names=fast_names,
        ttt_cfg=ttt_cfg,
        horizon=policy.cfg.horizon,
        rng=task_rng,
        device=device,
    )
    context_batch = _build_context_only_batch(
        task,
        device=device,
        rng=task_rng,
        num_context_episodes=ttt_cfg.outer_context_size,
    )

    context = context_batch["context"].repeat(int(cfg.eval.num_many_samples), 1, 1)
    context_mask = context_batch["context_mask"].repeat(int(cfg.eval.num_many_samples), 1)

    sample_generator = torch.Generator(device=device)
    sample_generator.manual_seed(int(cfg.eval.seed))
    samples = sample_quickdraw_tokens_encoder_decoder(
        policy=adapted_policy,
        max_tokens=max_tokens,
        demos={
            "context": context,
            "context_mask": context_mask,
        },
        generator=sample_generator,
        inference_steps=_resolve_inference_steps(cfg),
    )

    valid_ctx = context_batch["context"][0][context_batch["context_mask"][0]].detach().cpu()
    prompts = _split_context_prompts(valid_ctx, ttt_cfg.outer_context_size)
    _save_raw_plot_data(
        output_dir=cfg.logging.dir,
        plot_type="many_samples",
        split=split,
        name="panel",
        payload={
            "plot_type": "many_samples",
            "split": split,
            "coordinate_mode": str(coordinate_mode),
            "prompts": prompts,
            "samples": samples,
        },
    )

    total_plots = len(prompts) + len(samples)
    cols = min(total_plots, 3)
    rows = (total_plots + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), dpi=150)
    axes = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for prompt_idx, prompt_tokens in enumerate(prompts):
        _plot_tokens(
            axes[prompt_idx],
            prompt_tokens,
            title=f"Context {prompt_idx + 1}",
            coordinate_mode=coordinate_mode,
        )

    for sample_idx, sample_tokens in enumerate(samples):
        _plot_tokens(
            axes[len(prompts) + sample_idx],
            sample_tokens,
            title=f"Sample {sample_idx + 1}",
            coordinate_mode=coordinate_mode,
        )

    for ax in axes[total_plots:]:
        ax.axis("off")

    fig.tight_layout()
    plt.savefig(Path(cfg.logging.dir) / f"{split}_many_samples.png")
    plt.close(fig)


def _rasterize_sketch(sketch: torch.Tensor, rasterizer_config: RasterizerConfig) -> torch.Tensor:
    image = rasterize_absolute_points(sketch=sketch.cpu().numpy(), config=rasterizer_config)
    return torch.from_numpy(image).unsqueeze(0)


@torch.no_grad()
def _embed_sketches(
    sketches: list[torch.Tensor],
    *,
    embedding_model: ResNet18FeatureExtractor,
    rasterizer_config: RasterizerConfig,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    images = [_rasterize_sketch(sketch, rasterizer_config) for sketch in sketches]
    embeddings = []

    for start in range(0, len(images), batch_size):
        batch = torch.stack(images[start : start + batch_size], dim=0).to(device)
        embeddings.append(embedding_model(batch).cpu())

    return torch.cat(embeddings, dim=0), images


def _collect_generated_and_gt_queries(
    *,
    policy: MAMLDiTEncDecDiffusionPolicy,
    loader: DataLoader,
    cfg: ConfigDict,
    device: torch.device,
    ttt_cfg: TTTConfig,
    fast_names: List[str],
    max_tokens: int,
    adapt_fast_params: bool,
) -> tuple[list[torch.Tensor], list[torch.Tensor], Iterator[List[Dict[str, Any]]]]:
    iterator = iter(loader)
    generated: list[torch.Tensor] = []
    gt_queries: list[torch.Tensor] = []
    previous_mode = policy.training
    if adapt_fast_params:
        sampling_policy = copy.deepcopy(policy).to(device)
        sampling_policy.eval()
        for param in sampling_policy.parameters():
            param.requires_grad_(False)
    else:
        sampling_policy = policy
        sampling_policy.eval()
        generator = torch.Generator(device=device)
        generator.manual_seed(int(cfg.eval.seed))

    with tqdm(
        total=int(cfg.eval.fid.num_samples),
        desc=(
            "Generating samples for FID"
            if adapt_fast_params
            else "Generating samples for FID (no adaptation)"
        ),
        unit="sample",
    ) as progress:
        while len(generated) < int(cfg.eval.fid.num_samples):
            tasks, iterator = _next_or_restart_tasks(iterator, loader)
            if not adapt_fast_params:
                context_batch = _build_context_only_batch_for_tasks(
                    tasks,
                    device=device,
                    rng=generator,
                    num_context_episodes=ttt_cfg.outer_context_size,
                )
                samples = sample_quickdraw_tokens_encoder_decoder(
                    policy=sampling_policy,
                    max_tokens=max_tokens,
                    demos=context_batch,
                    generator=generator,
                    inference_steps=_resolve_inference_steps(cfg),
                )
                gt_batch = _query_sketches_from_tasks(tasks)
                remaining = int(cfg.eval.fid.num_samples) - len(generated)
                take = min(remaining, len(samples), len(gt_batch))
                generated.extend(samples[:take])
                gt_queries.extend(gt_batch[:take])
                progress.update(take)
                continue

            for task in tasks:
                task_seed = int(cfg.eval.seed) + len(generated)
                task_rng = torch.Generator(device=device)
                task_rng.manual_seed(task_seed)
                if adapt_fast_params:
                    _prepare_adapted_policy_for_task(
                        base_policy=policy,
                        adapted_policy=sampling_policy,
                        task=task,
                        fast_names=fast_names,
                        ttt_cfg=ttt_cfg,
                        horizon=policy.cfg.horizon,
                        rng=task_rng,
                        device=device,
                    )
                context_batch = _build_context_only_batch(
                    task,
                    device=device,
                    rng=task_rng,
                    num_context_episodes=ttt_cfg.outer_context_size,
                )
                sample_generator = torch.Generator(device=device)
                sample_generator.manual_seed(task_seed)
                sample = sample_quickdraw_tokens_encoder_decoder(
                    policy=sampling_policy,
                    max_tokens=max_tokens,
                    demos=context_batch,
                    generator=sample_generator,
                    inference_steps=_resolve_inference_steps(cfg),
                )[0]
                generated.append(sample)
                gt_queries.append(_query_sketch_from_task(task))
                progress.update(1)
                if len(generated) >= int(cfg.eval.fid.num_samples):
                    break

    if not adapt_fast_params and previous_mode:
        policy.train()

    return generated, gt_queries, iterator


def _collect_gt_queries(
    *,
    iterator: Iterator[List[Dict[str, Any]]],
    loader: DataLoader,
    num_samples: int,
) -> tuple[list[torch.Tensor], Iterator[List[Dict[str, Any]]]]:
    gt_queries: list[torch.Tensor] = []

    while len(gt_queries) < num_samples:
        tasks, iterator = _next_or_restart_tasks(iterator, loader)
        for task in tasks:
            gt_queries.append(_query_sketch_from_task(task))
            if len(gt_queries) >= num_samples:
                break

    return gt_queries, iterator


@torch.no_grad()
def _compute_fid_for_split(
    *,
    policy: MAMLDiTEncDecDiffusionPolicy,
    loader: DataLoader,
    split: str,
    cfg: ConfigDict,
    device: torch.device,
    embedding_model: ResNet18FeatureExtractor,
    rasterizer_config: RasterizerConfig,
    ttt_cfg: TTTConfig,
    fast_names: List[str],
    max_tokens: int,
    adapt_fast_params: bool,
    result_key: str,
) -> tuple[float, float]:
    generated, gt_queries, iterator = _collect_generated_and_gt_queries(
        policy=policy,
        loader=loader,
        cfg=cfg,
        device=device,
        ttt_cfg=ttt_cfg,
        fast_names=fast_names,
        max_tokens=max_tokens,
        adapt_fast_params=adapt_fast_params,
    )
    reference_gt_queries, _ = _collect_gt_queries(
        iterator=iterator,
        loader=loader,
        num_samples=int(cfg.eval.fid.num_samples),
    )

    generated_embeddings, generated_images = _embed_sketches(
        generated,
        embedding_model=embedding_model,
        rasterizer_config=rasterizer_config,
        batch_size=int(cfg.eval.fid.feature_batch_size),
        device=device,
    )
    gt_embeddings, gt_images = _embed_sketches(
        gt_queries,
        embedding_model=embedding_model,
        rasterizer_config=rasterizer_config,
        batch_size=int(cfg.eval.fid.feature_batch_size),
        device=device,
    )
    reference_gt_embeddings, _ = _embed_sketches(
        reference_gt_queries,
        embedding_model=embedding_model,
        rasterizer_config=rasterizer_config,
        batch_size=int(cfg.eval.fid.feature_batch_size),
        device=device,
    )

    fid = compute_fid(
        generated_features=generated_embeddings.numpy(),
        gt_features=gt_embeddings.numpy(),
    )
    reference_fid = compute_fid(
        generated_features=reference_gt_embeddings.numpy(),
        gt_features=gt_embeddings.numpy(),
    )

    _save_raw_plot_data(
        output_dir=cfg.logging.dir,
        plot_type="fid_grids",
        split=split,
        name=f"{result_key}_generated",
        payload={
            "plot_type": "fid_grid",
            "split": split,
            "result_key": result_key,
            "kind": "generated",
            "images": torch.stack([img.squeeze(0).cpu() for img in generated_images[:64]], dim=0),
        },
    )
    _save_raw_plot_data(
        output_dir=cfg.logging.dir,
        plot_type="fid_grids",
        split=split,
        name=f"{result_key}_ground_truth",
        payload={
            "plot_type": "fid_grid",
            "split": split,
            "result_key": result_key,
            "kind": "ground_truth",
            "images": torch.stack([img.squeeze(0).cpu() for img in gt_images[:64]], dim=0),
        },
    )
    _save_raw_plot_data(
        output_dir=cfg.logging.dir,
        plot_type="fid_metrics",
        split=split,
        name=f"{result_key}_metrics",
        payload={
            "plot_type": "fid_metrics",
            "split": split,
            "result_key": result_key,
            "fid": float(fid),
            "reference_fid": float(reference_fid),
            "num_samples": int(cfg.eval.fid.num_samples),
            "feature_batch_size": int(cfg.eval.fid.feature_batch_size),
        },
    )

    plot_image_grid(
        images=[img.squeeze().numpy() for img in generated_images[:64]],
        name=f"{result_key}_generated_{split}.png",
        output_dir=cfg.logging.dir,
    )
    plot_image_grid(
        images=[img.squeeze().numpy() for img in gt_images[:64]],
        name=f"{result_key}_gt_{split}.png",
        output_dir=cfg.logging.dir,
    )

    return fid, reference_fid


def _compute_fid(
    *,
    policy: MAMLDiTEncDecDiffusionPolicy,
    loaders: Dict[str, DataLoader],
    cfg: ConfigDict,
    device: torch.device,
    ttt_cfg: TTTConfig,
    fast_names: List[str],
    max_tokens: int,
    coordinate_mode: str,
    adapt_fast_params: bool,
    result_key: str,
) -> Dict[str, Dict[str, float]]:
    if coordinate_mode != "absolute":
        raise ValueError(
            "FID evaluation currently supports only absolute coordinates. "
            "Delta-coordinate rasterization is not implemented here."
        )

    rasterizer_cfg = load_config(_RASTERIZER_CONFIG).rasterizer_config
    rasterizer_config = RasterizerConfig(**rasterizer_cfg)

    embedding_model = ResNet18FeatureExtractor(
        pretrained_checkpoint_path=cfg.eval.fid.resnet_checkpoint_path
    ).to(device)
    embedding_model.eval()

    results: Dict[str, Dict[str, float]] = {}
    for split in cfg.eval.fid.splits:
        fid, reference_fid = _compute_fid_for_split(
            policy=policy,
            loader=loaders[split],
            split=split,
            cfg=cfg,
            device=device,
            embedding_model=embedding_model,
            rasterizer_config=rasterizer_config,
            ttt_cfg=ttt_cfg,
            fast_names=fast_names,
            max_tokens=max_tokens,
            adapt_fast_params=adapt_fast_params,
            result_key=result_key,
        )
        print(
            f"[{result_key}:{split}] FID: {fid:.6f} | "
            f"Reference FID (query vs query): {reference_fid:.6f}"
        )
        results[str(split)] = {
            "fid": float(fid),
            "reference_fid": float(reference_fid),
        }
    return results


TASKS = {
    "empty_sketches": _save_empty_sketch_samples,
    "partial_sketches": _save_partial_sketch_samples,
    "many_samples": _save_many_samples,
}


def run_selected_tasks(
    *,
    tasks: Iterable[str],
    policy: MAMLDiTEncDecDiffusionPolicy,
    loaders: Dict[str, DataLoader],
    cfg: ConfigDict,
    device: torch.device,
    ttt_cfg: TTTConfig,
    fast_names: List[str],
    max_tokens: int,
    coordinate_mode: str,
) -> Dict[str, object]:
    results: Dict[str, object] = {}
    for name in tasks:
        if name == "fid":
            results["fid"] = _compute_fid(
                policy=policy,
                loaders=loaders,
                cfg=cfg,
                device=device,
                ttt_cfg=ttt_cfg,
                fast_names=fast_names,
                max_tokens=max_tokens,
                coordinate_mode=coordinate_mode,
                adapt_fast_params=True,
                result_key="fid",
            )
            continue
        if name == "fid_no_adaptation":
            results["fid_no_adaptation"] = _compute_fid(
                policy=policy,
                loaders=loaders,
                cfg=cfg,
                device=device,
                ttt_cfg=ttt_cfg,
                fast_names=fast_names,
                max_tokens=max_tokens,
                coordinate_mode=coordinate_mode,
                adapt_fast_params=False,
                result_key="fid_no_adaptation",
            )
            continue
        if name not in TASKS:
            raise ValueError(f"Unknown task: {name}")
        TASKS[name](
            policy=policy,
            loader=loaders[cfg.eval.qualitative_split],
            split=cfg.eval.qualitative_split,
            cfg=cfg,
            device=device,
            ttt_cfg=ttt_cfg,
            fast_names=fast_names,
            max_tokens=max_tokens,
            coordinate_mode=coordinate_mode,
        )
    return results


def _write_eval_summary(
    *,
    cfg: ConfigDict,
    checkpoint_path: Path,
    output_dir: Path,
    policy_cfg: DiTEncDecDiffusionPolicyConfig,
    ttt_cfg: TTTConfig,
    resolved_data_k: int,
    max_tokens: int,
    coordinate_mode: str,
    fast_names: List[str],
    results: Dict[str, object],
) -> None:
    inference_steps = _resolve_inference_steps(cfg)
    run_id = checkpoint_path.parent.name
    summary = {
        "checkpoint_path": str(checkpoint_path),
        "run_id": run_id,
        "output_dir": str(output_dir),
        "raw_data_dir": str(output_dir / "raw_data"),
        "timestamp": datetime.now().isoformat(),
        "tasks": [str(task) for task in cfg.eval.tasks],
        "coordinate_mode": str(coordinate_mode),
        "objective_type": str(policy_cfg.objective_type),
        "regression_loss": str(policy_cfg.regression_loss),
        "regression_fixed_variance": float(policy_cfg.regression_fixed_variance),
        "prediction_type": str(policy_cfg.prediction_type),
        "resolved_data_k": int(resolved_data_k),
        "outer_context_size": int(ttt_cfg.outer_context_size),
        "num_loo_per_task": int(ttt_cfg.num_loo_per_task),
        "inner_steps": int(ttt_cfg.inner_steps),
        "inner_lr": float(ttt_cfg.inner_lr),
        "max_grad_norm": float(ttt_cfg.max_grad_norm),
        "last_frac_fast": float(ttt_cfg.last_frac_fast),
        "reuse_diffusion_noise": bool(ttt_cfg.reuse_diffusion_noise),
        "use_math_attention": bool(ttt_cfg.use_math_attention),
        "fast_param_selection": {
            "include_decoder_mlp_fast": bool(ttt_cfg.include_decoder_mlp_fast),
            "include_ada_fast": bool(ttt_cfg.include_ada_fast),
            "include_final_norm_fast": bool(ttt_cfg.include_final_norm_fast),
            "include_decoder_self_attention_fast": bool(ttt_cfg.include_decoder_self_attention_fast),
            "include_decoder_cross_attention_fast": bool(ttt_cfg.include_decoder_cross_attention_fast),
            "include_encoder_fast": bool(ttt_cfg.include_encoder_fast),
            "include_input_projections_fast": bool(ttt_cfg.include_input_projections_fast),
            "include_output_head_fast": bool(ttt_cfg.include_output_head_fast),
            "include_diffusion_conditioning_fast": bool(ttt_cfg.include_diffusion_conditioning_fast),
        },
        "fast_param_names": [str(name) for name in fast_names],
        "fast_param_tensors": int(len(fast_names)),
        "inference_steps": int(
            inference_steps if inference_steps is not None else policy_cfg.num_inference_steps
        ),
        "max_tokens": int(max_tokens),
        "qualitative_split": str(cfg.eval.qualitative_split),
        "fid": {
            "enabled": "fid" in [str(task) for task in cfg.eval.tasks],
            "num_samples": int(cfg.eval.fid.num_samples),
            "feature_batch_size": int(cfg.eval.fid.feature_batch_size),
            "splits": [str(split) for split in cfg.eval.fid.splits],
            "resnet_checkpoint_path": str(cfg.eval.fid.resnet_checkpoint_path),
            "results": results.get("fid", {}),
        },
        "fid_no_adaptation": {
            "enabled": "fid_no_adaptation" in [str(task) for task in cfg.eval.tasks],
            "num_samples": int(cfg.eval.fid.num_samples),
            "feature_batch_size": int(cfg.eval.fid.feature_batch_size),
            "splits": [str(split) for split in cfg.eval.fid.splits],
            "resnet_checkpoint_path": str(cfg.eval.fid.resnet_checkpoint_path),
            "results": results.get("fid_no_adaptation", {}),
        },
    }
    summary_path = output_dir / "eval_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


_CONFIG_FILE = config_flags.DEFINE_config_file(
    "config",
    default="configs/diffusion/eval_ttt.py",
)
_RASTERIZER_CONFIG = config_flags.DEFINE_config_file(
    "rasterizer_config",
    default="configs/metrics/cache.py",
)


def main(_) -> None:
    cfg = load_config(_CONFIG_FILE)
    set_seed(cfg.run.seed)
    device = torch.device(cfg.run.device if torch.cuda.is_available() else "cpu")

    (
        policy,
        checkpoint,
        checkpoint_path,
        policy_cfg,
        ttt_cfg,
        fast_names,
        resolved_data_k,
        max_tokens,
        coordinate_mode,
    ) = _load_policy_from_checkpoint(cfg, device)
    output_dir = _resolve_output_dir(cfg, checkpoint_path)
    cfg.logging.dir = str(output_dir)
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Saving eval outputs to: {output_dir}")
    print(
        "Resolved TTT eval setup: "
        f"data.K={resolved_data_k}, "
        f"outer_context_size={ttt_cfg.outer_context_size}, "
        f"num_loo_per_task={ttt_cfg.num_loo_per_task}, "
        f"inner_steps={ttt_cfg.inner_steps}, "
        f"inner_lr={ttt_cfg.inner_lr}, "
        f"fast_param_tensors={len(fast_names)}, "
        f"objective_type={policy_cfg.objective_type}, "
        f"regression_loss={policy_cfg.regression_loss}, "
        f"prediction_type={policy_cfg.prediction_type}, "
        f"inference_steps={_resolve_inference_steps(cfg) or policy_cfg.num_inference_steps}"
    )

    max_seq_len = _resolved_max_seq_len(cfg, checkpoint)
    loaders = {
        "train": _build_loader(
            cfg,
            split="train",
            resolved_data_k=resolved_data_k,
            max_seq_len=max_seq_len,
            coordinate_mode=coordinate_mode,
        ),
        "val": _build_loader(
            cfg,
            split="val",
            resolved_data_k=resolved_data_k,
            max_seq_len=max_seq_len,
            coordinate_mode=coordinate_mode,
        ),
    }

    results = run_selected_tasks(
        tasks=cfg.eval.tasks,
        policy=policy,
        loaders=loaders,
        cfg=cfg,
        device=device,
        ttt_cfg=ttt_cfg,
        fast_names=fast_names,
        max_tokens=max_tokens,
        coordinate_mode=coordinate_mode,
    )
    _write_eval_summary(
        cfg=cfg,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        policy_cfg=policy_cfg,
        ttt_cfg=ttt_cfg,
        resolved_data_k=resolved_data_k,
        max_tokens=max_tokens,
        coordinate_mode=coordinate_mode,
        fast_names=fast_names,
        results=results,
    )


if __name__ == "__main__":
    from absl import app

    app.run(main)

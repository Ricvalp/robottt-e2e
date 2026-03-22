#!/usr/bin/env python3
"""Evaluate encoder-decoder diffusion policies with qualitative samples and FID."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

import matplotlib
import torch
from ml_collections import ConfigDict, config_flags
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset import RasterizerConfig, rasterize_absolute_points
from dataset.episode_builder import EpisodeBuilderSimilar
from dataset.loader import QuickDrawEpisodes
from diffusion import DiTEncDecDiffusionPolicy, DiTEncDecDiffusionPolicyConfig
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


def _next_or_restart(
    iterator: Iterator[Dict[str, torch.Tensor]],
    loader: DataLoader,
) -> tuple[Dict[str, torch.Tensor], Iterator[Dict[str, torch.Tensor]]]:
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


def _pad_sequences(
    sequences: Iterable[torch.Tensor],
    *,
    feature_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    sequence_list = list(sequences)
    batch_size = len(sequence_list)
    max_len = max((seq.shape[0] for seq in sequence_list), default=0)

    padded = torch.zeros(batch_size, max_len, feature_dim, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    for idx, seq in enumerate(sequence_list):
        seq_len = seq.shape[0]
        if seq_len == 0:
            continue
        padded[idx, -seq_len:] = seq
        mask[idx, -seq_len:] = True

    return padded, mask


def _extract_episode_parts(tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    reset_positions = (tokens[:, 5] == 1.0).nonzero(as_tuple=True)[0]
    if reset_positions.numel() == 0:
        raise ValueError("Episode is missing the reset token.")

    reset_idx = int(reset_positions[0])
    tokens = torch.cat([tokens[:, :5], tokens[:, 6:]], dim=-1)

    context = tokens[:reset_idx].clone()
    query_separator = tokens[reset_idx + 1 : reset_idx + 2].clone()
    query = tokens[reset_idx + 2 : -1].clone()
    if query.shape[0] == 0:
        raise ValueError("Episode query sketch is empty.")

    return context, query_separator, query


def _history_prefix(query: torch.Tensor, query_separator: torch.Tensor, prefix_fraction: float) -> torch.Tensor:
    prefix_fraction = float(prefix_fraction)
    if not 0.0 <= prefix_fraction <= 1.0:
        raise ValueError("partial_prefix_fraction must lie in [0, 1].")

    if query.shape[0] <= 1:
        prefix_points = 0
    else:
        prefix_points = int(math.floor(query.shape[0] * prefix_fraction))
        prefix_points = min(prefix_points, query.shape[0] - 1)

    return torch.cat([query_separator, query[:prefix_points]], dim=0)


def _strip_special_tokens(tokens: torch.Tensor) -> torch.Tensor:
    if tokens.numel() == 0:
        return tokens.new_zeros((0, 3))
    keep = (tokens[:, 4] < 0.5) & (tokens[:, 5] < 0.5)
    return tokens[keep][:, :3]


def _query_sketches_from_batch(batch: Dict[str, torch.Tensor]) -> list[torch.Tensor]:
    sketches = []
    query = batch["query"]
    query_mask = batch["query_mask_gt"]
    for idx in range(query.shape[0]):
        sketches.append(query[idx][query_mask[idx]])
    return sketches


def _history_sketch_from_batch(batch: Dict[str, torch.Tensor], idx: int) -> torch.Tensor:
    history = batch["history"][idx]
    history_len = history.shape[0]
    valid_mask = batch["query_mask"][idx, :history_len]
    return _strip_special_tokens(history[valid_mask])


class EncoderDecoderEvalCollator:
    def __init__(self, horizon: int, partial_prefix_fraction: float) -> None:
        self.horizon = int(horizon)
        self.partial_prefix_fraction = float(partial_prefix_fraction)

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        contexts: List[torch.Tensor] = []
        histories: List[torch.Tensor] = []
        queries: List[torch.Tensor] = []

        for sample in batch:
            context, query_separator, query = _extract_episode_parts(sample["tokens"])
            history = _history_prefix(
                query,
                query_separator,
                prefix_fraction=self.partial_prefix_fraction,
            )

            contexts.append(context)
            histories.append(history)
            queries.append(query[:, :3])

        context, context_mask = _pad_sequences(contexts, feature_dim=6)
        history, _ = _pad_sequences(histories, feature_dim=6)
        query, query_mask_gt = _pad_sequences(queries, feature_dim=3)

        query_mask = torch.zeros(
            history.shape[0], history.shape[1] + self.horizon, dtype=torch.bool
        )
        for idx, hist in enumerate(histories):
            query_mask[idx, -(hist.shape[0] + self.horizon) :] = True

        return {
            "context": context,
            "context_mask": context_mask,
            "history": history,
            "query_mask": query_mask,
            "query": query,
            "query_mask_gt": query_mask_gt,
        }


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


def _policy_cfg_from_checkpoint(saved_cfg: dict) -> DiTEncDecDiffusionPolicyConfig:
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
        prediction_type=model_cfg.get("prediction_type", "epsilon"),
        num_inference_steps=eval_cfg["num_inference_steps"],
        noise_scheduler_kwargs=noise_scheduler_kwargs,
    )


def _load_policy_from_checkpoint(
    cfg: ConfigDict,
    device: torch.device,
) -> tuple[DiTEncDecDiffusionPolicy, dict, Path]:
    checkpoint_path = _resolve_checkpoint_path(cfg)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_cfg = checkpoint["config"]

    policy_cfg = _policy_cfg_from_checkpoint(saved_cfg)
    policy = DiTEncDecDiffusionPolicy(policy_cfg).to(device)
    policy.load_state_dict(checkpoint["model"])
    policy.eval()
    return policy, saved_cfg, checkpoint_path


def _build_loader(
    cfg: ConfigDict,
    *,
    split: str,
    horizon: int,
) -> DataLoader:
    dataset = QuickDrawEpisodes(
        root=cfg.data.root,
        split=split,
        K=cfg.data.K,
        max_seq_len=cfg.data.max_seq_len,
        max_query_len=cfg.data.max_query_len,
        max_context_len=cfg.data.max_context_len,
        backend=cfg.data.backend,
        coordinate_mode=cfg.data.coordinate_mode,
        index_dir=cfg.data.index_dir,
        ids_dir=cfg.data.ids_dir,
        seed=cfg.run.seed,
        builder_cls=EpisodeBuilderSimilar,
        families_cache_path=cfg.data.families_cache_path,
    )
    collator = EncoderDecoderEvalCollator(
        horizon=horizon,
        partial_prefix_fraction=cfg.eval.partial_prefix_fraction,
    )
    return DataLoader(
        dataset,
        batch_size=cfg.eval.samples,
        shuffle=False,
        num_workers=cfg.loader.num_workers,
        pin_memory=(cfg.run.device == "cuda" and torch.cuda.is_available()),
        drop_last=False,
        collate_fn=collator,
    )


def _save_empty_sketch_samples(
    *,
    policy: DiTEncDecDiffusionPolicy,
    loader: DataLoader,
    split: str,
    cfg: ConfigDict,
    device: torch.device,
) -> None:
    iterator = iter(loader)
    batch, _ = _next_or_restart(iterator, loader)

    generator = torch.Generator(device=device)
    generator.manual_seed(cfg.eval.seed)
    samples = sample_quickdraw_tokens_encoder_decoder(
        policy=policy,
        max_tokens=cfg.data.max_query_len,
        demos={
            "context": batch["context"],
            "context_mask": batch["context_mask"],
        },
        generator=generator,
    )

    batch_size = len(samples)
    for idx in range(batch_size):
        ctx_tokens = batch["context"][idx]
        ctx_mask = batch["context_mask"][idx]
        valid_ctx = ctx_tokens[ctx_mask].detach().cpu()
        prompts = _split_context_prompts(valid_ctx, cfg.data.K)
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
                title=f"Context {prompt_idx + 1}",
                coordinate_mode=cfg.data.coordinate_mode,
            )

        _plot_tokens(
            axes[len(prompts)],
            sample_tokens,
            title="Sample",
            coordinate_mode=cfg.data.coordinate_mode,
        )

        for ax in axes[total_plots:]:
            ax.axis("off")

        fig.tight_layout()
        plt.savefig(Path(cfg.logging.dir) / f"{split}_empty_samples_{idx}.png")
        plt.close(fig)


def _save_partial_sketch_samples(
    *,
    policy: DiTEncDecDiffusionPolicy,
    loader: DataLoader,
    split: str,
    cfg: ConfigDict,
    device: torch.device,
) -> None:
    iterator = iter(loader)
    batch, _ = _next_or_restart(iterator, loader)

    generator = torch.Generator(device=device)
    generator.manual_seed(cfg.eval.seed)
    samples = sample_quickdraw_tokens_encoder_decoder_from_partial(
        policy=policy,
        max_tokens=cfg.data.max_query_len,
        demos={
            "context": batch["context"],
            "context_mask": batch["context_mask"],
            "history": batch["history"],
            "query_mask": batch["query_mask"],
        },
        generator=generator,
    )

    batch_size = len(samples)
    for idx in range(batch_size):
        ctx_tokens = batch["context"][idx]
        ctx_mask = batch["context_mask"][idx]
        valid_ctx = ctx_tokens[ctx_mask].detach().cpu()
        prompts = _split_context_prompts(valid_ctx, cfg.data.K)
        sample_tokens = samples[idx]
        history_tokens = _history_sketch_from_batch(batch, idx)

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
                coordinate_mode=cfg.data.coordinate_mode,
            )

        _plot_tokens(
            axes[len(prompts)],
            history_tokens,
            title="Sample",
            coordinate_mode=cfg.data.coordinate_mode,
            color="green",
            invert_axis=False,
        )
        _plot_tokens(
            axes[len(prompts)],
            sample_tokens,
            title="Sample",
            coordinate_mode=cfg.data.coordinate_mode,
        )

        for ax in axes[total_plots:]:
            ax.axis("off")

        fig.tight_layout()
        plt.savefig(Path(cfg.logging.dir) / f"{split}_partial_samples_{idx}.png")
        plt.close(fig)


def _save_many_samples(
    *,
    policy: DiTEncDecDiffusionPolicy,
    loader: DataLoader,
    split: str,
    cfg: ConfigDict,
    device: torch.device,
) -> None:
    iterator = iter(loader)
    batch, _ = _next_or_restart(iterator, loader)

    base_context = batch["context"][:1]
    base_context_mask = batch["context_mask"][:1]
    context = base_context.repeat(cfg.eval.num_many_samples, 1, 1)
    context_mask = base_context_mask.repeat(cfg.eval.num_many_samples, 1)

    generator = torch.Generator(device=device)
    generator.manual_seed(cfg.eval.seed)
    samples = sample_quickdraw_tokens_encoder_decoder(
        policy=policy,
        max_tokens=cfg.data.max_query_len,
        demos={
            "context": context,
            "context_mask": context_mask,
        },
        generator=generator,
    )

    valid_ctx = base_context[0][base_context_mask[0]].detach().cpu()
    prompts = _split_context_prompts(valid_ctx, cfg.data.K)

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
            coordinate_mode=cfg.data.coordinate_mode,
        )

    for sample_idx, sample_tokens in enumerate(samples):
        ax = axes[len(prompts) + sample_idx]
        _plot_tokens(
            ax,
            sample_tokens,
            title=f"Sample {sample_idx + 1}",
            coordinate_mode=cfg.data.coordinate_mode,
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
    policy: DiTEncDecDiffusionPolicy,
    loader: DataLoader,
    cfg: ConfigDict,
    device: torch.device,
    seed: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor], Iterator[Dict[str, torch.Tensor]]]:
    iterator = iter(loader)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    generated: list[torch.Tensor] = []
    gt_queries: list[torch.Tensor] = []

    with tqdm(
        total=cfg.eval.fid.num_samples,
        desc="Generating samples for FID",
        unit="sample",
    ) as progress:
        while len(generated) < cfg.eval.fid.num_samples:
            batch, iterator = _next_or_restart(iterator, loader)
            samples = sample_quickdraw_tokens_encoder_decoder(
                policy=policy,
                max_tokens=cfg.data.max_query_len,
                demos={
                    "context": batch["context"],
                    "context_mask": batch["context_mask"],
                },
                generator=generator,
            )
            gt_batch = _query_sketches_from_batch(batch)

            remaining = cfg.eval.fid.num_samples - len(generated)
            take = min(remaining, len(samples), len(gt_batch))
            generated.extend(samples[:take])
            gt_queries.extend(gt_batch[:take])
            progress.update(take)

    return generated, gt_queries, iterator


def _collect_gt_queries(
    *,
    iterator: Iterator[Dict[str, torch.Tensor]],
    loader: DataLoader,
    num_samples: int,
) -> tuple[list[torch.Tensor], Iterator[Dict[str, torch.Tensor]]]:
    gt_queries: list[torch.Tensor] = []

    while len(gt_queries) < num_samples:
        batch, iterator = _next_or_restart(iterator, loader)
        batch_queries = _query_sketches_from_batch(batch)
        remaining = num_samples - len(gt_queries)
        gt_queries.extend(batch_queries[:remaining])

    return gt_queries, iterator


@torch.no_grad()
def _compute_fid_for_split(
    *,
    policy: DiTEncDecDiffusionPolicy,
    loader: DataLoader,
    split: str,
    cfg: ConfigDict,
    device: torch.device,
    embedding_model: ResNet18FeatureExtractor,
    rasterizer_config: RasterizerConfig,
) -> tuple[float, float]:
    generated, gt_queries, iterator = _collect_generated_and_gt_queries(
        policy=policy,
        loader=loader,
        cfg=cfg,
        device=device,
        seed=cfg.eval.seed,
    )
    reference_gt_queries, _ = _collect_gt_queries(
        iterator=iterator,
        loader=loader,
        num_samples=cfg.eval.fid.num_samples,
    )

    generated_embeddings, generated_images = _embed_sketches(
        generated,
        embedding_model=embedding_model,
        rasterizer_config=rasterizer_config,
        batch_size=cfg.eval.fid.feature_batch_size,
        device=device,
    )
    gt_embeddings, gt_images = _embed_sketches(
        gt_queries,
        embedding_model=embedding_model,
        rasterizer_config=rasterizer_config,
        batch_size=cfg.eval.fid.feature_batch_size,
        device=device,
    )
    reference_gt_embeddings, _ = _embed_sketches(
        reference_gt_queries,
        embedding_model=embedding_model,
        rasterizer_config=rasterizer_config,
        batch_size=cfg.eval.fid.feature_batch_size,
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

    plot_image_grid(
        images=[img.squeeze().numpy() for img in generated_images[:64]],
        name=f"fid_generated_{split}.png",
        output_dir=cfg.logging.dir,
    )
    plot_image_grid(
        images=[img.squeeze().numpy() for img in gt_images[:64]],
        name=f"fid_gt_{split}.png",
        output_dir=cfg.logging.dir,
    )

    return fid, reference_fid


def _compute_fid(
    *,
    policy: DiTEncDecDiffusionPolicy,
    loaders: Dict[str, DataLoader],
    cfg: ConfigDict,
    device: torch.device,
) -> None:
    if cfg.data.coordinate_mode != "absolute":
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

    for split in cfg.eval.fid.splits:
        fid, reference_fid = _compute_fid_for_split(
            policy=policy,
            loader=loaders[split],
            split=split,
            cfg=cfg,
            device=device,
            embedding_model=embedding_model,
            rasterizer_config=rasterizer_config,
        )
        print(
            f"[{split}] FID: {fid:.6f} | "
            f"Reference FID (query vs query): {reference_fid:.6f}"
        )


TASKS = {
    "empty_sketches": _save_empty_sketch_samples,
    "partial_sketches": _save_partial_sketch_samples,
    "many_samples": _save_many_samples,
}


def run_selected_tasks(
    *,
    tasks: Iterable[str],
    policy: DiTEncDecDiffusionPolicy,
    loaders: Dict[str, DataLoader],
    cfg: ConfigDict,
    device: torch.device,
) -> None:
    for name in tasks:
        if name == "fid":
            _compute_fid(policy=policy, loaders=loaders, cfg=cfg, device=device)
            continue
        if name not in TASKS:
            raise ValueError(f"Unknown task: {name}")
        TASKS[name](
            policy=policy,
            loader=loaders[cfg.eval.qualitative_split],
            split=cfg.eval.qualitative_split,
            cfg=cfg,
            device=device,
        )


_CONFIG_FILE = config_flags.DEFINE_config_file(
    "config",
    default="configs/diffusion/eval_encoder_decoder.py",
)
_RASTERIZER_CONFIG = config_flags.DEFINE_config_file(
    "rasterizer_config",
    default="configs/metrics/cache.py",
)


def main(_) -> None:
    cfg = load_config(_CONFIG_FILE)
    set_seed(cfg.run.seed)
    device = torch.device(cfg.run.device if torch.cuda.is_available() else "cpu")
    Path(cfg.logging.dir).mkdir(parents=True, exist_ok=True)

    policy, checkpoint_cfg, checkpoint_path = _load_policy_from_checkpoint(cfg, device)
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(
        "Policy inference steps from checkpoint: "
        f"{checkpoint_cfg['eval']['num_inference_steps']}"
    )

    loaders = {
        "train": _build_loader(cfg, split="train", horizon=policy.cfg.horizon),
        "val": _build_loader(cfg, split="val", horizon=policy.cfg.horizon),
    }

    run_selected_tasks(
        tasks=cfg.eval.tasks,
        policy=policy,
        loaders=loaders,
        cfg=cfg,
        device=device,
    )


if __name__ == "__main__":
    from absl import app

    app.run(main)

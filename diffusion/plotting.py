from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.lines import Line2D


def set_paper_theme(
    *,
    context: str = "paper",
    style: str = "white",
    font_scale: float = 1.1,
) -> None:
    sns.set_theme(
        context=context,
        style=style,
        font_scale=font_scale,
        rc={
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 11,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        },
    )


def load_raw_plot_data(raw_data_path: str | Path) -> dict[str, Any]:
    return torch.load(Path(raw_data_path), map_location="cpu")


def _to_numpy_tokens(tokens: Any) -> np.ndarray:
    if isinstance(tokens, torch.Tensor):
        return tokens.detach().cpu().numpy()
    return np.asarray(tokens)


def _panel_shape(num_items: int, max_cols: int = 3) -> tuple[int, int]:
    cols = min(max_cols, max(1, num_items))
    rows = (num_items + cols - 1) // cols
    return rows, cols


def _draw_sketch(
    ax,
    tokens: Any,
    *,
    coordinate_mode: str,
    title: str | None = None,
    color: str = "black",
    inactive_color: str = "tab:red",
    linewidth: float = 1.6,
    inactive_linestyle: str = "--",
    invert_axis: bool = True,
) -> None:
    array = _to_numpy_tokens(tokens)
    if array.size == 0 or array.shape[0] == 0:
        if title is not None:
            ax.set_title(title)
        ax.set_aspect("equal")
        if invert_axis:
            ax.invert_yaxis()
        ax.axis("off")
        return

    coords = array[:, :2].cumsum(axis=0) if coordinate_mode == "delta" else array[:, :2]
    pen_state = array[:, 2]

    for token_idx in range(1, coords.shape[0]):
        start = coords[token_idx - 1]
        end = coords[token_idx]
        active = pen_state[token_idx] >= 0.5
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=color if active else inactive_color,
            linewidth=linewidth,
            linestyle="-" if active else inactive_linestyle,
            solid_capstyle="round",
        )

    if title is not None:
        ax.set_title(title)
    ax.set_aspect("equal")
    if invert_axis:
        ax.invert_yaxis()
    ax.axis("off")


def plot_empty_sketch_panel(
    raw_data_path: str | Path,
    *,
    dpi: int = 300,
    max_cols: int = 3,
    panel_scale: float = 2.6,
    linewidth: float = 1.6,
):
    set_paper_theme()
    payload = load_raw_plot_data(raw_data_path)
    prompts = payload["prompts"]
    sample = payload["sample"]
    coordinate_mode = str(payload["coordinate_mode"])

    total_plots = len(prompts) + 1
    rows, cols = _panel_shape(total_plots, max_cols=max_cols)
    fig, axes = plt.subplots(rows, cols, figsize=(panel_scale * cols, panel_scale * rows), dpi=dpi)
    axes = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for prompt_idx, prompt_tokens in enumerate(prompts):
        _draw_sketch(
            axes[prompt_idx],
            prompt_tokens,
            title=f"Context {prompt_idx + 1}",
            coordinate_mode=coordinate_mode,
            linewidth=linewidth,
        )

    _draw_sketch(
        axes[len(prompts)],
        sample,
        title="Sample",
        coordinate_mode=coordinate_mode,
        linewidth=linewidth,
    )

    for ax in axes[total_plots:]:
        ax.axis("off")

    fig.tight_layout()
    return fig, axes


def plot_partial_sketch_panel(
    raw_data_path: str | Path,
    *,
    dpi: int = 300,
    max_cols: int = 3,
    panel_scale: float = 2.6,
    linewidth: float = 1.6,
    history_color: str = "#1b9e77",
):
    set_paper_theme()
    payload = load_raw_plot_data(raw_data_path)
    prompts = payload["prompts"]
    history = payload["history"]
    sample = payload["sample"]
    coordinate_mode = str(payload["coordinate_mode"])

    total_plots = len(prompts) + 1
    rows, cols = _panel_shape(total_plots, max_cols=max_cols)
    fig, axes = plt.subplots(rows, cols, figsize=(panel_scale * cols, panel_scale * rows), dpi=dpi)
    axes = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for prompt_idx, prompt_tokens in enumerate(prompts):
        _draw_sketch(
            axes[prompt_idx],
            prompt_tokens,
            title=f"Context {prompt_idx + 1}",
            coordinate_mode=coordinate_mode,
            linewidth=linewidth,
        )

    sample_ax = axes[len(prompts)]
    _draw_sketch(
        sample_ax,
        history,
        title="Sample",
        coordinate_mode=coordinate_mode,
        color=history_color,
        linewidth=linewidth,
        invert_axis=False,
    )
    _draw_sketch(
        sample_ax,
        sample,
        title="Sample",
        coordinate_mode=coordinate_mode,
        linewidth=linewidth,
    )

    for ax in axes[total_plots:]:
        ax.axis("off")

    fig.tight_layout()
    return fig, axes


def plot_many_samples_panel(
    raw_data_path: str | Path,
    *,
    dpi: int = 300,
    max_cols: int = 3,
    panel_scale: float = 2.6,
    linewidth: float = 1.6,
):
    set_paper_theme()
    payload = load_raw_plot_data(raw_data_path)
    prompts = payload["prompts"]
    samples = payload["samples"]
    coordinate_mode = str(payload["coordinate_mode"])

    total_plots = len(prompts) + len(samples)
    rows, cols = _panel_shape(total_plots, max_cols=max_cols)
    fig, axes = plt.subplots(rows, cols, figsize=(panel_scale * cols, panel_scale * rows), dpi=dpi)
    axes = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for prompt_idx, prompt_tokens in enumerate(prompts):
        _draw_sketch(
            axes[prompt_idx],
            prompt_tokens,
            title=f"Context {prompt_idx + 1}",
            coordinate_mode=coordinate_mode,
            linewidth=linewidth,
        )

    for sample_idx, sample_tokens in enumerate(samples):
        _draw_sketch(
            axes[len(prompts) + sample_idx],
            sample_tokens,
            title=f"Sample {sample_idx + 1}",
            coordinate_mode=coordinate_mode,
            linewidth=linewidth,
        )

    for ax in axes[total_plots:]:
        ax.axis("off")

    fig.tight_layout()
    return fig, axes


def plot_fid_raster_grid(
    raw_data_path: str | Path,
    *,
    dpi: int = 300,
    panel_scale: float = 2.0,
    cmap: str = "gray",
):
    set_paper_theme(style="white")
    payload = load_raw_plot_data(raw_data_path)
    images = payload["images"]
    if isinstance(images, torch.Tensor):
        image_array = images.detach().cpu().numpy()
    else:
        image_array = np.asarray(images)

    num_images = int(image_array.shape[0])
    cols = math.ceil(math.sqrt(num_images))
    rows = math.ceil(num_images / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(panel_scale * cols, panel_scale * rows), dpi=dpi)
    axes = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for ax, image in zip(axes, image_array):
        ax.imshow(image, cmap=cmap, vmin=0.0, vmax=1.0)
        ax.axis("off")

    for ax in axes[num_images:]:
        ax.axis("off")

    fig.tight_layout()
    return fig, axes


def _normalize_fid_series(
    series_collection: Mapping[str, Sequence[tuple[float, float]]] | Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if isinstance(series_collection, Mapping):
        normalized = []
        for label, points in series_collection.items():
            timesteps = [float(point[0]) for point in points]
            fids = [float(point[1]) for point in points]
            normalized.append({"label": str(label), "timesteps": timesteps, "fids": fids})
        return normalized

    normalized = []
    for idx, series in enumerate(series_collection):
        if "timesteps" in series and "fids" in series:
            timesteps = [float(value) for value in series["timesteps"]]
            fids = [float(value) for value in series["fids"]]
        elif "points" in series:
            timesteps = [float(point[0]) for point in series["points"]]
            fids = [float(point[1]) for point in series["points"]]
        else:
            raise ValueError("Each series must contain either ('timesteps', 'fids') or 'points'.")

        if len(timesteps) != len(fids):
            raise ValueError("Each FID series must have the same number of timesteps and fids.")
        if len(timesteps) < 2:
            raise ValueError("Each FID series must contain at least one pretraining point and one finetuning point.")

        normalized.append(
            {
                "label": str(series.get("label", f"Series {idx + 1}")),
                "timesteps": timesteps,
                "fids": fids,
            }
        )
    return normalized


def plot_fid_vs_timesteps(
    series_collection: Mapping[str, Sequence[tuple[float, float]]] | Sequence[Mapping[str, Any]],
    *,
    ax=None,
    palette: str = "colorblind",
    finetune_color: str = "#d62728",
    linewidth: float = 2.0,
    marker_size: float = 6.0,
    xlabel: str = "Pretraining Timestep",
    ylabel: str = "FID",
):
    set_paper_theme()
    normalized = _normalize_fid_series(series_collection)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.2, 4.0), dpi=300)
    else:
        fig = ax.figure

    palette_colors = sns.color_palette(palette, n_colors=len(normalized))
    handles: list[Line2D] = []

    for color, series in zip(palette_colors, normalized):
        timesteps = np.asarray(series["timesteps"], dtype=float)
        fids = np.asarray(series["fids"], dtype=float)

        ax.plot(
            timesteps[:-1],
            fids[:-1],
            color=color,
            linewidth=linewidth,
            marker="o",
            markersize=marker_size,
            label=series["label"],
        )
        ax.plot(
            timesteps[-2:],
            fids[-2:],
            color=finetune_color,
            linewidth=linewidth,
            marker="o",
            markersize=marker_size,
        )
        ax.scatter(
            timesteps[-1],
            fids[-1],
            color=finetune_color,
            s=marker_size**2 * 1.5,
            zorder=5,
        )

    for color, series in zip(palette_colors, normalized):
        handles.append(Line2D([0], [0], color=color, linewidth=linewidth, marker="o", label=series["label"]))
    handles.append(
        Line2D([0], [0], color=finetune_color, linewidth=linewidth, marker="o", label="MAML Finetune")
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(handles=handles, frameon=False, loc="best")
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig, ax

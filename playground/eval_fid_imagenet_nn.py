#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import random
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from absl import app
from ml_collections import ConfigDict, config_flags
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import utils
import wandb
from tqdm import tqdm

from playground.dataset.imagenet_nearest_dataset import ImageNetNearestContextDataset
from playground.models.conditional_model import ConditionalUNet, DDPM


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def build_eval_dataset(cfg: ConfigDict) -> ImageNetNearestContextDataset:
    return ImageNetNearestContextDataset(
        imagenet_dir=str(cfg.data.eval_dir),
        embeddings_path=str(cfg.data.eval_embeddings),
        index_path=str(cfg.data.eval_faiss_index),
        index_meta_path=str(getattr(cfg.data, "eval_faiss_meta", "")).strip() or None,
        k=int(cfg.data.cond_batch_size),
        image_size=int(cfg.data.image_size),
        normalize_minus1_1=True,
        random_query=bool(cfg.data.random_query_eval),
        seed=int(cfg.run.seed) + 999_983,
        return_metadata=bool(getattr(cfg.data, "return_metadata_eval", True)),
        faiss_nprobe=int(getattr(cfg.data, "faiss_nprobe_eval", 64)),
        random_horizontal_flip_prob=0.0,
    )


def build_loader(
    dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    drop_last: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
    )


def next_batch(loader: DataLoader, it: Optional[Iterator]):
    if it is None:
        it = iter(loader)
    try:
        batch = next(it)
    except StopIteration:
        it = iter(loader)
        batch = next(it)
    return batch, it


def split_query_and_context(
    batch,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    meta: Dict[str, torch.Tensor] = {}
    if isinstance(batch, dict):
        if "images" not in batch:
            raise KeyError("Expected key 'images' in metadata batch.")
        images = batch["images"]
        meta = {k: v for k, v in batch.items() if k != "images"}
    else:
        images = batch

    if not torch.is_tensor(images):
        raise TypeError(f"Expected tensor batch, got {type(images)}")
    if images.ndim != 5:
        raise ValueError(f"Expected images (B, K+1, C, H, W), got {tuple(images.shape)}")

    x = images[:, 0].contiguous().to(device, non_blocking=True)
    cond = images[:, 1:].contiguous().to(device, non_blocking=True)
    return x, cond, meta


def _device_arg_for_pytorch_fid(device: torch.device) -> str:
    if device.type == "cuda":
        idx = 0 if device.index is None else int(device.index)
        return f"cuda:{idx}"
    return "cpu"


def _pytorch_fid_available() -> bool:
    return importlib.util.find_spec("pytorch_fid") is not None


def _run_pytorch_fid_command(cmd: List[str]) -> Tuple[bool, str]:
    try:
        res = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except Exception as exc:
        return False, str(exc)
    text = (res.stdout or "") + ("\n" + res.stderr if res.stderr else "")
    return res.returncode == 0, text


def _collect_image_files_for_fid(root_dir: str) -> List[str]:
    root = Path(root_dir)
    if not root.exists():
        return []
    allowed_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in allowed_ext:
            files.append(str(p))
    files.sort()
    return files


def _build_pytorch_fid_stats_from_files(
    files: List[str],
    stats_path: Path,
    cfg: ConfigDict,
    device: torch.device,
) -> bool:
    if len(files) == 0:
        print("No reference images found to build pytorch-fid stats.")
        return False
    try:
        from pytorch_fid.fid_score import InceptionV3, calculate_activation_statistics
    except Exception as exc:
        print(f"Failed to import pytorch_fid API: {exc}")
        return False

    dims = int(getattr(cfg.fid, "pytorch_fid_dims", 2048))
    batch_size = int(getattr(cfg.fid, "pytorch_fid_batch_size", 64))
    num_workers = int(getattr(cfg.fid, "pytorch_fid_num_workers", 0))
    if dims not in InceptionV3.BLOCK_INDEX_BY_DIM:
        print(f"Unsupported pytorch-fid dims={dims}.")
        return False

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)
    mu, sigma = calculate_activation_statistics(
        files,
        model,
        batch_size=batch_size,
        dims=dims,
        device=device,
        num_workers=num_workers,
    )
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(stats_path), mu=mu, sigma=sigma)
    return True


def ensure_pytorch_fid_stats(cfg: ConfigDict, device: torch.device) -> Path:
    stats_path = Path(str(cfg.fid.pytorch_fid_stats_file))
    if stats_path.exists():
        try:
            with np.load(str(stats_path)) as data:
                if "mu" in data and "sigma" in data:
                    return stats_path
        except Exception:
            pass
        print(f"Existing stats file is invalid; rebuilding: {stats_path}")
        try:
            stats_path.unlink()
        except Exception:
            pass

    ref_dir = str(getattr(cfg.fid, "reference_dir", "")).strip() or str(cfg.data.eval_dir)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "pytorch_fid",
        "--save-stats",
        "--device",
        _device_arg_for_pytorch_fid(device),
        "--batch-size",
        str(int(getattr(cfg.fid, "pytorch_fid_batch_size", 64))),
        "--num-workers",
        str(int(getattr(cfg.fid, "pytorch_fid_num_workers", 0))),
        "--dims",
        str(int(getattr(cfg.fid, "pytorch_fid_dims", 2048))),
        ref_dir,
        str(stats_path),
    ]
    ok, output = _run_pytorch_fid_command(cmd)
    if ok:
        return stats_path

    print("`pytorch_fid --save-stats` failed, trying API fallback.")
    if output.strip():
        print(output.strip())
    files = _collect_image_files_for_fid(ref_dir)
    ok_api = _build_pytorch_fid_stats_from_files(files, stats_path, cfg, device)
    if not ok_api:
        raise RuntimeError("Unable to create pytorch-fid reference stats.")
    return stats_path


def run_pytorch_fid(
    generated_dir: Path,
    reference_stats: Path,
    cfg: ConfigDict,
    device: torch.device,
) -> float:
    cmd = [
        sys.executable,
        "-m",
        "pytorch_fid",
        "--device",
        _device_arg_for_pytorch_fid(device),
        "--batch-size",
        str(int(getattr(cfg.fid, "pytorch_fid_batch_size", 64))),
        "--num-workers",
        str(int(getattr(cfg.fid, "pytorch_fid_num_workers", 0))),
        "--dims",
        str(int(getattr(cfg.fid, "pytorch_fid_dims", 2048))),
        str(generated_dir),
        str(reference_stats),
    ]
    ok, output = _run_pytorch_fid_command(cmd)
    if not ok:
        raise RuntimeError(f"pytorch-fid failed:\n{output}")
    match = re.search(r"FID:\s*([0-9eE+\-.]+)", output)
    if match is None:
        raise RuntimeError(f"Could not parse FID from output:\n{output}")
    return float(match.group(1))


def _load_state_dict_with_report(
    module: torch.nn.Module,
    state_dict: Dict[str, torch.Tensor],
    strict: bool,
    label: str,
) -> bool:
    module_keys = set(module.state_dict().keys())
    provided_keys = set(state_dict.keys())
    if len(module_keys.intersection(provided_keys)) == 0:
        print(f"{label}: no matching parameter keys, skipping this load path.")
        return False
    try:
        load_result = module.load_state_dict(state_dict, strict=strict)
    except RuntimeError as exc:
        print(f"Failed loading {label}: {exc}")
        return False

    if hasattr(load_result, "missing_keys") and hasattr(load_result, "unexpected_keys"):
        if len(load_result.missing_keys) > 0:
            print(f"{label}: missing keys ({len(load_result.missing_keys)}), e.g. {load_result.missing_keys[:5]}")
        if len(load_result.unexpected_keys) > 0:
            print(
                f"{label}: unexpected keys ({len(load_result.unexpected_keys)}), "
                f"e.g. {load_result.unexpected_keys[:5]}"
            )
    return True


def load_checkpoint(ddpm: DDPM, cfg: ConfigDict, device: torch.device):
    ckpt_path = Path(str(cfg.eval.checkpoint))
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    strict = bool(getattr(cfg.eval, "strict", True))
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    loaded = False

    if isinstance(ckpt, dict):
        if "model" in ckpt:
            loaded = _load_state_dict_with_report(ddpm, ckpt["model"], strict, "checkpoint/model_as_ddpm")
            if not loaded:
                loaded = _load_state_dict_with_report(ddpm.model, ckpt["model"], strict, "checkpoint/model_as_unet")
        if (not loaded) and ("unet" in ckpt):
            loaded = _load_state_dict_with_report(ddpm.model, ckpt["unet"], strict, "checkpoint/unet")
        if (not loaded) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            loaded = _load_state_dict_with_report(ddpm, ckpt, strict, "checkpoint/raw_ddpm")
            if not loaded:
                loaded = _load_state_dict_with_report(ddpm.model, ckpt, strict, "checkpoint/raw_unet")
    else:
        raise RuntimeError(f"Unsupported checkpoint format: {type(ckpt)}")

    if not loaded:
        raise RuntimeError(f"Could not load weights from {ckpt_path}")
    return ckpt, ckpt_path


def maybe_override_arch_from_checkpoint(cfg: ConfigDict, ckpt) -> None:
    if not bool(getattr(cfg.eval, "use_checkpoint_cfg", True)):
        return
    if not isinstance(ckpt, dict):
        return
    ckpt_cfg = ckpt.get("cfg", None)
    if not isinstance(ckpt_cfg, dict):
        return
    model_cfg = ckpt_cfg.get("model", None)
    data_cfg = ckpt_cfg.get("data", None)
    if isinstance(model_cfg, dict):
        for key in [
            "in_channels",
            "base_channels",
            "channel_mults",
            "num_res_blocks",
            "dropout",
            "attn_resolutions",
            "cross_attn_resolutions",
            "num_heads",
            "cond_dim",
        ]:
            if key in model_cfg:
                setattr(cfg.model, key, model_cfg[key])
    if isinstance(data_cfg, dict) and "image_size" in data_cfg:
        cfg.data.image_size = data_cfg["image_size"]


def evaluate(cfg: ConfigDict) -> None:
    if (not _pytorch_fid_available()) and (importlib.util.find_spec("pytorch_fid") is None):
        raise RuntimeError("pytorch-fid is not installed. Install with: pip install pytorch-fid")

    if torch.cuda.is_available() and str(cfg.run.device).startswith("cuda"):
        device = torch.device(str(cfg.run.device))
    else:
        device = torch.device("cpu")
    set_seed(int(cfg.run.seed))

    # Read checkpoint first so we can optionally inherit model/data shape.
    ckpt_raw = torch.load(str(cfg.eval.checkpoint), map_location="cpu", weights_only=False)
    maybe_override_arch_from_checkpoint(cfg, ckpt_raw)

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
        time_scale=1000.0,
        cond_dim=cfg.model.cond_dim,
    ).to(device)
    ddpm = DDPM(
        model,
        log_snr_max=cfg.diffusion.log_snr_max,
        log_snr_min=cfg.diffusion.log_snr_min,
        p_uncond=float(getattr(cfg.diffusion, "p_uncond", 0.1)),
    ).to(device)
    _, ckpt_path = load_checkpoint(ddpm, cfg, device)
    ddpm.eval()

    eval_ds = build_eval_dataset(cfg)
    eval_loader = build_loader(
        eval_ds,
        batch_size=int(cfg.fid.batch_size),
        num_workers=int(cfg.data.num_workers),
        shuffle=True,
        drop_last=False,
    )

    out_root = Path(str(cfg.fid.generated_samples_dir))
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_root / f"{ckpt_path.stem}_{run_tag}"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {int(cfg.fid.num_samples)} samples to {out_dir} ...")
    data_iter: Optional[Iterator] = None
    num_samples = int(cfg.fid.num_samples)
    generated = 0
    with torch.no_grad():
        pbar = tqdm(total=num_samples, desc="Generating samples", leave=False)
        while generated < num_samples:
            batch, data_iter = next_batch(eval_loader, data_iter)
            _, cond, _ = split_query_and_context(batch, device)
            curr_batch = min(cond.shape[0], num_samples - generated)
            if curr_batch <= 0:
                break
            samples = ddpm.sample(
                batch_size=curr_batch,
                shape=(cfg.model.in_channels, cfg.data.image_size, cfg.data.image_size),
                device=device,
                steps=cfg.sample.steps,
                cond=cond[:curr_batch],
                cfg_scale=float(getattr(cfg.sample, "cfg_scale", 1.0)),
            )
            samples_01 = ((samples.clamp(-1, 1) + 1) / 2).cpu()
            for i, img in enumerate(samples_01):
                utils.save_image(img, out_dir / f"{generated + i:06d}.png")
            generated += curr_batch
            pbar.update(curr_batch)
        pbar.close()

    stats_path = ensure_pytorch_fid_stats(cfg, device)
    print(f"Using reference stats: {stats_path}")
    fid_value = run_pytorch_fid(out_dir, stats_path, cfg, device)
    print(f"pytorch-FID: {fid_value:.4f}")

    if bool(getattr(cfg.wandb, "use", False)):
        run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.run_name,
            dir=cfg.wandb.dir,
            config=cfg.to_dict(),
        )
        run.log(
            {
                "eval/fid": fid_value,
                "eval/num_samples": int(cfg.fid.num_samples),
                "eval/checkpoint": str(ckpt_path),
                "eval/generated_dir": str(out_dir),
                "eval/reference_stats": str(stats_path),
            }
        )
        run.finish()

    if not bool(getattr(cfg.fid, "keep_generated_samples", False)):
        shutil.rmtree(out_dir, ignore_errors=True)
        print("Deleted generated samples directory (keep_generated_samples=False).")
    else:
        print(f"Kept generated samples at: {out_dir}")


_CONFIG = config_flags.DEFINE_config_file(
    "config",
    default="playground/configs/eval_fid_imagenet_nn.py",
    help_string="Path to a ml_collections config file.",
)


def main(argv=None):
    del argv
    cfg = _CONFIG.value
    evaluate(cfg)


if __name__ == "__main__":
    app.run(main)

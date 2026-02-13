#!/usr/bin/env python3
"""
Compute DINOv2 embeddings for an ImageNet ImageFolder split.

Outputs (in output_dir):
- <prefix>_embeddings.npy     float16/float32, shape (N, D)
- <prefix>_labels.npy         int64, shape (N,)
- <prefix>_source_indices.npy int64, shape (N,)
- <prefix>_paths.txt          N lines, relative image paths
- <prefix>_meta.json

Example:
  python playground/dataset/build_imagenet_dinov2_embeddings.py \
      --imagenet-dir "$IMAGENET_TRAIN_DIR" \
      --output-dir "$IMAGENET_EMBEDDINGS_DIR" \
      --split-name train
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.amp as amp
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm


class IndexedImageFolder(datasets.ImageFolder):
    """ImageFolder that also returns the source index."""

    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        return img, label, index


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build DINOv2 embeddings for ImageNet ImageFolder.")
    p.add_argument("--imagenet-dir", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--split-name", type=str, default="train")
    p.add_argument("--model-name", type=str, default="dinov2_vitb14")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--output-dtype", type=str, default="float16", choices=["float16", "float32"])
    p.add_argument("--resize-size", type=int, default=256)
    p.add_argument("--crop-size", type=int, default=224)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--subset-seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device_str.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_str)
    return torch.device("cpu")


def _imagenet_eval_transform(resize_size: int, crop_size: int):
    return transforms.Compose(
        [
            transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def _select_subset_indices(n: int, max_samples: int, seed: int) -> Optional[List[int]]:
    if max_samples <= 0 or max_samples >= n:
        return None
    rng = np.random.default_rng(seed)
    idxs = rng.choice(n, size=max_samples, replace=False)
    idxs.sort()
    return idxs.tolist()


def _prefix(split_name: str, model_name: str) -> str:
    return f"imagenet_{split_name}_{model_name}"


def main() -> None:
    args = parse_args()

    imagenet_dir = Path(args.imagenet_dir)
    if not imagenet_dir.exists():
        raise FileNotFoundError(f"ImageNet directory not found: {imagenet_dir}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = _prefix(args.split_name, args.model_name)
    emb_path = output_dir / f"{prefix}_embeddings.npy"
    labels_path = output_dir / f"{prefix}_labels.npy"
    src_idx_path = output_dir / f"{prefix}_source_indices.npy"
    paths_txt_path = output_dir / f"{prefix}_paths.txt"
    meta_path = output_dir / f"{prefix}_meta.json"

    out_paths = [emb_path, labels_path, src_idx_path, paths_txt_path, meta_path]
    if (not args.overwrite) and any(p.exists() for p in out_paths):
        existing = [str(p) for p in out_paths if p.exists()]
        raise FileExistsError(
            "Some output files already exist. Use --overwrite to replace them:\n"
            + "\n".join(existing)
        )

    transform = _imagenet_eval_transform(args.resize_size, args.crop_size)
    base_ds = IndexedImageFolder(root=str(imagenet_dir), transform=transform)

    subset_indices = _select_subset_indices(len(base_ds), args.max_samples, args.subset_seed)
    if subset_indices is not None:
        ds = Subset(base_ds, subset_indices)
        print(f"Using subset: {len(ds)} / {len(base_ds)} images")
    else:
        ds = base_ds

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    device = _resolve_device(args.device)
    print(f"Loading DINOv2 model '{args.model_name}' on {device}...")
    model = torch.hub.load("facebookresearch/dinov2", args.model_name)
    model = model.to(device)
    model.eval()

    total = len(ds)
    emb_mm = None
    labels_mm = np.lib.format.open_memmap(labels_path, mode="w+", dtype=np.int64, shape=(total,))
    src_idx_mm = np.lib.format.open_memmap(src_idx_path, mode="w+", dtype=np.int64, shape=(total,))

    np_dtype = np.float16 if args.output_dtype == "float16" else np.float32

    pos = 0
    use_amp = bool(args.amp and device.type == "cuda")
    with torch.no_grad():
        for images, labels, source_indices in tqdm(loader, desc="Embedding ImageNet"):
            images = images.to(device, non_blocking=True)
            with amp.autocast(device_type=device.type, enabled=use_amp):
                feats = model(images)
            feats = feats.float().cpu().numpy()

            if emb_mm is None:
                emb_mm = np.lib.format.open_memmap(
                    emb_path,
                    mode="w+",
                    dtype=np_dtype,
                    shape=(total, feats.shape[1]),
                )

            bsz = feats.shape[0]
            emb_mm[pos : pos + bsz] = feats.astype(np_dtype, copy=False)
            labels_mm[pos : pos + bsz] = labels.cpu().numpy().astype(np.int64, copy=False)
            src_idx_mm[pos : pos + bsz] = source_indices.cpu().numpy().astype(np.int64, copy=False)
            pos += bsz

    if emb_mm is None:
        raise RuntimeError("No embeddings were computed (empty dataset?).")
    emb_mm.flush()
    labels_mm.flush()
    src_idx_mm.flush()

    source_indices_np = np.load(src_idx_path, mmap_mode="r")
    with paths_txt_path.open("w", encoding="utf-8") as f:
        for src_idx in source_indices_np:
            full_path = base_ds.samples[int(src_idx)][0]
            rel_path = os.path.relpath(full_path, str(imagenet_dir))
            f.write(rel_path + "\n")

    meta = {
        "imagenet_dir": str(imagenet_dir),
        "split_name": args.split_name,
        "model_name": args.model_name,
        "num_samples": int(total),
        "embedding_dim": int(np.load(emb_path, mmap_mode="r").shape[1]),
        "embedding_dtype": args.output_dtype,
        "subset_indices_used": subset_indices is not None,
        "resize_size": int(args.resize_size),
        "crop_size": int(args.crop_size),
        "class_count": int(len(base_ds.classes)),
        "class_names": list(base_ds.classes),
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Saved embeddings and metadata:")
    print(f"  {emb_path}")
    print(f"  {labels_path}")
    print(f"  {src_idx_path}")
    print(f"  {paths_txt_path}")
    print(f"  {meta_path}")


if __name__ == "__main__":
    main()

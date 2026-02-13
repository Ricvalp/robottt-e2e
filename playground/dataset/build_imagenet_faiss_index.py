#!/usr/bin/env python3
"""
Build a FAISS index from precomputed ImageNet embeddings.

Supports:
- exact flat index (`flat`)
- IVF-Flat index (`ivf_flat`) for scalable search

Example:
  python playground/dataset/build_imagenet_faiss_index.py \
      --embeddings-path "$IMAGENET_EMBEDDINGS_DIR/imagenet_train_dinov2_vitb14_embeddings.npy" \
      --output-index "$IMAGENET_FAISS_DIR/imagenet_train_dinov2_vitb14_ivf.index" \
      --index-type ivf_flat --nlist 4096 --nprobe 64 --normalize
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import faiss
import numpy as np
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build FAISS index for ImageNet embeddings.")
    p.add_argument("--embeddings-path", type=str, required=True)
    p.add_argument("--output-index", type=str, required=True)
    p.add_argument("--output-meta", type=str, default="")
    p.add_argument("--index-type", type=str, default="ivf_flat", choices=["flat", "ivf_flat"])
    p.add_argument("--metric", type=str, default="l2", choices=["l2", "ip"])
    p.add_argument("--normalize", action="store_true", help="L2-normalize vectors before train/add/search.")
    p.add_argument("--nlist", type=int, default=4096)
    p.add_argument("--nprobe", type=int, default=64)
    p.add_argument("--train-samples", type=int, default=200000)
    p.add_argument("--add-batch-size", type=int, default=50000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _metric_enum(metric: str) -> int:
    return faiss.METRIC_L2 if metric == "l2" else faiss.METRIC_INNER_PRODUCT


def _normalize_if_needed(x: np.ndarray, normalize: bool) -> np.ndarray:
    if not normalize:
        return x
    y = np.asarray(x, dtype=np.float32, order="C")
    faiss.normalize_L2(y)
    return y


def _sample_training_vectors(
    embeddings: np.ndarray,
    train_samples: int,
    seed: int,
    normalize: bool,
) -> np.ndarray:
    n = embeddings.shape[0]
    m = min(int(train_samples), n)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=m, replace=False)
    x = np.asarray(embeddings[idx], dtype=np.float32, order="C")
    return _normalize_if_needed(x, normalize)


def main() -> None:
    args = parse_args()

    embeddings_path = Path(args.embeddings_path)
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")

    output_index = Path(args.output_index)
    output_index.parent.mkdir(parents=True, exist_ok=True)

    output_meta = Path(args.output_meta) if args.output_meta else output_index.with_suffix(output_index.suffix + ".meta.json")

    if not args.overwrite and (output_index.exists() or output_meta.exists()):
        raise FileExistsError(
            f"Output already exists (use --overwrite):\n{output_index}\n{output_meta}"
        )

    embeddings = np.load(embeddings_path, mmap_mode="r")
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape {embeddings.shape}")
    n, dim = embeddings.shape
    metric_enum = _metric_enum(args.metric)

    print(f"Embeddings: N={n}, D={dim}, dtype={embeddings.dtype}")
    print(f"Index type: {args.index_type}, metric: {args.metric}, normalize={args.normalize}")

    if args.index_type == "flat":
        if args.metric == "l2":
            index = faiss.IndexFlatL2(dim)
        else:
            index = faiss.IndexFlatIP(dim)
    else:
        if args.metric == "l2":
            quantizer = faiss.IndexFlatL2(dim)
        else:
            quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, int(args.nlist), metric_enum)

        print("Training IVF index...")
        train_x = _sample_training_vectors(
            embeddings,
            train_samples=args.train_samples,
            seed=args.seed,
            normalize=bool(args.normalize),
        )
        index.train(train_x)
        if not index.is_trained:
            raise RuntimeError("Index failed to train.")

    add_bs = int(args.add_batch_size)
    for start in tqdm(range(0, n, add_bs), desc="Adding vectors"):
        end = min(start + add_bs, n)
        x = np.asarray(embeddings[start:end], dtype=np.float32, order="C")
        x = _normalize_if_needed(x, bool(args.normalize))
        index.add(x)

    if hasattr(index, "nprobe"):
        index.nprobe = int(args.nprobe)

    faiss.write_index(index, str(output_index))

    meta = {
        "embeddings_path": str(embeddings_path),
        "num_vectors": int(n),
        "dim": int(dim),
        "index_type": args.index_type,
        "index_class": type(index).__name__,
        "metric": args.metric,
        "normalize_embeddings": bool(args.normalize),
        "nlist": int(args.nlist) if args.index_type == "ivf_flat" else None,
        "nprobe": int(args.nprobe) if hasattr(index, "nprobe") else None,
        "train_samples": int(args.train_samples) if args.index_type == "ivf_flat" else None,
    }
    with output_meta.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved FAISS index: {output_index}")
    print(f"Saved metadata: {output_meta}")


if __name__ == "__main__":
    main()

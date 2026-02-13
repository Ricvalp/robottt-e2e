#!/usr/bin/env python3
"""
Build and persist a FAISS index for CIFAR-100 enriched embeddings.

Source dataset:
  https://huggingface.co/datasets/renumics/cifar100-enriched

By default this builds an exact L2 index (`faiss.IndexFlatL2`) over the
`embedding` vectors and saves:
- `<output_dir>/cifar100_enriched_<split>.index`
- `<output_dir>/cifar100_enriched_<split>.meta.json`
- `<output_dir>/cifar100_enriched_<split>_fine_labels.npy`
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import faiss
import numpy as np
from datasets import load_dataset

DATASET_NAME = "renumics/cifar100-enriched"


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return x / norms


def default_index_paths(output_dir: str | Path, split: str) -> Tuple[Path, Path]:
    base = Path(output_dir)
    index_path = base / f"cifar100_enriched_{split}.index"
    meta_path = base / f"cifar100_enriched_{split}.meta.json"
    return index_path, meta_path


def load_enriched_embeddings(
    split: str,
    cache_dir: str | None = None,
) -> Tuple[object, np.ndarray]:
    ds = load_dataset(DATASET_NAME, split=split, cache_dir=cache_dir)
    embeddings = np.stack(ds["embedding"]).astype(np.float32, copy=False)
    return ds, embeddings


def build_faiss_index(
    embeddings: np.ndarray,
    *,
    metric: str = "l2",
    normalize_embeddings: bool = False,
) -> Tuple[faiss.Index, np.ndarray]:
    metric_key = metric.lower().strip()
    emb = np.ascontiguousarray(embeddings.astype(np.float32, copy=False))

    if normalize_embeddings:
        emb = np.ascontiguousarray(_l2_normalize(emb), dtype=np.float32)

    dim = int(emb.shape[1])
    if metric_key in {"l2", "flatl2", "indexflatl2"}:
        index = faiss.IndexFlatL2(dim)
    elif metric_key in {"ip", "inner_product", "flatip", "indexflatip"}:
        index = faiss.IndexFlatIP(dim)
    else:
        raise ValueError(f"Unsupported metric '{metric}'. Use 'l2' or 'ip'.")

    index.add(emb)
    return index, emb


def build_and_save_index(
    *,
    split: str = "train",
    output_dir: str | Path = "playground/data/cifar100_faiss",
    cache_dir: str | None = None,
    metric: str = "l2",
    normalize_embeddings: bool = False,
    save_labels: bool = True,
) -> Tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path, meta_path = default_index_paths(output_dir, split)

    ds, embeddings = load_enriched_embeddings(split=split, cache_dir=cache_dir)
    index, indexed_embeddings = build_faiss_index(
        embeddings,
        metric=metric,
        normalize_embeddings=normalize_embeddings,
    )

    faiss.write_index(index, str(index_path))

    meta: Dict[str, object] = {
        "dataset_name": DATASET_NAME,
        "split": split,
        "num_samples": int(indexed_embeddings.shape[0]),
        "embedding_dim": int(indexed_embeddings.shape[1]),
        "index_class": type(index).__name__,
        "metric": metric,
        "normalize_embeddings": bool(normalize_embeddings),
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if save_labels:
        labels = np.asarray(ds["fine_label"], dtype=np.int64)
        labels_path = output_dir / f"cifar100_enriched_{split}_fine_labels.npy"
        np.save(labels_path, labels)

    return index_path, meta_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build FAISS index for CIFAR100 enriched embeddings.")
    p.add_argument("--split", type=str, default="train", choices=["train", "test"])
    p.add_argument("--output-dir", type=str, default="playground/data/cifar100_faiss")
    p.add_argument("--cache-dir", type=str, default=None)
    p.add_argument("--metric", type=str, default="l2", choices=["l2", "ip"])
    p.add_argument(
        "--normalize-embeddings",
        action="store_true",
        help="L2-normalize embeddings before indexing (useful with IP).",
    )
    p.add_argument(
        "--no-save-labels",
        action="store_true",
        help="Do not save fine-label array alongside the index.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    index_path, meta_path = build_and_save_index(
        split=args.split,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        metric=args.metric,
        normalize_embeddings=bool(args.normalize_embeddings),
        save_labels=not bool(args.no_save_labels),
    )
    print(f"FAISS index written to: {index_path}")
    print(f"Meta written to: {meta_path}")


if __name__ == "__main__":
    main()

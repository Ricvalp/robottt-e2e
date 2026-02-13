#!/usr/bin/env python3
"""
ImageNet nearest-neighbor context dataset backed by FAISS search.

Each sample:
- query image (random by default)
- K nearest images in embedding space

Returned sample tensor shape:
- (K+1, 3, H, W), ordered as [query, nn1, ..., nnK]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, get_worker_info
from torchvision import datasets, transforms


class ImageNetNearestContextDataset(Dataset):
    def __init__(
        self,
        *,
        imagenet_dir: str,
        embeddings_path: str,
        index_path: str,
        index_meta_path: Optional[str] = None,
        k: int = 8,
        image_size: int = 64,
        normalize_minus1_1: bool = True,
        random_query: bool = True,
        seed: int = 0,
        return_metadata: bool = False,
        faiss_nprobe: Optional[int] = None,
        random_horizontal_flip_prob: float = 0.0,
    ) -> None:
        super().__init__()
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")

        self.k = int(k)
        self.imagenet_dir = str(imagenet_dir)
        self.embeddings_path = str(embeddings_path)
        self.index_path = str(index_path)
        self.index_meta_path = index_meta_path
        self.seed = int(seed)
        self.random_query = bool(random_query)
        self.return_metadata = bool(return_metadata)
        self.normalize_minus1_1 = bool(normalize_minus1_1)
        self.faiss_nprobe = faiss_nprobe

        transform_ops: List[transforms.Compose] = [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
        ]
        if random_horizontal_flip_prob > 0:
            transform_ops.append(transforms.RandomHorizontalFlip(p=float(random_horizontal_flip_prob)))
        transform_ops.append(transforms.ToTensor())
        if normalize_minus1_1:
            transform_ops.append(transforms.Lambda(lambda x: x * 2 - 1))

        self.image_transform = transforms.Compose(transform_ops)
        self.image_ds = datasets.ImageFolder(root=self.imagenet_dir, transform=self.image_transform)
        self.num_samples = len(self.image_ds)

        emb_path = Path(self.embeddings_path)
        if not emb_path.exists():
            raise FileNotFoundError(f"Embeddings not found: {emb_path}")
        self.embeddings = np.load(emb_path, mmap_mode="r")
        if self.embeddings.ndim != 2:
            raise ValueError(f"Expected 2D embeddings, got shape {self.embeddings.shape}")
        if self.embeddings.shape[0] != self.num_samples:
            raise ValueError(
                f"Embeddings length ({self.embeddings.shape[0]}) != dataset size ({self.num_samples})."
            )

        self.index_normalized = False
        if self.index_meta_path is None:
            default_meta = Path(self.index_path + ".meta.json")
            if default_meta.exists():
                self.index_meta_path = str(default_meta)
        if self.index_meta_path is not None and Path(self.index_meta_path).exists():
            with open(self.index_meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.index_normalized = bool(meta.get("normalize_embeddings", False))

        # Lazy-loaded per-process to avoid fork/pickle issues.
        self._index = None
        self._index_pid = None

        self.labels = np.asarray(self.image_ds.targets, dtype=np.int64)
        self.label_names = list(self.image_ds.classes)

    def __len__(self) -> int:
        return self.num_samples

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_index"] = None
        state["_index_pid"] = None
        return state

    def _ensure_index(self):
        pid = os.getpid()
        if self._index is not None and self._index_pid == pid:
            return
        index_file = Path(self.index_path)
        if not index_file.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_file}")

        self._index = faiss.read_index(str(index_file))
        if self.faiss_nprobe is not None and hasattr(self._index, "nprobe"):
            self._index.nprobe = int(self.faiss_nprobe)
        self._index_pid = pid

    def _query_index_for_item(self, index: int) -> int:
        if not self.random_query:
            return int(index % self.num_samples)

        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        seed = (self.seed + worker_id * 1_000_003 + index * 97_009) % (2**32)
        rng = np.random.default_rng(seed)
        return int(rng.integers(0, self.num_samples))

    def _search_neighbors(self, query_index: int) -> Tuple[List[int], List[float]]:
        self._ensure_index()

        q = np.asarray(self.embeddings[query_index], dtype=np.float32).reshape(1, -1)
        if self.index_normalized:
            faiss.normalize_L2(q)

        search_k = min(self.num_samples, max(self.k + 16, self.k + 1))
        neighbors: List[int] = []
        neighbor_dists: List[float] = []

        while True:
            dists, nn_idx = self._index.search(q, search_k)
            neighbors.clear()
            neighbor_dists.clear()
            for idx, dist in zip(nn_idx[0].tolist(), dists[0].tolist()):
                if idx < 0:
                    continue
                if int(idx) == int(query_index):
                    continue
                neighbors.append(int(idx))
                neighbor_dists.append(float(dist))
                if len(neighbors) == self.k:
                    return neighbors, neighbor_dists

            if search_k >= self.num_samples:
                break
            search_k = min(self.num_samples, search_k * 2)

        raise RuntimeError(
            f"Could not find {self.k} neighbors for query index {query_index}. "
            f"Found {len(neighbors)}."
        )

    def label_name(self, label: int) -> str:
        if 0 <= int(label) < len(self.label_names):
            return self.label_names[int(label)]
        return str(int(label))

    def __getitem__(self, index: int):
        query_index = self._query_index_for_item(index)
        neighbor_indices, distances = self._search_neighbors(query_index)

        all_indices = [query_index] + neighbor_indices
        images = torch.stack([self.image_ds[i][0] for i in all_indices], dim=0)

        if not self.return_metadata:
            return images

        query_label = int(self.labels[query_index])
        neighbor_labels = [int(self.labels[i]) for i in neighbor_indices]
        return {
            "images": images,
            "query_index": torch.tensor(query_index, dtype=torch.long),
            "neighbor_indices": torch.tensor(neighbor_indices, dtype=torch.long),
            "query_label": torch.tensor(query_label, dtype=torch.long),
            "neighbor_labels": torch.tensor(neighbor_labels, dtype=torch.long),
            "distances": torch.tensor(distances, dtype=torch.float32),
        }


def _to_01(x: torch.Tensor, normalize_minus1_1: bool) -> torch.Tensor:
    if normalize_minus1_1:
        return ((x + 1.0) / 2.0).clamp(0, 1)
    return x.clamp(0, 1)


def visualize_batch(
    batch: Dict[str, torch.Tensor],
    *,
    label_names: List[str],
    normalize_minus1_1: bool,
    out_path: Path,
    rows: int,
) -> None:
    images = batch["images"]
    query_labels = batch["query_label"]
    neighbor_labels = batch["neighbor_labels"]
    distances = batch["distances"]

    bsz, k_plus_one, _, _, _ = images.shape
    n_rows = min(rows, bsz)
    n_cols = k_plus_one

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.2 * n_cols, 2.2 * n_rows),
        squeeze=False,
    )

    def label_name(x: int) -> str:
        if 0 <= int(x) < len(label_names):
            return label_names[int(x)]
        return str(int(x))

    for r in range(n_rows):
        row_imgs = _to_01(images[r], normalize_minus1_1)
        for c in range(n_cols):
            img = row_imgs[c].permute(1, 2, 0).cpu().numpy()
            axes[r, c].imshow(img)
            axes[r, c].axis("off")
            if c == 0:
                q_lbl = int(query_labels[r].item())
                axes[r, c].set_title(f"Q: {label_name(q_lbl)}", fontsize=8)
            else:
                n_lbl = int(neighbor_labels[r, c - 1].item())
                dist = float(distances[r, c - 1].item())
                axes[r, c].set_title(f"N{c}: {label_name(n_lbl)}\nD={dist:.2f}", fontsize=7)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def label_agreement_at_k(ds: ImageNetNearestContextDataset, num_queries: int = 128) -> float:
    if not ds.return_metadata:
        raise ValueError("Dataset must be created with return_metadata=True for diagnostics.")

    total = 0
    same = 0
    for i in range(num_queries):
        sample = ds[i]
        q = int(sample["query_label"].item())
        nn = sample["neighbor_labels"].tolist()
        same += sum(int(lbl == q) for lbl in nn)
        total += len(nn)

    return float(same / max(total, 1))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ImageNet nearest-neighbor dataset smoke test.")
    p.add_argument("--imagenet-dir", type=str, required=True)
    p.add_argument("--embeddings-path", type=str, required=True)
    p.add_argument("--index-path", type=str, required=True)
    p.add_argument("--index-meta-path", type=str, default="")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--rows", type=int, default=4)
    p.add_argument("--faiss-nprobe", type=int, default=64)
    p.add_argument("--deterministic-query", action="store_true")
    p.add_argument("--num-queries-metric", type=int, default=64)
    p.add_argument(
        "--plot-path",
        type=str,
        default="playground/outputs/imagenet_nearest_neighbors_grid.png",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    ds = ImageNetNearestContextDataset(
        imagenet_dir=args.imagenet_dir,
        embeddings_path=args.embeddings_path,
        index_path=args.index_path,
        index_meta_path=args.index_meta_path if args.index_meta_path else None,
        k=args.k,
        image_size=args.image_size,
        normalize_minus1_1=True,
        random_query=not bool(args.deterministic_query),
        seed=args.seed,
        return_metadata=True,
        faiss_nprobe=args.faiss_nprobe,
    )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    batch = next(iter(loader))
    print(f"batch['images'].shape = {tuple(batch['images'].shape)}")

    agreement = label_agreement_at_k(ds, num_queries=args.num_queries_metric)
    print(f"Approx label agreement@{args.k} over {args.num_queries_metric} queries: {agreement:.3f}")

    out_path = Path(args.plot_path)
    visualize_batch(
        batch,
        label_names=ds.label_names,
        normalize_minus1_1=ds.normalize_minus1_1,
        out_path=out_path,
        rows=args.rows,
    )
    print(f"Saved nearest-neighbor grid to: {out_path}")


if __name__ == "__main__":
    main()

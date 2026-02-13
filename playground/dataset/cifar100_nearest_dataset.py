#!/usr/bin/env python3
"""
CIFAR-100 nearest-neighbor context dataset using FAISS over enriched embeddings.

Each dataset sample is built as:
- 1 query image sampled at random (or by deterministic index)
- K nearest images in embedding space (excluding the query itself)

Returned tensor shape per sample:
- (K+1, 3, H, W), ordered as [query, nn1, nn2, ..., nnK]

Typical DataLoader batch shape:
- (B, K+1, 3, H, W)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, get_worker_info
from torchvision import transforms

try:
    from playground.dataset.build_cifar100_faiss_index import (
        build_and_save_index,
        default_index_paths,
    )
except ImportError:
    from build_cifar100_faiss_index import (  # type: ignore
        build_and_save_index,
        default_index_paths,
    )


DATASET_NAME = "renumics/cifar100-enriched"


def _load_meta(meta_path: Path) -> Dict[str, object]:
    if not meta_path.exists():
        return {}
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


class CIFAR100NearestContextDataset(Dataset):
    """
    Dataset returning query + K nearest neighbors in embedding space.

    Parameters
    ----------
    k : int
        Number of nearest neighbors used as context.
    split : str
        Dataset split of `renumics/cifar100-enriched`.
    index_path : str | None
        Path to saved FAISS index. If missing and `build_index_if_missing=True`,
        the index is built automatically.
    meta_path : str | None
        Path to metadata json produced by index builder.
    index_dir : str
        Directory used for default index/meta paths.
    image_size : int
        Output image size.
    normalize_minus1_1 : bool
        If True, map image tensors from [0,1] to [-1,1].
    random_query : bool
        If True, each __getitem__ samples a random query image.
    seed : int
        Base seed for deterministic random query sampling.
    return_metadata : bool
        If True, return dict with tensor + labels/indices/distances.
    """

    def __init__(
        self,
        *,
        k: int,
        split: str = "train",
        index_path: str | None = None,
        meta_path: str | None = None,
        index_dir: str = "playground/data/cifar100_faiss",
        cache_dir: str | None = None,
        image_size: int = 32,
        normalize_minus1_1: bool = True,
        random_query: bool = True,
        seed: int = 0,
        return_metadata: bool = False,
        build_index_if_missing: bool = True,
    ) -> None:
        super().__init__()
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")

        self.k = int(k)
        self.split = split
        self.seed = int(seed)
        self.random_query = bool(random_query)
        self.return_metadata = bool(return_metadata)
        self.normalize_minus1_1 = bool(normalize_minus1_1)

        self.ds = load_dataset(DATASET_NAME, split=split, cache_dir=cache_dir)
        self.num_samples = len(self.ds)
        self.fine_labels = np.asarray(self.ds["fine_label"], dtype=np.int64)

        label_feature = self.ds.features.get("fine_label")
        self.label_names = list(getattr(label_feature, "names", []))

        if index_path is None or meta_path is None:
            default_index, default_meta = default_index_paths(index_dir, split)
            index_path = str(default_index) if index_path is None else index_path
            meta_path = str(default_meta) if meta_path is None else meta_path

        self.index_path = Path(index_path)
        self.meta_path = Path(meta_path)

        if not self.index_path.exists():
            if not build_index_if_missing:
                raise FileNotFoundError(
                    f"Index not found at {self.index_path}. Set build_index_if_missing=True or build it first."
                )
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            build_and_save_index(
                split=split,
                output_dir=self.index_path.parent,
                cache_dir=cache_dir,
                metric="l2",
                normalize_embeddings=False,
                save_labels=True,
            )

        self.index = faiss.read_index(str(self.index_path))
        if self.index.ntotal != self.num_samples:
            raise ValueError(
                f"Index sample count ({self.index.ntotal}) != dataset size ({self.num_samples}). "
                "Rebuild index for the same split/dataset version."
            )

        self.meta = _load_meta(self.meta_path)
        self.index_normalized = bool(self.meta.get("normalize_embeddings", False))

        ops: List[transforms.Compose] = []
        ops.append(transforms.Resize((image_size, image_size)))
        ops.append(transforms.ToTensor())
        if self.normalize_minus1_1:
            ops.append(transforms.Lambda(lambda x: x * 2 - 1))
        self.image_transform = transforms.Compose(ops)

    def __len__(self) -> int:
        return self.num_samples

    def _query_index_for_item(self, index: int) -> int:
        if not self.random_query:
            return int(index % self.num_samples)

        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        # Deterministic per (worker, index) for reproducibility across epochs.
        seed = (self.seed + worker_id * 1_000_003 + index * 97_009) % (2**32)
        rng = np.random.default_rng(seed)
        return int(rng.integers(0, self.num_samples))

    def _search_neighbors(self, query_index: int) -> Tuple[List[int], List[float]]:
        q = self.index.reconstruct(int(query_index)).reshape(1, -1).astype(np.float32, copy=False)
        if self.index_normalized:
            q /= np.clip(np.linalg.norm(q, axis=1, keepdims=True), 1e-12, None)

        # Search slightly more than K+1 to robustly skip self index.
        search_k = min(self.num_samples, max(self.k + 8, self.k + 1))
        dists, nn_idx = self.index.search(q, search_k)

        neighbors: List[int] = []
        neighbor_dists: List[float] = []
        for idx, dist in zip(nn_idx[0].tolist(), dists[0].tolist()):
            if idx < 0:
                continue
            if int(idx) == int(query_index):
                continue
            neighbors.append(int(idx))
            neighbor_dists.append(float(dist))
            if len(neighbors) == self.k:
                break

        if len(neighbors) < self.k:
            raise RuntimeError(
                f"Could not find {self.k} neighbors for query index {query_index}. "
                f"Found {len(neighbors)}."
            )

        return neighbors, neighbor_dists

    def _load_image_tensor(self, idx: int) -> torch.Tensor:
        image = self.ds[int(idx)]["full_image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        return self.image_transform(image)

    def label_name(self, label: int) -> str:
        if 0 <= int(label) < len(self.label_names):
            return self.label_names[int(label)]
        return str(label)

    def __getitem__(self, index: int):
        query_index = self._query_index_for_item(index)
        neighbor_indices, distances = self._search_neighbors(query_index)

        all_indices = [query_index] + neighbor_indices
        images = torch.stack([self._load_image_tensor(i) for i in all_indices], dim=0)

        if not self.return_metadata:
            return images

        query_label = int(self.fine_labels[query_index])
        neighbor_labels = [int(self.fine_labels[i]) for i in neighbor_indices]

        return {
            "images": images,  # (K+1, C, H, W): [query, neighbors]
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
    images = batch["images"]  # (B, K+1, C, H, W)
    query_labels = batch["query_label"]  # (B,)
    neighbor_labels = batch["neighbor_labels"]  # (B, K)
    distances = batch["distances"]  # (B, K)

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
                axes[r, c].set_title(f"Q: {label_name(q_lbl)}", fontsize=9)
            else:
                n_lbl = int(neighbor_labels[r, c - 1].item())
                dist = float(distances[r, c - 1].item())
                axes[r, c].set_title(f"N{c}: {label_name(n_lbl)}\nL2={dist:.2f}", fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def label_agreement_at_k(ds: CIFAR100NearestContextDataset, num_queries: int = 256) -> float:
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
    p = argparse.ArgumentParser(description="Nearest-neighbor CIFAR100 context dataset smoke test.")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--split", type=str, default="train", choices=["train", "test"])
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--rows", type=int, default=4)
    p.add_argument("--index-dir", type=str, default="playground/data/cifar100_faiss")
    p.add_argument("--index-path", type=str, default=None)
    p.add_argument("--meta-path", type=str, default=None)
    p.add_argument("--cache-dir", type=str, default=None)
    p.add_argument("--deterministic-query", action="store_true")
    p.add_argument("--no-normalize-minus1-1", action="store_true")
    p.add_argument("--num-queries-metric", type=int, default=128)
    p.add_argument(
        "--plot-path",
        type=str,
        default="playground/outputs/cifar100_nearest_neighbors_grid.png",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    dataset = CIFAR100NearestContextDataset(
        k=args.k,
        split=args.split,
        index_path=args.index_path,
        meta_path=args.meta_path,
        index_dir=args.index_dir,
        cache_dir=args.cache_dir,
        image_size=args.image_size,
        normalize_minus1_1=not bool(args.no_normalize_minus1_1),
        random_query=not bool(args.deterministic_query),
        seed=args.seed,
        return_metadata=True,
        build_index_if_missing=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    batch = next(iter(loader))
    print(f"batch['images'].shape = {tuple(batch['images'].shape)}")

    agreement = label_agreement_at_k(dataset, num_queries=args.num_queries_metric)
    print(f"Approx label agreement@{args.k} over {args.num_queries_metric} queries: {agreement:.3f}")

    plot_path = Path(args.plot_path)
    visualize_batch(
        batch,
        label_names=dataset.label_names,
        normalize_minus1_1=dataset.normalize_minus1_1,
        out_path=plot_path,
        rows=args.rows,
    )
    print(f"Saved nearest-neighbor grid to: {plot_path}")


if __name__ == "__main__":
    main()

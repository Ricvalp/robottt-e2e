#!/usr/bin/env python3
"""
Compute and save FID reference statistics for a dataset using a trained classifier.

Usage:
    python compute_fid_stats.py --dataset cifar100 --classifier_path playground/classifier_checkpoints/cifar100/cifar100_classifier.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from fid_utils import load_classifier_for_fid, compute_reference_stats, save_fid_stats


def main():
    parser = argparse.ArgumentParser(description="Compute FID reference statistics")
    parser.add_argument("--dataset", type=str, required=True, choices=["mnist", "cifar100", "cifar10", "imagenet64", "celeba64"])
    parser.add_argument("--classifier_path", type=str, required=True)
    parser.add_argument("--output", type=str, default="playground/fid_stats_classifier.json")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--classifier_width", type=int, default=64)
    parser.add_argument("--classifier_num_blocks", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset configuration
    dataset_configs = {
        "mnist": {"num_classes": 10, "in_channels": 1, "image_size": 28},
        "cifar10": {"num_classes": 10, "in_channels": 3, "image_size": 32},
        "cifar100": {"num_classes": 100, "in_channels": 3, "image_size": 32},
        "imagenet64": {"num_classes": 1000, "in_channels": 3, "image_size": 64},
        "celeba64": {"num_classes": 40, "in_channels": 3, "image_size": 64},  # 40 attributes
    }
    cfg = dataset_configs[args.dataset]
    image_size = args.image_size if args.image_size else cfg["image_size"]

    # Build transform
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])

    # Load dataset
    print(f"Loading {args.dataset} dataset...")
    if args.dataset == "mnist":
        ds = datasets.MNIST(args.data_root, train=True, download=True, transform=transform)
    elif args.dataset == "cifar10":
        ds = datasets.CIFAR10(args.data_root, train=True, download=True, transform=transform)
    elif args.dataset == "cifar100":
        ds = datasets.CIFAR100(args.data_root, train=True, download=True, transform=transform)
    elif args.dataset == "celeba64":
        ds = datasets.CelebA(args.data_root, split="train", download=True, transform=transform)
    elif args.dataset == "imagenet64":
        imagenet_path = Path(args.data_root) / "imagenet" / "train"
        if not imagenet_path.exists():
            raise ValueError(f"ImageNet not found at {imagenet_path}. Set --data_root correctly.")
        ds = datasets.ImageFolder(str(imagenet_path), transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Load classifier
    print(f"Loading classifier from {args.classifier_path}...")
    classifier = load_classifier_for_fid(
        checkpoint_path=args.classifier_path,
        width=args.classifier_width,
        num_blocks=args.classifier_num_blocks,
        num_classes=cfg["num_classes"],
        in_channels=cfg["in_channels"],
        device=device,
    )

    # Compute stats
    print("Computing reference statistics...")
    mu, sigma = compute_reference_stats(classifier, loader, device)
    print(f"Feature dimension: {mu.shape[0]}")

    # Save
    save_fid_stats(args.output, args.dataset, mu, sigma)
    print(f"Done! Stats saved to {args.output}")


if __name__ == "__main__":
    main()

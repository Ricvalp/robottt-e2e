"""
FID (Fréchet Inception Distance) computation utilities.

Supports two modes:
1. Custom classifier (SmallResNet) - for MNIST/CIFAR
2. InceptionV3 (pretrained) - for ImageNet/CelebA (standard FID)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple, Optional, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import scipy.linalg


# -------------------------
# InceptionV3 Feature Extractor
# -------------------------

class InceptionV3Features(nn.Module):
    """
    InceptionV3 feature extractor for FID computation.
    Uses the pool3 layer (2048-dim) as in the original FID paper.
    """
    def __init__(self):
        super().__init__()
        from torchvision.models import inception_v3, Inception_V3_Weights
        
        inception = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        
        # Extract layers up to pool3 (before aux and fc)
        self.Conv2d_1a_3x3 = inception.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = inception.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = inception.Conv2d_2b_3x3
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Conv2d_3b_1x1 = inception.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = inception.Conv2d_4a_3x3
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Mixed_5b = inception.Mixed_5b
        self.Mixed_5c = inception.Mixed_5c
        self.Mixed_5d = inception.Mixed_5d
        self.Mixed_6a = inception.Mixed_6a
        self.Mixed_6b = inception.Mixed_6b
        self.Mixed_6c = inception.Mixed_6c
        self.Mixed_6d = inception.Mixed_6d
        self.Mixed_6e = inception.Mixed_6e
        self.Mixed_7a = inception.Mixed_7a
        self.Mixed_7b = inception.Mixed_7b
        self.Mixed_7c = inception.Mixed_7c
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Inception expects 299x299, we resize if needed
        if x.shape[2] != 299 or x.shape[3] != 299:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        
        # Inception expects specific normalization
        # Input should be [0, 1], we convert to Inception's expected range
        x = x * 2 - 1  # [0, 1] -> [-1, 1]
        
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = self.maxpool1(x)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = self.maxpool2(x)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)
        x = self.avgpool(x)
        x = x.flatten(1)  # 2048-dim features
        return x


_inception_cache: Optional[InceptionV3Features] = None


def get_inception_model(device: torch.device) -> InceptionV3Features:
    """Get cached InceptionV3 model for FID computation."""
    global _inception_cache
    if _inception_cache is None:
        print("Loading InceptionV3 for FID computation...")
        _inception_cache = InceptionV3Features()
    return _inception_cache.to(device).eval()


# -------------------------
# Feature Extraction
# -------------------------

def get_features_from_classifier(
    classifier: nn.Module,
    images: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Extract penultimate layer features from SmallResNet classifier.
    """
    classifier.eval()
    with torch.no_grad():
        x = images.to(device)
        x = classifier.stem(x)
        x = classifier.blocks(x)
        x = classifier.pool(x).flatten(1)
    return x


def get_features_from_inception(
    inception: InceptionV3Features,
    images: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Extract 2048-dim features from InceptionV3.
    Images should be in [0, 1] range.
    """
    inception.eval()
    with torch.no_grad():
        x = images.to(device)
        features = inception(x)
    return features


# -------------------------
# Statistics
# -------------------------

def compute_statistics(features: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean and covariance of features."""
    features_np = features.cpu().numpy()
    mu = np.mean(features_np, axis=0)
    sigma = np.cov(features_np, rowvar=False)
    return mu, sigma


def compute_fid(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray, eps: float = 1e-6) -> float:
    """
    Compute Fréchet distance between two Gaussian distributions.
    FID = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1 @ sigma2))
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2
    
    covmean, _ = scipy.linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return float(diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


# -------------------------
# Reference Stats Computation
# -------------------------

def compute_reference_stats(
    feature_extractor: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    extractor_type: Literal["classifier", "inception"] = "classifier",
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute FID reference statistics from a dataset."""
    all_features = []
    feature_extractor.eval()
    
    get_features = get_features_from_classifier if extractor_type == "classifier" else get_features_from_inception
    
    for images, _ in tqdm(dataloader, desc="Computing reference stats"):
        # Ensure images are in [0, 1] range
        if images.min() < 0:
            images = (images + 1) / 2
        features = get_features(feature_extractor, images, device)
        all_features.append(features.cpu())
    
    all_features = torch.cat(all_features, dim=0)
    return compute_statistics(all_features)


def compute_reference_stats_inception(
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute FID reference statistics using InceptionV3."""
    inception = get_inception_model(device)
    return compute_reference_stats(inception, dataloader, device, extractor_type="inception")


# -------------------------
# FID Computation from Samples
# -------------------------

def compute_fid_from_samples(
    feature_extractor: nn.Module,
    samples: torch.Tensor,
    reference_stats: Tuple[np.ndarray, np.ndarray],
    device: torch.device,
    batch_size: int = 64,
    extractor_type: Literal["classifier", "inception"] = "classifier",
) -> float:
    """
    Compute FID between generated samples and reference statistics.
    
    Args:
        feature_extractor: Model for feature extraction
        samples: (N, C, H, W) tensor of generated images in [-1, 1] range
        reference_stats: (mu, sigma) tuple from reference dataset
        device: torch device
        batch_size: batch size for feature extraction
        extractor_type: "classifier" or "inception"
    
    Returns:
        FID score
    """
    # Convert samples from [-1, 1] to [0, 1]
    samples = (samples + 1) / 2
    
    all_features = []
    feature_extractor.eval()
    
    get_features = get_features_from_classifier if extractor_type == "classifier" else get_features_from_inception
    
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i+batch_size]
        features = get_features(feature_extractor, batch, device)
        all_features.append(features.cpu())
    
    all_features = torch.cat(all_features, dim=0)
    mu_gen, sigma_gen = compute_statistics(all_features)
    mu_ref, sigma_ref = reference_stats
    
    return compute_fid(mu_ref, sigma_ref, mu_gen, sigma_gen)


def compute_fid_inception(
    samples: torch.Tensor,
    reference_stats: Tuple[np.ndarray, np.ndarray],
    device: torch.device,
    batch_size: int = 64,
) -> float:
    """Compute FID using InceptionV3 (standard FID)."""
    inception = get_inception_model(device)
    return compute_fid_from_samples(inception, samples, reference_stats, device, batch_size, "inception")


# -------------------------
# Stats File I/O
# -------------------------

def load_fid_stats(stats_file: str, dataset_key: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load precomputed FID statistics from JSON file."""
    with open(stats_file, 'r') as f:
        data = json.load(f)
    
    if dataset_key in data:
        stats = data[dataset_key]
    else:
        # Backward compatibility
        stats = data
    
    mu = np.array(stats["mu"])
    sigma = np.array(stats["sigma"])
    return mu, sigma


def has_fid_stats(stats_file: str, dataset_key: str) -> bool:
    """Check if FID stats exist for a dataset."""
    stats_path = Path(stats_file)
    if not stats_path.exists():
        return False
    with open(stats_path, 'r') as f:
        data = json.load(f)
    return dataset_key in data


def save_fid_stats(stats_file: str, dataset_key: str, mu: np.ndarray, sigma: np.ndarray) -> None:
    """Save FID statistics to JSON file."""
    stats_path = Path(stats_file)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    
    if stats_path.exists():
        with open(stats_path, 'r') as f:
            data = json.load(f)
        if "mu" in data and "sigma" in data and dataset_key not in data:
            data = {"mnist": data}
    else:
        data = {}
    
    data[dataset_key] = {
        "mu": mu.tolist(),
        "sigma": sigma.tolist(),
    }
    
    with open(stats_path, 'w') as f:
        json.dump(data, f)
    print(f"Saved FID stats for '{dataset_key}' to {stats_file}")


# -------------------------
# Classifier Loading (for MNIST/CIFAR)
# -------------------------

def load_classifier_for_fid(
    checkpoint_path: str,
    width: int = 64,
    num_blocks: int = 4,
    num_classes: int = 100,
    in_channels: int = 3,
    device: torch.device = None,
) -> nn.Module:
    """Load a trained SmallResNet classifier for FID computation."""
    from playground.models.classifier_model import SmallResNet
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = SmallResNet(
        width=width,
        num_blocks=num_blocks,
        num_classes=num_classes,
        in_channels=in_channels,
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model

#!/bin/bash
# Environment variables for playground training scripts
# Source this file before running training: source env.sh

# =============================================================================
# DATA DIRECTORIES
# =============================================================================

# Root directory for downloaded datasets (MNIST, CIFAR, CelebA)
export PLAYGROUND_DATA_ROOT="${PLAYGROUND_DATA_ROOT:-data}"

# ImageNet directories (required for train_imagenet.py)
export IMAGENET_TRAIN_DIR="${IMAGENET_TRAIN_DIR:-/mnt/external_storage/torchvision_ImageFolder/train}"
export IMAGENET_VAL_DIR="${IMAGENET_VAL_DIR:-/mnt/external_storage/torchvision_ImageFolder/val}"

# =============================================================================
# OUTPUT DIRECTORIES
# =============================================================================

# Base directory for checkpoints
export PLAYGROUND_CHECKPOINT_DIR="${PLAYGROUND_CHECKPOINT_DIR:-checkpoints}"

# Base directory for outputs (samples, FID stats, etc.)
export PLAYGROUND_OUTPUT_DIR="${PLAYGROUND_OUTPUT_DIR:-.}"

# =============================================================================
# FID / CLASSIFIER PATHS
# =============================================================================

# FID statistics file
export FID_STATS_FILE="${FID_STATS_FILE:-${PLAYGROUND_OUTPUT_DIR}/fid_stats_classifier.json}"

# CIFAR-100 classifier checkpoint (for FID computation)
export CIFAR100_CLASSIFIER_CKPT="${CIFAR100_CLASSIFIER_CKPT:-${PLAYGROUND_CHECKPOINT_DIR}/classifier_cifar100/cifar100_classifier.pt}"

# MNIST classifier checkpoint (for counting/FID)
export MNIST_CLASSIFIER_CKPT="${MNIST_CLASSIFIER_CKPT:-${PLAYGROUND_CHECKPOINT_DIR}/classifier_mnist/classifier_epoch_010.pt}"

echo "Environment variables loaded:"
echo "  PLAYGROUND_DATA_ROOT=$PLAYGROUND_DATA_ROOT"
echo "  IMAGENET_TRAIN_DIR=$IMAGENET_TRAIN_DIR"
echo "  IMAGENET_VAL_DIR=$IMAGENET_VAL_DIR"
echo "  PLAYGROUND_CHECKPOINT_DIR=$PLAYGROUND_CHECKPOINT_DIR"
echo "  PLAYGROUND_OUTPUT_DIR=$PLAYGROUND_OUTPUT_DIR"
echo "  FID_STATS_FILE=$FID_STATS_FILE"

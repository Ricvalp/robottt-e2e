#!/bin/bash
# Environment variables for playground training scripts
# Source this file before running training: source env.sh

# =============================================================================
# DATA DIRECTORIES
# =============================================================================

# Root directory for downloaded datasets (MNIST, CIFAR, CelebA)
export PLAYGROUND_DATA_ROOT="data"

# ImageNet directories (required for train_imagenet.py)
export IMAGENET_TRAIN_DIR="/mnt/external_storage/torchvision_ImageFolder/train"
export IMAGENET_VAL_DIR="/mnt/external_storage/torchvision_ImageFolder/val"

# ImageNet nearest-neighbor cache root (embeddings + FAISS index)
# Change to scratch path on cluster if needed.
export IMAGENET_NN_CACHE_DIR="/mnt/external_storage/imagenet_nn"
export IMAGENET_EMBEDDINGS_DIR="${IMAGENET_NN_CACHE_DIR}/embeddings"
export IMAGENET_FAISS_DIR="${IMAGENET_NN_CACHE_DIR}/faiss"

# =============================================================================
# OUTPUT DIRECTORIES
# =============================================================================

# Base directory for checkpoints
export PLAYGROUND_CHECKPOINT_DIR="playground/checkpoints"

# Base directory for outputs (samples, FID stats, etc.)
export PLAYGROUND_OUTPUT_DIR="playground/outputs"

# =============================================================================
# FID / CLASSIFIER PATHS
# =============================================================================

# FID statistics file
export FID_STATS_FILE="${PLAYGROUND_OUTPUT_DIR}/fid_stats_classifier.json"
export IMAGENET_FID_STATS_FILE="${PLAYGROUND_OUTPUT_DIR}/fid_stats_inception.json"

# CIFAR-100 classifier checkpoint (for FID computation)
export CIFAR100_CLASSIFIER_CKPT="${PLAYGROUND_CHECKPOINT_DIR}/classifier_checkpoints/cifar100/cifar100_classifier.pt"

# MNIST classifier checkpoint (for counting/FID)
export MNIST_CLASSIFIER_CKPT="${PLAYGROUND_CHECKPOINT_DIR}/classifier_checkpoints/mnist/mnist_classifier.pt"

echo "Environment variables loaded:"
echo "  PLAYGROUND_DATA_ROOT=$PLAYGROUND_DATA_ROOT"
echo "  IMAGENET_TRAIN_DIR=$IMAGENET_TRAIN_DIR"
echo "  IMAGENET_VAL_DIR=$IMAGENET_VAL_DIR"
echo "  PLAYGROUND_CHECKPOINT_DIR=$PLAYGROUND_CHECKPOINT_DIR"
echo "  PLAYGROUND_OUTPUT_DIR=$PLAYGROUND_OUTPUT_DIR"
echo "  FID_STATS_FILE=$FID_STATS_FILE"
echo "  IMAGENET_NN_CACHE_DIR=$IMAGENET_NN_CACHE_DIR"
echo "  IMAGENET_EMBEDDINGS_DIR=$IMAGENET_EMBEDDINGS_DIR"
echo "  IMAGENET_FAISS_DIR=$IMAGENET_FAISS_DIR"
echo "  IMAGENET_FID_STATS_FILE=$IMAGENET_FID_STATS_FILE"
echo "  CIFAR100_CLASSIFIER_CKPT=$CIFAR100_CLASSIFIER_CKPT"

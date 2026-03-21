#!/bin/bash
# Environment variables for ICIL perceiver pretraining.
# Source before running training: source env.sh

# =============================================================================
# DATA DIRECTORIES
# =============================================================================

# Root of cached RLBench dense H5 variations.
export QRD_CACHE_ROOT="data/train-val-split"

# Root of cached QuickDraw faiss index.
export QRD_INDEX_ROOT="metrics/index"

# =============================================================================
# OUTPUT DIRECTORIES
# =============================================================================

# Parent directory for run outputs (each run uses a subdirectory named by wandb run id).
export QRD_OUTPUT_PARENT_DIR="/mnt/external_storage/robotics/quick_robot_draw/runs/outputs"

# Parent directory for checkpoints (each run uses a subdirectory named by wandb run id).
export QRD_CHECKPOINT_PARENT_DIR="/mnt/external_storage/robotics/quick_robot_draw/runs/checkpoints"

# =============================================================================
# PROFILING OUTPUT DIRECTORIES
# =============================================================================

# Base directory for all profiling traces.
export QRD_PROFILE_TRACE_DIR="/mnt/external_storage/robotics/quick_robot_draw/runs/profiles"

# Distinct trace file names within the same profiling directory.
export QRD_PRETRAIN_PROFILE_TRACE_FILE="pretrain_trace.json"
export QRD_PROFILE_TRACE_FILE="trace.json"

# =============================================================================
# WANDB
# =============================================================================

export WANDB_PROJECT="qrd-pretrain"
export WANDB_ENTITY="ricvalp"
export WANDB_MODE="online"

echo "[env.sh] QRD_CACHE_ROOT=${QRD_CACHE_ROOT}"
echo "[env.sh] QRD_OUTPUT_PARENT_DIR=${QRD_OUTPUT_PARENT_DIR}"
echo "[env.sh] QRD_CHECKPOINT_PARENT_DIR=${QRD_CHECKPOINT_PARENT_DIR}"
echo "[env.sh] QRD_PROFILE_TRACE_DIR=${QRD_PROFILE_TRACE_DIR}"
echo "[env.sh] QRD_PRETRAIN_PROFILE_TRACE_FILE=${QRD_PRETRAIN_PROFILE_TRACE_FILE}"
echo "[env.sh] QRD_PROFILE_TRACE_FILE=${QRD_PROFILE_TRACE_FILE}"
echo "[env.sh] WANDB_PROJECT=${WANDB_PROJECT}"
echo "[env.sh] WANDB_ENTITY=${WANDB_ENTITY}"
echo "[env.sh] WANDB_MODE=${WANDB_MODE}"

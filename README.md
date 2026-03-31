# MAML-ICIL for QuickDraw

This repository implements a sketch-generation pipeline on Google Quick, Draw! that combines:

1. diffusion pretraining with in-context conditioning on similar sketches
2. second-order MAML finetuning so the model can adapt its behavior after a few gradient steps on a small support set

The codebase is built around QuickDraw sketches stored as processed stroke sequences, a similarity-based episode builder, an encoder-decoder DiT-style diffusion policy, and evaluation scripts with qualitative plots and FID on rasterized sketches.

## Repository Summary

The main workflow is:

1. preprocess raw QuickDraw sketches into normalized stroke sequences
2. train a 1-channel ResNet18 classifier / feature extractor on rasterized sketches
3. compute per-sketch embeddings and build one FAISS index per class/family
4. pretrain an encoder-decoder diffusion policy with in-context conditioning on nearest-neighbor prompt sketches
5. finetune that pretrained policy with MAML using leave-one-out support losses on the in-context demos
6. evaluate pretrained and MAML-finetuned checkpoints with qualitative sampling and FID

## Main Entry Points

### Pretraining

- `diffusion/pretrain_encoder_decoder.py`
- config: `configs/diffusion/pretrain_encoder_decoder.py`

This script trains a diffusion encoder-decoder policy on flattened in-context episodes built with `EpisodeBuilderSimilar`, where:

- one query sketch is selected
- the prompt set is retrieved from FAISS nearest neighbors within the same class
- the full token sequence is composed as:
  - `[SEP, prompt_1, SEP, ..., prompt_K, SEP, RESET, SEP, query, STOP]`
- the collator samples a random split point in the query sketch
- the model sees:
  - `context`: all prompt sketches
  - `history`: the observed query prefix
  - `actions`: the next `horizon` query tokens to denoise

### MAML Finetuning

- `diffusion/train_maml_icil.py`
- config: `configs/diffusion/train_maml_icil.py`

This is the reference MAML script. It:

- loads a pretrained encoder-decoder checkpoint
- reconstructs the policy and diffusion scheduler settings from the checkpoint config
- builds MAML tasks from `QuickDrawEpisodesMAML`
- performs second-order MAML with leave-one-out support losses
- adapts only a small subset of decoder parameters in the inner loop

Two optimized variants are also present:

- `diffusion/train_maml_icil_v2.py`
  - batches the selected LOO support examples within each task
  - converts task episodes to torch tensors earlier in the pipeline
- `diffusion/train_maml_icil_v3.py`
  - preprocesses each outer batch into task-local support/query tensors before entering `maml_step`

All three MAML scripts use the same config file by default.

### Evaluation

- pretrained checkpoints: `diffusion/eval_encoder_decoder.py`
- MAML checkpoints: `diffusion/eval_maml_icil.py`

The MAML eval script supports:

- `empty_sketches`
- `partial_sketches`
- `many_samples`
- `fid`
  - adapts fast parameters for each task before sampling
- `fid_no_adaptation`
  - evaluates the finetuned checkpoint without any inner-loop adaptation

## Data Representation

### Dataset

The current cached dataset manifest in `data/train-val-split/DatasetManifest.json` indicates:

- 345 QuickDraw families/classes
- 311 train families
- 34 val families

### Preprocessed Sketch Format

`dataset/preprocess.py` converts each sketch into:

- `absolute`: normalized absolute coordinates
- `deltas`: first-order coordinate differences
- `pen`: pen-down / pen-up indicators
- `length`

### Token Conventions

There are two token spaces in the repo:

- 7-channel episode tokens for episode composition:
  - `[x, y, pen, not_pen, sep, reset, stop]`
- 6-channel model tokens after removing the `reset` channel:
  - `[x, y, pen, not_pen, sep, stop]`

The pretraining collator strips the reset channel before passing inputs to the model.

## Similarity Retrieval

Prompt selection is similarity-driven, not random.

The retrieval pipeline is:

1. rasterize sketches to images
2. embed them with a 1-channel ResNet18
3. L2-normalize embeddings
4. build one FAISS `IndexFlatIP` per family

Because the embeddings are normalized, inner-product search is effectively cosine similarity search.

Relevant scripts:

- `metrics/train_resnet18.py`
- `metrics/compute_embeddings.py`
- `metrics/build_faiss_index.py`

## Model Architecture

The policy is defined in `diffusion/policies/dit_encdec_policy.py`.

### Encoder-Decoder Structure

- context sketches are encoded with a Transformer encoder
- query history plus noisy future action tokens are processed by a Transformer decoder
- the decoder uses:
  - self-attention over `[history, noisy_actions]`
  - cross-attention to encoder memory
  - AdaLN-style modulation from the diffusion timestep embedding

### Components

- `point_feature_proj`
  - context token projection
- `history_feature_proj`
  - query-history projection
- `action_encoder`
  - projection for noised action tokens
- `world_time_embedder`
  - sinusoidal positional embedding for sketch/query positions
- `diffusion_time_embedder`
  - sinusoidal embedding for DDPM timestep
- `diffusion_proj`
  - MLP on diffusion-time embedding
- `output_head`
  - predicts diffusion target on the final `horizon` slice

### Decoder Conditioning

The decoder blocks use:

- self-attention
- cross-attention
- MLP
- per-block AdaLN modulation MLP producing shift/scale/gates for all three sublayers

The final decoder normalization is also AdaLN-style.

## Training Objectives

### Pretraining Objective

The diffusion objective is the same standard denoising objective used in the policy:

- sample a timestep `t`
- noise the clean action chunk `x0`
- predict one of:
  - `epsilon`
  - `x0`
  - `v`

The current default config uses:

- `prediction_type = "v_prediction"`

### MAML Objective

For each task:

1. sample `K_maml = K_pretrain + 1` context sketches plus one query sketch
2. choose `m = num_loo_per_task` leave-one-out support problems
3. for each support problem:
   - hold out one context sketch
   - condition on the remaining context sketches
   - compute the same diffusion loss used during pretraining
4. average the support losses
5. update fast parameters with one or more inner steps
6. build the outer query loss on the real query sketch, conditioned on a subset of size `K_pretrain`
7. average query losses across tasks to get `meta_loss`

Important implementation choices in the current setup:

- fast params:
  - last 25% of decoder blocks
  - decoder MLP weights
  - decoder AdaLN modulation MLPs
  - final decoder normalization/modulation MLP
- fast params exclude:
  - encoder
  - attention weights
  - early/input projection layers
- slow params / outer optimizer:
  - decoder
  - input projections
  - output head
  - diffusion conditioning MLPs
  - encoder frozen by default

## Configuration and Environment

The repo uses `ml_collections.ConfigDict` configs.

Machine-dependent paths are read from `env.sh` via environment variables such as:

- `QRD_CACHE_ROOT`
- `QRD_INDEX_ROOT`
- `QRD_RESNET_CHECKPOINT_PARENT_DIR`
- `QRD_OUTPUT_PARENT_DIR`
- `QRD_CHECKPOINT_PARENT_DIR`
- `QRD_PROFILE_TRACE_DIR`

Recommended usage:

```bash
source env.sh
```

## Typical Commands

### Pretrain

```bash
PYTHONPATH=. python diffusion/pretrain_encoder_decoder.py \
  --config=configs/diffusion/pretrain_encoder_decoder.py
```

### MAML Finetune

```bash
PYTHONPATH=. python diffusion/train_maml_icil.py \
  --config=configs/diffusion/train_maml_icil.py \
  --config.finetune.pretrained_checkpoint=eval_checkpoints/36a6n1g8/policy_epoch_010.pt
```

### Evaluate Pretrained Model

```bash
PYTHONPATH=. python diffusion/eval_encoder_decoder.py \
  --config=configs/diffusion/eval_encoder_decoder.py \
  --config.checkpoint.path=eval_checkpoints/36a6n1g8/policy_epoch_010.pt
```

### Evaluate MAML Model

```bash
PYTHONPATH=. python diffusion/eval_maml_icil.py \
  --config=configs/diffusion/eval_maml_icil.py \
  --config.checkpoint.path=eval_checkpoints/toxvieal/latest.pt
```

### FID Only on Validation Split

```bash
TASKS='("fid",)'
SPLITS='("val",)'

PYTHONPATH=. python diffusion/eval_maml_icil.py \
  --config=configs/diffusion/eval_maml_icil.py \
  --config.checkpoint.path=eval_checkpoints/toxvieal/latest.pt \
  --config.eval.tasks="$TASKS" \
  --config.eval.fid.splits="$SPLITS"
```

## Existing Artifacts in This Workspace

The workspace currently contains:

- pretrained checkpoints
  - `eval_checkpoints/36a6n1g8/policy_epoch_010.pt`
  - `eval_checkpoints/7r4lvabj/policy_epoch_011.pt`
- one MAML finetune checkpoint
  - `eval_checkpoints/toxvieal/latest.pt`
- ResNet18 feature extractor checkpoints up to step 90000
  - `metrics/checkpoints/resnet18_step90000.pt`
- saved eval summaries under `outputs/`

## Observed Runs and Saved Evaluation Results

The saved output summaries currently show:

| Run ID | Type | Key config recovered from checkpoint | Saved FID summary |
| --- | --- | --- | --- |
| `36a6n1g8` | pretrain | `K=4`, `hidden_dim=256`, `horizon=8`, `prediction_type=v_prediction` | train `4.4271`, val `5.2525` at `10` inference steps |
| `7r4lvabj` | pretrain | `K=4`, `hidden_dim=256`, `horizon=16`, `prediction_type=v_prediction` | one saved eval: train `4.4165`, val `5.1134`; a second repeated eval on the same checkpoint produced train `4.2165`, val `5.3854` |
| `toxvieal` | MAML finetune | initialized from `7r4lvabj`, `K_maml=5`, `outer_context_size=4`, `num_loo_per_task=2`, `inner_steps=1`, `inner_lr=3e-4` | adapted FID summary: train `4.7179`, val `5.8017` at `10` inference steps |

Notes:

- The visible pretrained checkpoints use `hidden_dim=256`, even though the current default pretraining config file now specifies `hidden_dim=512`.
- Checkpoints are the source of truth for model reconstruction during evaluation and MAML finetuning.
- `fid_no_adaptation` has been implemented in `eval_maml_icil.py`, but no saved summary for that task is currently present in `outputs/`.

## Practical Notes

- Pretraining dataset filtering currently uses `max_seq_len` for episode acceptance; `max_query_len` is mainly used for qualitative sampling length.
- MAML uses math SDPA on CUDA for second-order gradient stability.
- `train_maml_icil_v2.py` and `train_maml_icil_v3.py` are experimental performance-oriented variants and should be treated as engineering alternatives, not separate methods.

## Dependencies

There is no pinned requirements file in this repo. Based on imports, you should expect to need at least:

- PyTorch
- torchvision
- diffusers
- faiss
- ml-collections
- absl-py
- wandb
- tqdm
- matplotlib
- scipy
- numpy

Depending on the dataset backend you may also need LMDB-related dependencies.

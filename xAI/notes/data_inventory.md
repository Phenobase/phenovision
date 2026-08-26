# xAI Project: Available Training Data Inventory

Last updated: 2026-06-01

## Overview

This document catalogs training run data available for the "pretraining as preadaptation" analysis, which compares how different pretraining strategies (evolutionary histories) affect fine-tuning dynamics (adaptation to a new environment).

## Two-Phase Retraining Runs (March 2026): 214K Subset

The two-phase decomposition runs (PlantCLEF vs MAE, the basis for the `xai_*` figures and `analysis_log.md`) did **not** train on the full v1.1.0 split. To keep each run to ~9 hours, training was capped to a **random 214,000-image subset** (~10% of the full training split), applied identically to both models.

- **Actual training script**: `xAI/py/xai_train.py` (the R `xAI/R/xai_train.R` was an earlier design draft and was not the version run)
- **Subset flag**: `--max_train_samples 214000`, set in `xAI/scripts/submit_xai_plantclef.sh` and `submit_xai_mae.sh`
- **Sampling**: random **without replacement** (seeded RNG in `xai_train.py`), logged at runtime as `[TEST MODE] Limited training to 214000 samples`
- **Not curated**: a plain random 10% sample for speed — not stratified, not taxonomically filtered

| Data | Count | Source |
|------|-------|--------|
| Full v1.1.0 training split | 2,141,091 | `data/inat/train_v1.1.0.csv` |
| **Subset actually trained on** | **214,000** | random seeded sample (~10%) |
| Validation (frequent, fixed) | 10,000 | random subset of `data/inat/val_v1.1.0.csv` (713,697 rows) |
| Representation snapshots | 1,000 | first 1,000 of the val subset |

**Why this matters for interpretation:** Both runs used the *same* 214K cap, so PlantCLEF vs MAE remains a fair comparison — the only difference is the pretrained backbone. But absolute fitness/AUC values are from a 10%-data regime and should not be compared directly against the full-data March 2025 runs below.

**Run summaries** (`xAI/output/{mae,plantclef}_run/run_summary.csv`): 15 epochs, ~541 min each.

**Two batch sizes (one per phase):** The two phases run at different batch sizes because Phase 1 freezes the backbone (no backbone gradients → less VRAM → larger batch fits):

- **Phase 1** (frozen backbone, head only): batch size **768** → 278 batches/epoch (214000 ÷ 768)
- **Phase 2** (full model unfrozen): batch size **384** → 557 batches/epoch (214000 ÷ 384)

Both are confirmed by the `[x/N]` denominators in the SLURM logs (278 and 557). **Caveat:** the original `run_summary.csv` recorded only `batch_size` (= the Phase 1 value, 768) and silently omitted Phase 2's 384. This was a reporting bug — Phase 2 (the "evolution" phase analyzed in all the figures) actually ran at 384. Fixed in `xai_train.py` to write `phase1_batch_size` and `phase2_batch_size` separately; the two existing `run_summary.csv` files predate the fix and still show the single `batch_size` field.

**Layer-wise LR decay (Phase 2):** Phase 2 uses layer-wise learning-rate decay via `param_groups_lrd` (`PlantCLEF2022/util/lr_decay.py`) with `layer_decay = 0.65`. The per-layer scale is `0.65 ** (num_layers − layer_id)` with `num_layers = 24 blocks + 1 = 25`, so input-side layers get a near-zero LR and output-side layers get the full base LR (a ~17,000× spread):

| Layer | LR scale (× base LR) |
|-------|----------------------|
| `patch_embed` / `cls_token` / `pos_embed` (input) | 0.65²⁵ ≈ 3.7e-5 |
| `blocks.0` (deepest, near input) | 0.65²⁴ ≈ 5.7e-5 |
| `blocks.22` | 0.65² ≈ 0.42 |
| `blocks.23` (shallowest block) | 0.65¹ = 0.65 |
| `head` (output) | 0.65⁰ = 1.0 |

Phase 1 trains the head only (backbone frozen), so layer-wise decay is moot there. **Interpretation caveat:** this LR schedule partly *imposes* the "conserved deep layers / evolvable shallow layers" pattern seen in the weight-displacement analysis — deep layers barely move in part because their effective LR is ~10⁻⁵×, not purely because the pretrained features were already optimal. Keep this confound in mind for the "conserved core / evolvable interface" framing.

**Data provenance:** v1.1.0 snapshot (the original March 2025 training data was lost; see "Incomplete Runs (Lost)" below).

## Complete Runs: PlantCLEF vs MAE (March 2025)

These two runs form the primary comparison. Both were run on the same date with identical hyperparameters, differing only in pretraining strategy.

### PlantCLEF ("Virtual Taxonomist")
- **Log**: `output/plantclef_2025_03_09/pheno_vit_plantclef-60645825.out` (6,014 lines)
- **Checkpoints**: `output/plantclef_2025_03_09/checkpoints/vit_finetuned_plantclef_epoch{1..46}.pt`
- **Pretraining**: PlantCLEF 2022 competition model, trained on 2.9M plant images for species classification
- **Hypothesis**: Species classification relies on recognizing reproductive structures, creating a "preadaptation" for phenology detection
- **Epochs**: 47 (46 complete, partial epoch 47)
- **Batches/epoch**: 2,400 (logged every 20 batches = 121 data points/epoch)
- **Checkpoint format**: Full model object (requires `models_vit` on Python path), state_dict with 296 keys

### MAE ("ImageNet Generalist")
- **Log**: `output/mae_2025_03_09/pheno_vit_mae-60645826.out` (6,035 lines)
- **Checkpoints**: `output/mae_2025_03_09/checkpoints/vit_finetuned_mae_epoch{1..46}.pt`
- **Pretraining**: Masked Autoencoder (MAE), self-supervised on ImageNet
- **Epochs**: 47 (46 complete, partial epoch 47)
- **Batches/epoch**: 2,400 (logged every 20 batches)
- **Checkpoint format**: Same as PlantCLEF

### Shared Configuration (Both Runs)
- **Architecture**: ViT-L/16 (Vision Transformer Large, 16x16 patches)
- **Input**: 224x224 RGB images
- **Output**: 2 classes (flowers, fruits), sigmoid activation, BCE loss
- **Training data**: ~1.5M iNaturalist plant images with phenology annotations
- **Batch size**: 384 (inferred from 2,400 batches/epoch)
- **Initial learning rate**: 0.000150 (with warmup, cosine schedule)
- **GPU**: A100

### Data Available Per Run

| Data Type | Granularity | Source | Notes |
|-----------|------------|--------|-------|
| Per-batch training loss | Every 20 batches (~5,500 points) | SLURM log | Both per-batch and running average |
| Learning rate | Every 20 batches | SLURM log | Cosine schedule with warmup |
| Epoch-end training loss | Per epoch (47 points) | SLURM log (last batch running avg) |
| Validation accuracy (flower, fruit) | Per epoch | SLURM log ("Test:" blocks) |
| Validation J-index (flower, fruit) | Per epoch | SLURM log | = TSS = sensitivity + specificity - 1 |
| Validation loss | Per epoch | SLURM log |
| Per-epoch model weights | 46 checkpoints | `.pt` files | Full model, 296 parameter tensors |

### Key Metrics Summary

| Metric | PlantCLEF | MAE | Interpretation |
|--------|-----------|-----|----------------|
| Initial loss (batch 0) | 0.6931 | 0.6931 | ln(2) = random binary classifier |
| Epoch 1 mean training loss | 0.1954 | 0.2709 | PlantCLEF 28% lower |
| Final training loss (epoch 47) | 0.0250 | 0.0504 | PlantCLEF ~2x lower |
| Peak flower J-index | 0.908 (epoch 3) | 0.891 (epoch 11) | PlantCLEF peaks 8 epochs earlier |
| Peak fruit J-index | 0.821 (epoch 4) | 0.778 (epoch 11) | PlantCLEF 4.3 pp higher |
| Peak mean J-index | 0.864 (epoch 3) | 0.835 (epoch 11) | Consistent with published paper |

## Incomplete Runs (Lost)

### CLIP (March 2025)
- **Directory**: `output/clip_2025_03_09/`
- **Status**: Empty checkpoints dir, no log file
- **Pretraining**: CLIP (contrastive language-image pretraining)
- **Likely outcome**: Run failed or was cancelled

### SigLIP (March 2025)
- **Directories**: `output/siglip_256_v2_2025_03_09/`, `output/siglip_256_v2_2025_03_10/`
- **Status**: Empty checkpoints dirs, no log files
- **Pretraining**: SigLIP (sigmoid loss for language-image pretraining)
- **Likely outcome**: Both runs failed. Two directories suggest a retry on the 10th also failed.

### Implications
- We cannot compare 4 pretraining strategies as originally planned
- PlantCLEF vs MAE remains a strong comparison: domain-relevant vs domain-agnostic pretraining
- If CLIP/SigLIP are important for the story, they would need to be rerun

## Other Training Runs (Not Directly Comparable)

### MAE (November 2025)
- **Directories**: `output/mae_2025_11_13/`, `output/mae_2025_11_14/`
- **Summary**: `output/mae_2025_11_14/training_summary.txt` — 3 epochs only, batch_size=128
- **Status**: Short test run, different hyperparameters, not comparable to March 2025 runs

### Reproductive v1.1.0 (October 2025, via targets + GuildAI)
- **Directory**: `output/reproductive/v1.1.0/`
- **GuildAI run**: `1d6b36f5928c42d5a5aa1eb32af3637c`
- **Pretraining**: PlantCLEF (fine-tuned from v1.0.0 DOI)
- **Epochs**: 10 (best at epoch 7, avg DQI 0.8033)
- **Training data**: Updated 2025-10-27 data snapshot (different from March 2025 runs)
- **Not comparable**: Different data, different starting point (v1.0.0 model, not raw PlantCLEF pretrained)

## Recoverable Analyses from Existing Checkpoints

The 46 per-epoch checkpoints for both PlantCLEF and MAE enable several retrospective analyses without new training runs:

### 1. Weight Displacement Analysis
- Load pretrained init weights and each epoch's state_dict
- Compute ||theta_t - theta_0|| (L2 distance from initialization)
- Per-layer decomposition: where does adaptation concentrate?
- **Requires**: Original pretrained model weights (PlantCLEF and MAE base models)

### 2. Weight Displacement Efficiency
- delta_loss / delta_||theta|| = "fitness gain per unit evolutionary change"
- Analogous to Hansen's conditional evolvability
- **Requires**: Same as above

### 3. Representation Analysis
- PCA of activations at each layer for a held-out image set
- Track effective dimensionality of representations across epochs
- Compare how pretrained feature spaces evolve during fine-tuning
- **Requires**: GPU access, a set of evaluation images

### 4. Attention Head Specialization
- Extract per-head attention patterns from checkpoints
- Track head specialization/diversification across training
- **Requires**: GPU access, evaluation images

## Analyses Requiring New Training Runs

### 1. Intrinsic Dimensionality (d_90)
- Reparameterize: theta = theta_0 + P*v (random projection)
- Sweep dimensionality d, measure performance at convergence
- Compare d_90 for PlantCLEF vs MAE pretrained
- **Requires**: Modified training script, multiple GPU-hours per d value, 5-10 replicates

### 2. Ecological Fitting Decomposition
- Freeze backbone, train only classification head to convergence
- This isolates how good the pretrained representations are without adaptation
- Then unfreeze and track adaptation dynamics
- **Requires**: Modified training script, 2 additional training runs

### 3. Per-Batch Gradient Norms / Covariance
- Log gradient L2 norm and per-layer gradient covariance eigenspectrum during training
- **Requires**: Modified training script with gradient hooks

### 4. CLIP / SigLIP Comparisons
- Rerun the failed March 2025 experiments
- **Requires**: Access to CLIP and SigLIP pretrained ViT-L/16 weights

## Extracted Intermediate Data

Parsed and transformed data saved for reuse:
- `xAI/output/batch_fitness_combined.csv` — all batch-level training data with fitness (exp(-loss)), smoothed fitness, and gradient
- `xAI/output/epoch_validation_combined.csv` — per-epoch validation metrics for both models

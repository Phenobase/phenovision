# Inference Pipeline Run Report

**Date:** 2026-03-12
**Models:** Reproductive v1.1.0, Leaves v1.0.0
**Pipeline:** `_targets_inference.R` (project: `main`, store: `_targets_stores/inference`)

## Summary

| Metric | Value |
|--------|-------|
| Total runtime | ~58 hours across 2 SLURM jobs |
| This run (job 27048786) | 8h 35min |
| Prior run (job 26773093) | ~50h (GPU inference) |
| Targets completed | 787 (this run) + 850 skipped (cached) |
| Targets errored | 14 branch-level + 8 pattern-level |
| Partition | hpg-turin (3x NVIDIA L4 GPUs) |
| Resources | 10 CPUs, 12G/cpu (120 GB total), 72h limit |

## Configuration

| Parameter | Value |
|-----------|-------|
| Reproductive model version | v1.1.0 |
| Reproductive model DOI | 10.57967/hf/7952 |
| Leaves model version | v1.0.0 |
| Leaves model DOI | 10.57967/hf/5785 |
| GPU workers | 3 (L4 GPUs, filelock-based assignment) |
| CPU workers | 7 |
| Batch size (inference) | 1280 |
| Data loader workers | 6 |

## Results

### Leaves (v1.0.0) -- COMPLETE

- **94/94 GPU inference branches completed** -- zero errors
- All downstream targets (threshold, long, by_obs, internal, ingest, combined CSVs) completed

### Reproductive (v1.1.0) -- 92/94 branches complete

- **92/94 branches fully processed** (2 branches errored, see Errors section)
- Combined CSVs generated from the 92 completed branches

### Output Files

**Reproductive (v1.1.0):**

| File | Size | Rows | Location |
|------|------|------|----------|
| Internal (all predictions) | 51 GB | 113,753,406 | `output/reproductive/v1.1.0/inference/annotations_internal_all.csv` |
| Ingest (high-certainty detections) | 12 GB | 31,908,902 | `output/reproductive/v1.1.0/inference/annotations_ingest_all.csv` |
| Per-batch internal | 92 files | -- | `output/reproductive/v1.1.0/inference/final_internal/` |
| Per-batch ingest | 92 files | -- | `output/reproductive/v1.1.0/inference/final_ingest/` |

**Leaves (v1.0.0):**

| File | Size | Rows | Location |
|------|------|------|----------|
| Internal (all predictions) | 14 GB | 31,468,191 | `output/leaves/v1.0.0/inference/annotations_internal_all.csv` |
| Ingest (high-certainty detections) | 2.6 GB | 6,678,853 | `output/leaves/v1.0.0/inference/annotations_ingest_all.csv` |
| Per-batch internal | 94 files | -- | `output/leaves/v1.0.0/inference/final_internal/` |
| Per-batch ingest | 94 files | -- | `output/leaves/v1.0.0/inference/final_ingest/` |

### Data Quality Checks

- Combined CSVs have exactly 1 header row each (no duplicate headers from concatenation)
- Both ingest files have 22 columns matching Phenobase schema
- Repro combined CSV has 113.75M rows -- matches sum across 92 per-batch files
- Model URI correctly set to `10.57967/hf/7952` (repro v1.1.0)
- Streaming CSV concatenation completed without OOM: ingest 1m 15s, internal 3m 52s

## Errors

### 1. Truncated PNG File Read (1 branch)

- **Target:** `annotations_repro_7d806d3f8e19bc3a`
- **Error:** `OSError: Truncated File Read` in `PngImagePlugin.py:load_end` during `img.convert('RGB')`
- **Impact:** 1 repro branch + 7 cascade downstream targets (threshold, long, by_obs, final, ingest, internal, ingest_csv)
- **Root cause:** A PNG image has a valid header but truncated pixel data. `Image.open()` succeeds (lazy) but `.convert()` fails. The try/except in `PhenoDatasetInf.__getitem__` only wraps `Image.open()`, not `.convert()`.
- **Resolution:** Full-disk scan found 0 permanently truncated PNGs. Likely a transient shared filesystem issue. Will retry automatically on next `tar_make()`.
- **Note:** This reveals a bug in `py/custom_dataset.py` where the `.convert()` call is outside the try/except block in `PhenoDatasetInf`. Fixing this would invalidate all cached inference branches, so deferring to a future model version.

### 2. class_pred Conversion Error (1 branch)

- **Target:** `annotations_long_repro_704b996934e84be8`
- **Error:** `Can't convert <class_pred> to <character>` in `convert_to_long()` at `as.character(as.factor(.class_fl))`
- **Impact:** 1 branch + 5 cascade downstream targets
- **Root cause:** Transient -- could not reproduce on the actual data from the saved workspace. Likely a package loading issue in the crew worker.
- **Resolution:** Will retry automatically on next `tar_make()`.

## Infrastructure Notes

### Multi-GPU Setup

This was the first successful multi-GPU inference run. Key implementation details:

- **GPU assignment:** File-based lock queue using `filelock` package. Each crew worker acquires a GPU lock at startup, holds it for worker lifetime.
- **Environment variable:** Uses `PHENOVISION_GPU_ASSIGNED` flag (not `CUDA_VISIBLE_DEVICES` check) because SLURM pre-sets `CUDA_VISIBLE_DEVICES=0,1,2`.
- **CUDA cleanup:** `torch$cuda$empty_cache()` + `gc()` at end of each `annotate_batch()` call to free GPU memory for next branch.
- **Performance:** ~93 min per repro branch on L4 (vs ~57 min on B200). 3x parallelism gives effective ~31 min/branch throughput.

### CSV Concatenation

Switched from `bind_rows()` (loads all 94 branches into memory -- 50+ GB) to streaming approach: reads per-batch CSV files one at a time via `concatenate_csvs()`. This resolved the OOM crash from the prior run.

## Next Steps

1. Resubmit pipeline to retry the 2 errored repro branches (automatic, no invalidation needed)
2. Repro combined CSVs will regenerate with all 94 batches after retry
3. Consider fixing `PhenoDatasetInf` try/except scope for future model versions

#!/bin/bash
# §2.1 standard-benchmark SLURM ARRAY (analysis_plan.md §2.1).
#
# One array task = one config (one line of configs/experiment/benchmark_grid.txt) = one GPU job
# = one tidy CSV under runs/benchmarks/. The array is throttled to %3 so AT MOST 3 GPUs run at
# once — the guralnick allocation is shared and capped at ~5 concurrent GPUs; %3 leaves headroom.
# NEVER raise this above %3 without checking with the group.
#
# -------------------------------------------------------------------------------------------------
# BEFORE SUBMITTING, check the group's current GPU usage so you don't starve other jobs:
#     module load ufrc && slurmInfo            # allocation summary; look at GPUs running/pending
#     squeue -A guralnick -o '%.18i %.9P %.20j %.8u %.2t %.10M %.6D %R'   # who's using what
# -------------------------------------------------------------------------------------------------
#
# Generate / refresh the grid first (cifar100 pairs only by default):
#     mamba run -n two_noise python scripts/gen_benchmark_grid.py
#     # -> prints the exact "--array=0-N%3" to use below.
#
# Launch §2.1 (set --array to match the grid line count; current grid = 36 lines -> 0-35):
#     sbatch --array=0-35%3 scripts/submit_benchmark.sh
#
# §2.2 alpha-vs-batch (single GPU job per pair; NOT an array). Pass --avb then alpha_vs_batch args:
#     sbatch --job-name=tn_avb_vits scripts/submit_benchmark.sh --avb \
#         --model vit_s --dataset cifar100 --micro-batch 16 --budget-microbatches 8000 --amp
# Keep total concurrent GPU jobs (array tasks + any --avb jobs) <= 3.
#
#SBATCH --job-name=tn_bench
#SBATCH --account=guralnick
#SBATCH --qos=guralnick
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:l4:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=12:00:00
#SBATCH --output logs/%x-%A_%a.out
#SBATCH --error  logs/%x-%A_%a.err

set -euo pipefail

cd /blue/guralnick/share/r.dinnage/Projects/phenovision/xAI/two_noise
mkdir -p logs runs/benchmarks runs/alpha_vs_batch

# --- §2.2 mode: `--avb <alpha_vs_batch args...>` runs the sweep as a single (non-array) job ---
if [[ "${1:-}" == "--avb" ]]; then
    shift
    echo "[avb] alpha_vs_batch args: $*"
    mamba run -n two_noise python -m ml_experiments.alpha_vs_batch "$@"
    exit 0
fi

GRID=configs/experiment/benchmark_grid.txt
if [[ ! -f "$GRID" ]]; then
    echo "Grid $GRID not found; run: mamba run -n two_noise python scripts/gen_benchmark_grid.py"
    exit 1
fi

# Pick the config for this array index (0-based).
LINE=$(( SLURM_ARRAY_TASK_ID + 1 ))
ARGS=$(sed -n "${LINE}p" "$GRID")
if [[ -z "$ARGS" ]]; then
    echo "No config at line ${LINE} of $GRID (array index ${SLURM_ARRAY_TASK_ID})"
    exit 1
fi

echo "[task ${SLURM_ARRAY_TASK_ID}] config: $ARGS"
mamba run -n two_noise python -m ml_experiments.benchmarks $ARGS

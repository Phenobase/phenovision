#!/bin/bash
# α=1 TRUE-FISHER STABILITY test (notes_evolution_vs_ml.md §1, §4): does importing the true
# (sampled-label) Fisher curvature + relative LM damping stabilize the full-inverse step
# (SOAPFullPower precond_power=1.0) on a real ViT-S at SMALL batch (16), where the empirical
# Fisher diverges to NaN? Runs all THREE conditions (A empirical/lowdamp, B true-Fisher+damp,
# C empirical+damp) sequentially in ONE single-GPU job and writes runs/alpha1_stability/results.csv.
#
# -------------------------------------------------------------------------------------------------
# ONE GPU ONLY. A benchmark SLURM array (submit_benchmark.sh, throttled %3) may be running and the
# guralnick allocation is capped at ~5 concurrent GPUs. BEFORE submitting, check group usage:
#     squeue -A guralnick -t R -h -O "tres-alloc:200" | grep -oE 'gres/gpu=[0-9]+' | awk -F= '{s+=$2} END {print s+0}'
# Only submit if that prints <= 2 (so this job's 1 GPU keeps the total <= 3). Never request >1 here.
# -------------------------------------------------------------------------------------------------
#
#SBATCH --job-name=tn_alpha1
#SBATCH --account=guralnick
#SBATCH --qos=guralnick
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:l4:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=08:00:00
#SBATCH --output logs/%x-%j.out
#SBATCH --error  logs/%x-%j.err

set -euo pipefail

cd /blue/guralnick/share/r.dinnage/Projects/phenovision/xAI/two_noise
mkdir -p logs runs/alpha1_stability

# Defaults: ViT-S / CIFAR-100, batch 16, AMP, 2000 steps, all three conditions. Override by passing
# extra args:  sbatch scripts/submit_alpha1_stability.sh --max-steps 3000 --conditions A B C
ARGS=("--device" "cuda" "--model" "vit_s" "--dataset" "cifar100" "--batch" "16"
      "--max-steps" "2000" "--amp" "--conditions" "A" "B" "C")
if [[ $# -gt 0 ]]; then
    ARGS=("$@")
fi

echo "[alpha1] args: ${ARGS[*]}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

mamba run -n two_noise python -m ml_experiments.alpha1_stability "${ARGS[@]}"

echo "[alpha1] results:"
cat runs/alpha1_stability/results.csv || true

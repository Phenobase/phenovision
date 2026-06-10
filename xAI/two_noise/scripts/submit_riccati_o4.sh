#!/bin/bash
# O4 — evolving-M meta-loop (the headline): operative exponent climbs 1/2->1 in the
# RiccatiPrecond. Does the noise-dependent shrink schedule recover most of true-Fisher's gain at
# 1x cost? Runs the three conditions sequentially in ONE single-GPU job and writes
# runs/riccati_evolve_m/<model>_<dataset>_b<batch>.csv. Uses bf16 autocast (no GradScaler).
#
# -------------------------------------------------------------------------------------------------
# ONE GPU ONLY. The guralnick allocation is capped at ~5 concurrent GPUs; be fair (<=3 of ours).
# BEFORE submitting, check group usage:
#     squeue -A guralnick -t R -h -O "tres-alloc:200" | grep -oE 'gres/gpu=[0-9]+' | awk -F= '{s+=$2} END {print s+0}'
# Only submit if that prints <= 2 (so this job's 1 GPU keeps the total <= 3). Never request >1 here.
# -------------------------------------------------------------------------------------------------
#
#SBATCH --job-name=tn_ricc_o4
#SBATCH --account=guralnick
#SBATCH --qos=guralnick
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:l4:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=16:00:00
#SBATCH --output logs/%x-%j.out
#SBATCH --error  logs/%x-%j.err

set -euo pipefail

cd /blue/guralnick/share/r.dinnage/Projects/phenovision/xAI/two_noise
mkdir -p logs runs/riccati_evolve_m

# Defaults: ViT-S / CIFAR-100, batch 64 (small enough to stress gradient noise), AMP(bf16), 4000
# steps, all three conditions. Override:  sbatch scripts/submit_riccati_o2.sh --batch 32 ...
ARGS=("--device" "cuda" "--model" "vit_s" "--dataset" "cifar100" "--batch" "128"
      "--max-steps" "4000" "--amp" "--lanczos"
      "--conditions" "adamw" "soap" "whiten" "inverse" "evolve"
      "--cond-lrs" "adamw=1e-3,soap=1e-3,whiten=3e-3,inverse=3e-4,evolve=3e-4"
      "--eta-m-grid" "3e-4" "1e-3" "3e-3" "--meta-every-grid" "20")
if [[ $# -gt 0 ]]; then
    ARGS=("$@")
fi

echo "[o4] args: ${ARGS[*]}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

mamba run -n two_noise python -m ml_experiments.riccati_evolve_m "${ARGS[@]}"

echo "[o4] results tail:"
tail -n 8 runs/riccati_evolve_m/*.csv || true

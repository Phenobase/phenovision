#!/bin/bash
# DIAGNOSTIC -- does the matrix-free Newton-Schulz factor converge to C^{-1/2}/C^{-1} on a real ViT?
# Sweeps inner_steps x precond_every (x mode) and logs the operative exponent + NS fixed-point
# residual + loss. Answers whether O4's exponent~0 / weak-whiten is an NS-budget problem.
# ONE GPU ONLY (guralnick cap ~5, be fair <=3). Check group usage before submitting:
#   squeue -A guralnick -t R -h -O "tres-alloc:200" | grep -oE 'gres/gpu=[0-9]+' | awk -F= '{s+=$2} END {print s+0}'
#
#SBATCH --job-name=tn_ricc_conv
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
mkdir -p logs runs/riccati_precond_convergence

ARGS=("--device" "cuda" "--model" "vit_s" "--dataset" "cifar100" "--batch" "128"
      "--max-steps" "1200" "--amp"
      "--modes" "whiten" "inverse"
      "--inner-steps-grid" "2" "5" "10" "20"
      "--precond-every-grid" "1" "5")
if [[ $# -gt 0 ]]; then
    ARGS=("$@")
fi

echo "[conv] args: ${ARGS[*]}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mamba run -n two_noise python -m ml_experiments.riccati_precond_convergence "${ARGS[@]}"

echo "[conv] results tail:"
tail -n 8 runs/riccati_precond_convergence/*.csv || true

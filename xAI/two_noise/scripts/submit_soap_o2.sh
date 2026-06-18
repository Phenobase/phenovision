#!/bin/bash
# O2-on-SOAP — the SAME cheap-shrink-schedule (1x) vs true-Fisher (2x) vs whiten comparison, but
# with SOAPFullPower (EXACT eigendecomposition preconditioner) instead of the matrix-free Riccati.
# Motivation: the convergence diagnostic showed the matrix-free Newton-Schulz under-converges on a
# real ViT (G~I, operative exponent ~0), so SOAP -- which computes C^{-alpha} exactly from the
# spectrum -- is the confound-free vehicle for the shrinkage science. Shrink is applied to the
# in-basis second moment before the power (effective-exponent reduction). bf16, no GradScaler.
# Writes runs/riccati_schedule_vs_fisher/<model>_<dataset>_b<batch>.csv (optimizer column = soap).
#
# -------------------------------------------------------------------------------------------------
# ONE GPU ONLY. guralnick cap ~5 concurrent; be fair (<=3 of ours). Check before submitting:
#     squeue -A guralnick -t R -h -O "tres-alloc:200" | grep -oE 'gres/gpu=[0-9]+' | awk -F= '{s+=$2} END {print s+0}'
# Submit only if that prints <= 2. Never request >1 here.
# -------------------------------------------------------------------------------------------------
#
#SBATCH --job-name=tn_soap_o2
#SBATCH --account=guralnick
#SBATCH --qos=guralnick
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:l4:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=12:00:00
#SBATCH --output logs/%x-%j.out
#SBATCH --error  logs/%x-%j.err

set -euo pipefail
cd /blue/guralnick/share/r.dinnage/Projects/phenovision/xAI/two_noise
mkdir -p logs runs/riccati_schedule_vs_fisher

# ViT-S / CIFAR-100, batch 64 (stresses gradient noise -> where shrinkage matters). lr left unset
# so make_optimizer's per-alpha schedule applies (whiten alpha=0.5 -> base_lr ~1e-3; the alpha=1
# conditions -> ~1e-4). Four conditions: whiten / full_inverse(naive) / inverse_fisher(2x) /
# schedule(shrink, 1x). Override: sbatch scripts/submit_soap_o2.sh --batch 32 ...
ARGS=("--device" "cuda" "--optimizer" "soap" "--model" "vit_s" "--dataset" "cifar100"
      "--batch" "64" "--max-steps" "4000" "--amp"
      "--conditions" "whiten" "full_inverse" "inverse_fisher" "schedule")
if [[ $# -gt 0 ]]; then
    ARGS=("$@")
fi

echo "[o2-soap] args: ${ARGS[*]}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mamba run -n two_noise python -m ml_experiments.riccati_schedule_vs_fisher "${ARGS[@]}"

echo "[o2-soap] results tail:"
tail -n 8 runs/riccati_schedule_vs_fisher/*.csv || true

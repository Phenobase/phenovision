#!/bin/bash
# T1/T2 — the anisotropy sign-flip (notes_landscape_disaster_benefit.md §3.1/§3.3/§3.4).
# Isotropic disaster Ω; sweep severity (σ) × tail (gaussian/student_t) × N. Predict the evolved-M
# ratio m_flat/m_steep CROSSES 1: mild/gaussian/large-N -> A⁺ (ratio<1); severe/heavy-tail/small-N
# -> A⁻¹ (ratio>1). CPU-only (JAX vmap over replicate populations); burst QOS, NOT the GPU budget.
#SBATCH --job-name=tn_flip
#SBATCH --account=guralnick
#SBATCH --qos=guralnick-b
#SBATCH --partition=hpg-default
#SBATCH --output logs/%x-%j.out
#SBATCH --error  logs/%x-%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem-per-cpu=8G --time=24:00:00
#SBATCH --requeue

cd /blue/guralnick/share/r.dinnage/Projects/phenovision/xAI/two_noise
export JAX_PLATFORMS=cpu

# Production grid: severity from mild (arithmetic regime, A⁺) to severe (catastrophe, A⁻¹), both
# tails, N from small (catastrophe-prone) to large (arithmetic). Many replicates + long burn-in to
# pin the drift sign. Override by passing args.
ARGS=("--sigmas" "0.3" "0.5" "0.8" "1.5" "2.5" \
      "--tails" "gaussian" "student_t" "--df" "3.0" \
      "--Ns" "400" "1000" "2000" \
      "--strength" "1.0" "--anisotropy" "4.0" \
      "--L" "14" "--burn-in" "600" "--tau" "100" "--replicates" "192")
if [[ $# -gt 0 ]]; then ARGS=("$@"); fi

echo "[flip] args: ${ARGS[*]}"
mamba run -n two_noise_jax python -m sim.anisotropy_flip "${ARGS[@]}" \
    2>&1 | tee logs/anisotropy_flip_$(date +%Y%m%d_%H%M%S).log

echo "[flip] results:"; cat runs/anisotropy_flip/results.csv || true

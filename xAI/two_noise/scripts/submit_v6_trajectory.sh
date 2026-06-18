#!/bin/bash
# V3 — production-scale M-drift direction test (the core second-order-selection mechanism).
# CPU-friendly (JAX vmap over replicate populations); runs on the burst QOS, NOT using the GPU
# budget. Many replicates + long burn-in push the static-regime drift-vs-theory cosine toward
# the doc's >0.9 (the pilot resolves it only moderately).
#SBATCH --job-name=tn_v6
#SBATCH --account=guralnick
#SBATCH --qos=guralnick-b
#SBATCH --partition=hpg-default
#SBATCH --output logs/%x-%j.out
#SBATCH --error  logs/%x-%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem-per-cpu=8G --time=12:00:00
#SBATCH --requeue

cd /blue/guralnick/share/r.dinnage/Projects/phenovision/xAI/two_noise
export JAX_PLATFORMS=cpu

# Production scale: large N, many replicate populations, long fast-equilibration burn-in, a
# τ window above the fast correlation time but short enough to keep M near M0 (V0 magnitude trap).
ARGS=("--N" "800" "--L" "12" "--generations" "400" "--replicates" "128")
if [[ $# -gt 0 ]]; then ARGS=("$@"); fi

echo "[v3] args: ${ARGS[*]}"
mamba run -n two_noise_jax python -m sim.v6_trajectory "${ARGS[@]}" \
    2>&1 | tee logs/v6_trajectory_$(date +%Y%m%d_%H%M%S).log

echo "[v3] results:"; cat runs/v6_trajectory/results.csv || true

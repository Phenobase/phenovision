#!/bin/bash
# Flip stability diagnostic: start M0 AT each shape (A+, iso, A-1) and measure which way the drift
# moves the ratio, at MILD sigma / large N across selection STRENGTH. Separates the no-flip result
# (a real falsification of the arithmetic A+ regime vs the canalization-transient confound).
# CPU-only (JAX); burst QOS.
#SBATCH --job-name=tn_flipdiag
#SBATCH --account=guralnick
#SBATCH --qos=guralnick-b
#SBATCH --partition=hpg-default
#SBATCH --output logs/%x-%j.out
#SBATCH --error  logs/%x-%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem-per-cpu=8G --time=12:00:00
#SBATCH --requeue

cd /blue/guralnick/share/r.dinnage/Projects/phenovision/xAI/two_noise
export JAX_PLATFORMS=cpu

ARGS=("--sigma" "0.4" "--Ns" "2000" "--strengths" "0.1" "0.3" "1.0" \
      "--shapes" "aplus" "iso" "ainv" "--tail" "gaussian" \
      "--L" "14" "--burn-in" "600" "--tau" "120" "--replicates" "192")
if [[ $# -gt 0 ]]; then ARGS=("$@"); fi

echo "[flipdiag] args: ${ARGS[*]}"
mamba run -n two_noise_jax python -m sim.flip_diagnostic "${ARGS[@]}" \
    2>&1 | tee logs/flip_diagnostic_$(date +%Y%m%d_%H%M%S).log
echo "[flipdiag] results:"; cat runs/flip_diagnostic/results.csv || true

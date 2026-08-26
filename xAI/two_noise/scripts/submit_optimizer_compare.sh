#!/bin/bash
# Optimizer head-to-head (to equilibrium) SLURM ARRAY. One array task = one config line of
# configs/experiment/compare_grid.txt = one GPU job = one tidy per-step CSV under runs/benchmarks/.
# Throttled to %3 so AT MOST 3 GPUs run at once (the guralnick cap is ~5 concurrent; leave headroom).
# NEVER raise above %3 without checking the group (module load ufrc && slurmInfo).
#
# Generate the grid first:
#     mamba run -n two_noise python scripts/gen_compare_grid.py   # prints the exact --array=0-N%3
# Launch (set --array to the printed line count):
#     sbatch --array=0-Nx%3 scripts/submit_optimizer_compare.sh
#
#SBATCH --job-name=tn_cmp
#SBATCH --account=guralnick
#SBATCH --qos=guralnick
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:l4:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=12:00:00
#SBATCH --output logs/%x-%A_%a.out
#SBATCH --error  logs/%x-%A_%a.err

set -euo pipefail
cd /blue/guralnick/share/r.dinnage/Projects/phenovision/xAI/two_noise
mkdir -p logs runs/benchmarks

# grid file is selectable via TN_CMP_GRID (set per split job: stable_evo vs baselines)
GRID="${TN_CMP_GRID:-configs/experiment/compare_grid.txt}"
if [[ ! -f "$GRID" ]]; then
    echo "Grid $GRID not found; run: mamba run -n two_noise python scripts/gen_compare_grid.py"
    exit 1
fi
echo "[cmp] using grid $GRID"
LINE=$(( SLURM_ARRAY_TASK_ID + 1 ))
ARGS=$(sed -n "${LINE}p" "$GRID")
if [[ -z "$ARGS" ]]; then
    echo "No config at line ${LINE} of $GRID (array index ${SLURM_ARRAY_TASK_ID})"; exit 1
fi
# Stage the dataset ONCE across concurrent array tasks: flock serializes, and _harness only
# downloads/extracts if absent (read-only otherwise), so tasks never race on extraction.
DS=$(echo "$ARGS" | grep -oE '\-\-dataset[= ]\S+' | awk -F'[ =]' '{print $2}')
if [[ "$DS" == "cifar100" ]]; then
    flock /tmp/tn_prestage_${DS}.lock mamba run -n two_noise \
        python -m ml_experiments._harness --prestage "$DS" || true
fi

echo "[cmp task ${SLURM_ARRAY_TASK_ID}] config: $ARGS"
mamba run -n two_noise python -m ml_experiments.benchmarks $ARGS

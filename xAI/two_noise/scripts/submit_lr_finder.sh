#!/bin/bash
# Stage 1 of the optimizer comparison: the LR range test (lr_find) for every (optimizer, batch)
# cell, in ONE GPU job (each test is short, ~n_iter steps). Writes runs/lr_finder/suggested.csv,
# which gen_compare_grid.py reads to build the full-to-convergence runs (stage 2).
#SBATCH --job-name=tn_lrfind
#SBATCH --account=guralnick
#SBATCH --qos=guralnick
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:l4:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=4:00:00
#SBATCH --output logs/%x-%j.out
#SBATCH --error  logs/%x-%j.err

set -euo pipefail
cd /blue/guralnick/share/r.dinnage/Projects/phenovision/xAI/two_noise
mkdir -p logs runs/lr_finder
# fresh suggestions for this sweep (lr_finder APPENDS, so clear a stale file first)
rm -f runs/lr_finder/suggested.csv

ARGS=("--model" "vit_s" "--dataset" "cifar100" \
      "--optimizers" "sgd" "adamw" "soap@0.5" "soap@1.0" "stable_evo" \
      "--batch-sizes" "64" "256" "1024" "4096" \
      "--micro-batch" "256" "--n-iter" "200" "--amp")
if [[ $# -gt 0 ]]; then ARGS=("$@"); fi

echo "[lr_finder] args: ${ARGS[*]}"
mamba run -n two_noise python -m ml_experiments.lr_finder "${ARGS[@]}"
echo "[lr_finder] suggested.csv:"; cat runs/lr_finder/suggested.csv || true

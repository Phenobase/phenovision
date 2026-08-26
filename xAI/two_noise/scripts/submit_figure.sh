#!/bin/bash
# Auto-assembly of the §5 shared-prediction figure (analysis_plan.md §4/§5).
#
# Submit with a dependency on the runs that feed the figure so it fires AUTOMATICALLY when they
# finish (no polling):
#   sbatch --dependency=afterany:<bench_array>:<avb_job>:<sim_b_job> scripts/submit_figure.sh
# afterany = run regardless of their exit state (partial data still makes a useful figure).
#
#SBATCH --job-name=tn_figure
#SBATCH --account=guralnick
#SBATCH --qos=guralnick
#SBATCH --partition=hpg-default
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2 --mem-per-cpu=8G --time=00:30:00
#SBATCH --output logs/%x-%j.out
#SBATCH --error  logs/%x-%j.err

set -eo pipefail
cd /blue/guralnick/share/r.dinnage/Projects/phenovision/xAI/two_noise

echo "[figure] assembling tidy CSVs from completed runs..."
mamba run -n two_noise python -m figures.export_csv

echo "[figure] rendering shared-prediction figure (R/ggplot)..."
cd /blue/guralnick/share/r.dinnage/Projects/phenovision
mamba run -n reticulate-gpu2 Rscript xAI/R/two_noise_figures.R

echo "[figure] done -> xAI/figures/two_noise_shared_prediction.png"

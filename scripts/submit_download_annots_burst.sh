#!/bin/bash
# Resume variant of submit_download_annots.sh for the LARGE 2026-07-14 data update (9.6M new photos).
# The default (investment qos, 6x16G=96G) OOM'd the crew dispatcher during the image-download phase.
# This runs the SAME resumable pipeline on BURST (CPU-only work -> keeps off the investment allocation)
# with a big single-node memory pool. Cached targets skip; it continues from the last completed batch.
#SBATCH --job-name=pheno_download_annots
#SBATCH --mail-user=r.dinnage@gmail.com
#SBATCH --mail-type=FAIL,END
#SBATCH --account=guralnick
#SBATCH --qos=guralnick-b
#SBATCH --partition=hpg-default
#SBATCH --requeue

#SBATCH --output logs/%x-%j.out
#SBATCH --error logs/%x-%j.err

# Resources (CPU only, no GPU). Big --mem for the crew dispatcher; 6 workers = 6 concurrent batches.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=350G

# Burst walltime (downloads of ~9.6M images can exceed 24h)
#SBATCH --time=96:00:00

# Activate conda
source /home/${USER}/.bashrc
source activate reticulate-gpu2

# Force reticulate to use the conda environment's Python
export RETICULATE_PYTHON="${RETICULATE_PYTHON:-/blue/guralnick/r.dinnage/.conda/envs/reticulate-gpu2/bin/python}"

date;hostname;pwd

# Pass SLURM CPU allocation to targets
export TARGETS_WORKERS=$SLURM_CPUS_PER_TASK

# Run (resumes from cached targets)
Rscript run_pipeline.R --pipeline=download_annots

date

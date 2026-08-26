#!/bin/bash
#SBATCH --job-name=pheno_download_annots
#SBATCH --mail-user=r.dinnage@gmail.com
#SBATCH --mail-type=FAIL,END
#SBATCH --account=guralnick
#SBATCH --qos=guralnick

# Where to put the outputs: %j expands into the job number
#SBATCH --output logs/%x-%j.out
#SBATCH --error logs/%x-%j.err

# Resources (CPU only, no GPU needed for download)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=16G

# Job run time (downloads can be slow)
#SBATCH --time=24:00:00

# Activate conda
source /home/${USER}/.bashrc
source activate reticulate-gpu2

# Force reticulate to use the conda environment's Python
export RETICULATE_PYTHON="${RETICULATE_PYTHON:-/blue/guralnick/r.dinnage/.conda/envs/reticulate-gpu2/bin/python}"

# Save useful information
date;hostname;pwd

# Pass SLURM CPU allocation to targets (reads from --cpus-per-task above)
export TARGETS_WORKERS=$SLURM_CPUS_PER_TASK

# Run the download annotations pipeline
Rscript run_pipeline.R --pipeline=download_annots

# Report completion
date

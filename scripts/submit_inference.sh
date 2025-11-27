#!/bin/bash
#SBATCH --job-name=pheno_inference
#SBATCH --mail-user=r.dinnage@gmail.com
#SBATCH --mail-type=FAIL,END
#SBATCH --account=guralnick
#SBATCH --qos=guralnick
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:1

# Where to put the outputs: %j expands into the job number
#SBATCH --output logs/%x-%j.out
#SBATCH --error logs/%x-%j.err

# Resources (GPU inference on millions of images)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=8G

# Job run time (inference on full dataset can take days)
#SBATCH --time=72:00:00

# Activate conda
source /home/${USER}/.bashrc
source activate rstudio-gpu

# Save useful information
date;hostname;pwd

# Print GPU info
nvidia-smi

# Pass SLURM CPU allocation to targets (reads from --cpus-per-task above)
export TARGETS_WORKERS=$SLURM_CPUS_PER_TASK

# Run the inference pipeline
Rscript run_pipeline.R --pipeline=inference

# Report completion
date
nvidia-smi

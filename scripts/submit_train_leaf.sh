#!/bin/bash
#SBATCH --job-name=pheno_train_leaf
#SBATCH --mail-user=r.dinnage@gmail.com
#SBATCH --mail-type=FAIL,END
#SBATCH --account=guralnick
#SBATCH --qos=guralnick
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1

# Where to put the outputs: %j expands into the job number
#SBATCH --output logs/%x-%j.out
#SBATCH --error logs/%x-%j.err

# Resources (GPU training needs significant memory and time)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G

# Job run time (leaf training includes two rounds, may take longer)
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

# Run the leaf training pipeline
Rscript run_pipeline.R --pipeline=train_leaf

# Report completion
date
nvidia-smi

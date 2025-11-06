#!/bin/bash
#SBATCH --job-name=pheno_train_repro
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

# Job run time (training can take days)
#SBATCH --time=72:00:00

# Activate conda
source /home/${USER}/.bashrc
source activate rstudio-gpu

# Save useful information
date;hostname;pwd

# Print GPU info
nvidia-smi

# Run the reproductive training pipeline
Rscript run_pipeline.R --pipeline=train_repro --workers=4

# Report completion
date
nvidia-smi

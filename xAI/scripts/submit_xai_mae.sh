#!/bin/bash
#SBATCH --job-name=xai_mae
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=14:00:00
#SBATCH --output xAI/logs/%x-%j.out
#SBATCH --error xAI/logs/%x-%j.err
#SBATCH --mail-user=r.dinnage@gmail.com
#SBATCH --mail-type=FAIL,END
#SBATCH --account=guralnick
#SBATCH --qos=guralnick

echo "$(date)"
echo "$(hostname)"
echo "$(pwd)"

# Activate conda environment
source activate reticulate-gpu2
export PYTHONPATH="${PWD}/PlantCLEF2022:${PWD}:${PYTHONPATH}"

echo "Running xAI MAE training..."
python -u xAI/py/xai_train.py --pretrained_model mae --max_train_samples 214000

echo "Done: $(date)"

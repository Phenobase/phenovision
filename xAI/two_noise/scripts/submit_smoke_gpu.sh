#!/bin/bash
#SBATCH --job-name=tn_smoke_gpu
#SBATCH --account=guralnick
#SBATCH --qos=guralnick
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:l4:1
#SBATCH --output logs/%x-%j.out
#SBATCH --error  logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=00:15:00

# two_noise GPU coexistence smoke test (1 L4, ~1 min of work).
cd /blue/guralnick/share/r.dinnage/Projects/phenovision/xAI/two_noise
echo "host: $(hostname)  gpu: $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
mamba run -n two_noise python scripts/smoke_gpu.py

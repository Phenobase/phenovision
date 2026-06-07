#!/bin/bash
#SBATCH --job-name=tn_smoke_jax
#SBATCH --account=guralnick
#SBATCH --qos=guralnick
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:l4:1
#SBATCH --output logs/%x-%j.out
#SBATCH --error  logs/%x-%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem-per-cpu=8G --time=00:10:00
cd /blue/guralnick/share/r.dinnage/Projects/phenovision/xAI/two_noise
nvidia-smi --query-gpu=name --format=csv,noheader || true
mamba run -n two_noise_jax python scripts/smoke_jax_gpu.py

#!/bin/bash
# Publication-scale Sim A sweep (analysis_plan.md §3.1). The sims are CPU-friendly (JAX vmap
# over replicates × scan over generations); a CPU job on the burst QOS does NOT use the GPU
# budget. Override JAX_PLATFORMS to use a GPU if one is requested instead.
#SBATCH --job-name=tn_sim_a
#SBATCH --account=guralnick
#SBATCH --qos=guralnick-b
#SBATCH --partition=hpg-turin
#SBATCH --output logs/%x-%j.out
#SBATCH --error  logs/%x-%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem-per-cpu=8G --time=08:00:00

cd /blue/guralnick/share/r.dinnage/Projects/phenovision/xAI/two_noise

# CPU run (no GPU requested). JAX threads across the allocated CPUs.
export JAX_PLATFORMS=cpu

# Publication-scale: large N, many replicates, wide anisotropy sweep, long burn-in.
# (Adjust on the CLI; defaults below match the build-report recommendation.)
mamba run -n two_noise_jax python -m sim.sim_a_exponent \
    --N 4000 --L 25 \
    --generations 1200 --burn-in 700 \
    --replicates 32 \
    --n-boot 1000 \
    --low-mu 1e-4 --high-mu 1e-1 \
    2>&1 | tee logs/sim_a_$(date +%Y%m%d_%H%M%S).log

#!/bin/bash
# Publication-scale Sim B — the N* phase diagram (analysis_plan.md §3.2).
#
# CPU-ONLY ARRAY job, one (N*, Ne) cell per array task. Sim B is CPU-friendly (JAX vmap over
# replicates × scan over generations) and is run on the BURST QOS (guralnick-b): idle cores at
# LOW priority, time-limited (4 d), and PREEMPTIBLE. GPU is FORBIDDEN here — no --gres=gpu — and
# JAX_PLATFORMS=cpu pins JAX to CPU so a stray GPU is never touched.
#
# Preemption safety: each task writes a per-cell .done marker + per-cell csv when it finishes
# (sim/sim_b_phase.py: run_one_cell_checkpointed). A requeued/preempted task SKIPS already-.done
# cells, so the whole array is idempotent — just resubmit (or rely on --requeue) and it resumes.
#
#SBATCH --job-name=tn_sim_b
#SBATCH --account=guralnick
#SBATCH --qos=guralnick-b           # BURST: CPU-only, low-priority, preemptible -> checkpointed
#SBATCH --partition=hpg-default     # standard CPU partition (hpg-turin is a GPU node; NO GPU on burst)
#SBATCH --requeue                   # auto-requeue on preemption; .done markers make it resume
#SBATCH --output logs/%x-%A_%a.out
#SBATCH --error  logs/%x-%A_%a.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem-per-cpu=8G --time=12:00:00
#
# Grid is 5 N* × 4 Ne = 20 cells -> array 0-19. %8 caps concurrent tasks at 8 (be a good burst
# citizen; raise/lower to taste). If you change the grids below, update --array to match
# (n_cells = len(N*_grid) * len(Ne_grid)).
#SBATCH --array=0-14%8

cd /blue/guralnick/share/r.dinnage/Projects/phenovision/xAI/two_noise

# CPU-only (burst QOS forbids GPU). JAX threads across the allocated CPUs.
export JAX_PLATFORMS=cpu

# Publication grid (must match the default grid in sim_b_phase.py, or pass explicitly here AND
# in --array sizing). Defaults: N*_grid = 600,100,30,10,3 ; Ne_grid = 600,200,60,20  (=20 cells).
N_STAR_GRID="600,200,80,30,15"
NE_GRID="600,120,30"

# Per-cell publication settings: N=600 individuals, L=25 loci, anisotropic A (ratio 8),
# epistasis on (sigma_eps=0.3 so M can become anisotropic and then be compressed by gradient
# noise), 32 replicates per cell, long burn-in to M-equilibrium.
mamba run -n two_noise_jax python -m sim.sim_b_phase \
    --cell "${SLURM_ARRAY_TASK_ID}" \
    --n-star-grid "${N_STAR_GRID}" \
    --Ne-grid "${NE_GRID}" \
    --N 600 --L 25 \
    --A-eigratio 8.0 --sigma-eps 0.3 --env-amp 0.4 --mu 5e-2 \
    --generations 1200 --burn-in 700 \
    --replicates 32 \
    --seed 0 \
    2>&1 | tee "logs/sim_b_cell${SLURM_ARRAY_TASK_ID}_$(date +%Y%m%d_%H%M%S).log"

# After the whole array finishes, aggregate the per-cell csvs into results.csv + surfaces.csv:
#   JAX_PLATFORMS=cpu mamba run -n two_noise_jax python -m sim.sim_b_phase \
#       --aggregate-only --n-star-grid "${N_STAR_GRID}" --Ne-grid "${NE_GRID}"
# (or just re-run run_phase_diagram, which skips all .done cells and aggregates.)

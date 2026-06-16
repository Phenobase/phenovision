#!/bin/bash
# init-d90 intrinsic-dimension sweep — SLURM ARRAY (briefing Part II §9.1, plan C6).
#
# One array task = one config (one line of xAI/py/d90/d90_grid.txt) = one GPU job = one
# run_subspace.py invocation, which appends one row to the d90 results table. The d90 runs are
# INDEPENDENT of each other and of the main block (they depend only on the three init
# checkpoints, which exist before any fine-tuning), so the array is embarrassingly parallel.
#
# The array is throttled to %3 so AT MOST 3 GPUs run at once — the guralnick allocation is a
# FIRM <=3 concurrent-GPU cap (leave >=2 for the group). NEVER raise this above %3.
#
# SCHEDULING (briefing §9.1 "Scheduling under Plan A"): run this batch BEFORE and/or AFTER the
# main 2-trainer-+-1-collector block — NEVER concurrent with it. If you launch d90 while the
# main block is up, the combined GPU count would exceed the <=3 cap. To chain it AFTER the main
# block without exceeding the cap, submit with a dependency on the main jobs, e.g.:
#     sbatch --dependency=afterany:<main_jobids> --array=0-119%3 xAI/scripts/submit_d90.sh
#
# d90 'v'-optimization is LIGHT (a low-dim Adam over R^d plus ViT-L forward/backward at the
# main batch size), so L4s on hpg-turin are sufficient — reserve the B200s for the main
# trainers. ViT-L forward/backward fits comfortably on an L4 at the d90 batch size.
#
# -------------------------------------------------------------------------------------------------
# BEFORE SUBMITTING, check the group's current GPU usage so you don't starve other jobs:
#     module load ufrc && slurmInfo            # allocation summary; look at GPUs running/pending
#     squeue -A guralnick -o '%.18i %.9P %.20j %.8u %.2t %.10M %.6D %R'   # who's using what
# -------------------------------------------------------------------------------------------------
#
# Generate / refresh the grid first (writes xAI/py/d90/d90_grid.txt and prints the exact array):
#     mamba run -n reticulate-gpu2 python xAI/py/d90/gen_d90_grid.py            # placeholder targets
#     mamba run -n reticulate-gpu2 python xAI/py/d90/gen_d90_grid.py \
#         --perf-table xAI/output/preadapt/perf_table.json                     # real per-condition targets
#     # -> prints "--array=0-119%3" to use below (120-line coarse grid: 3 conditions x 8 d x 5 seeds).
#
# Launch (set --array to match the grid line count; coarse grid = 120 lines -> 0-119):
#     sbatch --array=0-119%3 xAI/scripts/submit_d90.sh
#
#SBATCH --job-name=d90
#SBATCH --account=guralnick
#SBATCH --qos=guralnick
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:l4:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=12:00:00
#SBATCH --output xAI/logs/%x-%A_%a.out
#SBATCH --error  xAI/logs/%x-%A_%a.err
#SBATCH --mail-user=r.dinnage@gmail.com
#SBATCH --mail-type=FAIL

# Env init FIRST, with NO `set -u` active: a bare SLURM shell sourcing /home/$USER/.bashrc trips
# over unbound vars (e.g. BASHRCSOURCED) under `set -u`, which killed a smoke run. Source the
# profile + activate conda before enabling strict mode, then turn on `set -eo pipefail` (no -u).
# Conda env + import path (same pattern as submit_xai_plantclef.sh; preadapt_* add two_noise
# to sys.path themselves, but export it too so a bare run_subspace.py import resolves cleanly).
source /home/${USER}/.bashrc
source activate reticulate-gpu2
set -eo pipefail

PROJECT_ROOT=/blue/guralnick/share/r.dinnage/Projects/phenovision
cd "$PROJECT_ROOT"
mkdir -p xAI/logs xAI/output/d90

echo "$(date)  host=$(hostname)  cwd=$(pwd)"

export PYTHONPATH="${PWD}/PlantCLEF2022:${PWD}:${PWD}/xAI/two_noise:${PYTHONPATH:-}"

GRID=xAI/py/d90/d90_grid.txt
if [[ ! -f "$GRID" ]]; then
    echo "Grid $GRID not found; run: mamba run -n reticulate-gpu2 python xAI/py/d90/gen_d90_grid.py"
    exit 1
fi

# Pick the config for this array index (0-based -> 1-based line).
LINE=$(( SLURM_ARRAY_TASK_ID + 1 ))
ARGS=$(sed -n "${LINE}p" "$GRID")
if [[ -z "$ARGS" ]]; then
    echo "No config at line ${LINE} of $GRID (array index ${SLURM_ARRAY_TASK_ID})"
    exit 1
fi

echo "[task ${SLURM_ARRAY_TASK_ID}] run_subspace: $ARGS"
python -u xAI/py/d90/run_subspace.py $ARGS

echo "Done: $(date)"

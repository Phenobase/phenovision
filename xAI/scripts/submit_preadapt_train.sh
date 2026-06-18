#!/bin/bash
# PhenoVision preadapt TRAINER array (plan component C5).
#
# One array task = one condition's two-phase trainer (one line of the per-variant grid file) =
# one B200 GPU job. The array is throttled to %2 so AT MOST 2 trainers run at once. The single
# collector (submit_preadapt_collector.sh) takes the 3rd GPU, for 2 + 1 = 3 GPUs total.
#
# ==> GPU ACCOUNTING IS FIRM: 2 trainers (%2) + 1 collector = 3 concurrent GPUs.
#     The guralnick allocation is capped at ~3-5 concurrent GPUs and SHARED with the group.
#     NEVER raise %2 here (or run a second variant's trainers/collector concurrently) without
#     explicit group sign-off. Run variant 2 AFTER variant 1 (see run_preadapt_wave.sh).
#
# -------------------------------------------------------------------------------------------------
# BEFORE SUBMITTING, check the group's current GPU usage so you don't starve other jobs:
#     module load ufrc && slurmInfo            # allocation summary; look at GPUs running/pending
#     squeue -A guralnick -o '%.18i %.9P %.20j %.8u %.2t %.10M %.6D %R'   # who's using what
# -------------------------------------------------------------------------------------------------
#
# Generate / refresh the grids first (writes preadapt_grid_{adamw,stable_evo}.txt, 3 lines each):
#     mamba run -n reticulate-gpu2 python xAI/scripts/gen_preadapt_grid.py
#     # -> prints the exact "--array=0-2%2" to use below.
#
# Launch ONE variant (the grid file is passed via $PREADAPT_GRID; the array is 0-2 for 3 conditions):
#     PREADAPT_GRID=xAI/two_noise/configs/experiment/preadapt_grid_adamw.txt \
#         sbatch --array=0-2%2 xAI/scripts/submit_preadapt_train.sh
#
# (run_preadapt_wave.sh wraps this + the collector for a whole variant; prefer that launcher.)
#
#SBATCH --job-name=preadapt_train
#SBATCH --account=guralnick
#SBATCH --qos=guralnick
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=13G   # 8x13=104G. 2026-06-18: with num_workers=4 (grid) the trainer peaks ~91G
                            # (was ~155G at 8 workers -- DataLoader workers each fork a ~7.6G dataset
                            # copy). 104G req leaves QOS room for a 210G heavy-capable GPU collector:
                            # 2 trainers(104)+collector(210)=418 < 437. Raise back if a save OOMs.
#SBATCH --time=96:00:00
#SBATCH --output xAI/logs/%x-%A_%a.out
#SBATCH --error  xAI/logs/%x-%A_%a.err
#SBATCH --mail-user=r.dinnage@gmail.com
#SBATCH --mail-type=FAIL,END

# Env init FIRST, with NO `set -u` active: a bare SLURM shell sourcing /home/$USER/.bashrc trips
# over unbound vars (e.g. BASHRCSOURCED) under `set -u`, which killed a smoke run. Source the
# profile + activate conda before enabling strict mode, then turn on `set -eo pipefail` (no -u).
# Conda env + import paths: PlantCLEF2022 (ViT arch), project root (py modules), and xAI/py
# (preadapt_train imports preadapt_models / preadapt_common / xai_train). preadapt_train.py also
# sys.path.inserts xAI/two_noise for StableEvolutionSOAP, but we add it here too for safety.
source /home/${USER}/.bashrc
source activate reticulate-gpu2
set -eo pipefail

PROJECT_ROOT=/blue/guralnick/share/r.dinnage/Projects/phenovision
cd "$PROJECT_ROOT"
mkdir -p xAI/logs

echo "$(date)"
echo "host=$(hostname)"
echo "pwd=$(pwd)"

export PYTHONPATH="${PWD}/PlantCLEF2022:${PWD}:${PWD}/xAI/py:${PWD}/xAI/two_noise:${PYTHONPATH:-}"

# Two ways to supply the trainer arg-line:
#
#  (A) REDESIGN single-run path (run_preadapt_run.sh): PREADAPT_ARGS holds the FULL arg-line for
#      ONE (condition x variant) run. This is the non-array path -- ONE trainer GPU per run, the 6
#      runs launched SEQUENTIALLY so GPU usage never exceeds 1 trainer + 2 collector workers = 3.
#      PREADAPT_ARGS takes PRECEDENCE; when set we do NOT touch the grid or SLURM_ARRAY_TASK_ID.
#
#  (B) Legacy array path (run_preadapt_wave.sh, kept for the plumbing-check): PREADAPT_GRID names a
#      per-variant grid file and SLURM_ARRAY_TASK_ID selects the line. Used only when PREADAPT_ARGS
#      is unset.
ARGS="${PREADAPT_ARGS:-}"
if [[ -n "$ARGS" ]]; then
    echo "[single] PREADAPT_ARGS supplied (REDESIGN single-run path)."
    echo "[single] args: $ARGS"
else
    # Legacy array path. Grid file for THIS variant comes from the environment (set by
    # run_preadapt_wave.sh or sbatch --export). Optional positional override: arg 1 = grid path.
    GRID="${PREADAPT_GRID:-${1:-}}"
    if [[ -z "$GRID" ]]; then
        echo "ERROR: no trainer args. Set PREADAPT_ARGS=<full arg-line> (REDESIGN single-run), or" >&2
        echo "       PREADAPT_GRID=... with a --array index (legacy wave path)." >&2
        echo "       e.g. PREADAPT_GRID=xAI/two_noise/configs/experiment/preadapt_grid_adamw.txt" >&2
        exit 1
    fi
    if [[ ! -f "$GRID" ]]; then
        echo "ERROR: grid file '$GRID' not found. Run: mamba run -n reticulate-gpu2 python xAI/scripts/gen_preadapt_grid.py" >&2
        exit 1
    fi
    # Pick the config line for this array index (0-based -> 1-based sed line).
    LINE=$(( SLURM_ARRAY_TASK_ID + 1 ))
    ARGS=$(sed -n "${LINE}p" "$GRID")
    if [[ -z "$ARGS" ]]; then
        echo "ERROR: no config at line ${LINE} of $GRID (array index ${SLURM_ARRAY_TASK_ID})." >&2
        exit 1
    fi
    echo "[task ${SLURM_ARRAY_TASK_ID}] grid=$GRID"
    echo "[task ${SLURM_ARRAY_TASK_ID}] args: $ARGS"
fi

# B200 note (user CLAUDE.md): a "CUDA capability sm_100 not compatible" warning is a harmless
# red herring — the GPU still runs at full utilization.
mamba run -n reticulate-gpu2 python xAI/py/preadapt_train.py $ARGS

echo "Done: $(date)"

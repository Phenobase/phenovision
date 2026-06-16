#!/bin/bash
# submit_posthoc.sh — single-GPU (L4) helper for the C7 post-hoc analyses over the retention
# ladder (briefing Part II §9.2): linear mode connectivity (LMC) or filter-normalized
# loss-landscape slices. Both are POST-HOC — they run on already-retained ladder checkpoints,
# never alongside the trainers — so this is a small standalone single-GPU job.
#
# These do NOT count against the 2-trainers + 1-collector orchestration; run them after a block
# finishes (or on a 4th idle GPU within the firm <=3 concurrent cap — check slurmInfo first).
#
# -------------------------------------------------------------------------------------------------
# BEFORE SUBMITTING, check the group's current GPU usage so you don't starve other jobs:
#     module load ufrc && slurmInfo            # allocation summary; GPUs running/pending
#     squeue -A guralnick -o '%.18i %.9P %.20j %.8u %.2t %.10M %.6D %R'
# Keep total concurrent GPU jobs (trainers + collector + any post-hoc) <= 3.
# -------------------------------------------------------------------------------------------------
#
# Usage — dispatch on the first arg (lmc | slice), the rest are passed straight through:
#
#   # Linear mode connectivity (same condition, adamw vs stable_evo at one ladder step):
#   sbatch --job-name=ph_lmc scripts/submit_posthoc.sh lmc \
#       --ckpt-a xAI/output/preadapt/mae__adamw__s42/checkpoints/step00001234.pt --label-a mae_adamw \
#       --ckpt-b xAI/output/preadapt/mae__stable_evo__s42/checkpoints/step00001234.pt --label-b mae_se \
#       --steps 11 --eval-max-batches 40 --val-csv data/inat/val_v1.1.0.csv \
#       --out xAI/output/posthoc/lmc/mae_adamw_vs_se.csv
#
#   # Cross-condition LMC (annotate the verdict — barrier is EXPECTED, not the within-init test):
#   sbatch --job-name=ph_lmc scripts/submit_posthoc.sh lmc --cross-init \
#       --ckpt-a .../mae__stable_evo__s42/checkpoints/step00001234.pt --label-a mae \
#       --ckpt-b .../plantclef__stable_evo__s42/checkpoints/step00001234.pt --label-b plantclef \
#       --steps 11 --eval-max-batches 40 --val-csv data/inat/val_v1.1.0.csv \
#       --out xAI/output/posthoc/lmc/mae_vs_plantclef.csv
#
#   # Filter-normalized 2-D loss-landscape slice of one checkpoint:
#   sbatch --job-name=ph_slice scripts/submit_posthoc.sh slice \
#       --ckpt xAI/output/preadapt/mae__stable_evo__s42/checkpoints/step00001234.pt \
#       --val-csv data/inat/val_v1.1.0.csv --n-dirs 2 --grid-points 21 --alpha-range 1.0 \
#       --eval-max-batches 20 --out xAI/output/posthoc/landscape/mae_se_step1234_2d.csv
#
#SBATCH --job-name=ph_posthoc
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
#SBATCH --mail-type=FAIL,END

# Env init FIRST, with NO `set -u` active: a bare SLURM shell sourcing /home/$USER/.bashrc trips
# over unbound vars (e.g. BASHRCSOURCED) under `set -u`, which killed a smoke run. Source the
# profile + activate conda before enabling strict mode, then turn on `set -eo pipefail` (no -u).
source /home/${USER}/.bashrc
source activate reticulate-gpu2
set -eo pipefail

echo "$(date)"; echo "$(hostname)"; echo "$(pwd)"

cd /blue/guralnick/share/r.dinnage/Projects/phenovision
mkdir -p xAI/logs xAI/output/posthoc/lmc xAI/output/posthoc/landscape

# Conda env + import paths (mirror submit_xai_plantclef.sh; preadapt modules add two_noise on
# sys.path internally, but export it too so `python -m posthoc.<x>` resolves cleanly).
export PYTHONPATH="${PWD}/PlantCLEF2022:${PWD}:${PWD}/xAI/py:${PWD}/xAI/two_noise:${PYTHONPATH:-}"

MODE="${1:-}"; shift || true
if [[ "$MODE" == "lmc" ]]; then
    echo "[posthoc] mode=lmc args: $*"
    python -u -m posthoc.mode_connectivity "$@"
elif [[ "$MODE" == "slice" ]]; then
    echo "[posthoc] mode=slice args: $*"
    python -u -m posthoc.landscape_slices "$@"
else
    echo "Usage: sbatch scripts/submit_posthoc.sh <lmc|slice> [args passed through]" >&2
    echo "  lmc   -> posthoc.mode_connectivity  (LMC between two ladder checkpoints)" >&2
    echo "  slice -> posthoc.landscape_slices   (filter-normalized loss-landscape slice)" >&2
    exit 2
fi

echo "Done: $(date)"

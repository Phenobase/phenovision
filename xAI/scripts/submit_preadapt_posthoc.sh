#!/bin/bash
# PhenoVision preadapt POST-HOC pass (REDESIGN contract D) -- ONE run, runs ONCE after the trainer
# + both collector workers finish.
#
# In the REDESIGN the inherently CROSS-CHECKPOINT diagnostics are pulled OUT of the live extraction
# path (so the two collectors can process checkpoints statelessly, concurrently, out of order) and
# computed here, once per run, from the already-written per-run store + the trainer's inline log:
#   * §6.1 QL/QR rotation  -- principal angles of the stored top-k QL/QR eigvectors vs the
#                             previous step and vs the earliest/init reference.
#   * §6.3 trajectory      -- MSD(tau) + slope + plateau, straightness, velocity autocorrelation,
#                             motion flat/steep, from the stored per-step proj_coords + eigvecs and
#                             the trainer's inline velocity/path-length log.
# It writes these back into the SAME per-run store, so downstream analysis sees one unified store.
#
# This is a SINGLE L4 GPU job. run_preadapt_run.sh submits it with an afterany dependency on BOTH
# collector workers, so it starts only AFTER the run's 3 GPUs (1 trainer + 2 collectors) have freed
# -- it therefore NEVER adds a 4th concurrent GPU. The <= 3 concurrent-GPU invariant holds.
#
# Usage (prefer run_preadapt_run.sh, which wires the dependency):
#     PREADAPT_RUN_DIR=xAI/output/preadapt/mae__stable_evo__s42 \
#     PREADAPT_VARIANT=stable_evo PREADAPT_SEED=42 \
#         sbatch --dependency=afterany:<cw0>:<cw1> xAI/scripts/submit_preadapt_posthoc.sh
#
#SBATCH --job-name=preadapt_ph
#SBATCH --account=guralnick
#SBATCH --qos=guralnick
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:l4:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --time=12:00:00
#SBATCH --output xAI/logs/%x-%j.out
#SBATCH --error  xAI/logs/%x-%j.err
#SBATCH --mail-user=r.dinnage@gmail.com
#SBATCH --mail-type=FAIL,END

# Env init FIRST, with NO `set -u` active (bare SLURM shell + /home/$USER/.bashrc trips on unbound
# vars under set -u). Source profile + conda, THEN enable strict mode (no -u).
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

RUN_DIR="${PREADAPT_RUN_DIR:-${1:-}}"
if [[ -z "$RUN_DIR" ]]; then
    echo "ERROR: no run dir. Set PREADAPT_RUN_DIR=xAI/output/preadapt/<run_id> (or pass as arg 1)." >&2
    exit 1
fi
VARIANT="${PREADAPT_VARIANT:-}"
SEED="${PREADAPT_SEED:-42}"
OUT_STORE_BASE="${PREADAPT_OUT_STORE:-xAI/output/preadapt/_extract_store}"

# posthoc_sequential.py CLI: --out-store is the PER-RUN store (<base>/<run_id>, holding arrays.zarr +
# scalar parts written by the collectors); --inline-scalars is the trainer's inline log; --run-id tags rows.
RUN_ID="$(basename "$RUN_DIR")"                       # = {condition}__{variant}__s{seed}
PER_RUN_STORE="${OUT_STORE_BASE}/${RUN_ID}"
INLINE_SCALARS="${RUN_DIR}/metrics/scalars.parquet"   # trainer's velocity/path-length/proj log

echo "=== preadapt post-hoc pass ==="
echo "run_dir=$RUN_DIR variant=${VARIANT:-<unset>} seed=$SEED run_id=$RUN_ID"
echo "per_run_store=$PER_RUN_STORE  inline_scalars=$INLINE_SCALARS"

# Reads the per-run store + the trainer's inline log; writes §6.1 rotation + §6.3 trajectory back into
# the store. Robust to sparse/missing data (skips with a warning). Invoked once per run.
mamba run -n reticulate-gpu2 python xAI/py/extractor/posthoc_sequential.py \
    --out-store "$PER_RUN_STORE" \
    --inline-scalars "$INLINE_SCALARS" \
    --run-id "$RUN_ID"

echo "Done: $(date)"

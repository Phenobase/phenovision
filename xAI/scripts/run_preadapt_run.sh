#!/bin/bash
# PhenoVision preadapt RUN launcher (REDESIGN: collector throughput is the limiter).
#
# Submits ONE run = ONE (condition x variant) experiment as:
#
#     1 trainer            (submit_preadapt_train.sh -- SINGLE job, NOT an array)   -- hpg-b200 (1 GPU)
#   + 2 collector workers  (submit_preadapt_collector2.sh, worker 0 and worker 1)   -- hpg-turin (2 GPUs)
#   + 1 post-hoc job       (submit_preadapt_posthoc.sh, afterany on the 2 workers)  -- hpg-turin (1 GPU)
#                                                                                       ^^^^^^^^^^^^^^^^^
#   GPU INVARIANT: the trainer + 2 workers run TOGETHER = 3 concurrent GPUs (the FIRM <= 3 cap,
#   shared with the group). The post-hoc job depends (afterany) on BOTH collector workers, so it
#   only starts AFTER the run's GPUs have freed -- it never adds a 4th concurrent GPU. The 6 runs
#   are launched SEQUENTIALLY (run_preadapt_all.sh), so only ONE run is ever active. NEVER launch a
#   second run's jobs while this run's trainer or collectors are still running.
#
# The grids must exist first (they supply the trainer arg-line per condition/variant):
#     mamba run -n reticulate-gpu2 python xAI/scripts/gen_preadapt_grid.py
#
# Usage:
#     bash xAI/scripts/run_preadapt_run.sh <condition> <variant> [dep]
#
#   condition  mae | plantclef | naive
#   variant    adamw | stable_evo
#   dep        (optional) a SLURM dependency clause applied to the TRAINER, so this run only starts
#              after a prior run's jobs finish (the sequential-chain mechanism; run_preadapt_all.sh
#              passes "afterany:<trainer>:<cw0>:<cw1>:<posthoc>" of the prior run). Pass the full
#              SLURM form, e.g. "afterany:123456:123457". If omitted, the run starts immediately
#              (only safe when NO other preadapt run is active).
#
# Prints all submitted job ids. Does the actual sbatch (run_preadapt_all.sh is the print-only
# planner). Set DRY_RUN=1 to print the sbatch commands without submitting.
#
# -------------------------------------------------------------------------------------------------
# PRE-SUBMIT CHECK -- ALWAYS look at the group's GPU usage before launching:
#     module load ufrc && slurmInfo
#     squeue -A guralnick -o '%.18i %.9P %.20j %.8u %.2t %.10M %.6D %R'
# Confirm headroom for 3 GPUs (1 B200 trainer + 2 L4 collector workers) before you submit.
# -------------------------------------------------------------------------------------------------

set -eo pipefail

PROJECT_ROOT=/blue/guralnick/share/r.dinnage/Projects/phenovision
cd "$PROJECT_ROOT"

CONDITION="${1:-}"
VARIANT="${2:-}"
DEP="${3:-}"

case "$CONDITION" in
    mae|plantclef|naive) ;;
    *) echo "ERROR: condition must be mae | plantclef | naive (got '${CONDITION}')." >&2
       echo "Usage: bash xAI/scripts/run_preadapt_run.sh <condition> <variant> [dep]" >&2
       exit 1 ;;
esac
case "$VARIANT" in
    adamw|stable_evo) ;;
    *) echo "ERROR: variant must be adamw | stable_evo (got '${VARIANT}')." >&2
       echo "Usage: bash xAI/scripts/run_preadapt_run.sh <condition> <variant> [dep]" >&2
       exit 1 ;;
esac

SEED="${SEED:-42}"
HEAVY_EVERY="${HEAVY_EVERY:-3}"
RUN_ID="${CONDITION}__${VARIANT}__s${SEED}"
RUN_DIR="xAI/output/preadapt/${RUN_ID}"
WATCH_DIR="${RUN_DIR}/checkpoints"

# Trainer arg-line for this (condition x variant): the matching line of the per-variant grid.
GRID="xAI/two_noise/configs/experiment/preadapt_grid_${VARIANT}.txt"
if [[ ! -f "$GRID" ]]; then
    echo "ERROR: grid '$GRID' not found. Run:" >&2
    echo "  mamba run -n reticulate-gpu2 python xAI/scripts/gen_preadapt_grid.py" >&2
    exit 1
fi
# Conditions are emitted in this fixed order (gen_preadapt_grid.CONDITIONS); line = index + 1.
case "$CONDITION" in
    mae)       LINE=1 ;;
    plantclef) LINE=2 ;;
    naive)     LINE=3 ;;
esac
TRAIN_ARGS=$(sed -n "${LINE}p" "$GRID")
if [[ -z "$TRAIN_ARGS" ]]; then
    echo "ERROR: no config at line ${LINE} of $GRID for condition '$CONDITION'." >&2
    exit 1
fi

# Sanity: the line's condition/variant must match (guards against a stale / reordered grid).
if ! grep -q -- "--condition ${CONDITION} " <<<"$TRAIN_ARGS " \
   || ! grep -q -- "--variant ${VARIANT} " <<<"$TRAIN_ARGS "; then
    echo "ERROR: grid line ${LINE} of $GRID does not match condition=${CONDITION} variant=${VARIANT}:" >&2
    echo "  $TRAIN_ARGS" >&2
    echo "  Regenerate the grids: python xAI/scripts/gen_preadapt_grid.py" >&2
    exit 1
fi

DEP_ARGS=()
if [[ -n "$DEP" ]]; then
    DEP_ARGS=(--dependency="$DEP")
fi

echo "================================================================================"
echo "preadapt RUN: condition=$CONDITION variant=$VARIANT  run_id=$RUN_ID"
echo "  run_dir   : $RUN_DIR"
echo "  watch_dir : $WATCH_DIR  (PURE WORK QUEUE; kept ladder -> $RUN_DIR/kept/)"
echo "  trainer   : 1 B200 GPU (single job)   |  collectors: 2 L4 GPUs (worker 0 + worker 1)"
echo "  posthoc   : 1 L4 GPU, afterany on both collectors (starts only AFTER GPUs free)"
echo "  GPU invariant: 1 trainer + 2 collector workers = 3 concurrent GPUs (FIRM <= 3)."
[[ -n "$DEP" ]] && echo "  trainer dependency: $DEP"
echo "  trainer args: $TRAIN_ARGS"
echo "--------------------------------------------------------------------------------"

DRY_RUN="${DRY_RUN:-0}"

submit() {
    # echo + (optionally) run an sbatch; capture --parsable job id on stdout.
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "[dry-run] $*" >&2
        echo "DRYRUN_JID"
    else
        sbatch "$@"
    fi
}

# 1) Trainer (B200), SINGLE job (no --array). PREADAPT_ARGS carries the full arg-line; the trainer
#    submit script takes the PREADAPT_ARGS path (non-array) when it is set.
TRAIN_JID=$(submit --parsable "${DEP_ARGS[@]}" \
    --job-name="preadapt_train_${CONDITION}_${VARIANT}" \
    --export=ALL,PREADAPT_ARGS="$TRAIN_ARGS" \
    xAI/scripts/submit_preadapt_train.sh)
echo "[run] submitted trainer        job=$TRAIN_JID  ($RUN_ID)"

# 2) Two collector workers (L4 each). Both watch THIS run's checkpoints dir; the atomic claim makes
#    concurrent processing of different checkpoints safe. They start alongside the trainer (they
#    tolerate a not-yet-created watch dir) and receive PREADAPT_TRAIN_JID so they can exit if the
#    trainer fails without a sentinel. They depend on the SAME [dep] so the whole run starts together.
# D90_QUEUE: the persistent, SHARED d90-backfill work-queue. Both workers do d90-backfill when
# there is no extraction backlog (including the entire B200 queue-wait BEFORE the trainer starts
# and idle gaps during the run). The queue is shared across all runs so successive runs' workers
# keep chewing through the ~120-240 configs without redoing any. Generate it once with:
#   mamba run -n reticulate-gpu2 python xAI/scripts/gen_d90_backfill_queue.py
# (If the queue dir does not exist yet, a worker simply finds nothing to claim and idle-collects.)
D90_QUEUE="${D90_QUEUE:-xAI/output/preadapt/_d90_queue}"

CW_JIDS=()
for W in 0 1; do
    CW_JID=$(submit --parsable "${DEP_ARGS[@]}" \
        --job-name="preadapt_cw${W}_${CONDITION}_${VARIANT}" \
        --export=ALL,PREADAPT_RUN_DIR="$RUN_DIR",PREADAPT_WORKER_ID="$W",PREADAPT_VARIANT="$VARIANT",PREADAPT_SEED="$SEED",HEAVY_EVERY="$HEAVY_EVERY",PREADAPT_TRAIN_JID="$TRAIN_JID",D90_QUEUE="$D90_QUEUE" \
        xAI/scripts/submit_preadapt_collector2.sh)
    CW_JIDS+=("$CW_JID")
    echo "[run] submitted collector w$W   job=$CW_JID  (watching $WATCH_DIR; d90_queue=$D90_QUEUE)"
done

# 3) Post-hoc pass (L4), afterany on BOTH collector workers -> starts only after the run's GPUs free
#    (so it NEVER adds a 4th concurrent GPU). It computes the cross-checkpoint diagnostics (§6.1
#    QL/QR rotation, §6.3 trajectory) from the per-run store + the trainer's inline velocity/path log.
PH_DEP="afterany:${CW_JIDS[0]}:${CW_JIDS[1]}"
PH_JID=$(submit --parsable \
    --job-name="preadapt_ph_${CONDITION}_${VARIANT}" \
    --dependency="$PH_DEP" \
    --export=ALL,PREADAPT_RUN_DIR="$RUN_DIR",PREADAPT_VARIANT="$VARIANT",PREADAPT_SEED="$SEED" \
    xAI/scripts/submit_preadapt_posthoc.sh)
echo "[run] submitted posthoc         job=$PH_JID  (dependency=$PH_DEP)"

echo "--------------------------------------------------------------------------------"
echo "[run] JOB IDS  trainer=$TRAIN_JID  collectors=${CW_JIDS[0]},${CW_JIDS[1]}  posthoc=$PH_JID"
# Emit a machine-parseable line so run_preadapt_all.sh can chain the NEXT run on ALL of these.
echo "RUN_JIDS ${CONDITION} ${VARIANT} ${TRAIN_JID} ${CW_JIDS[0]} ${CW_JIDS[1]} ${PH_JID}"
echo "[run] monitor:"
echo "  squeue -j ${TRAIN_JID},${CW_JIDS[0]},${CW_JIDS[1]},${PH_JID}"
echo "  squeue -A guralnick -o '%.18i %.9P %.20j %.8u %.2t %.10M %.6D %R'"
echo "  tail -f xAI/logs/preadapt_train_${CONDITION}_${VARIANT}-${TRAIN_JID}.out"
echo "  tail -f xAI/logs/preadapt_cw0_${CONDITION}_${VARIANT}-${CW_JIDS[0]}.out"
echo "================================================================================"

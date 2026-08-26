#!/bin/bash
# PhenoVision preadapt WAVE launcher (plan component C5) — thin orchestration wrapper.
#
# Submits, for ONE variant, the collector job + the 3-condition trainer array (throttled %2):
#
#     2 trainers (%2, hpg-b200)  +  1 collector (hpg-turin)  =  3 concurrent GPUs.
#                                                               ^^^^^^^^^^^^^^^^^^^^^
#   FIRM CAP. The guralnick allocation is capped at ~3 concurrent GPUs and SHARED with the group.
#   ==> Run ONE variant at a time. NEVER launch variant 2's wave while variant 1's wave is still
#       running (that would be 4 trainers + 2 collectors = 6 GPUs). See "RUNNING VARIANT 2" below.
#       %2 + 1 collector = 3 GPUs is the whole budget; do NOT raise %2 without group sign-off.
#
# The grids must exist first:
#     mamba run -n reticulate-gpu2 python xAI/scripts/gen_preadapt_grid.py
#
# Usage:
#     bash xAI/scripts/run_preadapt_wave.sh <adamw|stable_evo> [afterok|afterany:<jobid>]
#
#   arg 1  variant to run this wave for.
#   arg 2  (optional) a SLURM dependency for BOTH submitted jobs, so this wave only starts after a
#          prior wave's jobs finish — the mechanism for "variant 2 after variant 1" without ever
#          exceeding 3 GPUs. Pass the form SLURM expects, e.g. "afterany:123456" or
#          "afterany:123456:123457". If omitted, the wave is submitted to start immediately
#          (only safe when NO other preadapt wave is running).
#
# -------------------------------------------------------------------------------------------------
# PRE-SUBMIT CHECK (briefing) — ALWAYS look at the group's GPU usage before launching:
#     module load ufrc && slurmInfo            # allocation summary; GPUs running / pending
#     squeue -A guralnick -o '%.18i %.9P %.20j %.8u %.2t %.10M %.6D %R'
# Confirm there is headroom for 3 GPUs (2 B200 trainers + 1 L4 collector) before you sbatch.
# -------------------------------------------------------------------------------------------------

# NOTE: this launcher does NOT activate conda (it only sbatches). We still drop `-u` for
# consistency with the worker scripts and because it dereferences optional positional args.
set -eo pipefail

PROJECT_ROOT=/blue/guralnick/share/r.dinnage/Projects/phenovision
cd "$PROJECT_ROOT"

VARIANT="${1:-}"
DEP="${2:-}"
case "$VARIANT" in
    adamw|stable_evo) ;;
    *) echo "ERROR: variant must be 'adamw' or 'stable_evo' (got '${VARIANT}')." >&2
       echo "Usage: bash xAI/scripts/run_preadapt_wave.sh <adamw|stable_evo> [afterany:<jobid>]" >&2
       exit 1 ;;
esac

GRID="xAI/two_noise/configs/experiment/preadapt_grid_${VARIANT}.txt"
if [[ ! -f "$GRID" ]]; then
    echo "ERROR: grid '$GRID' not found. Run:" >&2
    echo "  mamba run -n reticulate-gpu2 python xAI/scripts/gen_preadapt_grid.py" >&2
    exit 1
fi
N_LINES=$(grep -cve '^[[:space:]]*$' "$GRID")
ARRAY_SPEC="0-$(( N_LINES - 1 ))%2"   # 3 conditions -> 0-2%2 (2 concurrent trainers)

# Optional dependency clause applied to BOTH jobs (variant-2-after-variant-1 scheduling).
DEP_ARGS=()
if [[ -n "$DEP" ]]; then
    DEP_ARGS=(--dependency="$DEP")
    echo "[wave] dependency: jobs will start only after '$DEP'."
fi

echo "=== preadapt wave: variant=$VARIANT  grid=$GRID  array=$ARRAY_SPEC ==="
echo "[wave] PRE-SUBMIT: verify GPU headroom for 3 GPUs (2 B200 trainers + 1 L4 collector):"
echo "       module load ufrc && slurmInfo"
echo "       squeue -A guralnick -o '%.18i %.9P %.20j %.8u %.2t %.10M %.6D %R'"
echo

# 1) Trainer array (B200), throttled %2 = 2 concurrent trainers. Submit FIRST so we can capture
#    its job id and hand it to the collector as PREADAPT_TRAIN_JID — that lets the collector
#    terminate (after a final drain) if a trainer FAILS without writing its RUN_COMPLETE
#    sentinel. Producer/consumer are filesystem-decoupled: the collector tolerates watch dirs
#    that don't exist yet, so submitting the trainers before the collector is safe.
TRAIN_JID=$(sbatch --parsable "${DEP_ARGS[@]}" \
    --job-name="preadapt_train_${VARIANT}" \
    --array="$ARRAY_SPEC" \
    --export=ALL,PREADAPT_GRID="$GRID" \
    xAI/scripts/submit_preadapt_train.sh)
echo "[wave] submitted trainers   job=$TRAIN_JID  array=$ARRAY_SPEC  (variant=$VARIANT)"

# 2) Collector (L4). It skips not-yet-created watch dirs until the trainers make them, and it
#    receives the trainer array job id (PREADAPT_TRAIN_JID) so it can exit once the trainer array
#    has left the queue even if some task failed before writing a sentinel.
COLLECT_JID=$(sbatch --parsable "${DEP_ARGS[@]}" \
    --job-name="preadapt_collect_${VARIANT}" \
    --export=ALL,PREADAPT_VARIANT="$VARIANT",PREADAPT_TRAIN_JID="$TRAIN_JID" \
    xAI/scripts/submit_preadapt_collector.sh)
echo "[wave] submitted collector  job=$COLLECT_JID  (variant=$VARIANT, watching train job=$TRAIN_JID)"

echo
echo "[wave] monitor:"
echo "  squeue -A guralnick -o '%.18i %.9P %.20j %.8u %.2t %.10M %.6D %R'"
echo "  squeue -j ${TRAIN_JID},${COLLECT_JID}"
echo "  module load ufrc && slurmInfo"
echo "  tail -f xAI/logs/preadapt_train_${VARIANT}-${TRAIN_JID}_*.out"
echo "  tail -f xAI/logs/preadapt_collect_${VARIANT}-${COLLECT_JID}.out"
echo
echo "================================================================================"
echo "RUNNING VARIANT 2 (after variant 1) — keep <=3 concurrent GPUs at all times"
echo "================================================================================"
echo "The collector exits once all 3 RUN_COMPLETE sentinels exist and the final drain is empty —"
echo "or, if a trainer FAILS without a sentinel, once the trainer array (job=${TRAIN_JID}) leaves the"
echo "queue (it knows the id via PREADAPT_TRAIN_JID). Either way a wave's GPUs free up only when"
echo "BOTH its trainer array AND its collector have finished."
echo "Launch the second variant with a dependency on BOTH of THIS wave's job ids:"
echo
echo "  bash xAI/scripts/run_preadapt_wave.sh <other_variant> afterany:${TRAIN_JID}:${COLLECT_JID}"
echo
echo "Or simply wait for this wave to finish (squeue empty for these job ids) and run the second"
echo "wave with no dependency. Either way: ONE wave (3 GPUs) at a time — never two waves at once."

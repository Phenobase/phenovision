#!/bin/bash
# PhenoVision preadapt v2 WAVE launcher (CPU/GPU two-pass collector split).
#
# Launches ONE wave = up to 2 trainers (selected by grid line number) + 1 SHARED GPU two-pass
# collector + N CPU two-pass collectors (default N=4) + a per-run posthoc (afterany on the wave's
# collectors). All of the wave's trainers + collectors share ONE GPU two-pass collector and N CPU
# collectors, all watching the wave's run dirs.
#
#   GPU ACCOUNTING PER WAVE (FIRM <= 3):
#       2 trainers (B200)            = 2 GPU
#     + 1 GPU two-pass collector (L4) = 1 GPU
#     + N CPU two-pass collectors     = 0 GPU (hpg-default, CPU-only)
#     + per-run posthoc (afterany)    = 0 GPU concurrently (starts only AFTER the collectors free)
#       ----------------------------------------------------------------
#       = 3 concurrent GPUs at peak. With a 1-trainer (solo) wave it is 1 + 1 = 2 GPUs.
#   NEVER raise the GPU collector count or the trainer count per wave without group sign-off.
#
# TWO-PASS CORRECTNESS (FIRM): all collectors run --require-passes gpu,cpu. v2 lives in a DISTINCT
# tree xAI/output/preadapt_v2/ and is NEVER watched by a legacy single-pass collector (and the live
# v1 tree xAI/output/preadapt/ is NEVER watched by a two-pass worker). A checkpoint is disposed only
# after BOTH the .gpu.complete and .cpu.complete sentinels exist.
#
# v2 GRID: xAI/two_noise/configs/experiment/preadapt_v2_grid.txt (6 lines, stable_evo only):
#     round1 {mae,plantclef,naive}@s42 = lines 1,2,3 ; round2 @s43 = lines 4,5,6.
# run_id = {condition}__stable_evo__s{seed} (derived from each line's --condition + --seed). The
# grid lines do NOT include --output_dir; we append --output_dir xAI/output/preadapt_v2/<run_id>.
#
# Usage:
#     bash xAI/scripts/run_preadapt_v2_wave.sh <line1> [line2] [dep]            # print plan (default)
#     bash xAI/scripts/run_preadapt_v2_wave.sh <line1> [line2] [dep] --submit   # actually submit
#
#   <lineN>  1-based line number(s) into preadapt_v2_grid.txt (1 or 2 lines).
#   dep      (optional) a SLURM dependency clause applied to the TRAINERS + collectors so this wave
#            starts only after a prior wave's jobs finish (the round-barrier mechanism;
#            run_preadapt_v2_all.sh passes "afterany:<...>"). Pass the full SLURM form.
#
# MODE: PRINT-ONLY by default (echo the sbatch plan, do NOT submit). Pass --submit to submit.
# Set N_CPU_COLLECTORS to change the CPU collector count (default 3, sized to the QOS). Set DRY_RUN=1 in submit mode
# to echo the sbatch commands without running them.
#
# -------------------------------------------------------------------------------------------------
# PRE-SUBMIT CHECK -- ALWAYS verify the group's GPU usage before launching:
#     module load ufrc && slurmInfo
#     squeue -A guralnick -o '%.18i %.9P %.20j %.8u %.2t %.10M %.6D %R'
# Confirm headroom for up to 3 GPUs (2 B200 trainers + 1 L4 GPU collector).
# -------------------------------------------------------------------------------------------------

set -eo pipefail

PROJECT_ROOT=/blue/guralnick/share/r.dinnage/Projects/phenovision
cd "$PROJECT_ROOT"

GRID="xAI/two_noise/configs/experiment/preadapt_v2_grid.txt"
OUT_ROOT="xAI/output/preadapt_v2"
OUT_STORE="${PREADAPT_OUT_STORE:-${OUT_ROOT}/_extract_store}"
D90_QUEUE="${D90_QUEUE:-${OUT_ROOT}/_d90_queue}"
REQUIRE_PASSES="${PREADAPT_REQUIRE_PASSES:-gpu,cpu}"
N_CPU_COLLECTORS="${N_CPU_COLLECTORS:-3}"   # sized to the investment QOS: 2 trainers(8cpu) +
                                            # 1 GPU collector(8cpu) + 3 CPU collectors(8cpu) = 48
                                            # CPU (<=56); mem 224+112+3*24=408G (<=437). 3 CPU
                                            # workers absorb the ~85s/ckpt CPU pass under the
                                            # ~80s/emit cadence. Raise via env only on burst QOS.
HEAVY_EVERY="${HEAVY_EVERY:-3}"

# --- parse args: 1-2 line numbers, an optional dep clause, and an optional --submit flag ---------
MODE="print"
DEP=""
declare -a LINES=()
for a in "$@"; do
    case "$a" in
        --submit)  MODE="submit" ;;
        --print|--dry-run) MODE="print" ;;
        afterany:*|after:*|afterok:*|afternotok:*|singleton)
            DEP="$a" ;;
        ''|*[!0-9]*)
            echo "ERROR: unrecognized arg '$a' (expected grid line number, a SLURM dep clause, or --submit)." >&2
            exit 1 ;;
        *)  LINES+=("$a") ;;
    esac
done

if [[ ${#LINES[@]} -lt 1 || ${#LINES[@]} -gt 2 ]]; then
    echo "ERROR: give 1 or 2 grid line numbers (got ${#LINES[@]})." >&2
    echo "Usage: bash xAI/scripts/run_preadapt_v2_wave.sh <line1> [line2] [dep] [--submit]" >&2
    exit 1
fi
if [[ ! -f "$GRID" ]]; then
    echo "ERROR: v2 grid '$GRID' not found." >&2
    exit 1
fi

# --- derive per-line run_id / run_dir / trainer arg-line from the grid ----------------------------
declare -a RUN_IDS RUN_DIRS TRAIN_ARGS_LIST
for ln in "${LINES[@]}"; do
    ARGS=$(sed -n "${ln}p" "$GRID")
    if [[ -z "$ARGS" ]]; then
        echo "ERROR: no config at line ${ln} of $GRID." >&2
        exit 1
    fi
    # Parse --condition and --seed from the arg-line (token after the flag).
    COND=$(sed -n 's/.*--condition \([^ ]*\).*/\1/p' <<<"$ARGS")
    SEED=$(sed -n 's/.*--seed \([^ ]*\).*/\1/p' <<<"$ARGS")
    if [[ -z "$COND" || -z "$SEED" ]]; then
        echo "ERROR: grid line ${ln} missing --condition and/or --seed: $ARGS" >&2
        exit 1
    fi
    RID="${COND}__stable_evo__s${SEED}"
    RDIR="${OUT_ROOT}/${RID}"
    RUN_IDS+=("$RID")
    RUN_DIRS+=("$RDIR")
    # Grid lines do NOT include --output_dir; append it here.
    TRAIN_ARGS_LIST+=("$ARGS --output_dir $RDIR")
done

# Colon-separated run dirs for the shared collectors (watch every run in this wave).
RUN_DIRS_COLON="$(IFS=:; echo "${RUN_DIRS[*]}")"

N_GPU=$(( ${#LINES[@]} + 1 ))   # trainers + 1 GPU collector
echo "================================================================================"
echo "preadapt v2 WAVE  (lines: ${LINES[*]})   mode: $MODE"
echo "  runs        : ${RUN_IDS[*]}"
echo "  out_root    : $OUT_ROOT      out_store: $OUT_STORE"
echo "  require-passes: $REQUIRE_PASSES   (two-pass; v2 tree only)"
echo "  collectors  : 1 GPU (L4) + ${N_CPU_COLLECTORS} CPU (hpg-default), all watch: $RUN_DIRS_COLON"
echo "  d90_queue   : $D90_QUEUE   (GPU collector only)"
[[ -n "$DEP" ]] && echo "  dependency  : $DEP"
echo "  GPU accounting: ${#LINES[@]} trainer(s) + 1 GPU collector = ${N_GPU} concurrent GPU (CPU collectors + posthoc add 0) <= 3"
echo "--------------------------------------------------------------------------------"

DRY_RUN="${DRY_RUN:-0}"

submit() {
    # echo + (optionally) run an sbatch; capture --parsable job id on stdout.
    if [[ "$MODE" == "print" || "$DRY_RUN" == "1" ]]; then
        echo "[plan] sbatch $*" >&2
        echo "PLAN_JID"
    else
        sbatch "$@"
    fi
}

DEP_ARGS=()
if [[ -n "$DEP" ]]; then
    DEP_ARGS=(--dependency="$DEP")
fi

# 1) Trainers (B200), one SINGLE job per grid line (no --array). PREADAPT_ARGS carries the full
#    arg-line incl. --output_dir; the trainer submit script takes the PREADAPT_ARGS path when set.
declare -a TRAIN_JIDS=()
for i in "${!LINES[@]}"; do
    RID="${RUN_IDS[$i]}"
    TARGS="${TRAIN_ARGS_LIST[$i]}"
    TJ=$(submit --parsable "${DEP_ARGS[@]}" \
        --job-name="preadapt_train_${RID}" \
        --export=ALL,PREADAPT_ARGS="$TARGS" \
        xAI/scripts/submit_preadapt_train.sh)
    TRAIN_JIDS+=("$TJ")
    echo "[wave] trainer        ${RID}  job=$TJ"
done
TRAIN_JIDS_CSV="$(IFS=,; echo "${TRAIN_JIDS[*]}")"

# 2) ONE shared GPU two-pass collector (L4). Watches every run dir in this wave; terminates on this
#    wave's trainers + RUN_COMPLETE sentinels. --require-passes gpu,cpu (FIRM). D90_QUEUE enables
#    idle-GPU d90 backfill. It depends on the SAME [dep] so the whole wave starts together.
GPU_CW_JID=$(submit --parsable "${DEP_ARGS[@]}" \
    --job-name="preadapt_cwgpu_w${LINES[0]}" \
    --export=ALL,PREADAPT_RUN_DIRS="$RUN_DIRS_COLON",PREADAPT_WORKER_ID=0,PREADAPT_OUT_STORE="$OUT_STORE",PREADAPT_REQUIRE_PASSES="$REQUIRE_PASSES",PREADAPT_TRAIN_JIDS="$TRAIN_JIDS_CSV",HEAVY_EVERY="$HEAVY_EVERY",D90_QUEUE="$D90_QUEUE" \
    xAI/scripts/submit_preadapt_collector_gpu.sh)
echo "[wave] GPU collector  job=$GPU_CW_JID  (watch=$RUN_DIRS_COLON; require-passes=$REQUIRE_PASSES; d90_queue=$D90_QUEUE)"

# 3) N CPU two-pass collectors (hpg-default, 0 GPU). Each watches every run dir; same termination.
#    --require-passes gpu,cpu (FIRM). No d90 (d90 is GPU work). Depend on the SAME [dep].
declare -a CPU_CW_JIDS=()
for ((w=0; w<N_CPU_COLLECTORS; w++)); do
    CJ=$(submit --parsable "${DEP_ARGS[@]}" \
        --job-name="preadapt_cwcpu${w}_w${LINES[0]}" \
        --export=ALL,PREADAPT_RUN_DIRS="$RUN_DIRS_COLON",PREADAPT_WORKER_ID="$w",PREADAPT_OUT_STORE="$OUT_STORE",PREADAPT_REQUIRE_PASSES="$REQUIRE_PASSES",PREADAPT_TRAIN_JIDS="$TRAIN_JIDS_CSV",HEAVY_EVERY="$HEAVY_EVERY" \
        xAI/scripts/submit_preadapt_collector_cpu.sh)
    CPU_CW_JIDS+=("$CJ")
    echo "[wave] CPU collector ${w}  job=$CJ"
done

# All collector job ids (GPU + CPU) -> the posthoc afterany dependency + the wave's collector set.
declare -a COLLECTOR_JIDS=("$GPU_CW_JID" "${CPU_CW_JIDS[@]}")
COLLECTORS_AFTERANY="afterany:$(IFS=:; echo "${COLLECTOR_JIDS[*]}")"

# 4) Per-run posthoc (L4), afterany on ALL of this wave's collectors -> starts only AFTER the wave's
#    GPUs free, so it adds no concurrent GPU. One posthoc per run (each run has its own per-run store).
declare -a POSTHOC_JIDS=()
for i in "${!RUN_IDS[@]}"; do
    RID="${RUN_IDS[$i]}"
    RDIR="${RUN_DIRS[$i]}"
    SEED=$(sed -n 's/.*--seed \([^ ]*\).*/\1/p' <<<"${TRAIN_ARGS_LIST[$i]}")
    PJ=$(submit --parsable \
        --job-name="preadapt_ph_${RID}" \
        --dependency="$COLLECTORS_AFTERANY" \
        --export=ALL,PREADAPT_RUN_DIR="$RDIR",PREADAPT_OUT_STORE="$OUT_STORE",PREADAPT_VARIANT=stable_evo,PREADAPT_SEED="$SEED" \
        xAI/scripts/submit_preadapt_posthoc.sh)
    POSTHOC_JIDS+=("$PJ")
    echo "[wave] posthoc        ${RID}  job=$PJ  (dependency=$COLLECTORS_AFTERANY)"
done

echo "--------------------------------------------------------------------------------"
# Machine-parseable line: trainer ids / collector ids / posthoc ids (each space-joined within field;
# fields separated by ' | '). run_preadapt_v2_all.sh parses this to build the round barrier.
echo "WAVE_JIDS ${TRAIN_JIDS[*]} | ${COLLECTOR_JIDS[*]} | ${POSTHOC_JIDS[*]}"
echo "[wave] GPU concurrency: ${#LINES[@]} trainer(s) + 1 GPU collector = ${N_GPU} (<= 3). CPU collectors + posthoc add 0."
if [[ "$MODE" == "print" ]]; then
    echo "[wave] PRINT-ONLY: nothing submitted. Re-run with --submit to launch this wave."
fi
echo "[wave] monitor:"
echo "  squeue -A guralnick -o '%.18i %.9P %.20j %.8u %.2t %.10M %.6D %R'"
echo "================================================================================"

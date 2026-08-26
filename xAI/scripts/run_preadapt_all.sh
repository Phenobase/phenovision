#!/bin/bash
# PhenoVision preadapt ALL-RUNS launcher (REDESIGN: collector throughput is the limiter).
#
# Chains ALL 6 runs (3 conditions x 2 variants) STRICTLY SEQUENTIALLY so that only ONE run -- and
# thus at most 3 concurrent GPUs (1 B200 trainer + 2 L4 collector workers) -- is ever active.
#
#   GPU INVARIANT (enforced by the sequential chain): each run is 1 trainer + 2 collector workers =
#   3 concurrent GPUs (the FIRM <= 3 cap, shared with the group). Run N+1's trainer + collectors
#   depend (afterany) on EVERY job of run N -- its trainer, BOTH collector workers, AND its post-hoc
#   job -- so run N+1 cannot start any GPU job until run N has fully finished and freed its GPUs.
#   The per-run post-hoc job itself runs afterany on that run's 2 collectors (so it does not add a
#   4th GPU within a run); chaining run N+1 on run N's post-hoc too keeps the runs cleanly serial.
#
#   At NO point are two runs' GPU jobs co-resident. Total GPUs in flight <= 3 at all times.
#
# The 6 runs are ordered by condition then variant:
#     1. mae       x adamw
#     2. mae       x stable_evo
#     3. plantclef x adamw
#     4. plantclef x stable_evo
#     5. naive     x adamw
#     6. naive     x stable_evo
#
# DEFAULT BEHAVIOR: PRINT-ONLY. This script prints the full 6-run sequential submission PLAN and the
# monitor commands but does NOT sbatch anything -- so you can review it and run the commands (or
# `--submit`) yourself. Pass --submit to actually submit the chain (each run's jobs become the next
# run's dependency, captured live). --dry-run is an explicit alias for the default print-only mode.
#
# The grids must exist first:
#     mamba run -n reticulate-gpu2 python xAI/scripts/gen_preadapt_grid.py
#
# Usage:
#     bash xAI/scripts/run_preadapt_all.sh              # print the plan (default; no sbatch)
#     bash xAI/scripts/run_preadapt_all.sh --dry-run    # same as default (explicit)
#     bash xAI/scripts/run_preadapt_all.sh --submit     # actually submit the sequential chain
#
# -------------------------------------------------------------------------------------------------
# PRE-SUBMIT CHECK -- ALWAYS verify the group's GPU usage before launching:
#     module load ufrc && slurmInfo
#     squeue -A guralnick -o '%.18i %.9P %.20j %.8u %.2t %.10M %.6D %R'
# Confirm headroom for 3 GPUs (1 B200 + 2 L4). The chain self-limits to 3, but other group jobs
# share the allocation.
# -------------------------------------------------------------------------------------------------

set -eo pipefail

PROJECT_ROOT=/blue/guralnick/share/r.dinnage/Projects/phenovision
cd "$PROJECT_ROOT"

MODE="print"   # print (default) | submit
case "${1:-}" in
    ""|--dry-run|--print) MODE="print" ;;
    --submit)             MODE="submit" ;;
    *) echo "ERROR: unknown arg '${1}'. Use --submit, --dry-run, or no arg (print)." >&2
       echo "Usage: bash xAI/scripts/run_preadapt_all.sh [--submit|--dry-run]" >&2
       exit 1 ;;
esac

# The 6 runs in sequential order: "<condition> <variant>".
RUNS=(
    "mae adamw"
    "mae stable_evo"
    "plantclef adamw"
    "plantclef stable_evo"
    "naive adamw"
    "naive stable_evo"
)

# Verify both grids exist (run_preadapt_run.sh reads the trainer arg-line from them).
for v in adamw stable_evo; do
    g="xAI/two_noise/configs/experiment/preadapt_grid_${v}.txt"
    if [[ ! -f "$g" ]]; then
        echo "ERROR: grid '$g' not found. Run:" >&2
        echo "  mamba run -n reticulate-gpu2 python xAI/scripts/gen_preadapt_grid.py" >&2
        exit 1
    fi
done

echo "================================================================================"
echo "preadapt ALL-RUNS plan  (6 runs SEQUENTIAL; <= 3 concurrent GPUs at all times)"
echo "  mode: $MODE"
echo "  each run = 1 trainer (B200) + 2 collector workers (2x L4) = 3 GPUs"
echo "  run N+1's jobs depend (afterany) on ALL of run N's jobs -> strictly one run active"
echo "================================================================================"
echo

if [[ "$MODE" == "print" ]]; then
    # PRINT-ONLY: show the exact ordered invocations + how the dependency chain threads through.
    echo "Submission PLAN (run these in order, or re-run with --submit to do it automatically):"
    echo
    echo "  # First run starts immediately (no dependency):"
    n=1
    prev_desc=""
    for spec in "${RUNS[@]}"; do
        read -r cond var <<<"$spec"
        if [[ -z "$prev_desc" ]]; then
            echo "  ${n}. bash xAI/scripts/run_preadapt_run.sh ${cond} ${var}"
        else
            echo "  ${n}. # AFTER run $((n-1)) ($prev_desc) fully finishes (trainer + 2 collectors + posthoc):"
            echo "     bash xAI/scripts/run_preadapt_run.sh ${cond} ${var} \\"
            echo "         afterany:<run$((n-1))_trainer>:<run$((n-1))_cw0>:<run$((n-1))_cw1>:<run$((n-1))_posthoc>"
        fi
        prev_desc="${cond} x ${var}"
        n=$((n+1))
    done
    echo
    echo "Each run_preadapt_run.sh prints a 'RUN_JIDS <cond> <var> <trainer> <cw0> <cw1> <posthoc>'"
    echo "line; use those four ids as the NEXT run's afterany dependency (as shown). --submit does"
    echo "this automatically by capturing each run's ids."
    echo
    echo "GPU invariant: with the afterany chain, only ONE run's jobs (1 trainer + 2 collectors = 3"
    echo "GPUs) are ever co-resident. The per-run posthoc runs after that run's collectors free, so"
    echo "it adds no 4th GPU. Total concurrent GPUs <= 3 throughout."
    echo
    echo "MONITOR:"
    echo "  squeue -A guralnick -o '%.18i %.9P %.20j %.8u %.2t %.10M %.6D %R'"
    echo "  module load ufrc && slurmInfo"
    echo "  watch -n 60 \"squeue -A guralnick -o '%.18i %.20j %.2t %.10M %R'\""
    echo
    echo "To submit the whole chain automatically:  bash xAI/scripts/run_preadapt_all.sh --submit"
    echo "================================================================================"
    exit 0
fi

# --- MODE == submit: actually launch the sequential chain, threading dependencies ----------------
echo "Submitting the 6-run sequential chain. Each run depends on ALL of the previous run's jobs."
echo

PREV_DEP=""           # SLURM dependency clause for the next run's trainer + collectors
n=1
for spec in "${RUNS[@]}"; do
    read -r cond var <<<"$spec"
    echo "--------------------------------------------------------------------------------"
    echo "[all] launching run ${n}/6: ${cond} x ${var}  ${PREV_DEP:+(dep=$PREV_DEP)}"
    # run_preadapt_run.sh does the actual sbatch of trainer + 2 collectors + posthoc and prints a
    # 'RUN_JIDS <cond> <var> <trainer> <cw0> <cw1> <posthoc>' line we parse to chain the next run.
    OUT="$(bash xAI/scripts/run_preadapt_run.sh "$cond" "$var" "$PREV_DEP")"
    echo "$OUT"
    JIDLINE="$(grep -E '^RUN_JIDS ' <<<"$OUT" | tail -n1 || true)"
    if [[ -z "$JIDLINE" ]]; then
        echo "[all] ERROR: run_preadapt_run.sh did not emit a RUN_JIDS line for ${cond} x ${var}." >&2
        echo "[all] Aborting the chain; runs already submitted are unaffected." >&2
        exit 1
    fi
    # RUN_JIDS <cond> <var> <trainer> <cw0> <cw1> <posthoc>
    read -r _tag _c _v TJ CW0 CW1 PHJ <<<"$JIDLINE"
    # The NEXT run depends on EVERY job of THIS run (trainer + both collectors + posthoc) so the
    # next run cannot start any GPU job until this run is entirely done -> strictly serial, <= 3 GPUs.
    PREV_DEP="afterany:${TJ}:${CW0}:${CW1}:${PHJ}"
    echo "[all] run ${n}/6 (${cond} x ${var}) submitted: trainer=$TJ cw0=$CW0 cw1=$CW1 posthoc=$PHJ"
    echo "[all] next run will depend on: $PREV_DEP"
    n=$((n+1))
done

echo "--------------------------------------------------------------------------------"
echo "[all] all 6 runs submitted as a sequential chain (<= 3 concurrent GPUs throughout)."
echo "[all] MONITOR:"
echo "  squeue -A guralnick -o '%.18i %.9P %.20j %.8u %.2t %.10M %.6D %R'"
echo "  module load ufrc && slurmInfo"
echo "================================================================================"

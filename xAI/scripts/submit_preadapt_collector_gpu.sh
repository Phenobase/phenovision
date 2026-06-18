#!/bin/bash
# PhenoVision preadapt v2 GPU two-pass COLLECTOR WORKER (CPU/GPU split, C4/C5).
#
# Companion to submit_preadapt_collector_cpu.sh (the CPU worker). This worker runs ONLY the GPU
# block set -- curvature / fitness / interp / probes: it builds the probe/Hessian batches and does
# the model forward / HVP, so it NEEDS a GPU (L4) and the val CSV (the fixed probe set).
#
# CPU/GPU SPLIT (two-pass refcount): every worker -- GPU and CPU -- passes --require-passes gpu,cpu.
# A checkpoint is disposed (deleted / moved to kept/) only once BOTH the .gpu.complete and the
# .cpu.complete sentinels exist, so the GPU worker and the CPU worker(s) can process the SAME
# checkpoint concurrently and out of order. The atomic per-(checkpoint, pass) claim in
# preadapt_common guarantees exactly-once within each pass; the finalize sweep (one-winner rename)
# disposes once both passes complete.
#
# ==> CORRECTNESS INVARIANT (FIRM): in two-pass mode EVERY collector watching a dir MUST pass
#     --require-passes gpu,cpu. A LEGACY single-pass worker (--require-passes all) on a v2 dir would
#     finalize/delete a checkpoint after ONE pass and corrupt the refcount. So this worker runs
#     --block-set gpu --require-passes gpu,cpu, and v2 dirs (xAI/output/preadapt_v2/) must NEVER be
#     watched by a legacy collector (and the LIVE v1 tree xAI/output/preadapt/ -- single-pass -- must
#     NEVER be watched by a two-pass worker).
#
# This worker can watch MULTIPLE run dirs at once: set PREADAPT_RUN_DIRS to a colon-separated list of
# run output dirs; each expands to a <dir>/checkpoints --watch-dirs argument. (PREADAPT_RUN_DIR --
# singular -- is still accepted for a single run, matching the CPU script.)
#
# v2 RUN SET (StableEvo only): 6 runs = {mae,plantclef,naive} x {seed 42, seed 43}, run_id =
# <condition>__stable_evo__s<seed>, output ROOT xAI/output/preadapt_v2/<run_id>. Example watching a
# wave's two runs:
#     PREADAPT_RUN_DIRS="xAI/output/preadapt_v2/mae__stable_evo__s42:xAI/output/preadapt_v2/plantclef__stable_evo__s42"
#
# TERMINATION (driven from bash; collector.py is NOT modified for this): we call `collector.py
# --once` (drain-then-exit) in a loop every $POLL seconds. We stop only when BOTH hold:
#   (1) EVERY watched run's RUN_COMPLETE sentinel exists -- written by preadapt_train.py main() at
#       <output_dir>/RUN_COMPLETE AFTER the last checkpoint, AND
#   (2) a final drain `--once` AFTER all sentinels appear processes 0 new checkpoints AND the
#       aggregate GPU-PASS work queue is EMPTY.
# PASS-AWARE QUEUE/TERMINATION (key difference from a single-pass worker): "pending FOR THIS WORKER"
# = checkpoints ready for the *gpu* pass (no <base>.pt.gpu.complete yet). A checkpoint that already
# has THIS worker's own .gpu.complete is NOT pending-for-this-worker even if the cpu pass still lags
# -- so the GPU worker can terminate (and free its GPU) once its own pass is done on every emitted
# checkpoint, without waiting for the CPU worker. The eventual FINALIZE sweep (run by whichever
# worker observes both .complete sentinels) still disposes the checkpoint once both passes finish.
# Failed-trainer guard (PREADAPT_TRAIN_JIDS, comma-separated): if NONE of the trainer jobs is still
# in the queue (they FAILED/finished without all sentinels), we do one final drain and exit rather
# than spin to the wall limit.
#
# d90 BACKFILL: when GPU-pass extraction has no backlog the worker claims + runs ONE d90 config on
# its GPU (guarded by D90_QUEUE). The queue is persistent + shared across runs (gen_d90_backfill_queue.py).
#
# Usage (typically launched once per wave for that wave's run dirs):
#     PREADAPT_RUN_DIRS="xAI/output/preadapt_v2/mae__stable_evo__s42:..." \
#     PREADAPT_WORKER_ID=0 D90_QUEUE=xAI/output/preadapt_v2/_d90_queue \
#         sbatch xAI/scripts/submit_preadapt_collector_gpu.sh
#     # optional overrides: PREADAPT_TRAIN_JIDS=<id1>,<id2>  HEAVY_EVERY=3  POLL=15  PREADAPT_VAL_CSV=...
#
#SBATCH --job-name=preadapt_cwgpu
#SBATCH --account=guralnick
#SBATCH --qos=guralnick
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:l4:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=14G   # 8x14=112G; GPU worker peak ~104G measured (heavy ckpt w/ probe+Hessian).
#SBATCH --time=96:00:00
#SBATCH --output xAI/logs/%x-%j.out
#SBATCH --error  xAI/logs/%x-%j.err
#SBATCH --mail-user=r.dinnage@gmail.com
#SBATCH --mail-type=FAIL,END

# Env init FIRST, with NO `set -u` active: a bare SLURM shell sourcing /home/$USER/.bashrc trips
# over unbound vars (e.g. BASHRCSOURCED) under `set -u`, which killed a smoke run. Source the
# profile + activate conda before enabling strict mode, then turn on `set -eo pipefail` (no -u).
source /home/${USER}/.bashrc
source activate reticulate-gpu2
set -eo pipefail

# Derive PROJECT_ROOT robustly from this script's location (.../<root>/xAI/scripts/<this>) so the
# script works regardless of where it is invoked from; fall back to the known path if resolution
# fails (e.g. when SLURM stages a copy outside the tree).
_SCRIPT_SRC="${BASH_SOURCE[0]:-$0}"
_SCRIPT_DIR="$(cd "$(dirname "$_SCRIPT_SRC")" 2>/dev/null && pwd || true)"
if [[ -n "$_SCRIPT_DIR" && -d "$_SCRIPT_DIR/../.." ]]; then
    PROJECT_ROOT="$(cd "$_SCRIPT_DIR/../.." && pwd)"
fi
if [[ -z "$PROJECT_ROOT" || ! -d "$PROJECT_ROOT/xAI" ]]; then
    PROJECT_ROOT=/blue/guralnick/share/r.dinnage/Projects/phenovision
fi
cd "$PROJECT_ROOT"
mkdir -p xAI/logs

echo "$(date)"
echo "host=$(hostname)"
echo "pwd=$(pwd)"

export PYTHONPATH="${PWD}/PlantCLEF2022:${PWD}:${PWD}/xAI/py:${PWD}/xAI/two_noise:${PYTHONPATH:-}"

# --- which run(s) this worker collects ---
# PREADAPT_RUN_DIRS (colon-separated, plural) is preferred for the v2 multi-run set; PREADAPT_RUN_DIR
# (singular) is accepted for one run, matching the CPU script / arg-1 fallback.
RUN_DIRS_RAW="${PREADAPT_RUN_DIRS:-${PREADAPT_RUN_DIR:-${1:-}}}"
if [[ -z "$RUN_DIRS_RAW" ]]; then
    echo "ERROR: no run dir(s). Set PREADAPT_RUN_DIRS=dir1:dir2:... (or PREADAPT_RUN_DIR=<dir>)." >&2
    exit 1
fi

# Expand the colon-separated run dirs into parallel arrays of run dirs / watch dirs / sentinels.
declare -a RUN_DIRS WATCH_DIRS SENTINELS
IFS=':' read -r -a RUN_DIRS <<< "$RUN_DIRS_RAW"
WATCH_DIRS_ARGS=""
for rd in "${RUN_DIRS[@]}"; do
    [[ -n "$rd" ]] || continue
    WATCH_DIRS+=("${rd}/checkpoints")
    SENTINELS+=("${rd}/RUN_COMPLETE")
    WATCH_DIRS_ARGS+=" ${rd}/checkpoints"
done
if [[ ${#WATCH_DIRS[@]} -eq 0 ]]; then
    echo "ERROR: PREADAPT_RUN_DIRS expanded to zero watch dirs." >&2
    exit 1
fi

WORKER_ID="${PREADAPT_WORKER_ID:-0}"
POLL="${POLL:-15}"                          # seconds between drain passes
HEAVY_EVERY="${HEAVY_EVERY:-3}"             # heavy interp blocks every Nth checkpoint
RETENTION_LATEST_KEEP="${RETENTION_LATEST_KEEP:-2}"
KEEP_FULL_LATEST="${KEEP_FULL_LATEST:-2}"

# Per-run-set out store: ScalarStore + ArrayStore live under <out-store>/<run_id>/. All workers
# (GPU + CPU) write the SAME store; rows are keyed by (run_id, step) and each checkpoint is
# processed by exactly one worker per pass (atomic claim), so writes never collide.
OUT_STORE="${PREADAPT_OUT_STORE:-xAI/output/preadapt_v2/_extract_store}"
mkdir -p "$OUT_STORE"

# Probe/val CSV: reuse the trainer's val split as the fixed probe set (briefing §4 invariant 1).
# The GPU pass builds the probe/Hessian batches, so KEEP this (the CPU pass drops it).
VAL_CSV="${PREADAPT_VAL_CSV:-data/inat/val_v1.1.0.csv}"

# Two-pass refcount: this worker's pass tag. FIRM: must be the two-pass form on every v2 dir.
REQUIRE_PASSES="${PREADAPT_REQUIRE_PASSES:-gpu,cpu}"
PASS_TAG="gpu"   # the GPU block set's pass tag (matches collector.py --block-set gpu pass_tag)

# Optional trainer job ids (comma-separated) so we can terminate if the trainers FAILED without
# sentinels. v2 launches one trainer per run; pass them all.
TRAIN_JIDS="${PREADAPT_TRAIN_JIDS:-${PREADAPT_TRAIN_JID:-}}"

# Optional d90 BACKFILL queue (HYBRID design). When set, an idle GPU worker (GPU-pass backlog
# drained to 0) claims and runs ONE d90 config on its GPU instead of just sleeping. The queue is
# persistent + shared across runs (gen_d90_backfill_queue.py). Unset => idle-sleep only.
D90_QUEUE="${D90_QUEUE:-}"

echo "=== preadapt v2 GPU collector worker ${WORKER_ID} (two-pass: $REQUIRE_PASSES, pass=$PASS_TAG) ==="
echo "run_dirs=${RUN_DIRS[*]}"
echo "watch_dirs=${WATCH_DIRS[*]}"
echo "sentinels=${SENTINELS[*]}"
echo "out_store=$OUT_STORE"
echo "val_csv=$VAL_CSV"
echo "poll=${POLL}s heavy_every=$HEAVY_EVERY retention_latest_keep=$RETENTION_LATEST_KEEP keep_full_latest=$KEEP_FULL_LATEST"
echo "train_jids=${TRAIN_JIDS:-<unset>}"
echo "d90_queue=${D90_QUEUE:-<unset>}"

# One --once drain pass over ALL watch dirs, GPU block set only, two-pass refcount (gpu,cpu).
# --val-csv KEPT: the GPU block set builds the probe/Hessian batches. --device cuda.
# --worker-id is prefixed "gpu" so its ScalarStore part dir never collides with a CPU worker's.
drain_once() {
    mamba run -n reticulate-gpu2 python xAI/py/extractor/collector.py \
        --once \
        --device cuda \
        --block-set gpu \
        --require-passes "$REQUIRE_PASSES" \
        --worker-id "gpu${WORKER_ID}" \
        --watch-dirs ${WATCH_DIRS_ARGS} \
        --out-store "$OUT_STORE" \
        --retention-latest-keep "$RETENTION_LATEST_KEEP" \
        --keep-full-latest "$KEEP_FULL_LATEST" \
        --heavy-every "$HEAVY_EVERY" \
        --val-csv "$VAL_CSV"
}

# Run one --once drain, capture its output, echo it through, and set LAST_DRAIN_N to the number of
# checkpoints it processed (parsed from the collector's "--once drained N checkpoint(s)" print).
# Defaults to 0 if the line is absent (e.g. the pass errored).
LAST_DRAIN_N=0
drain_once_counted() {
    local out
    out="$(drain_once 2>&1 || true)"
    printf '%s\n' "$out"
    LAST_DRAIN_N="$(printf '%s\n' "$out" \
        | sed -n 's/.*--once drained \([0-9][0-9]*\) checkpoint.*/\1/p' | tail -n1)"
    [[ -n "$LAST_DRAIN_N" ]] || LAST_DRAIN_N=0
}

# --- d90 BACKFILL: claim + run ONE config on THIS GPU when there is no GPU-pass extraction backlog.
# Atomic claim primitive: `mv` (the rename syscall) of a single .args file OUT of pending/ into
# claimed/<file>.<jobid>_w<id>. Two workers racing the same source: only one rename succeeds; the
# loser's `mv` returns non-zero because the source no longer exists -> exactly-once. Returns 0 on a
# successful claim+run path; non-zero when nothing was claimed.
CLAIMED_ARGS_FILE=""
run_one_d90() {
    [[ -n "$D90_QUEUE" ]] || return 1
    local pend="$D90_QUEUE/pending" claimed="$D90_QUEUE/claimed"
    local done_dir="$D90_QUEUE/done" failed_dir="$D90_QUEUE/failed"
    local state_dir="$D90_QUEUE/state"
    [[ -d "$pend" ]] || return 1
    mkdir -p "$claimed" "$done_dir" "$failed_dir" "$state_dir"
    # Per-claim training wall-time budget (s). Time-box so a worker stuck in one d90 call yields back
    # to collection; run_subspace saves resumable state and exits 2 (we re-queue to resume).
    local d90_max_wall="${D90_MAX_WALL:-600}"

    local tag="${SLURM_JOB_ID:-nojob}_w${WORKER_ID}"
    local src base dst args
    shopt -s nullglob
    for src in "$pend"/*.args; do
        base="$(basename "$src")"
        dst="$claimed/${base}.${tag}"
        # Atomic claim: rename out of pending/. If another worker won, `mv` fails -> try next.
        if mv "$src" "$dst" 2>/dev/null; then
            shopt -u nullglob
            CLAIMED_ARGS_FILE="$dst"
            args="$(cat "$dst")"
            echo "[cwgpu${WORKER_ID}] d90 START $base  (claim=$tag)"
            echo "[cwgpu${WORKER_ID}] d90 args: $args"
            # NEVER let a d90 failure kill the worker: run in a subshell guard (capturing rc in a
            # condition keeps this set -e safe). expandable_segments reduces L4 (22 GB) fragmentation
            # for the ViT-L train forward+backward. --max_wall_seconds + --state_dir TIME-BOX the claim.
            local rc
            if ( set +e; PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
                    mamba run -n reticulate-gpu2 python xAI/py/d90/run_subspace.py $args \
                    --max_wall_seconds "$d90_max_wall" --state_dir "$state_dir" ); then
                rc=0
            else
                rc=$?
            fi
            if [[ "$rc" -eq 0 ]]; then
                mv "$dst" "$done_dir/${base}.${tag}" 2>/dev/null || true
                echo "[cwgpu${WORKER_ID}] d90 DONE  $base -> done/"
            elif [[ "$rc" -eq 2 ]]; then
                # Time-boxed (INCOMPLETE): re-queue to pending/ (state saved) to resume next claim.
                mv "$dst" "$pend/${base}" 2>/dev/null || true
                echo "[cwgpu${WORKER_ID}] d90 YIELD $base -> pending/ (time-boxed; resume later)"
            else
                mv "$dst" "$failed_dir/${base}.${tag}" 2>/dev/null || true
                echo "[cwgpu${WORKER_ID}] d90 FAILED $base -> failed/ (worker continues, rc=$rc)" >&2
            fi
            return 0
        fi
    done
    shopt -u nullglob
    return 1   # nothing claimable -> queue exhausted (or D90_QUEUE has no pending configs)
}

# PASS-AWARE pending GPU-pass work-queue depth across ALL watch dirs: a checkpoint is pending FOR
# THIS (gpu) WORKER iff its <base>.pt exists, <base>.pt.done exists, it is not <base>.pt.processing
# (not being finalized), AND it does NOT yet have <base>.pt.gpu.complete (this pass not yet done).
# This is exactly is_ready_for_pass(path, "gpu") modulo the (rare) live-claim window -- close enough
# for termination, and the final drain re-checks. A checkpoint whose .gpu.complete already exists is
# NOT counted here (the GPU worker is done with it; the CPU pass / finalize sweep handle the rest).
queue_depth() {
    local n=0 wd done base
    shopt -s nullglob
    for wd in "${WATCH_DIRS[@]}"; do
        [[ -d "$wd" ]] || continue
        for done in "$wd"/*.pt.done; do
            base="${done%.done}"                        # ".../stepNNN.pt.done" -> ".../stepNNN.pt"
            [[ -f "$base" ]] || continue                 # base consumed/finalized -> not pending
            [[ -e "${base}.processing" ]] && continue    # being finalized -> not pending
            [[ -e "${base}.${PASS_TAG}.complete" ]] && continue  # this pass already done -> not pending-for-us
            n=$(( n + 1 ))
        done
    done
    shopt -u nullglob
    echo "$n"
}

# True when EVERY watched run has its RUN_COMPLETE sentinel.
all_sentinels_present() {
    local s
    for s in "${SENTINELS[@]}"; do
        [[ -f "$s" ]] || return 1
    done
    return 0
}

# True when we have trainer job ids AND NONE of them is still in the queue (all failed/finished).
trainers_gone() {
    [[ -n "$TRAIN_JIDS" ]] || return 1
    local q
    # squeue accepts a comma-separated job list; empty output => none of them are queued/running.
    q="$(squeue -j "$TRAIN_JIDS" -h 2>/dev/null || true)"
    [[ -z "$q" ]]
}

# --- collection loop: GPU-pass extraction has PRIORITY; d90-backfill fills idle GPU time; exit only
#     when all runs are COMPLETE and the GPU-PASS queue is drained (CPU pass may still lag -- the
#     finalize sweep disposes later). ---
D90_EXHAUSTED_LOGGED=0
while true; do
    drain_once_counted || echo "[cwgpu${WORKER_ID}] WARNING: a --once drain pass returned non-zero; retrying next cycle."

    # 1) GPU-pass extraction backlog has PRIORITY: if we processed any checkpoints, loop now (no d90).
    if [[ "$LAST_DRAIN_N" -gt 0 ]]; then
        echo "[cwgpu${WORKER_ID}] drained ${LAST_DRAIN_N} checkpoint(s); GPU-pass backlog present -> skip d90, loop."
        continue
    fi

    # 2a) Normal termination: ALL RUN_COMPLETE sentinels present, final drain finds nothing,
    #     GPU-pass queue empty (i.e. every emitted checkpoint has this worker's .gpu.complete).
    if all_sentinels_present; then
        echo "[cwgpu${WORKER_ID}] all RUN_COMPLETE sentinels present; performing FINAL drain..."
        drain_once_counted >/dev/null
        DEPTH="$(queue_depth)"
        if [[ "$LAST_DRAIN_N" -eq 0 ]] && [[ "$DEPTH" -eq 0 ]]; then
            echo "[cwgpu${WORKER_ID}] final drain found nothing new and GPU-pass queue depth=0 -> GPU pass complete on all runs. Exiting (CPU pass may still finalize)."
            break
        else
            echo "[cwgpu${WORKER_ID}] final drain still had GPU-pass work (drained=${LAST_DRAIN_N} depth=$DEPTH); looping."
            continue
        fi
    fi

    # 2b) Failed-trainer guard: NONE of the trainers is in the queue -> final drain and exit.
    if trainers_gone; then
        echo "[cwgpu${WORKER_ID}] none of the trainer jobs (${TRAIN_JIDS}) are in the queue; performing FINAL drain before exit..."
        drain_once || echo "[cwgpu${WORKER_ID}] WARNING: final drain returned non-zero."
        if all_sentinels_present; then
            echo "[cwgpu${WORKER_ID}] ...all RUN_COMPLETE sentinels present; clean end. Exiting."
        else
            echo "[cwgpu${WORKER_ID}] ...not all RUN_COMPLETE present: a trainer FAILED without a sentinel. Drained what was emitted; exiting." >&2
        fi
        break
    fi

    # 3) Idle (drained 0, runs not all done): backfill ONE d90 config on this GPU if the queue has work.
    if run_one_d90; then
        D90_EXHAUSTED_LOGGED=0
        continue   # re-check GPU-pass extraction priority + termination before the next d90.
    fi

    # 4) Truly idle: d90 unset OR exhausted. Sleep and loop.
    if [[ -n "$D90_QUEUE" && "$D90_EXHAUSTED_LOGGED" -eq 0 ]]; then
        echo "[cwgpu${WORKER_ID}] d90 backfill queue exhausted (no pending configs); idle-collecting only."
        D90_EXHAUSTED_LOGGED=1
    fi
    sleep "$POLL"
done

echo "Done: $(date)"

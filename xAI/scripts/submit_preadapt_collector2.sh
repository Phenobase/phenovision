#!/bin/bash
# PhenoVision preadapt COLLECTOR WORKER (REDESIGN: collector throughput is the limiter).
#
# ONE single-GPU collector WORKER (1 L4). The REDESIGN runs TWO of these per run (worker 0 and
# worker 1), each on its own L4, both watching the SAME single run's checkpoints dir:
#
#     GPU 1      : 1 trainer        (submit_preadapt_train.sh -- single, NOT an array)   -- hpg-b200
#     GPU 2      : collector worker 0 (THIS script, PREADAPT_WORKER_ID=0)                 -- hpg-turin
#     GPU 3      : collector worker 1 (THIS script, PREADAPT_WORKER_ID=1)                 -- hpg-turin
#                  ------------------------------------------------------------------------------
#                  = 3 concurrent GPUs total for ONE run. FIRM CAP (<= 3). The 6 runs (3 conditions
#                  x 2 variants) run SEQUENTIALLY (run_preadapt_all.sh), so only one run -- and thus
#                  at most 3 GPUs -- is ever active. NEVER run two runs' jobs at once.
#
# WHY TWO WORKERS (not one): the collector is the bottleneck in the REDESIGN. Two workers process
# DIFFERENT checkpoints CONCURRENTLY and out of order. This is safe because (contract B) extraction
# is STATELESS per checkpoint (extract_checkpoint builds its ctx from the checkpoint + fixed shared
# resources, no cross-checkpoint in-memory state) and (contract A) the atomic claim
# (preadapt_common.claim_for_processing, rename to .processing) guarantees EXACTLY ONE worker
# processes each checkpoint -- the loser of the claim race simply moves on. The trainer applies
# BACKPRESSURE (it pauses emission/training when PENDING > high), so the queue never grows unbounded
# even when both workers are momentarily busy.
#
# TERMINATION (driven from bash; collector.py is NOT modified for this): we call `collector.py
# --once` (drain-then-exit) in a loop every $POLL seconds. We stop only when BOTH hold:
#   (1) this run's RUN_COMPLETE sentinel exists  -- written by preadapt_train.py main() at
#       <output_dir>/RUN_COMPLETE AFTER the last checkpoint, AND
#   (2) a final drain `--once` AFTER the sentinel appears processes 0 new checkpoints AND the
#       work queue (count via the *.pt.done sentinels still present) is EMPTY -- i.e. the other
#       worker has not left anything behind. Each worker checks the queue itself, so whichever
#       finishes the last drain exits cleanly; the other exits on its own next cycle.
# Failed-trainer guard (PREADAPT_TRAIN_JID): if the trainer job has left the queue without a
# sentinel (it FAILED), we do one final drain and exit rather than spin to the wall limit.
#
# Usage (prefer run_preadapt_run.sh, which submits the trainer + BOTH workers + posthoc together):
#     PREADAPT_RUN_DIR=xAI/output/preadapt/mae__stable_evo__s42 \
#     PREADAPT_WORKER_ID=0 PREADAPT_VARIANT=stable_evo \
#         sbatch xAI/scripts/submit_preadapt_collector2.sh
#     # optional overrides: PREADAPT_TRAIN_JID=<id>  HEAVY_EVERY=3  POLL=15
#
#SBATCH --job-name=preadapt_cw
#SBATCH --account=guralnick
#SBATCH --qos=guralnick
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:l4:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=14G   # 8x14=112G; collector peak ~104G measured (heavy ckpt). REDESIGN: 1 trainer
                            # + 2 collector workers = 3x112=336G; +~84G other group jobs = ~420G < 437G QOS
                            # cap. (128G each = 384+84=468>437 -> 2nd collector stuck on QOSGrpMemLimit.)
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

PROJECT_ROOT=/blue/guralnick/share/r.dinnage/Projects/phenovision
cd "$PROJECT_ROOT"
mkdir -p xAI/logs

echo "$(date)"
echo "host=$(hostname)"
echo "pwd=$(pwd)"

export PYTHONPATH="${PWD}/PlantCLEF2022:${PWD}:${PWD}/xAI/py:${PWD}/xAI/two_noise:${PYTHONPATH:-}"

# --- which single run this worker collects ---
RUN_DIR="${PREADAPT_RUN_DIR:-${1:-}}"
if [[ -z "$RUN_DIR" ]]; then
    echo "ERROR: no run dir. Set PREADAPT_RUN_DIR=xAI/output/preadapt/<run_id> (or pass as arg 1)." >&2
    exit 1
fi
WORKER_ID="${PREADAPT_WORKER_ID:-0}"
VARIANT="${PREADAPT_VARIANT:-}"
SEED="${PREADAPT_SEED:-42}"
POLL="${POLL:-15}"                          # seconds between drain passes
HEAVY_EVERY="${HEAVY_EVERY:-3}"             # heavy interp blocks every Nth checkpoint (REDESIGN: 3)
RETENTION_LATEST_KEEP="${RETENTION_LATEST_KEEP:-2}"
KEEP_FULL_LATEST="${KEEP_FULL_LATEST:-2}"

# The run's watch dir is a PURE WORK QUEUE (contract A): kept ladder checkpoints get MOVED to
# <run>/kept/ after extraction; only to-be-processed *.pt(+.done) live in <run>/checkpoints/.
WATCH_DIR="${RUN_DIR}/checkpoints"
SENTINEL="${RUN_DIR}/RUN_COMPLETE"

# Per-run out store: ScalarStore + ArrayStore live under <out-store>/<run_id>/. Both workers write
# the SAME store (rows keyed by run_id, step); since each checkpoint is processed by exactly one
# worker (atomic claim) their writes never collide on a step.
OUT_STORE="${PREADAPT_OUT_STORE:-xAI/output/preadapt/_extract_store}"
mkdir -p "$OUT_STORE"

# Probe/val CSV: reuse the trainer's val split as the fixed probe set (briefing §4 invariant 1).
VAL_CSV="${PREADAPT_VAL_CSV:-data/inat/val_v1.1.0.csv}"

# Optional trainer job id so we can terminate if the trainer FAILED without a sentinel.
TRAIN_JID="${PREADAPT_TRAIN_JID:-}"

# Optional d90 BACKFILL queue (HYBRID design). When set, an idle worker (extraction backlog
# drained to 0) claims and runs ONE d90 config on its GPU instead of just sleeping. When the
# shared queue is exhausted the worker is a plain collector again. Unset => exactly today's
# behaviour. The queue is persistent + shared across runs (see gen_d90_backfill_queue.py).
D90_QUEUE="${D90_QUEUE:-}"

echo "=== preadapt collector worker ${WORKER_ID} ==="
echo "run_dir=$RUN_DIR variant=${VARIANT:-<unset>} seed=$SEED"
echo "watch_dir=$WATCH_DIR"
echo "sentinel=$SENTINEL"
echo "out_store=$OUT_STORE"
echo "val_csv=$VAL_CSV"
echo "poll=${POLL}s heavy_every=$HEAVY_EVERY retention_latest_keep=$RETENTION_LATEST_KEEP keep_full_latest=$KEEP_FULL_LATEST"
echo "train_jid=${TRAIN_JID:-<unset>}"
echo "d90_queue=${D90_QUEUE:-<unset>}"

# One --once drain pass over this run's watch dir. Prints "[collector] --once drained N checkpoint(s)".
# --worker-id distinguishes the two concurrent workers in the log; the atomic claim does the locking.
drain_once() {
    mamba run -n reticulate-gpu2 python xAI/py/extractor/collector.py \
        --once \
        --device cuda \
        --worker-id "$WORKER_ID" \
        --watch-dirs "$WATCH_DIR" \
        --out-store "$OUT_STORE" \
        --retention-latest-keep "$RETENTION_LATEST_KEEP" \
        --keep-full-latest "$KEEP_FULL_LATEST" \
        --heavy-every "$HEAVY_EVERY" \
        --val-csv "$VAL_CSV"
}

# Run one --once drain, capture its output, echo it through, and set LAST_DRAIN_N to the number
# of checkpoints it processed (parsed from the collector's "--once drained N checkpoint(s)"
# print). LAST_DRAIN_N defaults to 0 if the line is absent (e.g. the pass errored) so a failed
# drain is treated as "no extraction backlog cleared" (we then fall through to d90 / sleep).
LAST_DRAIN_N=0
drain_once_counted() {
    local out
    out="$(drain_once 2>&1 || true)"
    printf '%s\n' "$out"
    LAST_DRAIN_N="$(printf '%s\n' "$out" \
        | sed -n 's/.*--once drained \([0-9][0-9]*\) checkpoint.*/\1/p' | tail -n1)"
    [[ -n "$LAST_DRAIN_N" ]] || LAST_DRAIN_N=0
}

# --- d90 BACKFILL: claim + run ONE config on THIS GPU when there is no extraction backlog ----
# Atomic claim primitive: `mv` (the rename syscall) of a single .args file OUT of pending/ into
# claimed/<file>.<jobid>_w<id>. Two workers racing the same source: only one rename succeeds; the
# loser's `mv` returns non-zero because the source no longer exists -> exactly-once. (We do NOT
# use `mv -n`, whose test-then-rename is a TOCTOU race; a bare rename is the atomic lock, the same
# primitive preadapt_common.claim_for_processing uses via os.rename.) Returns 0 and sets
# CLAIMED_ARGS_FILE on a successful claim+run path; non-zero when nothing was claimed.
CLAIMED_ARGS_FILE=""
run_one_d90() {
    [[ -n "$D90_QUEUE" ]] || return 1
    local pend="$D90_QUEUE/pending" claimed="$D90_QUEUE/claimed"
    local done_dir="$D90_QUEUE/done" failed_dir="$D90_QUEUE/failed"
    local state_dir="$D90_QUEUE/state"
    [[ -d "$pend" ]] || return 1
    mkdir -p "$claimed" "$done_dir" "$failed_dir" "$state_dir"
    # Per-claim training wall-time budget (s). A d90 config can take 25 min-2 h; without a box a
    # worker stuck in one d90 call cannot collect, so if Phase 2 starts emitting the trainer stalls
    # on backpressure. The box makes the worker yield back to the collect-first loop every
    # ~D90_MAX_WALL s; run_subspace saves resumable state and exits 2 (we re-queue to resume).
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
            echo "[cw${WORKER_ID}] d90 START $base  (claim=$tag)"
            echo "[cw${WORKER_ID}] d90 args: $args"
            # NEVER let a d90 failure kill the worker: run in a subshell guard (capturing rc in a
            # condition keeps this set -e safe). expandable_segments reduces L4 (22 GB)
            # fragmentation for the ViT-L train forward+backward (batch is already L4-sized in the
            # arg-line). --max_wall_seconds + --state_dir TIME-BOX this claim: on timeout the run
            # saves resumable state and exits 2, so the worker yields back to collection promptly.
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
                echo "[cw${WORKER_ID}] d90 DONE  $base -> done/"
            elif [[ "$rc" -eq 2 ]]; then
                # Time-boxed (INCOMPLETE): re-queue to pending/ (state saved) to resume next claim.
                mv "$dst" "$pend/${base}" 2>/dev/null || true
                echo "[cw${WORKER_ID}] d90 YIELD $base -> pending/ (time-boxed; resume later)"
            else
                mv "$dst" "$failed_dir/${base}.${tag}" 2>/dev/null || true
                echo "[cw${WORKER_ID}] d90 FAILED $base -> failed/ (worker continues, rc=$rc)" >&2
            fi
            return 0
        fi
    done
    shopt -u nullglob
    return 1   # nothing claimable -> queue exhausted (or D90_QUEUE has no pending configs)
}

# PENDING work-queue depth: count *.pt.done sentinels whose *.pt exists and is not *.processing.
# This is the same set preadapt_common.count_pending() counts; we read it in bash so termination
# does not need a python round-trip. 0 => the queue is fully drained.
queue_depth() {
    local n=0 done base
    [[ -d "$WATCH_DIR" ]] || { echo 0; return; }
    shopt -s nullglob
    for done in "$WATCH_DIR"/*.pt.done; do
        base="${done%.done}"               # ".../stepNNN.pt.done" -> ".../stepNNN.pt"
        [[ -f "$base" ]] || continue        # base consumed -> not pending
        [[ -e "${base}.processing" ]] && continue   # claimed by a worker -> not pending
        n=$(( n + 1 ))
    done
    shopt -u nullglob
    echo "$n"
}

# True when we have a trainer job id AND it is no longer in the queue (failed/finished).
trainer_gone() {
    [[ -n "$TRAIN_JID" ]] || return 1
    local q
    q="$(squeue -j "$TRAIN_JID" -h 2>/dev/null || true)"
    [[ -z "$q" ]]
}

# --- collection loop: extraction has PRIORITY; d90-backfill fills idle GPU time; exit only
#     when the run is COMPLETE and drained. ---
# Order per cycle:
#   1. drain --once (extraction). If it processed >0 checkpoints -> loop immediately (PRIORITY:
#      do NOT do d90 while there is extraction backlog).
#   2. Otherwise (drained 0): check the termination conditions FIRST (between d90 runs) so the
#      worker exits promptly once the run is done; d90 leftovers are picked up by the NEXT run's
#      workers via the shared persistent queue (we never block run termination on d90).
#   3. If not terminating and D90_QUEUE has a claimable config -> run ONE d90 config on this GPU.
#   4. If not terminating and d90 is unset/exhausted -> sleep $POLL (the only truly-idle case).
D90_EXHAUSTED_LOGGED=0
while true; do
    drain_once_counted || echo "[cw${WORKER_ID}] WARNING: a --once drain pass returned non-zero; retrying next cycle."

    # 1) Extraction backlog has PRIORITY: if we just processed any checkpoints, loop now (no d90).
    if [[ "$LAST_DRAIN_N" -gt 0 ]]; then
        echo "[cw${WORKER_ID}] drained ${LAST_DRAIN_N} checkpoint(s); extraction backlog present -> skip d90, loop."
        continue
    fi

    # 2a) Normal termination: RUN_COMPLETE present, a final drain finds nothing new, queue empty.
    if [[ -f "$SENTINEL" ]]; then
        echo "[cw${WORKER_ID}] RUN_COMPLETE present; performing FINAL drain..."
        drain_once_counted >/dev/null
        DEPTH="$(queue_depth)"
        if [[ "$LAST_DRAIN_N" -eq 0 ]] && [[ "$DEPTH" -eq 0 ]]; then
            echo "[cw${WORKER_ID}] final drain found nothing new and queue depth=0 -> run collected. Exiting."
            break
        else
            echo "[cw${WORKER_ID}] final drain still had work (drained=${LAST_DRAIN_N} depth=$DEPTH); the other worker may be busy. Looping."
            continue
        fi
    fi

    # 2b) Failed-trainer guard: trainer left the queue without a sentinel -> final drain and exit.
    if trainer_gone; then
        echo "[cw${WORKER_ID}] trainer job ${TRAIN_JID} no longer in queue; performing FINAL drain before exit..."
        drain_once || echo "[cw${WORKER_ID}] WARNING: final drain returned non-zero."
        if [[ -f "$SENTINEL" ]]; then
            echo "[cw${WORKER_ID}] ...RUN_COMPLETE was present; clean end. Exiting."
        else
            echo "[cw${WORKER_ID}] ...no RUN_COMPLETE: trainer FAILED without a sentinel. Drained what was emitted; exiting." >&2
        fi
        break
    fi

    # 3) Idle (drained 0, run not done): backfill ONE d90 config on this GPU if the queue has work.
    #    A d90 run takes ~10-30 min; that is fine (the other worker keeps collecting and trainer
    #    backpressure covers any backlog). We re-check extraction + termination on the next cycle.
    if run_one_d90; then
        D90_EXHAUSTED_LOGGED=0
        continue   # re-check extraction priority + termination before the next d90.
    fi

    # 4) Truly idle: d90 unset OR exhausted. Sleep and loop.
    if [[ -n "$D90_QUEUE" && "$D90_EXHAUSTED_LOGGED" -eq 0 ]]; then
        echo "[cw${WORKER_ID}] d90 backfill queue exhausted (no pending configs); idle-collecting only."
        D90_EXHAUSTED_LOGGED=1
    fi
    sleep "$POLL"
done

echo "Done: $(date)"

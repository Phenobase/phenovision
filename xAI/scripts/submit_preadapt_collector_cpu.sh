#!/bin/bash
# PhenoVision preadapt v2 CPU-ONLY COLLECTOR WORKER (CPU/GPU split, C4/C5).
#
# Companion to submit_preadapt_collector2.sh (the GPU worker). This worker runs ONLY the CPU
# block set -- optim / weights / trajectory / circuits: pure linalg on the saved checkpoint
# tensors, NO model forward / HVP and NO probe/Hessian build. It therefore needs NO GPU.
#
# CPU/GPU SPLIT (two-pass refcount): every worker -- GPU and CPU -- passes --require-passes gpu,cpu.
# A checkpoint is disposed (deleted / moved to kept/) only once BOTH .complete sentinels exist, so
# the CPU worker and the GPU worker(s) can process the SAME checkpoint concurrently and out of
# order. The atomic per-(checkpoint, pass) claim in preadapt_common guarantees exactly-once within
# each pass. This is the v2 way to keep the GPU collector(s) from being the bottleneck: the cheap
# GPU-free blocks run on a CPU node that does not count against the ~3-GPU group cap.
#
# This worker can watch MULTIPLE run dirs at once (one CPU node is cheap and the CPU blocks are
# light): set PREADAPT_RUN_DIRS to a colon-separated list of run output dirs; each expands to a
# <dir>/checkpoints --watch-dirs argument. (PREADAPT_RUN_DIR -- singular -- is still accepted for a
# single run, matching the GPU script.)
#
# v2 RUN SET (StableEvo only): 6 runs = {mae,plantclef,naive} x {seed 42, seed 43}, run_id =
# <condition>__stable_evo__s<seed>, output ROOT xAI/output/preadapt_v2/<run_id>. Example watching
# all six:
#     PREADAPT_RUN_DIRS="xAI/output/preadapt_v2/mae__stable_evo__s42:xAI/output/preadapt_v2/mae__stable_evo__s43:xAI/output/preadapt_v2/plantclef__stable_evo__s42:xAI/output/preadapt_v2/plantclef__stable_evo__s43:xAI/output/preadapt_v2/naive__stable_evo__s42:xAI/output/preadapt_v2/naive__stable_evo__s43"
#
# TERMINATION (driven from bash; collector.py is NOT modified for this): we call `collector.py
# --once` (drain-then-exit) in a loop every $POLL seconds. We stop only when BOTH hold:
#   (1) EVERY watched run's RUN_COMPLETE sentinel exists -- written by preadapt_train.py main() at
#       <output_dir>/RUN_COMPLETE AFTER the last checkpoint, AND
#   (2) a final drain `--once` AFTER all sentinels appear processes 0 new checkpoints AND the
#       aggregate CPU work queue (count via the *.pt.done sentinels still present across all watch
#       dirs) is EMPTY -- i.e. no GPU worker has left a checkpoint that still needs the CPU pass.
# Failed-trainer guard (PREADAPT_TRAIN_JIDS, comma-separated): if NONE of the trainer jobs is still
# in the queue (they FAILED/finished without all sentinels), we do one final drain and exit rather
# than spin to the wall limit.
#
# Usage (typically launched once for the whole v2 run set):
#     PREADAPT_RUN_DIRS="xAI/output/preadapt_v2/mae__stable_evo__s42:..." \
#     PREADAPT_WORKER_ID=0 \
#         sbatch xAI/scripts/submit_preadapt_collector_cpu.sh
#     # optional overrides: PREADAPT_TRAIN_JIDS=<id1>,<id2>  HEAVY_EVERY=3  POLL=15
#
#SBATCH --job-name=preadapt_cwcpu
#SBATCH --account=guralnick
#SBATCH --qos=guralnick
#SBATCH --partition=hpg-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=3G    # 8x3=24G. CPU blocks are pure linalg on saved tensors (no model fwd /
                            # probe / Hessian); peak ~12-16G (full-ckpt load + block_optim float64
                            # eigh + block_weights projection). SIZED for the v2 wave: per wave on
                            # the investment QOS = 2 trainers (8 CPU/112G ea) + 1 GPU collector
                            # (8 CPU/112G) + N_CPU CPU collectors. With N_CPU=3: 16+8+24=48 CPU
                            # (<=56) and 224+112+72=408G (<=437) -> fits with margin. block_optim's
                            # ~67s CPU eigh (the §6.1 bottleneck, CPU-by-design) still gets 8 BLAS
                            # threads; 3 workers absorb the ~85s/ckpt CPU pass (~28s effective) well
                            # under the ~80s/emit cadence of the 2-trainer wave.
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

# CPU threading: pin BLAS/OpenMP thread counts to the SLURM core allocation so the CPU linalg
# blocks use the whole node without oversubscribing.
NTHREADS="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS="$NTHREADS"
export MKL_NUM_THREADS="$NTHREADS"

# --- which run(s) this worker collects ---
# PREADAPT_RUN_DIRS (colon-separated, plural) is preferred for the v2 multi-run set; PREADAPT_RUN_DIR
# (singular) is accepted for one run, matching the GPU script / arg-1 fallback.
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
HEAVY_EVERY="${HEAVY_EVERY:-3}"             # heavy blocks every Nth checkpoint (kept for parity)
RETENTION_LATEST_KEEP="${RETENTION_LATEST_KEEP:-2}"
KEEP_FULL_LATEST="${KEEP_FULL_LATEST:-2}"

# Per-run-set out store: ScalarStore + ArrayStore live under <out-store>/<run_id>/. All workers
# (GPU + CPU) write the SAME store; rows are keyed by (run_id, step) and each checkpoint is
# processed by exactly one worker per pass (atomic claim), so writes never collide.
OUT_STORE="${PREADAPT_OUT_STORE:-xAI/output/preadapt_v2/_extract_store}"
mkdir -p "$OUT_STORE"

# Optional trainer job ids (comma-separated) so we can terminate if the trainers FAILED without
# sentinels. v2 launches one trainer per run; pass them all.
TRAIN_JIDS="${PREADAPT_TRAIN_JIDS:-${PREADAPT_TRAIN_JID:-}}"

echo "=== preadapt CPU collector worker ${WORKER_ID} ==="
echo "run_dirs=${RUN_DIRS[*]}"
echo "watch_dirs=${WATCH_DIRS[*]}"
echo "sentinels=${SENTINELS[*]}"
echo "out_store=$OUT_STORE"
echo "nthreads=$NTHREADS"
echo "poll=${POLL}s heavy_every=$HEAVY_EVERY retention_latest_keep=$RETENTION_LATEST_KEEP keep_full_latest=$KEEP_FULL_LATEST"
echo "train_jids=${TRAIN_JIDS:-<unset>}"

# One --once drain pass over ALL watch dirs, CPU block set only, two-pass refcount (gpu,cpu).
# No --probe-csv/--val-csv: the CPU block set builds no probe/Hessian. --device cpu.
# --worker-id is prefixed "cpu" so its ScalarStore part dir never collides with a GPU worker's.
drain_once() {
    mamba run -n reticulate-gpu2 python xAI/py/extractor/collector.py \
        --once \
        --device cpu \
        --block-set cpu \
        --require-passes gpu,cpu \
        --worker-id "cpu${WORKER_ID}" \
        --watch-dirs ${WATCH_DIRS_ARGS} \
        --out-store "$OUT_STORE" \
        --retention-latest-keep "$RETENTION_LATEST_KEEP" \
        --keep-full-latest "$KEEP_FULL_LATEST" \
        --heavy-every "$HEAVY_EVERY"
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

# Aggregate PENDING CPU work-queue depth across ALL watch dirs: count *.pt.done sentinels whose
# *.pt exists and is not *.processing. (Same set count_pending() counts; read in bash so
# termination needs no python round-trip.) 0 => every watched queue is fully drained.
queue_depth() {
    local n=0 wd done base
    shopt -s nullglob
    for wd in "${WATCH_DIRS[@]}"; do
        [[ -d "$wd" ]] || continue
        for done in "$wd"/*.pt.done; do
            base="${done%.done}"                    # ".../stepNNN.pt.done" -> ".../stepNNN.pt"
            [[ -f "$base" ]] || continue             # base consumed -> not pending
            [[ -e "${base}.processing" ]] && continue # claimed by a worker -> not pending
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

# --- collection loop: CPU pass; exit only when all runs are COMPLETE and the aggregate queue is
#     drained. CPU workers have no d90 backfill (d90 is GPU work) -- they idle-sleep when drained.
while true; do
    drain_once_counted || echo "[cwcpu${WORKER_ID}] WARNING: a --once drain pass returned non-zero; retrying next cycle."

    # 1) If we just processed any checkpoints, loop now (more may be ready).
    if [[ "$LAST_DRAIN_N" -gt 0 ]]; then
        echo "[cwcpu${WORKER_ID}] drained ${LAST_DRAIN_N} checkpoint(s); looping."
        continue
    fi

    # 2a) Normal termination: ALL RUN_COMPLETE sentinels present, final drain finds nothing, queue empty.
    if all_sentinels_present; then
        echo "[cwcpu${WORKER_ID}] all RUN_COMPLETE sentinels present; performing FINAL drain..."
        drain_once_counted >/dev/null
        DEPTH="$(queue_depth)"
        if [[ "$LAST_DRAIN_N" -eq 0 ]] && [[ "$DEPTH" -eq 0 ]]; then
            echo "[cwcpu${WORKER_ID}] final drain found nothing new and aggregate queue depth=0 -> all runs collected. Exiting."
            break
        else
            echo "[cwcpu${WORKER_ID}] final drain still had work (drained=${LAST_DRAIN_N} depth=$DEPTH); a GPU worker may still be emitting/claiming. Looping."
            continue
        fi
    fi

    # 2b) Failed-trainer guard: NONE of the trainers is in the queue -> final drain and exit.
    if trainers_gone; then
        echo "[cwcpu${WORKER_ID}] none of the trainer jobs (${TRAIN_JIDS}) are in the queue; performing FINAL drain before exit..."
        drain_once || echo "[cwcpu${WORKER_ID}] WARNING: final drain returned non-zero."
        if all_sentinels_present; then
            echo "[cwcpu${WORKER_ID}] ...all RUN_COMPLETE sentinels present; clean end. Exiting."
        else
            echo "[cwcpu${WORKER_ID}] ...not all RUN_COMPLETE present: a trainer FAILED without a sentinel. Drained what was emitted; exiting." >&2
        fi
        break
    fi

    # 3) Idle (drained 0, runs not all done): CPU worker has no d90 backfill -> sleep and loop.
    sleep "$POLL"
done

echo "Done: $(date)"

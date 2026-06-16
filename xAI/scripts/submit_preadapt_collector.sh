#!/bin/bash
# PhenoVision preadapt COLLECTOR job (plan component C5).
#
# ONE single-GPU collector that consumes the checkpoints written by the 2 concurrent trainers of
# ONE variant. It runs on an L4 (hpg-turin) so the 2 trainers keep both B200s:
#
#     GPU 1 + 2  : 2 trainers   (submit_preadapt_train.sh, --array=0-2%2)   -- hpg-b200
#     GPU 3      : 1 collector  (THIS job)                                   -- hpg-turin
#                  ------------------------------------------------------------------
#                  = 3 concurrent GPUs total. FIRM CAP. NEVER raise without group OK.
#
# The collector watches the 3 run checkpoint dirs of the active variant and processes ready
# checkpoints sequentially (one model on the GPU at a time; briefing Part II §2.3 Plan A).
#
# TERMINATION (driven from bash; collector.py is NOT modified): we call `collector.py --once`
# (drain-then-exit) in a loop every $POLL seconds. We stop only when BOTH hold:
#   (1) every run dir has its RUN_COMPLETE sentinel  -- written by preadapt_train.py main() at
#       <output_dir>/RUN_COMPLETE AFTER the last checkpoint (confirmed by reading preadapt_train.py),
#   AND
#   (2) a final drain `--once` AFTER all sentinels appear processes 0 new checkpoints (the queue
#       is empty / fully drained).
# This guarantees we never exit while a trainer is still emitting or while drainable work remains.
#
# Usage (prefer run_preadapt_wave.sh, which submits this + the trainer array together):
#     PREADAPT_VARIANT=adamw sbatch xAI/scripts/submit_preadapt_collector.sh
#     # optional overrides: PREADAPT_SEED=42  POLL=30
#
#SBATCH --job-name=preadapt_collect
#SBATCH --account=guralnick
#SBATCH --qos=guralnick
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:l4:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G   # 8x16=128G; collector peak ~104G measured (heavy ckpt), returns to ~13G baseline (no leak).
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

# --- which variant's runs to collect ---
VARIANT="${PREADAPT_VARIANT:-${1:-}}"
if [[ -z "$VARIANT" ]]; then
    echo "ERROR: no variant. Set PREADAPT_VARIANT=adamw|stable_evo (or pass as arg 1)." >&2
    exit 1
fi
SEED="${PREADAPT_SEED:-42}"
POLL="${POLL:-30}"                         # seconds between drain passes
RETENTION_LATEST_KEEP="${RETENTION_LATEST_KEEP:-2}"
# Keep the latest N ladder checkpoints FULL (optimizer + rng -> resumable); every other kept
# ladder checkpoint is rewritten MODEL-ONLY (~6.6 GB -> ~1.2 GB) to conserve /blue (locked).
KEEP_FULL_LATEST="${KEEP_FULL_LATEST:-2}"

# Optional trainer array job id (set by run_preadapt_wave.sh via --export). When present, it lets
# us terminate if the trainer array has left the queue even though a task FAILED without writing
# its RUN_COMPLETE sentinel — otherwise the collector would spin until its own 96h wall limit.
TRAIN_JID="${PREADAPT_TRAIN_JID:-}"

# The 3 conditions of the LOCKED design (ImageNet dropped). run_id = {condition}__{variant}__s{seed}
# matches preadapt_train.make_run_id; output dir = xAI/output/preadapt/<run_id>; the trainer's watch
# dir is <output_dir>/checkpoints and its completion sentinel is <output_dir>/RUN_COMPLETE.
CONDITIONS=(mae plantclef naive)

PREADAPT_BASE="xAI/output/preadapt"
WATCH_DIRS=()
SENTINELS=()
for cond in "${CONDITIONS[@]}"; do
    run_id="${cond}__${VARIANT}__s${SEED}"
    out_dir="${PREADAPT_BASE}/${run_id}"
    WATCH_DIRS+=("${out_dir}/checkpoints")
    SENTINELS+=("${out_dir}/RUN_COMPLETE")
done

# Per-variant out store (briefing §7: ScalarStore + ArrayStore live under <out-store>/<run_id>/).
OUT_STORE="${PREADAPT_BASE}/_extract_store/${VARIANT}"
mkdir -p "$OUT_STORE"

# Probe/val CSVs: reuse the trainer's val split as the fixed probe set (briefing §4 invariant 1).
# Same images, same order, across all runs/conditions (the collector subsamples a fixed n-probe).
VAL_CSV="${PREADAPT_VAL_CSV:-data/inat/val_v1.1.0.csv}"

echo "=== preadapt collector ==="
echo "variant=$VARIANT seed=$SEED poll=${POLL}s retention_latest_keep=$RETENTION_LATEST_KEEP keep_full_latest=$KEEP_FULL_LATEST"
echo "out_store=$OUT_STORE"
echo "val_csv=$VAL_CSV"
echo "train_jid=${TRAIN_JID:-<unset>}"
for i in "${!WATCH_DIRS[@]}"; do
    echo "  watch[$i]=${WATCH_DIRS[$i]}   sentinel=${SENTINELS[$i]}"
done

# The watch dirs may not exist yet when this job starts before the trainers create them; the
# collector's _build_queue skips non-existent dirs, so a leading run with empty dirs is fine.

# One --once drain pass over all 3 watch dirs. Prints "[collector] --once drained N checkpoint(s)".
drain_once() {
    mamba run -n reticulate-gpu2 python xAI/py/extractor/collector.py \
        --once \
        --device cuda \
        --watch-dirs "${WATCH_DIRS[@]}" \
        --out-store "$OUT_STORE" \
        --retention-latest-keep "$RETENTION_LATEST_KEEP" \
        --keep-full-latest "$KEEP_FULL_LATEST" \
        --val-csv "$VAL_CSV"
}

all_sentinels_present() {
    for s in "${SENTINELS[@]}"; do
        [[ -f "$s" ]] || return 1
    done
    return 0
}

# True when we have a trainer array job id AND it is no longer in the queue (no running OR pending
# tasks). `squeue -j <jid> -h` prints one line per live array task and nothing once the whole array
# has finished/failed/been cancelled, so empty output => training is over regardless of sentinels.
trainer_array_gone() {
    [[ -n "$TRAIN_JID" ]] || return 1
    local q
    q="$(squeue -j "$TRAIN_JID" -h 2>/dev/null || true)"
    [[ -z "$q" ]]
}

# --- collection loop: drain, sleep, repeat; exit only when all runs are COMPLETE and drained ---
while true; do
    drain_once || echo "[wave] WARNING: a --once drain pass returned non-zero; will retry next cycle."

    # Normal exit: every run wrote RUN_COMPLETE, then a final drain finds nothing new.
    if all_sentinels_present; then
        echo "[wave] all ${#SENTINELS[@]} RUN_COMPLETE sentinels present; performing FINAL drain..."
        # Final drain AFTER every trainer finished. Capture whether it processed anything new.
        FINAL_OUT="$(drain_once 2>&1 | tee /dev/stderr || true)"
        if echo "$FINAL_OUT" | grep -qE '\-\-once drained 0 checkpoint'; then
            echo "[wave] final drain found nothing new -> all runs collected. Exiting."
            break
        else
            echo "[wave] final drain still processed work; looping once more to drain the remainder."
        fi
    fi

    # Failed-trainer guard: if the trainer array has left the queue (no running/pending tasks) but
    # not all sentinels appeared, a trainer FAILED without writing RUN_COMPLETE. Training is over
    # regardless, so do one FINAL drain (collect whatever the failed/finished trainers did emit)
    # and exit — otherwise we'd loop until the 96h wall limit. Skipped when PREADAPT_TRAIN_JID is
    # unset (trainer_array_gone returns false), preserving the original sentinel-only behavior.
    if trainer_array_gone; then
        echo "[wave] trainer array job ${TRAIN_JID} no longer in queue; performing FINAL drain before exit..."
        drain_once || echo "[wave] WARNING: final drain returned non-zero."
        if all_sentinels_present; then
            echo "[wave] ...all ${#SENTINELS[@]} RUN_COMPLETE sentinels were present; clean end. Exiting."
        else
            present=0
            for s in "${SENTINELS[@]}"; do [[ -f "$s" ]] && present=$(( present + 1 )); done
            echo "[wave] ...only ${present}/${#SENTINELS[@]} RUN_COMPLETE sentinels present: at least one" \
                 "trainer FAILED without a sentinel. Drained what was emitted; exiting (training is over)." >&2
        fi
        break
    fi

    sleep "$POLL"
done

echo "Done: $(date)"

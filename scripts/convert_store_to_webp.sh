#!/bin/bash
# =============================================================================
# convert_store_to_webp.sh -- ONE-TIME bulk conversion of the iNaturalist "medium" image store to
# WebP q82. SLURM ARRAY: one task per batch dir. Each task converts every <id>.{jpg,jpeg,png} ->
# <id>.webp (q82/method6) via py/webp_convert.py (ProcessPoolExecutor), DELETES the originals, and
# DELETES corrupt/unconvertible sources (self-cleaning store). Idempotent/resumable: existing
# <id>.webp are skipped, so re-running only does remaining work. Writes a per-batch manifest row.
#
# Space self-funds (webp ~1/3 of jpg, delete-as-you-go); S3 is the re-download backstop for any
# corrupt photo_id later needed.
#
# SETUP + LAUNCH (run from the repo root of the webp-conversion worktree):
#   bash scripts/convert_store_to_webp.sh --init      # writes the batch list, prints the array size
#   sbatch --array=0-<N-1>%40 scripts/convert_store_to_webp.sh
#   # resume: just re-sbatch the same array; done batches no-op.
# Monitor:  cat .webp_convert/convert_store_manifest.csv ; squeue -u $USER | grep convert_webp
#
#SBATCH --job-name=convert_webp
#SBATCH --account=guralnick
#SBATCH --qos=guralnick-b
#SBATCH --partition=hpg-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output .webp_convert/logs/%x-%A_%a.out
#SBATCH --error  .webp_convert/logs/%x-%A_%a.err
#SBATCH --mail-user=r.dinnage@gmail.com
#SBATCH --mail-type=FAIL

IMAGES_ROOT=/blue/guralnick/share/phenobase_inat_data/images/medium
# Stable work dir in the PRIMARY project tree (shared across worktrees; gitignored scratch):
WORKDIR=/blue/guralnick/share/r.dinnage/Projects/phenovision/.webp_convert
# SLURM copies the batch script to spool, so $0-relative paths don't work. Resolve the converter
# from the sbatch submit dir (override with CONVERTER=/abs/path/py/webp_convert.py if needed).
CONVERTER="${CONVERTER:-${SLURM_SUBMIT_DIR:-$(pwd)}/py/webp_convert.py}"
BATCH_LIST="$WORKDIR/batches_all.txt"
ROWDIR="$WORKDIR/.rows"; FAILDIR="$WORKDIR/failed"
MANIFEST="$WORKDIR/convert_store_manifest.csv"
WORKERS="${WORKERS:-16}"

mkdir -p "$WORKDIR/logs" "$ROWDIR" "$FAILDIR"

# --- --init: build the batch list (one batch NUMBER per line) + assemble any existing rows ---
if [[ "$1" == "--init" ]]; then
  ls -d "$IMAGES_ROOT"/batch_* 2>/dev/null | sed -E 's#.*/batch_##' | sort -n > "$BATCH_LIST"
  N=$(wc -l < "$BATCH_LIST")
  echo "wrote $BATCH_LIST : $N batches"
  echo "launch:  sbatch --array=0-$((N-1))%40 scripts/convert_store_to_webp.sh"
  exit 0
fi
if [[ "$1" == "--manifest" ]]; then
  { echo "batch,n_src,n_webp,n_failed,status,timestamp"; cat "$ROWDIR"/*.row 2>/dev/null | sort -t, -k1 -n; } > "$MANIFEST"
  echo "wrote $MANIFEST ($(($(wc -l < "$MANIFEST")-1)) batches)"; awk -F, 'NR>1{c[$5]++} END{for(s in c) print "  "s": "c[s]}' "$MANIFEST"
  exit 0
fi

# --- array task: convert one batch ---
[[ -f "$BATCH_LIST" ]] || { echo "ERROR: $BATCH_LIST missing; run --init first" >&2; exit 1; }
source /home/${USER}/.bashrc
source activate reticulate-gpu2
set -eo pipefail

LINE=$(( SLURM_ARRAY_TASK_ID + 1 ))
BATCH=$(sed -n "${LINE}p" "$BATCH_LIST")
[[ -n "$BATCH" ]] || { echo "no batch at line $LINE" >&2; exit 1; }
BDIR="$IMAGES_ROOT/batch_$BATCH"
TS=$(date '+%Y-%m-%dT%H:%M:%S')
echo "$(date) host=$(hostname) converting batch_$BATCH ($BDIR)"

# n_src BEFORE (remaining jpg/jpeg/png); if 0 and webp present -> already done
N_SRC=$(find "$BDIR" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) 2>/dev/null | wc -l)
if [[ "$N_SRC" -eq 0 ]]; then
  N_WEBP=$(find "$BDIR" -maxdepth 1 -type f -iname '*.webp' 2>/dev/null | wc -l)
  echo "batch_$BATCH,0,$N_WEBP,0,already_done,$TS" > "$ROWDIR/batch_$BATCH.row"
  echo "[batch_$BATCH] already_done (webp=$N_WEBP)"; exit 0
fi

OUT=$(python "$CONVERTER" "$BDIR" --workers "$WORKERS" --fail-log "$FAILDIR/batch_$BATCH.failed")
# OUT: "n_src=.. n_webp=.. n_failed=.."
NS=$(sed -E 's/.*n_src=([0-9]+).*/\1/' <<<"$OUT")
NW=$(sed -E 's/.*n_webp=([0-9]+).*/\1/' <<<"$OUT")
NF=$(sed -E 's/.*n_failed=([0-9]+).*/\1/' <<<"$OUT")
REMAIN=$(find "$BDIR" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) 2>/dev/null | wc -l)
STATUS=$([[ "$REMAIN" -eq 0 ]] && echo converted || echo "INCOMPLETE_remain${REMAIN}")
echo "batch_$BATCH,$NS,$NW,$NF,$STATUS,$TS" > "$ROWDIR/batch_$BATCH.row"
echo "[batch_$BATCH] $STATUS (src=$NS webp=$NW failed=$NF remain=$REMAIN)"

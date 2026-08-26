#!/bin/bash
# =============================================================================
# recover_deleted_webp.sh -- SLURM array (one task per damaged batch) that refetches the images the
# WebP bulk conversion deleted and reconverts them to webp on ingest. Burst QOS, CPU-only.
# Batch list: .webp_convert/recover_batches.txt (one batch number per line).
# Launch (from the phenovision-webp worktree root):
#   N=$(wc -l < /blue/guralnick/share/r.dinnage/Projects/phenovision/.webp_convert/recover_batches.txt)
#   sbatch --array=0-$((N-1))%13 scripts/recover_deleted_webp.sh
#
#SBATCH --job-name=recover_webp
#SBATCH --account=guralnick
#SBATCH --qos=guralnick-b
#SBATCH --partition=hpg-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output /blue/guralnick/share/r.dinnage/Projects/phenovision/.webp_convert/logs/%x-%A_%a.out
#SBATCH --error  /blue/guralnick/share/r.dinnage/Projects/phenovision/.webp_convert/logs/%x-%A_%a.err
#SBATCH --mail-user=r.dinnage@gmail.com
#SBATCH --mail-type=FAIL

set -eo pipefail
WORKTREE=/blue/guralnick/share/r.dinnage/Projects/phenovision-webp
BATCH_LIST=/blue/guralnick/share/r.dinnage/Projects/phenovision/.webp_convert/recover_batches.txt
cd "$WORKTREE"

source /home/${USER}/.bashrc
source activate reticulate-gpu2

LINE=$(( SLURM_ARRAY_TASK_ID + 1 ))
BATCH=$(sed -n "${LINE}p" "$BATCH_LIST")
[[ -n "$BATCH" ]] || { echo "no batch at line $LINE" >&2; exit 1; }

echo "$(date) host=$(hostname) recovering batch_$BATCH"
Rscript scripts/recover_batch_from_faillog.R "$BATCH"

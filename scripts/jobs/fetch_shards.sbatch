#!/bin/bash -l
# FULL PULL: each array task packs a contiguous block of WebDataset shards.
#
# -l is REQUIRED (non-login shell has no `module`). --output is ABSOLUTE (a relative
# %x_%j.out resolves against the submit dir; if that is gone SLURM kills the job in ~1s
# with NO log). NO `set -u`: /etc/profile aborts under it and `|| true` cannot catch it --
# that is what killed iNat-leps' first launch with 52 zero-byte outputs.
#
# Walltime is 3h ON PURPOSE. Vulcan's partition ladder (measured 2026-08-26):
#   gpubase_bynode_b1  3:00:00   210 nodes   <- largest pool
#   gpubase_bynode_b2 12:00:00   196 nodes
#   gpubase_bynode_b3  1-00:00   130 nodes
# Asking for more than 3h drops into a smaller pool for no benefit: the pilot measured
# ~155 s/shard, so 50 shards ≈ 2.15 h and fits b1 with headroom.
# The cpubase_* partitions hold only TWO nodes total, so a CPU-only job belongs here on
# the GPU nodes -- it requests no GPU, leaving all four L40S per node free for others.
#SBATCH --account=aip-dinnage
#SBATCH --job-name=pheno-fetch
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --array=0-196%20
#SBATCH --output=/home/dinnage/phenovision/jobout/fetch-%A_%a.out

set -o pipefail

REPO="$HOME/phenovision"
PY="$HOME/venvs/phenovision/bin/python"
STORE=/project/aip-dinnage/phenovision/data/phenobase_inat_data/images/shards

# Which set: permissive (default) or restricted. Both are numbered from shard 0 and live
# in separate directories, so training can point at permissive/ and the exclusion is a
# directory boundary rather than a filter someone must remember to apply.
SET="${SET:-permissive}"
case "$SET" in
  permissive) export SHARD_LISTS="$REPO/manifests/shard_lists" ;;
  restricted) export SHARD_LISTS="$REPO/manifests_restricted/shard_lists" ;;
  *) echo "FATAL: SET must be permissive or restricted, got '$SET'"; exit 1 ;;
esac
export SHARD_OUT="$STORE/$SET"
mkdir -p "$SHARD_OUT"

SPT="${SPT:-50}"                  # shards per array task -- FIXED, see the note below
export THREADS="${THREADS:-16}"

# Vulcan compute nodes reach the internet only through http(s)_proxy=http://squid:3128,
# which the Slurm job env already exports. requests picks it up automatically. Do NOT
# source /etc/profile to get it.
start=$(( SLURM_ARRAY_TASK_ID * SPT ))
end=$(( start + SPT - 1 ))

echo "host      : $(hostname)"
echo "started   : $(date -u +%FT%TZ)"
echo "set       : $SET"
echo "lists     : $SHARD_LISTS"
echo "out       : $SHARD_OUT"
echo "shards    : $start .. $end   (SPT=$SPT, THREADS=$THREADS)"
echo "proxy     : ${https_proxy:-<unset>}"
echo

# NOTE ON RECOVERY -- this is deliberately NOT the iNat-leps design.
# There, shards-per-task was DERIVED from SLURM_ARRAY_TASK_COUNT, so re-running a single
# failed task as `--array=13` set the count to 1, computed a start past the end of the
# corpus, printed "nothing to do" and exited 0 -- leaving an 80-shard hole while every
# task reported success. Here SPT is a FIXED constant, so `--array=13` always means
# shards 650..699 whether it runs alone or with 196 siblings. Re-running any single task
# is safe by construction, and fetch_shard.py skips shards that already have a .done.
for s in $(seq "$start" "$end"); do
  [ -d "$SHARD_LISTS/shard=$s" ] || continue    # past the end of this set
  "$PY" "$REPO/scripts/fetch_shard.py" "$s" || echo "  shard $s FAILED (continuing)"
done

echo
echo "finished  : $(date -u +%FT%TZ)"
echo "done markers in range: $(ls "$SHARD_OUT"/*.done 2>/dev/null | wc -l) total in $SET"

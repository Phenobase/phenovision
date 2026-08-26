#!/bin/bash -l
# PILOT: fetch a handful of shards end-to-end before committing to the full ~22h array.
# Validates the Squid proxy, S3 reachability from a compute node, WebP q82 encoding, tar
# integrity, sidecar contents and throughput.
#
# -l is REQUIRED (non-login shell has no `module`). --output is ABSOLUTE (a relative
# %x_%j.out resolves against the submit dir; if that is gone SLURM kills the job in ~1s
# with no log). NO `set -u`: /etc/profile aborts under it, and `|| true` cannot catch it --
# that is what silently killed iNat-leps' first launch with 52 zero-byte outputs.
#SBATCH --account=aip-dinnage
#SBATCH --job-name=pheno-fetch-pilot
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/dinnage/phenovision/jobout/fetch-pilot-%j.out

set -o pipefail

REPO="$HOME/phenovision"
PY="$HOME/venvs/phenovision/bin/python"
FIRST="${FIRST:-0}"
COUNT="${COUNT:-5}"
export THREADS="${THREADS:-16}"

# Vulcan compute nodes have no direct internet; the Slurm job env already exports
# http(s)_proxy=http://squid:3128. Do NOT source /etc/profile to get it -- requests picks
# it up from the inherited environment automatically.
echo "host        : $(hostname)"
echo "started     : $(date -u +%FT%TZ)"
echo "proxy       : ${https_proxy:-<unset>} / ${http_proxy:-<unset>}"
echo "shards      : $FIRST .. $((FIRST + COUNT - 1))"
echo "threads     : $THREADS"
echo

# Prove the proxy reaches S3 before doing anything expensive. This is the single
# assumption the whole fused-download design rests on.
echo "=== proxy reachability probe ==="
"$PY" - <<'PYEOF'
import os, requests
url = "https://inaturalist-open-data.s3.amazonaws.com/photos/16/medium.jpg"
try:
    r = requests.get(url, timeout=30)
    print("  GET %s -> HTTP %d, %d bytes" % (url.rsplit("/", 3)[-3], r.status_code, len(r.content)))
    print("  PROXY_OK" if r.ok else "  PROXY_REACHABLE_BUT_NOT_OK")
except Exception as e:
    print("  PROXY_FAIL:", type(e).__name__, e)
    raise SystemExit(1)
PYEOF
[ $? -ne 0 ] && { echo "FATAL: cannot reach S3 from this compute node"; exit 1; }

echo
echo "=== fetching $COUNT shards ==="
for s in $(seq "$FIRST" $((FIRST + COUNT - 1))); do
  "$PY" "$REPO/scripts/fetch_shard.py" "$s" || echo "  shard $s FAILED"
done

echo
echo "=== results ==="
STORE=/project/aip-dinnage/phenovision/data/phenobase_inat_data/images/shards
for s in $(seq "$FIRST" $((FIRST + COUNT - 1))); do
  f=$(printf "%s/shard-%05d.tar" "$STORE" "$s")
  [ -f "$f" ] || { echo "  shard $s: MISSING"; continue; }
  printf "  shard %-5s %10s  %s\n" "$s" "$(du -h "$f" | cut -f1)" "$(cat "$f.done" 2>/dev/null)"
done

echo
echo "finished    : $(date -u +%FT%TZ)"

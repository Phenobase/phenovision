#!/bin/bash -l
# -l is REQUIRED: a plain #!/bin/bash sbatch script gets a NON-LOGIN shell where `module`
# is undefined. --output is ABSOLUTE because %x_%j.out resolves against the SUBMIT
# directory, and if that is gone SLURM fails the job in ~1s with no log at all.
# Do NOT add `set -u`: /etc/profile references unbound vars and aborts the shell before
# the payload runs, producing a 0-byte .out that looks like a payload failure.
#SBATCH --account=aip-dinnage
#SBATCH --job-name=pheno-manifest
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/dinnage/phenovision/jobout/manifest-%j.out

set -o pipefail

REPO="$HOME/phenovision"
PY="$HOME/venvs/phenovision/bin/python"
SNAP="${SNAP:-/project/aip-dinnage/shared/inat-open-data/latest}"
OUT="$REPO/manifests"

echo "host      : $(hostname)"
echo "started   : $(date -u +%FT%TZ)"
echo "snapshot  : $SNAP -> $(readlink -f "$SNAP")"
echo "output    : $OUT"
echo

# The snapshot must be COMPLETE. A half-staged dump would silently yield a short frame.
if ! grep -q ALL_DONE "$(readlink -f "$SNAP")/stage.log" 2>/dev/null; then
  echo "FATAL: $SNAP has no ALL_DONE in stage.log -- staging incomplete or failed."
  cat "$(readlink -f "$SNAP")/stage.log" 2>/dev/null
  exit 1
fi

# duckdb spills to disk on the big joins; keep that on fast node-local storage, not
# /project (SLURM_TMPDIR is cleaned up automatically at job end).
TMPD="${SLURM_TMPDIR:-/tmp}/duckdb-$$"
mkdir -p "$TMPD"

"$PY" "$REPO/scripts/build_shard_manifest.py" \
    --snapshot "$SNAP" \
    --out      "$OUT" \
    --threads  "${SLURM_CPUS_PER_TASK:-8}" \
    --memory   56GB \
    --temp     "$TMPD"
rc=$?

rm -rf "$TMPD"
echo
echo "finished  : $(date -u +%FT%TZ)  exit=$rc"
if [ "$rc" -eq 0 ]; then
  echo "shard count: $(ls -1d "$OUT"/shard_lists/shard=* 2>/dev/null | wc -l)"
  cat "$OUT/full_summary.txt" 2>/dev/null
fi
exit $rc

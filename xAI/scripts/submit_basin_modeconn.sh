#!/bin/bash
# submit_basin_modeconn.sh -- CONVERGENCE basin-geometry + cross-run mode-connectivity for the
# v2 StableEvo run set (plan component C7; briefing Part II §9.2). POST-HOC / single-GPU.
#
# Given the 6 v2 run dirs (stable_evo x {mae,plantclef,naive} x {seed 42, seed 43}, under
# xAI/output/preadapt_v2/<run_id>), this job:
#
#   (1) BASIN GEOMETRY -- runs xAI/py/extractor/block_basin.py on each run's CONVERGED checkpoints
#       (the FINAL + --last-k kept ladder checkpoints), writing basin radius/anisotropy + Hessian
#       flatness/volume scalars + per-direction radii arrays into a per-run basin store under
#       xAI/output/preadapt_v2/_basin_store/<run_id>/.
#
#   (2) MODE CONNECTIVITY -- runs xAI/py/posthoc/mode_connectivity.py (linear mode connectivity)
#       between EVERY unordered pair of the 6 v2 run FINAL models (15 pairs), writing each pair's
#       λ-curve CSV under xAI/output/preadapt_v2/_modeconn/curves/ and aggregating the barrier
#       heights (train/val loss barrier + train/val AUC-PR dip + verdict) into one summary CSV
#       xAI/output/preadapt_v2/_modeconn/barriers.csv.
#
# Same-condition pairs (mae42-vs-mae43 etc.) share the Phase-2 init + data order only up to the seed,
# so a barrier flags a basin split; cross-condition pairs do NOT share an init, so their barrier is
# the generic independent-solutions barrier and is run with --cross-init (the verdict is annotated).
#
# IDEMPOTENT: a basin run is skipped if its per-run ``.basin_done`` sentinel exists; an LMC pair is
# skipped if its λ-curve CSV already exists. Re-running only does the missing work, then always
# rebuilds the aggregate barriers.csv from whatever curve CSVs are present. Force a redo by deleting
# the relevant sentinel / curve CSV (or pass FORCE=1 to clear them all).
#
# This is ONE L4 GPU job (qos=guralnick). It is POST-HOC -- run it AFTER the v2 trainers + collectors
# finish (the converged kept checkpoints must exist). It NEVER runs alongside the trainers, so the
# firm <=3 concurrent-GPU cap is respected. Check `module load ufrc && slurmInfo` before submitting.
#
# Usage:
#   sbatch xAI/scripts/submit_basin_modeconn.sh
#   # or override the v2 root / run set / last-k / val csv via env:
#   V2_ROOT=xAI/output/preadapt_v2 LAST_K=3 \
#   VAL_CSV=data/inat/val_v1.1.0.csv \
#       sbatch xAI/scripts/submit_basin_modeconn.sh
#   # or pass an explicit run-dir list as args (overrides auto-discovery):
#   sbatch xAI/scripts/submit_basin_modeconn.sh xAI/output/preadapt_v2/mae__stable_evo__s42 ...
#
#SBATCH --job-name=basin_modeconn
#SBATCH --account=guralnick
#SBATCH --qos=guralnick
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:l4:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=12G
#SBATCH --time=24:00:00
#SBATCH --output xAI/logs/%x-%j.out
#SBATCH --error  xAI/logs/%x-%j.err
#SBATCH --mail-user=r.dinnage@gmail.com
#SBATCH --mail-type=FAIL,END

# Env init FIRST, with NO `set -u` active (a bare SLURM shell sourcing /home/$USER/.bashrc trips on
# unbound vars under set -u). Source the profile + conda, THEN enable strict mode (no -u).
source /home/${USER}/.bashrc
source activate reticulate-gpu2
set -eo pipefail

PROJECT_ROOT=/blue/guralnick/share/r.dinnage/Projects/phenovision
cd "$PROJECT_ROOT"
mkdir -p xAI/logs

echo "$(date)  host=$(hostname)  cwd=$(pwd)"

export PYTHONPATH="${PWD}/PlantCLEF2022:${PWD}:${PWD}/xAI/py:${PWD}/xAI/two_noise:${PYTHONPATH:-}"

# ---- config (env-overridable) ------------------------------------------------------------------
V2_ROOT="${V2_ROOT:-xAI/output/preadapt_v2}"
LAST_K="${LAST_K:-3}"
VAL_CSV="${VAL_CSV:-data/inat/val_v1.1.0.csv}"
LMC_STEPS="${LMC_STEPS:-11}"
LMC_EVAL_MAX_BATCHES="${LMC_EVAL_MAX_BATCHES:-40}"
BASIN_STORE="${BASIN_STORE:-${V2_ROOT}/_basin_store}"
MODECONN_DIR="${MODECONN_DIR:-${V2_ROOT}/_modeconn}"
CURVES_DIR="${MODECONN_DIR}/curves"
BARRIERS_CSV="${MODECONN_DIR}/barriers.csv"
FORCE="${FORCE:-0}"

mkdir -p "$BASIN_STORE" "$CURVES_DIR"

if [[ ! -f "$VAL_CSV" ]]; then
    echo "ERROR: VAL_CSV not found: $VAL_CSV (set VAL_CSV=<held-out probe csv>)." >&2
    exit 1
fi

# ---- discover the v2 run dirs (args override auto-discovery) ------------------------------------
RUN_DIRS=()
if [[ $# -gt 0 ]]; then
    RUN_DIRS=("$@")
else
    # the 6 v2 runs: stable_evo x {mae,plantclef,naive} x {s42,s43}. Auto-discover any present.
    for cond in mae plantclef naive; do
        for sd in 42 43; do
            d="${V2_ROOT}/${cond}__stable_evo__s${sd}"
            if [[ -d "$d" ]]; then
                RUN_DIRS+=("$d")
            fi
        done
    done
fi

if [[ ${#RUN_DIRS[@]} -eq 0 ]]; then
    echo "ERROR: no v2 run dirs found under ${V2_ROOT} (and none passed as args)." >&2
    echo "       expected e.g. ${V2_ROOT}/mae__stable_evo__s42 ..." >&2
    exit 1
fi

echo "=== v2 basin + mode-connectivity ==="
echo "v2_root=$V2_ROOT  last_k=$LAST_K  val_csv=$VAL_CSV"
echo "runs (${#RUN_DIRS[@]}):"
for d in "${RUN_DIRS[@]}"; do echo "  $d"; done

# helper: resolve a run's FINAL kept checkpoint path (highest-step stepNNNNNNNN.pt under kept/, run
# dir, or checkpoints/). Prints the path or nothing.
final_ckpt_for() {
    local run_dir="$1"
    python - "$run_dir" <<'PYEOF'
import os, sys
run_dir = os.path.abspath(sys.argv[1].rstrip("/"))
best_step, best_path = -1, None
for d in (os.path.join(run_dir, "kept"), run_dir, os.path.join(run_dir, "checkpoints")):
    if not os.path.isdir(d):
        continue
    for e in os.listdir(d):
        if e.startswith("step") and e.endswith(".pt"):
            digits = e[len("step"):-len(".pt")]
            if digits.isdigit():
                s = int(digits)
                p = os.path.join(d, e)
                if os.path.exists(p) and s > best_step:
                    best_step, best_path = s, p
if best_path:
    print(best_path)
PYEOF
}

# =================================================================================================
# (1) BASIN GEOMETRY -- block_basin.py on each run's converged checkpoints (idempotent).
# =================================================================================================
echo
echo "=== (1) basin geometry on converged checkpoints ==="
for run_dir in "${RUN_DIRS[@]}"; do
    run_id="$(basename "$run_dir")"
    per_run_store="${BASIN_STORE}/${run_id}"
    sentinel="${per_run_store}/.basin_done"
    if [[ "$FORCE" != "1" && -f "$sentinel" ]]; then
        echo "[basin] $run_id already done (sentinel present); skipping. (FORCE=1 to redo)"
        continue
    fi
    mkdir -p "$per_run_store"
    echo "[basin] $run_id -> $per_run_store"
    python -u xAI/py/extractor/block_basin.py \
        --run-dir "$run_dir" \
        --out-store "$per_run_store" \
        --val-csv "$VAL_CSV" \
        --last-k "$LAST_K"
    : > "$sentinel"
done

# =================================================================================================
# (2) MODE CONNECTIVITY -- all unordered pairs of the 6 v2 FINAL models (idempotent per pair).
# =================================================================================================
echo
echo "=== (2) mode connectivity between all pairs of v2 final models ==="

# Resolve each run's final checkpoint + label once.
LABELS=()
CKPTS=()
for run_dir in "${RUN_DIRS[@]}"; do
    run_id="$(basename "$run_dir")"
    cp="$(final_ckpt_for "$run_dir")"
    if [[ -z "$cp" ]]; then
        echo "[lmc] WARNING: no final step checkpoint for $run_id; excluding from pairs." >&2
        continue
    fi
    LABELS+=("$run_id")
    CKPTS+=("$cp")
    echo "[lmc] final($run_id) = $cp"
done

N=${#LABELS[@]}
if [[ "$N" -lt 2 ]]; then
    echo "[lmc] fewer than 2 runs have a final checkpoint ($N); cannot pair. Skipping LMC." >&2
else
    for ((i=0; i<N; i++)); do
        for ((j=i+1; j<N; j++)); do
            la="${LABELS[$i]}";  lb="${LABELS[$j]}"
            ca="${CKPTS[$i]}";   cb="${CKPTS[$j]}"
            pair_csv="${CURVES_DIR}/${la}__VS__${lb}.csv"
            if [[ "$FORCE" == "1" ]]; then rm -f "$pair_csv"; fi
            if [[ -f "$pair_csv" ]]; then
                echo "[lmc] $la vs $lb already done ($pair_csv); skipping."
                continue
            fi
            # cross-init unless the two runs share a condition (same pretrained start). run_id is
            # "{condition}__stable_evo__s{seed}", so the condition is the field before the first "__".
            cond_a="${la%%__*}"; cond_b="${lb%%__*}"
            cross_flag=""
            if [[ "$cond_a" != "$cond_b" ]]; then cross_flag="--cross-init"; fi
            echo "[lmc] $la vs $lb  (cross-init=${cross_flag:-no})"
            python -u -m posthoc.mode_connectivity \
                --ckpt-a "$ca" --label-a "$la" \
                --ckpt-b "$cb" --label-b "$lb" \
                --steps "$LMC_STEPS" \
                --eval-max-batches "$LMC_EVAL_MAX_BATCHES" \
                --val-csv "$VAL_CSV" \
                $cross_flag \
                --out "$pair_csv"
        done
    done
fi

# ---- aggregate barrier heights from every present pair λ-curve CSV into ONE summary CSV ---------
echo
echo "=== aggregating barriers -> $BARRIERS_CSV ==="
python - "$CURVES_DIR" "$BARRIERS_CSV" <<'PYEOF'
import csv, os, sys
sys.path.insert(0, os.path.join(os.getcwd(), "xAI", "py"))
sys.path.insert(0, os.path.join(os.getcwd(), "xAI", "two_noise"))
sys.path.insert(0, os.path.join(os.getcwd(), "PlantCLEF2022"))
from posthoc.mode_connectivity import loss_barrier, auc_dip, verdict  # noqa: E402

curves_dir, out_csv = sys.argv[1], sys.argv[2]
rows_out = []
for fn in sorted(os.listdir(curves_dir)):
    if not fn.endswith(".csv"):
        continue
    path = os.path.join(curves_dir, fn)
    with open(path, newline="") as f:
        rd = list(csv.DictReader(f))
    if not rd:
        continue
    # coerce numeric fields used by the barrier/dip helpers.
    rows = []
    for r in rd:
        rr = dict(r)
        for k in ("lam", "train_loss", "train_auc_pr", "train_auc_roc",
                  "val_loss", "val_auc_pr", "val_auc_roc"):
            try:
                rr[k] = float(r.get(k, "nan"))
            except (TypeError, ValueError):
                rr[k] = float("nan")
        rows.append(rr)
    label_a = rd[0].get("label_a", "")
    label_b = rd[0].get("label_b", "")
    cross = (label_a.split("__")[0] != label_b.split("__")[0])
    rows_out.append(dict(
        label_a=label_a,
        label_b=label_b,
        cross_init=int(cross),
        train_loss_barrier=loss_barrier(rows, "train_loss"),
        val_loss_barrier=loss_barrier(rows, "val_loss"),
        train_auc_pr_dip=auc_dip(rows, "train_auc_pr"),
        val_auc_pr_dip=auc_dip(rows, "val_auc_pr"),
        verdict=verdict(rows, cross_init=cross),
        curve_csv=os.path.basename(path),
    ))

os.makedirs(os.path.dirname(os.path.abspath(out_csv)) or ".", exist_ok=True)
fieldnames = ["label_a", "label_b", "cross_init",
              "train_loss_barrier", "val_loss_barrier",
              "train_auc_pr_dip", "val_auc_pr_dip", "verdict", "curve_csv"]
with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in rows_out:
        w.writerow(r)
print(f"[barriers] wrote {out_csv} ({len(rows_out)} pair(s))")
for r in rows_out:
    print(f"[barriers] {r['label_a']} vs {r['label_b']}  "
          f"val_loss_barrier=+{r['val_loss_barrier']:.4f}  "
          f"val_auc_pr_dip={r['val_auc_pr_dip']:.4f}  cross_init={r['cross_init']}")
PYEOF

echo
echo "Done: $(date)"
echo "basin stores: $BASIN_STORE/<run_id>/"
echo "lmc curves:   $CURVES_DIR/"
echo "barriers:     $BARRIERS_CSV"

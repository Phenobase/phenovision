#!/bin/bash
# =============================================================================
# migration_setup.sh
# Stand up the xAI preadapt-v2 experiment in a FRESH HiPerGator allocation
# (e.g. the rdinnage.fiu account/login, where /blue/guralnick is NOT readable).
#
# USAGE — run from the repo root AFTER cloning:
#     git clone https://github.com/Phenobase/phenovision.git
#     cd phenovision
#     git checkout two_noise-build
#     bash xAI/scripts/migration_setup.sh
#
# -----------------------------------------------------------------------------
# AGENT END-TO-END RUNBOOK — an agent in a fresh rdinnage.fiu environment can run
# ALL of this. The ONLY step a human must do is (2): place the CSV zip (it is not
# in git and cannot be re-fetched). Everything else is scripted.
#   1. git clone https://github.com/Phenobase/phenovision.git && cd phenovision
#      git checkout two_noise-build
#   2. [HUMAN] drop v2_migration_csvs.zip in the repo root (download the zip this
#      project produced, upload it to the new repo). An agent cannot fetch it
#      (not in git, not re-downloadable).
#   3. CONDA_OVERRIDE_CUDA="11.8" mamba env create -f xAI/environment.yml
#      mamba activate reticulate-gpu2
#      (the CONDA_OVERRIDE_CUDA prefix lets the CUDA build solve on a CPU login node)
#   4. bash xAI/scripts/migration_setup.sh
#        -> auto: unzips CSVs + gdowns the PlantCLEF .pth + downloads the
#           228k-image subset.  (timm MAE/ImageNet weights auto-download run 1.)
#   5. bash xAI/scripts/launch_v2_round.sh 1 rdinnage.fiu rdinnage.fiu rdinnage.fiu-b
#        -> submits round 1 (naive->plantclef->mae s42) as an unattended chain.
#           For round 2 (s43): same command with '2'. Run both for parallelism if
#           the allocation's 4 GPU / 375 GB allow.
#   6. monitor:  squeue -u $USER | grep preadapt
#      results:  xAI/output/preadapt_v2/<run>/  and  .../_extract_store/
# -----------------------------------------------------------------------------
#
# WHAT THIS SCRIPT AUTOMATES:
#   Downloads ONLY the 228,392-image subset the v2 trainer actually reads — from the
#   EXPLICIT committed manifest xAI/data/v2_image_subset.csv.gz (file_name,photo_id,
#   extension), the union of both seeds' train+val picks (~31 GB, ~8% of the 2.85M-row
#   CSVs). The manifest is the SOURCE OF TRUTH (not an RNG re-derivation), so the set is
#   byte-identical in every environment. Images land at the paths the CSVs reference:
#   data/phenobase_inat_data/images/medium/batch_*/<photo_id>.<ext>.
#   Re-running is safe/resumable (existing files are skipped). To regenerate the manifest
#   on the guralnick side (full CSVs present): python xAI/scripts/gen_image_subset.py
#
# ---------------------------------------------------------------------------
# MANUAL PREREQUISITES (NOT in git / cannot be auto-fetched) — do these FIRST:
#
#   (1) CSVs  [REQUIRED — this script reads them to know which images to fetch]
#       data/inat/{train,val}_v1.1.0.csv are 352 MB / 117 MB, over GitHub's 100 MB
#       limit, so they are NOT in the repo. Just drop the provided zip at the
#       REPO ROOT  ->  ./v2_migration_csvs.zip
#       and this script AUTO-UNZIPS it into data/inat/ (it also checks data/inat/
#       and migration/). No manual unzip needed.
#
#   (2) Conda env  (SLIM Python-only stack; the R/RStudio dev stack is NOT needed)
#           CONDA_OVERRIDE_CUDA="11.8" mamba env create -f xAI/environment.yml
#           mamba activate reticulate-gpu2
#       CONDA_OVERRIDE_CUDA is only needed when creating on a CPU login node (the CUDA
#       torch build otherwise refuses to solve: "__cuda ... missing on the system").
#       (torch 2.6 + CUDA, timm, zarr, numpy/pandas/pyarrow, scipy, sklearn, ...)
#       PlantCLEF2022 is VENDORED in this repo with the torch-2.6 / numpy compat
#       fixes already applied — no submodule init needed.
#
#   (3) PlantCLEF pretrained weights  [REQUIRED for the 'plantclef' condition; 4.4 GB]
#       models/PlantCLEF2022_MAE_vit_large_patch16_epoch100.pth is NOT in git. This
#       script AUTO-FETCHES it via gdown from the direct Drive file id
#       1djr0WkA1zn1nsjCPJPMcxX4Yt4yF4vYw. If gdown is blocked (no internet / Drive
#       quota), get it manually:
#         https://drive.google.com/file/d/1djr0WkA1zn1nsjCPJPMcxX4Yt4yF4vYw/view
#       and place it at the path above. (Path hardcoded in preadapt_models.py.)
#
#   (4) timm ViT-L weights (MAE + ImageNet, ~1.2 GB each) AUTO-download from the
#       HuggingFace Hub on the first training run (needs compute-node internet).
#       Nothing to do — or pre-warm with:
#           python -c "import timm; timm.create_model('vit_large_patch16_224.mae', pretrained=True); \
#                      timm.create_model('vit_large_patch16_224', pretrained=True)"
#
#   (5) SLURM — the submit scripts default to --account=guralnick. Override at
#       launch to run under your allocation, e.g.:
#           sbatch --account=rdinnage.fiu --qos=rdinnage.fiu  ... xAI/scripts/submit_preadapt_train.sh
#           sbatch --account=rdinnage.fiu --qos=rdinnage.fiu  ... xAI/scripts/submit_preadapt_collector_gpu.sh   # hpg-turin (L4)
#           sbatch --account=rdinnage.fiu --qos=rdinnage.fiu-b ... xAI/scripts/submit_preadapt_collector_cpu.sh  # burst CPU pass
#       (rdinnage.fiu QOS = 4 GPU / 375 GB — fits a full trainer + heavy collector.)
# =============================================================================
set -eo pipefail
cd "$(dirname "$0")/../.."          # -> repo root
PARALLEL="${PARALLEL:-24}"          # concurrent downloads (override: PARALLEL=48 bash ...)

TRAIN=data/inat/train_v1.1.0.csv
VAL=data/inat/val_v1.1.0.csv
SUBSET=xAI/data/v2_image_subset.csv.gz   # committed, explicit image manifest (the source of truth)

# --- (1) CSVs: auto-unzip the provided v2_migration_csvs.zip if the CSVs aren't present ---
if [ ! -s "$TRAIN" ] || [ ! -s "$VAL" ]; then
  ZIP=""
  for cand in v2_migration_csvs.zip data/inat/v2_migration_csvs.zip migration/v2_migration_csvs.zip; do
    [ -s "$cand" ] && { ZIP="$cand"; break; }
  done
  if [ -n "$ZIP" ]; then
    echo "[migration] unzipping CSVs from $ZIP -> data/inat/"
    mkdir -p data/inat && unzip -o -q "$ZIP" -d data/inat/
  fi
fi
for f in "$TRAIN" "$VAL"; do
  [ -s "$f" ] || { echo "ERROR: $f missing and no v2_migration_csvs.zip found. Put v2_migration_csvs.zip in the REPO ROOT (see header step 1), then re-run."; exit 1; }
done

# --- (3) PlantCLEF weights: gdown the direct file id (manual fallback) ---
PCLEF=models/PlantCLEF2022_MAE_vit_large_patch16_epoch100.pth
GDRIVE_FILE_ID="1djr0WkA1zn1nsjCPJPMcxX4Yt4yF4vYw"   # direct .pth (epoch100) from the PlantCLEF2022 Drive
if [ ! -s "$PCLEF" ]; then
  echo "[migration] fetching PlantCLEF weights via gdown (direct file id) ..."
  mkdir -p models
  python3 -c "import gdown" 2>/dev/null || pip install -q gdown 2>/dev/null || true
  if python3 -c "import gdown" 2>/dev/null; then
    gdown "$GDRIVE_FILE_ID" -O "$PCLEF" 2>/dev/null \
      || gdown "https://drive.google.com/uc?id=$GDRIVE_FILE_ID" -O "$PCLEF" 2>/dev/null || true
  fi
  if [ -s "$PCLEF" ]; then
    echo "[migration] PlantCLEF .pth -> $PCLEF ($(du -h "$PCLEF" | cut -f1))"
  else
    echo "[migration] WARN: gdown failed. Get it manually from
    https://drive.google.com/file/d/$GDRIVE_FILE_ID/view  and place it at $PCLEF .
    Only the 'plantclef' condition needs it."
  fi
fi

echo "[migration] downloading the EXPERIMENT IMAGE SUBSET from the committed manifest $SUBSET (PARALLEL=$PARALLEL, resumable) ..."
[ -s "$SUBSET" ] || { echo "ERROR: image manifest $SUBSET missing. It is committed in the repo — 'git pull' the two_noise-build branch, or regenerate on the guralnick side: python xAI/scripts/gen_image_subset.py"; exit 1; }
PARALLEL="$PARALLEL" python3 - "$SUBSET" <<'PY'
import csv, gzip, os, sys, socket, urllib.request, concurrent.futures
socket.setdefaulttimeout(30)
PAR = int(os.environ.get("PARALLEL", "24"))
S3 = "https://inaturalist-open-data.s3.amazonaws.com/photos/{pid}/medium.{ext}"
SUBSET = sys.argv[1]

# Read the EXPLICIT committed manifest (file_name,photo_id,extension) — the SOURCE OF TRUTH. No RNG
# and no full-CSV parse, so the downloaded set is byte-identical in every environment. (Provenance:
# regenerable from the grid seeds + the trainer's RandomState via xAI/scripts/gen_image_subset.py.)
need = {}                                   # dest_path -> (photo_id, extension)
_open = gzip.open if SUBSET.endswith(".gz") else open
with _open(SUBSET, "rt", newline="") as f:
    for r in csv.DictReader(f):
        need[r["file_name"]] = (r["photo_id"], r["extension"])
todo = [(d, v) for d, v in need.items() if not (os.path.exists(d) and os.path.getsize(d) > 0)]
print(f"[migration] manifest = {len(need)} images (experiment subset); {len(todo)} to fetch ({len(need)-len(todo)} already present)", flush=True)

def fetch(item):
    dest, (pid, ext) = item
    url = S3.format(pid=pid, ext=ext)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    for attempt in range(3):
        try:
            urllib.request.urlretrieve(url, dest + ".part")
            os.replace(dest + ".part", dest)
            return None
        except Exception as e:
            try: os.remove(dest + ".part")
            except OSError: pass
            if attempt == 2:
                return f"{url}\t{dest}\t{e}"

fails, done = [], 0
with concurrent.futures.ThreadPoolExecutor(max_workers=PAR) as ex:
    for res in ex.map(fetch, todo):
        done += 1
        if res: fails.append(res)
        if done % 5000 == 0:
            print(f"[migration]   {done}/{len(todo)} done, {len(fails)} failed", flush=True)
print(f"[migration] download complete: {done-len(fails)} ok, {len(fails)} failed", flush=True)
if fails:
    with open("img_download_failures.tsv", "w") as f:
        f.write("\n".join(fails))
    print("[migration] failures written to ./img_download_failures.tsv "
          "(re-run this script to retry; a few may be deleted/withdrawn observations).", flush=True)
PY

echo "[migration] images are in data/phenobase_inat_data/images/medium/ ."
echo "[migration] Remaining manual prereqs: (1) CSVs done if you got here; (3) PlantCLEF .pth; (2) conda env."
echo "[migration] Then launch with --account=<your_alloc> --qos=<your_qos> (see header, step 5)."

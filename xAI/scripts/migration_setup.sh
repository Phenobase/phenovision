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
# WHAT THIS SCRIPT AUTOMATES:
#   Re-downloads ONLY the ~228k-image subset the v2 trainer actually reads
#   (the union of both seeds' train+val picks, ~31 GB, ~0.3% of the full pool)
#   from the iNaturalist open-data S3 bucket, into the exact paths the CSVs
#   reference: data/phenobase_inat_data/images/medium/batch_*/<photo_id>.<ext>.
#   Re-running is safe/resumable (existing files are skipped).
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
#   (2) Conda env
#           mamba env create -f xAI/environment.yml     # recreates 'reticulate-gpu2'
#           mamba activate reticulate-gpu2
#       (torch 2.6 + CUDA, timm, zarr, numpy/pandas/pyarrow, scipy, sklearn, ...)
#       PlantCLEF2022 is VENDORED in this repo with the torch-2.6 / numpy compat
#       fixes already applied — no submodule init needed.
#
#   (3) PlantCLEF pretrained weights  [REQUIRED for the 'plantclef' condition; 4.4 GB]
#       models/PlantCLEF2022_MAE_vit_large_patch16_epoch100.pth is NOT in git. This
#       script ATTEMPTS to fetch it automatically via gdown from the Drive folder
#       linked in the PlantCLEF2022 README (folder 1JCVX58oVZFuIttPHaeAjs_zkMXkzzJeA).
#       gdown folder-downloads of multi-GB files often fail or are ambiguous (the
#       folder holds 2 epoch-100 variants), so if it doesn't land, get it manually:
#         https://drive.google.com/drive/folders/1JCVX58oVZFuIttPHaeAjs_zkMXkzzJeA
#       and place the 'late submission epoch 100' .pth at the path above. VERIFY the
#       auto-fetched file is the right variant. (Path hardcoded in preadapt_models.py.)
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

# --- (3) PlantCLEF weights: best-effort gdown from the Drive folder (manual fallback) ---
PCLEF=models/PlantCLEF2022_MAE_vit_large_patch16_epoch100.pth
GDRIVE_FOLDER="https://drive.google.com/drive/folders/1JCVX58oVZFuIttPHaeAjs_zkMXkzzJeA"
if [ ! -s "$PCLEF" ]; then
  echo "[migration] fetching PlantCLEF weights via gdown (Google-Drive folder) ..."
  mkdir -p models
  python3 -c "import gdown" 2>/dev/null || pip install -q gdown 2>/dev/null || true
  if python3 -c "import gdown" 2>/dev/null; then
    tmpd=$(mktemp -d)
    gdown --folder "$GDRIVE_FOLDER" -O "$tmpd" 2>/dev/null || true
    cand=$(find "$tmpd" -iname '*epoch*100*.pth' 2>/dev/null | head -1)
    [ -z "$cand" ] && cand=$(find "$tmpd" -iname '*.pth' 2>/dev/null | head -1)
    if [ -n "$cand" ]; then mv "$cand" "$PCLEF"; echo "[migration] placed PlantCLEF .pth -> $PCLEF (VERIFY it's the epoch100 variant)"; fi
    rm -rf "$tmpd"
  fi
  [ -s "$PCLEF" ] || echo "[migration] WARN: could not auto-fetch PlantCLEF weights (gdown folder downloads of
    multi-GB files often fail). Download manually from $GDRIVE_FOLDER (the 'late submission epoch 100'
    .pth) and place it at $PCLEF . Only the 'plantclef' condition needs it."
fi

echo "[migration] re-downloading the CSV-referenced iNat image subset (PARALLEL=$PARALLEL, resumable) ..."
PARALLEL="$PARALLEL" python3 - "$TRAIN" "$VAL" <<'PY'
import csv, os, sys, socket, urllib.request, concurrent.futures
socket.setdefaulttimeout(30)
PAR = int(os.environ.get("PARALLEL", "24"))
S3 = "https://inaturalist-open-data.s3.amazonaws.com/photos/{pid}/medium.{ext}"

need = {}                                   # dest_path -> (photo_id, extension)
for csvf in sys.argv[1:]:
    with open(csvf, newline="") as f:
        for r in csv.DictReader(f):
            need[r["file_name"]] = (r["photo_id"], r["extension"])
todo = [(d, v) for d, v in need.items() if not (os.path.exists(d) and os.path.getsize(d) > 0)]
print(f"[migration] {len(need)} referenced images; {len(todo)} to fetch ({len(need)-len(todo)} already present)", flush=True)

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

#!/usr/bin/env python3
# =============================================================================
# gen_image_subset.py — generate the EXPLICIT image manifest for preadapt-v2.
#
# The trainer (xai_train.create_data_loaders) selects, PER SEED, a random subset
# of the full v1.1.0 CSVs:
#     train = RandomState(seed+1).choice(n_train, max_train_samples, replace=False)
#     val   = RandomState(seed).choice(n_val,   val_subset_size,   replace=False)
#     repr  = first repr_subset_size of the val pick  (no new images)
# The set of images the experiment actually touches is the UNION over all grid
# seeds. Rather than re-derive that from the RNG at download time (fragile across
# numpy/pandas versions + CSV-parse quirks), we materialize it ONCE here into an
# explicit, committed manifest that every environment reads verbatim.
#
# Run on the guralnick side (full CSVs present), then commit the .csv.gz:
#     python xAI/scripts/gen_image_subset.py
# Output: xAI/data/v2_image_subset.csv.gz  (columns: file_name,photo_id,extension)
# =============================================================================
import os, re, sys, gzip, csv
import numpy as np, pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TRAIN = os.path.join(ROOT, "data/inat/train_v1.1.0.csv")
VAL   = os.path.join(ROOT, "data/inat/val_v1.1.0.csv")
GRID  = os.path.join(ROOT, "xAI/two_noise/configs/experiment/preadapt_v2_grid.txt")
OUT   = os.path.join(ROOT, "xAI/data/v2_image_subset.csv.gz")

# --- read seeds + subset sizes FROM THE GRID so the manifest matches what is launched ---
seeds, max_train, val_sub = set(), 0, 10000   # 10000 = trainer default --val_subset_size
for ln in open(GRID):
    m = re.search(r"--seed (\d+)", ln)
    if m: seeds.add(int(m.group(1)))
    m = re.search(r"--max_train_samples (\d+)", ln)
    if m: max_train = max(max_train, int(m.group(1)))
    m = re.search(r"--val_subset_size (\d+)", ln)
    if m: val_sub = int(m.group(1))
seeds = sorted(seeds)
assert seeds, f"no --seed found in {GRID}"
print(f"grid -> seeds={seeds} max_train_samples={max_train} val_subset_size={val_sub}")

tcols = pd.read_csv(TRAIN, usecols=["file_name", "photo_id", "extension"])
vcols = pd.read_csv(VAL,   usecols=["file_name", "photo_id", "extension"])
n_train, n_val = len(tcols), len(vcols)

sel_t, sel_v = set(), set()
for s in seeds:
    if max_train and max_train < n_train:                  # mirrors the trainer's guard exactly
        sel_t |= set(np.random.RandomState(s + 1).choice(n_train, max_train, replace=False).tolist())
    else:
        sel_t |= set(range(n_train))
    sel_v |= set(np.random.RandomState(s).choice(n_val, min(val_sub, n_val), replace=False).tolist())

rows = {}   # file_name -> (photo_id, extension)   (dict de-dups train/val overlap)
for df, idx in ((tcols, sel_t), (vcols, sel_v)):
    fn = df["file_name"].to_numpy(); pid = df["photo_id"].to_numpy(); ext = df["extension"].to_numpy()
    for i in idx:
        rows[fn[i]] = (pid[i], ext[i])

print(f"full sets: train={n_train} val={n_val}; SUBSET (union of seeds {seeds}) = {len(rows)} unique images")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with gzip.open(OUT, "wt", newline="") as f:
    w = csv.writer(f); w.writerow(["file_name", "photo_id", "extension"])
    for fnm in sorted(rows):
        pid, ext = rows[fnm]; w.writerow([fnm, pid, ext])
print(f"wrote {OUT} ({os.path.getsize(OUT)/1e6:.1f} MB)")

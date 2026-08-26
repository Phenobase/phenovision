#!/usr/bin/env python3
"""MEASURE-ONLY equivocal-aware (3-way) decision-change analysis: original vs WebP q82.

Applies the PRODUCTION 3-way thresholds (final_buffer_params.csv):
  Detected     if pred >  equivocal_upper
  Not Detected if pred <  equivocal_lower
  Equivocal    otherwise            (the pipeline filters these out / marks low-certainty)
The pipeline REPORTS only Detected+Unequivocal, so the decision-relevant metric is "does an
image's Detected call change?" (and the worst case, a full Detected<->NotDetected reversal).
Loads phenobase/phenovision offline; GPU if available; no files written; originals untouched.
"""
import os, sys, io, glob, csv, random
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
import numpy as np, torch, timm
from PIL import Image
from transformers import AutoModelForImageClassification

WEBPQ = 82
_args = sys.argv[1:]
N = int(_args[-1]) if _args and _args[-1].isdigit() else 3000
DIRS = [a for a in _args if not a.isdigit()] or ["/blue/guralnick/share/phenobase_inat_data/images/medium/batch_400"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(16); print(f"device: {DEVICE}")

# --- production 3-way bands ---
bands = {}
with open("output/reproductive/v1.1.0/final_buffer_params.csv") as f:
    for r in csv.DictReader(f):
        bands[r["class"]] = (float(r["equivocal_lower"]), float(r["equivocal_upper"]))
print("bands:", bands)

snaps = [s for s in glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--phenobase--phenovision/snapshots/*/"))
    if os.path.exists(os.path.join(s, "model.safetensors"))]
pick = sorted(snaps, key=os.path.getmtime, reverse=True)[0]
print("snapshot:", os.path.basename(pick.rstrip("/")))
model = AutoModelForImageClassification.from_pretrained(pick, local_files_only=True).eval().to(DEVICE)
ncls = model.classifier.out_features
names = ["fruit", "flower"] if ncls == 2 else [f"class{i}" for i in range(ncls)]   # R convention idx0=fruit,idx1=flower
tf = timm.data.create_transform(input_size=224, is_training=False)

files = []
per = max(1, N // len(DIRS))
for d in DIRS:
    fs = [os.path.join(d, f) for f in os.listdir(d) if not f.startswith(".")]
    random.seed(0); files += random.sample(fs, min(per, len(fs)))
print(f"{len(files)} images from {len(DIRS)} dirs\n")

@torch.no_grad()
def probs(pils):
    x = torch.stack([tf(p) for p in pils]).to(DEVICE)
    return torch.sigmoid(model(pixel_values=x).logits).float().cpu().numpy()

PO, PQ, bad, buf = [], [], 0, []
def flush():
    if buf:
        PO.append(probs([b[0] for b in buf])); PQ.append(probs([b[1] for b in buf])); buf.clear()
for f in files:
    try:
        im = Image.open(f).convert("RGB")
        b = io.BytesIO(); im.save(b, "WEBP", quality=WEBPQ, method=6); b.seek(0)
        buf.append((im, Image.open(b).convert("RGB")))
        if len(buf) == 64: flush()
    except Exception:
        bad += 1
flush()
po = np.concatenate(PO); pq = np.concatenate(PQ); n = len(po)

def cat(p, lo, hi):  # 'D' detected, 'N' not-detected, 'E' equivocal
    return np.where(p > hi, "D", np.where(p < lo, "N", "E"))

print(f"=== equivocal-aware (3-way) change: original vs WebP q{WEBPQ}  ({n} imgs, {bad} skipped) ===")
for i, name in enumerate(names):
    lo, hi = bands[name]
    co, cq = cat(po[:, i], lo, hi), cat(pq[:, i], lo, hi)
    dist = {k: int((co == k).sum()) for k in "DEN"}
    any_change = int((co != cq).sum())
    det_change = int(((co == "D") != (cq == "D")).sum())        # gained or lost a reported Detected call
    reversal   = int((((co == "D") & (cq == "N")) | ((co == "N") & (cq == "D"))).sum())  # full flip across the band
    print(f"  {name:7} (band [{lo:.3f},{hi:.3f}])  orig dist D/E/N = {dist['D']}/{dist['E']}/{dist['N']}")
    print(f"           any 3-way change: {any_change}/{n} ({100*any_change/n:.2f}%) | "
          f"Detected-call change: {det_change}/{n} ({100*det_change/n:.2f}%) | "
          f"full D<->N reversal: {reversal}/{n} ({100*reversal/n:.2f}%)")
# combined: an OBSERVATION-level report changes if EITHER class's Detected call changes
det_change_any = np.zeros(n, dtype=bool)
rev_any = np.zeros(n, dtype=bool)
for i, name in enumerate(names):
    lo, hi = bands[name]
    co, cq = cat(po[:, i], lo, hi), cat(pq[:, i], lo, hi)
    det_change_any |= ((co == "D") != (cq == "D"))
    rev_any |= (((co == "D") & (cq == "N")) | ((co == "N") & (cq == "D")))
print(f"\n  IMAGE-LEVEL (either class): Detected-call change {int(det_change_any.sum())}/{n} "
      f"({100*det_change_any.sum()/n:.2f}%) | full reversal {int(rev_any.sum())}/{n} ({100*rev_any.sum()/n:.2f}%)")
print("  (vs the ~2.5% RAW point-threshold flip rate; this is the production-relevant number)")

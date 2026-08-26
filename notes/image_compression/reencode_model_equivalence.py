#!/usr/bin/env python3
"""MEASURE-ONLY: does WebP q82 change the PRODUCTION model's predictions vs the originals?

Loads the real phenobase/phenovision reproductive model (from the local HF cache, offline),
runs a sample of images through the SAME val transform twice -- (a) original bytes, (b) WebP-q82
re-encoded bytes -- and compares per-class sigmoid probabilities + decision-category flips at the
production thresholds. No files written, originals untouched. CPU (keeps off the GPU cap).
"""
import os, sys, io, json, glob, random
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
import numpy as np, torch, timm
from PIL import Image
from transformers import AutoModelForImageClassification

_args = sys.argv[1:]
N = int(_args[-1]) if _args and _args[-1].isdigit() else 300
DIRS = [a for a in _args if not a.isdigit()] or ["/blue/guralnick/share/phenobase_inat_data/images/medium/batch_400"]
WEBPQ = 82
THRESH = {"flower": 0.48, "fruit": 0.60}     # from output/reproductive/v1.1.0/final_buffer_params.csv
torch.set_num_threads(16)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {DEVICE}")

# --- load the production model offline (cached main ref; config lacks num_labels/id2label) ---
print("loading phenobase/phenovision (offline, cached)...")
try:
    model = AutoModelForImageClassification.from_pretrained("phenobase/phenovision", local_files_only=True).eval()
    print("  loaded via repo main ref")
except Exception as e:
    snaps = [s for s in glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--phenobase--phenovision/snapshots/*/"))
        if os.path.exists(os.path.join(s, "model.safetensors"))]
    pick = sorted(snaps, key=os.path.getmtime, reverse=True)[0]
    print(f"  main-ref load failed ({e}); fallback snapshot {os.path.basename(pick.rstrip('/'))}")
    model = AutoModelForImageClassification.from_pretrained(pick, local_files_only=True).eval()
ncls = model.classifier.out_features          # truth is the head width (config has no num_labels)
# R convention (ml_annotation_targets.r: V1=fruit, V2=flower): output idx0=fruit, idx1=flower
names = ["fruit", "flower"] if ncls == 2 else [f"class{i}" for i in range(ncls)]
idx2name = {i: names[i] for i in range(ncls)}
thr = np.array([THRESH.get(names[i], 0.5) for i in range(ncls)])
print(f"num labels: {ncls}; index->class->thr: " + ", ".join(f"{i}:{names[i]}={thr[i]}" for i in range(ncls)))
print("NOTE: class order is by R convention (idx0=fruit,idx1=flower); |dp| is order-independent.")
model = model.to(DEVICE)

tf = timm.data.create_transform(input_size=224, is_training=False)  # Resize256+CenterCrop224+norm

files = []
per = max(1, N // len(DIRS))
for _d in DIRS:
    _fs = [os.path.join(_d, f) for f in os.listdir(_d) if not f.startswith(".")]
    random.seed(0); files += random.sample(_fs, min(per, len(_fs)))
print(f"sampling {len(files)} images from {len(DIRS)} batch dir(s)")

@torch.no_grad()
def probs(pils):
    x = torch.stack([tf(p) for p in pils]).to(DEVICE)
    return torch.sigmoid(model(pixel_values=x).logits).float().cpu().numpy()

po, pq, bad = [], [], 0
B = 64
batch_o, batch_q = [], []
def flush():
    global po, pq
    if batch_o:
        po.append(probs(batch_o)); pq.append(probs(batch_q)); batch_o.clear(); batch_q.clear()
for f in files:
    try:
        im = Image.open(f).convert("RGB")
        b = io.BytesIO(); im.save(b, "WEBP", quality=WEBPQ, method=6); b.seek(0)
        imq = Image.open(b).convert("RGB")
        batch_o.append(im); batch_q.append(imq)
        if len(batch_o) == B: flush()
    except Exception:
        bad += 1
flush()
po = np.concatenate(po); pq = np.concatenate(pq)
d = np.abs(po - pq)

print(f"\n=== WebP q{WEBPQ} vs ORIGINAL -- production model predictions ({len(po)} imgs, {bad} skipped) ===")
for i in range(po.shape[1]):
    name = idx2name[i]
    flips = int(np.sum((po[:, i] > thr[i]) != (pq[:, i] > thr[i])))
    r = np.corrcoef(po[:, i], pq[:, i])[0, 1]
    print(f"  {name:7} (thr {thr[i]}): mean|dp|={d[:,i].mean():.5f}  max|dp|={d[:,i].max():.5f}  "
          f"p95|dp|={np.percentile(d[:,i],95):.5f}  Pearson r={r:.5f}  decision flips={flips}/{len(po)} ({100*flips/len(po):.2f}%)")
print(f"\n  overall: mean|dp|={d.mean():.5f}  max|dp|={d.max():.5f}")
print("  (interpretation: tiny dp and ~0 flips => q82 is safe for inference; large flips => red flag)")

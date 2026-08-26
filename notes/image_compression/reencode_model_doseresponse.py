#!/usr/bin/env python3
"""MEASURE-ONLY dose-response: how much do the PRODUCTION model's predictions change vs the
original as WebP quality drops? Control = near-lossless q98 (should be ~0 if the test is faithful).
Loads phenobase/phenovision offline; original vs WebP q in {98,92,85,82,75}; reports per-q
mean/p95/max |dp| and decision flips at production thresholds. GPU if available. No files written."""
import os, sys, io, glob, random
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
import numpy as np, torch, timm
from PIL import Image
from transformers import AutoModelForImageClassification

QUALS = [98, 92, 85, 82, 75]
THRESH = {"flower": 0.48, "fruit": 0.60}
_args = sys.argv[1:]
N = int(_args[-1]) if _args and _args[-1].isdigit() else 1500
DIRS = [a for a in _args if not a.isdigit()] or ["/blue/guralnick/share/phenobase_inat_data/images/medium/batch_400"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(16)
print(f"device: {DEVICE}")

snaps = [s for s in glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--phenobase--phenovision/snapshots/*/"))
    if os.path.exists(os.path.join(s, "model.safetensors"))]
pick = sorted(snaps, key=os.path.getmtime, reverse=True)[0]
print(f"snapshot: {os.path.basename(pick.rstrip('/'))}")
model = AutoModelForImageClassification.from_pretrained(pick, local_files_only=True).eval().to(DEVICE)
ncls = model.classifier.out_features
names = ["fruit", "flower"] if ncls == 2 else [f"class{i}" for i in range(ncls)]
thr = np.array([THRESH.get(names[i], 0.5) for i in range(ncls)])
tf = timm.data.create_transform(input_size=224, is_training=False)

files = []
per = max(1, N // len(DIRS))
for d in DIRS:
    fs = [os.path.join(d, f) for f in os.listdir(d) if not f.startswith(".")]
    random.seed(0); files += random.sample(fs, min(per, len(fs)))
print(f"{len(files)} images from {len(DIRS)} dirs; quals={QUALS}\n")

@torch.no_grad()
def probs(pils):
    x = torch.stack([tf(p) for p in pils]).to(DEVICE)
    return torch.sigmoid(model(pixel_values=x).logits).float().cpu().numpy()

# accumulate predictions: original + each quality
P = {q: [] for q in QUALS}; PO = []; bad = 0
chunk = 64; buf = []
def flush():
    global PO
    if not buf: return
    PO.append(probs([b["o"] for b in buf]))
    for q in QUALS: P[q].append(probs([b[q] for b in buf]))
    buf.clear()
for f in files:
    try:
        im = Image.open(f).convert("RGB"); rec = {"o": im}
        for q in QUALS:
            b = io.BytesIO(); im.save(b, "WEBP", quality=q, method=4); b.seek(0)
            rec[q] = Image.open(b).convert("RGB")
        buf.append(rec)
        if len(buf) == chunk: flush()
    except Exception:
        bad += 1
flush()
po = np.concatenate(PO)
print(f"=== dose-response: WebP quality vs ORIGINAL ({len(po)} imgs, {bad} skipped) ===")
print(f"  {'qual':>5} {'mean|dp|':>9} {'p95|dp|':>9} {'max|dp|':>9} {'flips_fruit':>12} {'flips_flower':>13}")
for q in QUALS:
    pq = np.concatenate(P[q]); d = np.abs(po - pq)
    ff = int(np.sum((po[:,0] > thr[0]) != (pq[:,0] > thr[0])))
    fl = int(np.sum((po[:,1] > thr[1]) != (pq[:,1] > thr[1])))
    print(f"  q{q:>4} {d.mean():9.5f} {np.percentile(d,95):9.5f} {d.max():9.5f} "
          f"{ff:5d}/{len(po)} ({100*ff/len(po):4.2f}%) {fl:5d}/{len(po)} ({100*fl/len(po):4.2f}%)")
print("\n  read: if q98 (near-lossless) already shows flips ~= q82, it's model boundary-jitter/test")
print("        artifact (not q82); if flips rise as quality drops, it's a real compression effect.")

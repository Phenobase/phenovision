#!/usr/bin/env python3
"""
reencode_fidelity_test.py -- MEASURE-ONLY (no files written, originals untouched).

(a) Size savings at several SHORTER-SIDE downscale targets (so a 224 crop never upscales at
    val time): for each target, encode in memory as JPEG q90 and WebP q85, sum bytes.
(b) Print the EXACT timm transforms the training scripts build (shows RandomResizedCrop
    scale/ratio + the full aug stack), and run a few re-encoded samples through BOTH
    transform_train and transform_val to confirm they pass through to a valid 224x224x3 tensor.

Usage: python reencode_fidelity_test.py <batchdir> [<batchdir> ...] [--sample N] [--targets 256 320 384]
"""
import os, sys, io, argparse, random
from concurrent.futures import ProcessPoolExecutor
from PIL import Image

def measure(args_tuple):
    path, targets, jpegq, webpq = args_tuple
    try:
        orig = os.path.getsize(path)
        im = Image.open(path).convert("RGB")
        w, h = im.size; short = min(w, h)
        out = {"orig": orig, "n": 1, "corrupt": 0}
        for t in targets:
            s = t / short
            im2 = im.resize((max(1, round(w*s)), max(1, round(h*s))), Image.LANCZOS) if s < 1 else im
            bj = io.BytesIO(); im2.save(bj, "JPEG", quality=jpegq)
            bw = io.BytesIO(); im2.save(bw, "WEBP", quality=webpq, method=6)
            out[f"jpeg{t}"] = bj.tell(); out[f"webp{t}"] = bw.tell()
        return out
    except Exception:
        z = {"orig": 0, "n": 1, "corrupt": 1}
        for t in targets: z[f"jpeg{t}"] = 0; z[f"webp{t}"] = 0
        return z

def run(a):
    # --- (b) print the actual transforms + passthrough check ---
    import timm, torch
    tt = timm.data.create_transform(input_size=224, is_training=True,
                                    auto_augment="rand-m9-mstd0.5-inc1",
                                    re_prob=0.25, re_mode="pixel", re_count=1)
    tv = timm.data.create_transform(input_size=224, is_training=False)
    print("=== TRAIN transform (timm create_transform, is_training=True) ===")
    print(tt)
    print("\n=== VAL transform (is_training=False) ===")
    print(tv)

    # grab a few sample images for passthrough
    sample_paths = []
    for d in a.batches:
        fs = [os.path.join(d, f) for f in os.listdir(d) if not f.startswith(".")][:5]
        sample_paths += fs
    print("\n=== passthrough check: re-encode -> reload -> transform -> tensor shape ===")
    ok = True
    for p in sample_paths[:6]:
        im = Image.open(p).convert("RGB"); w, h = im.size; s = 256/min(w, h)
        im2 = im.resize((round(w*s), round(h*s)), Image.LANCZOS) if s < 1 else im
        for fmt, kw in (("JPEG", dict(quality=90)), ("WEBP", dict(quality=85, method=6))):
            b = io.BytesIO(); im2.save(b, fmt, **kw); b.seek(0)
            re = Image.open(b).convert("RGB")
            xt = tt(re); xv = tv(re)
            good = tuple(xt.shape) == (3, 224, 224) and tuple(xv.shape) == (3, 224, 224)
            ok = ok and good
            print(f"  {os.path.basename(p)} 256px {fmt}: train{tuple(xt.shape)} val{tuple(xv.shape)} -> {'OK' if good else 'FAIL'}")
    print(f"  PASSTHROUGH: {'ALL OK' if ok else 'FAILURES PRESENT'}")

    # --- (a) size sweep over shorter-side targets ---
    g = lambda x, y: 0.0 if y == 0 else x/y
    tot = {"orig": 0, "n": 0, "corrupt": 0}
    for t in a.targets: tot[f"jpeg{t}"] = 0; tot[f"webp{t}"] = 0
    for d in a.batches:
        files = [os.path.join(d, f) for f in os.listdir(d) if not f.startswith(".")]
        files = [f for f in files if os.path.isfile(f)]
        if a.sample and len(files) > a.sample:
            random.seed(0); files = random.sample(files, a.sample)
        work = [(f, a.targets, a.jpegq, a.webpq) for f in files]
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for r in ex.map(measure, work, chunksize=64):
                for k, v in r.items(): tot[k] = tot.get(k, 0) + v
    o, n, c = tot["orig"], tot["n"], tot["corrupt"]
    print(f"\n=== SIZE SWEEP  (sample={a.sample or 'ALL'}/batch, {n} imgs, {c} corrupt) ===")
    print(f"  original (medium)        : {o/1e9:7.3f} GB  (100.0%)")
    for t in a.targets:
        j, wv = tot[f"jpeg{t}"], tot[f"webp{t}"]
        print(f"  short-{t} JPEG q{a.jpegq}        : {j/1e9:7.3f} GB  ({100*g(j,o):5.1f}% of orig -> saves {100*(1-g(j,o)):4.1f}%)")
        print(f"  short-{t} WebP q{a.webpq}        : {wv/1e9:7.3f} GB  ({100*g(wv,o):5.1f}% of orig -> saves {100*(1-g(wv,o)):4.1f}%; +{100*(1-g(wv,j)):4.1f}% over JPEG)")
    print(f"\n  EXTRAPOLATION to {a.store_tb} TB store:")
    for t in a.targets:
        print(f"    short-{t}: JPEG ~{a.store_tb*g(tot[f'jpeg{t}'],o):.2f} TB | WebP ~{a.store_tb*g(tot[f'webp{t}'],o):.2f} TB")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("batches", nargs="+")
    ap.add_argument("--targets", type=int, nargs="+", default=[256, 320, 384])
    ap.add_argument("--jpegq", type=int, default=90)
    ap.add_argument("--webpq", type=int, default=85)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--sample", type=int, default=8000, help="images/batch (0=all)")
    ap.add_argument("--store-tb", type=float, default=15.0)
    run(ap.parse_args())

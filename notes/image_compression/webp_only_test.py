#!/usr/bin/env python3
"""MEASURE-ONLY: full-resolution WebP re-encode (NO downscale) vs original JPEG.
Isolates the format-only saving. Reports a few WebP qualities + lossless."""
import os, sys, io, argparse, random
from concurrent.futures import ProcessPoolExecutor
from PIL import Image

def measure(t):
    path, qs = t
    try:
        orig = os.path.getsize(path)
        im = Image.open(path).convert("RGB")
        out = {"orig": orig, "n": 1, "corrupt": 0}
        for q in qs:
            b = io.BytesIO(); im.save(b, "WEBP", quality=q, method=6); out[f"q{q}"] = b.tell()
        bl = io.BytesIO(); im.save(bl, "WEBP", lossless=True, method=6); out["lossless"] = bl.tell()
        return out
    except Exception:
        d = {"orig": 0, "n": 1, "corrupt": 1, "lossless": 0}
        for q in qs: d[f"q{q}"] = 0
        return d

def run(a):
    g = lambda x, y: 0.0 if y == 0 else x/y
    tot = {}
    for d in a.batches:
        files = [os.path.join(d, f) for f in os.listdir(d) if not f.startswith(".")]
        files = [f for f in files if os.path.isfile(f)]
        if a.sample and len(files) > a.sample:
            random.seed(0); files = random.sample(files, a.sample)
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for r in ex.map(measure, [(f, a.q) for f in files], chunksize=64):
                for k, v in r.items(): tot[k] = tot.get(k, 0) + v
    o, n, c = tot["orig"], tot["n"], tot["corrupt"]
    print(f"=== FULL-RES WebP (no downscale)  ({n} imgs, {c} corrupt) ===")
    print(f"  original JPEG     : {o/1e9:7.3f} GB  (100.0%)")
    for q in a.q:
        v = tot[f"q{q}"]
        print(f"  WebP q{q}          : {v/1e9:7.3f} GB  ({100*g(v,o):5.1f}% of orig -> saves {100*(1-g(v,o)):5.1f}%)")
    v = tot["lossless"]
    print(f"  WebP lossless     : {v/1e9:7.3f} GB  ({100*g(v,o):5.1f}% of orig -> saves {100*(1-g(v,o)):5.1f}%)")
    print(f"\n  EXTRAPOLATION to {a.store_tb} TB store:")
    for q in a.q:
        print(f"    WebP q{q}: ~{a.store_tb*g(tot[f'q{q}'],o):.2f} TB")
    print(f"    WebP lossless: ~{a.store_tb*g(tot['lossless'],o):.2f} TB")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("batches", nargs="+")
    ap.add_argument("--q", type=int, nargs="+", default=[80, 85, 90])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--sample", type=int, default=8000)
    ap.add_argument("--store-tb", type=float, default=15.0)
    run(ap.parse_args())

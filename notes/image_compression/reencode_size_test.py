#!/usr/bin/env python3
"""
reencode_size_test.py -- MEASURE-ONLY (no files written, originals untouched).

For each image in the given batch dir(s), encode IN MEMORY at three stages and sum the bytes:
  1. original                         (on-disk JPEG, as-is)
  2. downscaled JPEG  (long<=256, q90)   -> isolates the DOWNSCALE saving
  3. downscaled WebP  (long<=256, q85)   -> isolates the WEBP saving ON TOP of downscaling

Prints per-batch and combined totals + ratios, and extrapolates to a given full-store size.
Usage:
    python reencode_size_test.py /path/batch_400 /path/batch_800 [--long 256] [--jpegq 90] [--webpq 85] [--workers 16] [--sample N]
"""
import os, sys, io, argparse, random
from concurrent.futures import ProcessPoolExecutor
from PIL import Image

def measure(path, long_side, jpegq, webpq):
    try:
        orig = os.path.getsize(path)
        im = Image.open(path); im = im.convert("RGB")
        w, h = im.size
        s = long_side / max(w, h)
        im2 = im.resize((max(1, round(w*s)), max(1, round(h*s))), Image.LANCZOS) if s < 1 else im
        bj = io.BytesIO(); im2.save(bj, "JPEG", quality=jpegq); jpeg = bj.tell()
        bw = io.BytesIO(); im2.save(bw, "WEBP", quality=webpq, method=6); webp = bw.tell()
        return (orig, jpeg, webp, 0)
    except Exception:
        try: return (os.path.getsize(path), 0, 0, 1)
        except OSError: return (0, 0, 0, 1)

def run(args):
    g = (lambda a, b: 0.0 if b == 0 else a/b)
    grand = [0, 0, 0, 0, 0]  # orig, jpeg, webp, corrupt, n
    for d in args.batches:
        files = [os.path.join(d, f) for f in os.listdir(d) if not f.startswith(".")]
        files = [f for f in files if os.path.isfile(f)]
        if args.sample and len(files) > args.sample:
            random.seed(0); files = random.sample(files, args.sample)
        o = j = wv = c = 0
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for (oo, jj, ww, cc) in ex.map(measure, files,
                                           [args.long]*len(files), [args.jpegq]*len(files),
                                           [args.webpq]*len(files), chunksize=64):
                o += oo; j += jj; wv += ww; c += cc
        n = len(files)
        print(f"\n=== {os.path.basename(d)}  ({n} files, {c} corrupt/skipped) ===")
        print(f"  original           : {o/1e9:7.3f} GB")
        print(f"  256px JPEG q{args.jpegq}     : {j/1e9:7.3f} GB   ({100*g(j,o):5.1f}% of orig  ->  downscale saves {100*(1-g(j,o)):4.1f}%)")
        print(f"  256px WebP q{args.webpq}     : {wv/1e9:7.3f} GB   ({100*g(wv,o):5.1f}% of orig  ->  total saves {100*(1-g(wv,o)):4.1f}%;  WebP adds {100*(1-g(wv,j)):4.1f}% over the JPEG)")
        grand[0]+=o; grand[1]+=j; grand[2]+=wv; grand[3]+=c; grand[4]+=n
    o, j, wv, c, n = grand
    print(f"\n=== COMBINED  ({n} files, {c} corrupt/skipped) ===")
    print(f"  original           : {o/1e9:7.3f} GB")
    print(f"  256px JPEG q{args.jpegq}     : {j/1e9:7.3f} GB   ({100*g(j,o):5.1f}% of orig  ->  downscale saves {100*(1-g(j,o)):4.1f}%)")
    print(f"  256px WebP q{args.webpq}     : {wv/1e9:7.3f} GB   ({100*g(wv,o):5.1f}% of orig  ->  total saves {100*(1-g(wv,o)):4.1f}%;  WebP adds {100*(1-g(wv,j)):4.1f}% over the JPEG)")
    if args.store_tb:
        print(f"\n  EXTRAPOLATION to a {args.store_tb} TB store at these ratios:")
        print(f"    -> 256px JPEG: ~{args.store_tb*g(j,o):.2f} TB    256px WebP: ~{args.store_tb*g(wv,o):.2f} TB")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("batches", nargs="+")
    ap.add_argument("--long", type=int, default=256)
    ap.add_argument("--jpegq", type=int, default=90)
    ap.add_argument("--webpq", type=int, default=85)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--sample", type=int, default=0, help="sample N images per batch (0 = all)")
    ap.add_argument("--store-tb", type=float, default=15.0, help="extrapolate ratios to this store size (TB)")
    run(ap.parse_args())

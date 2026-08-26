#!/usr/bin/env python3
"""MEASURE-ONLY iso-perceptual-quality codec test (full-res, no downscale, no files written).

For a sample of original iNat JPEGs, sweep JPEG and WebP quality; for each setting encode in
memory, decode back, and compute SSIM vs the ORIGINAL (the reference the model sees). Aggregate
mean(SSIM) and mean(bytes) per setting, then interpolate the WebP size at JPEG q85/q90's SSIM
=> "WebP is X% smaller at EQUAL perceptual quality" (removes the cross-codec q-parameter ambiguity).
"""
import os, sys, io, random
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from PIL import Image
from scipy.ndimage import gaussian_filter

JQ = [75, 80, 85, 90, 95]
WQ = [65, 70, 75, 80, 85, 90]

def ssim(a, b):                       # textbook gaussian-window SSIM on luminance, range [0,1]
    C1, C2 = 0.01**2, 0.03**2
    ma, mb = gaussian_filter(a, 1.5), gaussian_filter(b, 1.5)
    ma2, mb2, mab = ma*ma, mb*mb, ma*mb
    va = gaussian_filter(a*a, 1.5) - ma2
    vb = gaussian_filter(b*b, 1.5) - mb2
    vab = gaussian_filter(a*b, 1.5) - mab
    s = ((2*mab + C1)*(2*vab + C2)) / ((ma2 + mb2 + C1)*(va + vb + C2))
    return float(s.mean())

def work(p):
    try:
        ob = os.path.getsize(p)
        im = Image.open(p).convert("RGB")
        ref = np.asarray(im.convert("L"), dtype=np.float64) / 255.0
        out = {"orig": ob}
        for q in JQ:
            b = io.BytesIO(); im.save(b, "JPEG", quality=q); nb = b.tell()
            b.seek(0); dec = np.asarray(Image.open(b).convert("L"), dtype=np.float64) / 255.0
            out[f"j{q}"] = (nb, ssim(ref, dec))
        for q in WQ:
            b = io.BytesIO(); im.save(b, "WEBP", quality=q, method=6); nb = b.tell()
            b.seek(0); dec = np.asarray(Image.open(b).convert("L"), dtype=np.float64) / 255.0
            out[f"w{q}"] = (nb, ssim(ref, dec))
        return out
    except Exception:
        return None

def main():
    batches = sys.argv[1:] or []
    per = 2000
    files = []
    for d in batches:
        fs = [os.path.join(d, f) for f in os.listdir(d) if not f.startswith(".")]
        random.seed(0); files += random.sample(fs, min(per, len(fs)))
    agg = {}; orig_ok = 0; nok = 0; nbad = 0
    with ProcessPoolExecutor(max_workers=16) as ex:
        for r in ex.map(work, files, chunksize=32):
            if r is None: nbad += 1; continue
            nok += 1; orig_ok += r["orig"]
            for k, v in r.items():
                if k == "orig": continue
                bsum, ssum = agg.get(k, (0.0, 0.0))
                agg[k] = (bsum + v[0], ssum + v[1])
    pct = lambda k: 100*agg[k][0]/orig_ok
    sm  = lambda k: agg[k][1]/nok
    print(f"=== iso-quality SSIM test (full-res, {nok} imgs, {nbad} skipped) ===")
    print(f"  reference = original on-disk JPEG; SSIM of each re-encode vs original (1.0=identical)\n")
    print(f"  {'setting':10} {'%bytes':>8} {'mean_SSIM':>10}")
    for q in JQ: print(f"  JPEG q{q:<4} {pct(f'j{q}'):8.1f} {sm(f'j{q}'):10.4f}")
    for q in WQ: print(f"  WebP q{q:<4} {pct(f'w{q}'):8.1f} {sm(f'w{q}'):10.4f}")
    # iso-SSIM: interpolate WebP %bytes at each JPEG ref's SSIM
    w_ssim = np.array([sm(f"w{q}") for q in WQ]); w_pct = np.array([pct(f"w{q}") for q in WQ])
    order = np.argsort(w_ssim); w_ssim, w_pct = w_ssim[order], w_pct[order]
    print(f"\n  === WebP vs JPEG at EQUAL perceptual quality (SSIM-matched) ===")
    for q in (85, 90, 95):
        ts = sm(f"j{q}"); jp = pct(f"j{q}")
        clamp = " (EXTRAPOLATED: WebP sweep doesn't reach this SSIM)" if ts > w_ssim.max() else ""
        wp = float(np.interp(ts, w_ssim, w_pct))
        print(f"  match JPEG q{q} (SSIM={ts:.4f}, {jp:.1f}% bytes): WebP needs {wp:.1f}% bytes -> WebP {100*(1-wp/jp):4.1f}% smaller{clamp}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Generate the EXPERIMENT-B noise-SHAPE grid: same injected temperature, different noise shape.

The demographic (pSGLD) noise has covariance ∝ the applied preconditioner P. Experiment B asks
whether the convergence penalty is the noise MAGNITUDE or its SHAPE, by re-shaping the injection
(trace-matched, so the temperature is held fixed) to:
  * pSGLD   (default, no --demographic-shape-exp): variance ∝ P  (FDT / current)
  * iso     (--demographic-shape-exp 0):           variance ∝ 1  (flat in param space)
  * fisher  (--demographic-shape-exp 1):           variance ∝ v̂  (curvature-aligned — the shape
                                                   that MINIBATCH noise has)
If fisher hurts LESS than pSGLD at matched T -> it was the shape, not the magnitude (the injected
noise just wasn't curvature-aligned like the natural gradient noise). Runs the breeder's-equation
testbed (large batch -> gradient noise ~0 -> injected noise dominates) plus a small-batch contrast.

Each line is one `python -m ml_experiments.benchmarks` invocation; pair with
scripts/submit_optimizer_compare.sh (TN_CMP_GRID=<out>).
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "configs" / "experiment" / "shape_grid.txt"

# shape tag -> (flag fragment, label-suffix). pSGLD is the default injection (no shape flag).
SHAPES = {
    "psgld":  ("", "psgld"),
    "iso":    ("--demographic-shape-exp 0", "iso"),
    "fisher": ("--demographic-shape-exp 1", "fisher"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--temp", type=float, required=True,
                    help="pSGLD temperature T (pick the value tn_mech shows clearly penalizes)")
    ap.add_argument("--lr", type=float, default=2.5e-3, help="fixed lr (match the disentangle sweep)")
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[4096, 64],
                    help="4096 = breeder's testbed (gradient noise ~0); 64 = high-grad-noise contrast")
    ap.add_argument("--shapes", nargs="+", default=["psgld", "fisher", "iso"], choices=list(SHAPES))
    ap.add_argument("--with-base", action="store_true", default=True,
                    help="also emit the no-demo baseline per batch (reference)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--dataset-size", type=int, default=50000)
    ap.add_argument("--micro-batch", type=int, default=256)
    ap.add_argument("--demo-warmup", type=int, default=200)
    ap.add_argument("--swa-start-frac", type=float, default=0.75)
    ap.add_argument("--noise-scale-k", type=int, default=8)
    ap.add_argument("--runs-dir", default=str(ROOT / "runs" / "benchmarks_shape"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    lines = []
    for bs in args.batch_sizes:
        micro = min(bs, args.micro_batch)
        accum = max(1, bs // micro)
        spe = max(1, math.ceil(args.dataset_size / bs))
        ms = args.epochs * spe
        ee = max(20, ms // 50)
        head = (f"--model vit_s --dataset cifar100 --optimizer stable_evo --lr {args.lr:g} "
                f"--max-update-norm 2.0 --batch-size {micro} --accum-steps {accum} "
                f"--max-steps {ms} --eval-every {ee}")
        probe = (f"--noise-scale-every {ee} --noise-scale-k {args.noise_scale_k} "
                 f"--swa-start-frac {args.swa_start_frac:g}")
        tail = (f"--log-every 50 --early-stop-patience 0 --seed {args.seed} "
                f"--out-dir {args.runs_dir} --amp")
        if args.with_base:
            lines.append(f"{head} {probe} {tail} --label-suffix base")
        for shp in args.shapes:
            frag, sfx = SHAPES[shp]
            demo = (f"--demographic-noise --demographic-temperature {args.temp:g} "
                    f"--demographic-warmup {args.demo_warmup} {frag}".strip())
            lines.append(f"{head} {probe} {tail} --label-suffix {sfx} {demo}")

    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(lines)} experiment-B configs to {args.out}")
    print(f"  T={args.temp:g}  lr={args.lr:g}  batches={args.batch_sizes}  shapes={args.shapes}"
          + ("  (+base)" if args.with_base else ""))
    print(f"SLURM array: TN_CMP_GRID={args.out} sbatch --array=0-{len(lines)-1}%3 "
          f"scripts/submit_optimizer_compare.sh")


if __name__ == "__main__":
    main()

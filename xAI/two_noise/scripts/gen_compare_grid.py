#!/usr/bin/env python
"""Generate the FULL-RUN optimizer-comparison grid (stage 2), using the lr_finder's suggested lrs.

Two-stage protocol:
  Stage 1 (ml_experiments.lr_finder): a short LR range test per (optimizer, batch) -> a tuned lr,
           written to runs/lr_finder/suggested.csv.
  Stage 2 (THIS): one FULL run TO CONVERGENCE per (optimizer, batch) at that tuned lr, with early
           stopping (so large batches don't over-train), tracking loss/accuracy/speed and (for
           stable_evo) the realized exponent over time. Plus demographic-noise (pSGLD) variants to
           measure how injected noise moves alpha / final accuracy / speed.

Each line is one `python -m ml_experiments.benchmarks` invocation; the array
(scripts/submit_optimizer_compare.sh) runs one line per task.

Usage:
    python scripts/gen_compare_grid.py                       # reads runs/lr_finder/suggested.csv
    python scripts/gen_compare_grid.py --demo-on stable_evo soap@1.0 --demo-temps 1e-4 1e-3
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = ROOT / "configs" / "experiment" / "compare_grid.txt"
SUGGESTED = ROOT / "runs" / "lr_finder" / "suggested.csv"


def load_suggested():
    """{(optimizer_tag, eff_batch): suggested_lr} from the lr_finder, or {} if absent."""
    if not SUGGESTED.exists():
        return {}
    out = {}
    with open(SUGGESTED) as f:
        for r in csv.DictReader(f):
            try:
                out[(r["optimizer"], int(r["eff_batch"]))] = float(r["suggested_lr"])
            except (KeyError, ValueError):
                continue
    return out


def _base_parts(tag, lr):
    if tag.startswith("soap@"):
        return f"--optimizer soap --alpha {tag.split('@')[1]} --lr {lr:g} --max-update-norm 2.0"
    if tag == "stable_evo":
        return f"--optimizer stable_evo --lr {lr:g} --max-update-norm 2.0"
    return f"--optimizer {tag} --lr {lr:g}"


def _lr_for(sug, tag, bs, default_lr_fn):
    if (tag, bs) in sug:
        return sug[(tag, bs)], False
    return default_lr_fn(tag), True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vit_s")
    ap.add_argument("--dataset", default="cifar100")
    ap.add_argument("--optimizers", nargs="+",
                    default=["sgd", "adamw", "soap@0.5", "soap@1.0", "stable_evo"])
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[64, 256, 1024, 4096])
    ap.add_argument("--micro-batch", type=int, default=256)
    ap.add_argument("--max-steps", type=int, default=8000, help="fixed step budget (if --epochs<=0)")
    ap.add_argument("--epochs", type=int, default=0,
                    help="TRUE-CONVERGENCE budget: per-batch max_steps = epochs*ceil(dataset/eff_batch), "
                         "eval_every scaled to ~50 evals. Equalizes data seen across batches and lets "
                         "early stopping trim plateaus. Overrides --max-steps when >0.")
    ap.add_argument("--dataset-size", type=int, default=50000, help="train-set size for --epochs budgeting")
    ap.add_argument("--runs-dir", default=str(ROOT / "runs" / "benchmarks"),
                    help="output dir for the per-config CSVs (use a fresh dir to isolate a run)")
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--early-stop-patience", type=int, default=8,
                    help="stop after N evals with no val improvement (bounds large-batch cost)")
    ap.add_argument("--early-stop-min-delta", type=float, default=1e-3)
    # demographic-noise (pSGLD) variants: measure how injected noise affects alpha/accuracy/speed
    ap.add_argument("--demo-on", nargs="*", default=["stable_evo"],
                    help="base optimizers to also run WITH demographic noise (e.g. stable_evo soap@1.0)")
    ap.add_argument("--demo-temps", type=float, nargs="*", default=[1e-3],
                    help="pSGLD temperatures T for the demo variants")
    ap.add_argument("--demo-warmup", type=int, default=200)
    ap.add_argument("--stable-evo-at-soap-lr", action="store_true",
                    help="also run stable_evo at soap@0.5's (much higher) lr, tagged 'soaplr' — "
                         "tests whether the finder's stable_evo lr is just too low.")
    ap.add_argument("--fixed-lr", type=float, default=None,
                    help="override the per-cell lr with ONE value across ALL batches (tagged 'fixedlr'). "
                         "Holds lr — and thus injected noise std sqrt(T*lr) — constant so only gradient "
                         "noise (proportional to lr/batch) varies: disentangles whether the demo-noise "
                         "penalty depends on the gradient-noise level (magnitude) or not (shape/floor).")
    ap.add_argument("--group", default="all", choices=["all", "stable_evo", "baselines"],
                    help="split the run into separate jobs: 'stable_evo' (stable_evo + its demo "
                         "variants) | 'baselines' (sgd/adamw/soap) | 'all'. Writes "
                         "compare_grid_<group>.txt for non-all groups.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-amp", action="store_true")
    # mechanism probes: SWA/iterate-averaging eval (MAP estimate) + gradient-noise-scale tracking
    ap.add_argument("--swa-start-frac", type=float, default=0.0,
                    help="eval an iterate-average over the final (1-frac) of training (0=off). "
                         "NOTE: pair with --early-stop-patience 0 so the budget (hence the SWA "
                         "start step) is actually reached.")
    ap.add_argument("--noise-scale", action="store_true",
                    help="measure the gradient noise scale tr(Σ)/|g|² at each eval cadence")
    ap.add_argument("--noise-scale-k", type=int, default=8,
                    help="independent micro-batch grads per noise-scale estimate")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    # group split (separate SLURM jobs, independent tracking)
    if args.group == "stable_evo":
        args.optimizers = ["stable_evo"]; args.demo_on = ["stable_evo"]
    elif args.group == "baselines":
        args.optimizers = [o for o in args.optimizers if o != "stable_evo"]; args.demo_on = []
    out_path = Path(args.out)
    if args.group != "all" and out_path == OUT:
        out_path = out_path.with_name(f"compare_grid_{args.group}.txt")

    from ml_experiments._harness import default_lr
    from ml_experiments.lr_finder import parse_opt_tag

    def default_lr_fn(tag):
        name, alpha = parse_opt_tag(tag)
        return default_lr(name, alpha if alpha is not None else 0.5, 1e-3)

    sug = load_suggested()
    if not sug:
        print(f"[warn] {SUGGESTED} not found -> falling back to default_lr for every cell. "
              f"Run stage 1 first: python -m ml_experiments.lr_finder")

    def _budget(bs):
        """(max_steps, eval_every) for this batch — epoch-budgeted (true convergence) if --epochs>0."""
        if args.epochs > 0:
            spe = max(1, math.ceil(args.dataset_size / bs))    # optimizer steps per epoch
            ms = args.epochs * spe
            return ms, max(20, ms // 50)                       # ~50 evals/run
        return args.max_steps, args.eval_every

    tail = (f"--log-every {args.log_every} --early-stop-patience {args.early_stop_patience} "
            f"--early-stop-min-delta {args.early_stop_min_delta:g} --seed {args.seed} "
            f"--out-dir {args.runs_dir}" + ("" if args.no_amp else " --amp")
            + (f" --swa-start-frac {args.swa_start_frac:g}" if args.swa_start_frac > 0 else ""))

    lines, fellback = [], 0
    for bs in args.batch_sizes:
        micro = min(bs, args.micro_batch)
        accum = max(1, bs // micro)
        ms, ee = _budget(bs)
        bsargs = f"--batch-size {micro} --accum-steps {accum} --max-steps {ms} --eval-every {ee}"
        if args.noise_scale:    # probe at the per-batch eval cadence so rows align with the val rows
            bsargs += f" --noise-scale-every {ee} --noise-scale-k {args.noise_scale_k}"
        for tag in args.optimizers:
            sfx = ""
            if args.fixed_lr is not None:
                # hold lr (hence injected noise sqrt(T*lr)) constant across batches; only
                # gradient noise (lr/batch) varies — read the WITHIN-cell demo-base delta.
                lr, fb = args.fixed_lr, False; sfx = " --label-suffix fixedlr"
            elif tag == "stable_evo" and args.stable_evo_at_soap_lr:
                # stable_evo's own finder lr is far too low (the generative-lag confound in the LR
                # range test), so run it at soap@0.5's lr instead, tagged 'soaplr'. The demo variant
                # is then applied to THIS (the good lr) — a fair posterior-sampling test.
                lr, fb = _lr_for(sug, "soap@0.5", bs, default_lr_fn); sfx = " --label-suffix soaplr"
            else:
                lr, fb = _lr_for(sug, tag, bs, default_lr_fn)
            fellback += fb
            base = f"--model {args.model} --dataset {args.dataset} {_base_parts(tag, lr)} {bsargs} {tail}{sfx}"
            lines.append(base)
            if tag in args.demo_on:
                for T in args.demo_temps:
                    lines.append(base + f" --demographic-noise --demographic-temperature {T:g} "
                                        f"--demographic-warmup {args.demo_warmup}")

    out = out_path
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(lines)} full-run configs to {out}  (group={args.group})"
          + (f"  ({fellback} cells used default_lr fallback)" if fellback else "  (all lrs from lr_finder)"))
    print(f"  optimizers={args.optimizers}  batches={args.batch_sizes}  "
          f"demo_on={args.demo_on}  demo_temps={args.demo_temps}")
    print(f"SLURM array: TN_CMP_GRID={out} sbatch --array=0-{len(lines) - 1}%2 "
          f"scripts/submit_optimizer_compare.sh")


if __name__ == "__main__":
    main()

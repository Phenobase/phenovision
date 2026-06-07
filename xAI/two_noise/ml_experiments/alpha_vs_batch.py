"""§2.2 (real-model half) — the noise-dependent optimal exponent α*(batch size).

Hypothesis (analysis_plan.md §2.2): α* is monotone increasing in effective batch size (decreasing
in gradient noise), because the diffusion term C^{1-2α}/S over-amplifies low-curvature directions
exactly when α→1 and S is small. This is the ML image of the biological N* prediction and produces
the LEFT half of the §5 shared-prediction figure.

What this script does:
    For a (model, dataset), sweep α × batch_size. For each (α, batch) it trains at a FIXED COMPUTE
    BUDGET (same number of micro-batches processed, so wall/flops are comparable across batch
    sizes) and records best val loss. For each batch size, α* = the α minimizing val loss. Writes
    one tidy CSV runs/alpha_vs_batch/<model>_<dataset>.csv with every (α, batch) cell plus an
    `is_alpha_star` flag, ready for R to plot α* vs batch.

Effective-batch via gradient accumulation (the §2.2 "hold optimizer state comparable" requirement):
    We fix a MICRO-batch that fits in memory (--micro-batch) and realize each target batch size B as
    accum_steps = B / micro_batch. The optimizer sees one update per B examples regardless of B, so
    its momentum/preconditioner timescales (in *updates*) are identical; only the gradient noise
    1/S changes with B — which is exactly the knob the hypothesis is about. Compute budget is held
    fixed in MICRO-batches (--budget-microbatches), i.e. same number of examples and forward/backward
    passes for every cell, so larger B simply means fewer optimizer steps.

Smoke example (CPU):
    python -m ml_experiments.alpha_vs_batch --model tiny_vision --dataset synthetic_vision \
        --alphas 0.5 1.0 --batch-sizes 16 32 --micro-batch 16 \
        --budget-microbatches 8 --device cpu
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml_experiments._harness import (RUNS_DIR, default_lr, make_data, make_model,
                                     make_optimizer, train_eval)

CSV_COLUMNS = [
    "model", "dataset", "optimizer", "alpha", "lr", "batch_size", "micro_batch",
    "accum_steps", "budget_microbatches", "opt_steps", "val_loss", "val_metric",
    "val_metric_name", "best_train_loss", "peak_mem_mb", "mean_step_time_ms",
    "is_alpha_star", "seed",
]


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_COLUMNS})


def run_cell(model_name, dataset, alpha, batch_size, micro_batch, budget_microbatches,
             args, device):
    """Train one (alpha, batch_size) cell at the fixed compute budget; return a result row."""
    assert batch_size % micro_batch == 0, (
        f"batch_size {batch_size} must be a multiple of micro_batch {micro_batch}")
    accum_steps = batch_size // micro_batch
    # Fixed compute budget (micro-batches) => opt_steps shrinks as batch grows.
    opt_steps = max(1, budget_microbatches // accum_steps)

    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    train_loader, val_loader, meta = make_data(
        dataset, batch_size=micro_batch, num_workers=args.num_workers,
        block_size=args.block_size, generator=gen, synthetic_n=args.synthetic_n,
    )
    if meta.task == "lm":
        model = make_model(model_name, vocab_size=meta.vocab_size)
    else:
        model = make_model(model_name, num_classes=meta.num_classes)

    lr = args.lr if args.lr is not None else default_lr("soap", alpha, args.base_lr)
    # Full-inverse (alpha->1) over-steps at small batch (high gradient noise) and can diverge;
    # an update-norm trust region keeps it finite so the cell yields a real datapoint. Harmless
    # at small alpha (updates are already small).
    optimizer, lr_actual = make_optimizer(
        "soap", model.parameters(), alpha=alpha, lr=lr,
        weight_decay=args.weight_decay, base_lr=args.base_lr,
        max_update_norm=2.0,
    )

    try:
        result = train_eval(
            model, optimizer, train_loader, val_loader, device,
            max_steps=opt_steps, accum_steps=accum_steps,
            log_every=max(1, opt_steps // 4), eval_every=0,
            eval_max_batches=args.eval_max_batches, grad_clip=args.grad_clip,
            amp=args.amp, lr=lr_actual,
        )
    except FloatingPointError as e:
        # Divergence (e.g. alpha=1 at tiny batch): record this cell as worst, keep the sweep alive
        # so alpha* is still found from the finite cells (run() already filters non-finite).
        print(f"[alpha_vs_batch] DIVERGED B={batch_size} alpha={alpha:g}: {e}")
        return dict(
            model=model_name, dataset=dataset, optimizer="soap", alpha=alpha, lr=lr_actual,
            batch_size=batch_size, micro_batch=micro_batch, accum_steps=accum_steps,
            budget_microbatches=budget_microbatches, opt_steps=opt_steps,
            val_loss=float("inf"), val_metric=float("nan"), val_metric_name="diverged",
            best_train_loss=float("inf"), peak_mem_mb=float("nan"),
            mean_step_time_ms=float("nan"), is_alpha_star=False, seed=args.seed,
        )
    best_train = min((r["train_loss"] for r in result.records), default=float("nan"))
    return dict(
        model=model_name, dataset=dataset, optimizer="soap", alpha=alpha, lr=lr_actual,
        batch_size=batch_size, micro_batch=micro_batch, accum_steps=accum_steps,
        budget_microbatches=budget_microbatches, opt_steps=opt_steps,
        val_loss=result.final_val_loss, val_metric=result.final_val_metric,
        val_metric_name=result.val_metric_name, best_train_loss=best_train,
        peak_mem_mb=result.peak_mem_mb, mean_step_time_ms=result.mean_step_time_ms,
        is_alpha_star=False, seed=args.seed,
    )


def run(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    rows = []
    for batch_size in args.batch_sizes:
        cells = []
        for alpha in args.alphas:
            row = run_cell(args.model, args.dataset, alpha, batch_size, args.micro_batch,
                           args.budget_microbatches, args, device)
            print(f"[alpha_vs_batch] B={batch_size} alpha={alpha:g} "
                  f"val_loss={row['val_loss']:.4f} opt_steps={row['opt_steps']}")
            cells.append(row)
        # alpha* = min val_loss at this batch size.
        finite = [c for c in cells if c["val_loss"] == c["val_loss"]]  # drop NaN
        if finite:
            star = min(finite, key=lambda c: c["val_loss"])
            star["is_alpha_star"] = True
            print(f"[alpha_vs_batch] B={batch_size} alpha* = {star['alpha']:g}")
        rows.extend(cells)

    out_dir = Path(args.out_dir) if args.out_dir else (RUNS_DIR / "alpha_vs_batch")
    csv_path = out_dir / f"{args.model}_{args.dataset}.csv"
    write_csv(csv_path, rows)
    print(f"[alpha_vs_batch] wrote {csv_path} ({len(rows)} cells)")
    return csv_path


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="vit_s|vit_b|nanogpt|nanogpt_m|tiny_vision")
    p.add_argument("--dataset", required=True,
                   help="cifar100|tiny_imagenet|tinystories|synthetic_vision|synthetic_lm")
    p.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[16, 64, 256, 1024, 4096])
    p.add_argument("--micro-batch", type=int, default=16,
                   help="micro-batch that fits in memory; each batch size = accum * micro_batch")
    p.add_argument("--budget-microbatches", type=int, default=4000,
                   help="fixed compute budget in micro-batches (same examples per cell)")
    p.add_argument("--lr", type=float, default=None, help="explicit lr; else default_lr(soap, alpha)")
    p.add_argument("--base-lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-max-batches", type=int, default=0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--synthetic-n", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="")
    p.add_argument("--out-dir", default="")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    main()

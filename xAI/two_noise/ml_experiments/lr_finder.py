"""LR range test (Smith 2015 / fastai lr_find) — the standard quick learning-rate estimator.

Stage 1 of the optimizer comparison: for each (optimizer, batch) cell, exponentially ramp the lr
over a few hundred steps and record the (smoothed) training loss; the suggested lr is the point of
STEEPEST loss descent (the fastai default), capped at the loss-minimizing lr. This is short (no run
to convergence) and gives a model-specific lr to use for the full to-convergence runs.

Notes for the preconditioned optimizers:
  * the update-norm trust region is DISABLED during the test (max_update_norm=0) so the high-lr
    divergence is visible — otherwise the clip flattens it and the test can't find the edge;
  * the curvature/eigenbasis warms up over the first steps, which the leading-edge trim skips.

Writes one per-cell CSV (lr, loss, smoothed) to runs/lr_finder/ and a tidy runs/lr_finder/suggested.csv
(optimizer, eff_batch, suggested_lr, steepest_lr, min_loss_lr). gen_compare_grid.py reads
suggested.csv to build the full-run grid.

Smoke (CPU):
    python -m ml_experiments.lr_finder --model tiny_vision --dataset synthetic_vision \
        --optimizers adamw stable_evo --batch-sizes 16 --n-iter 40 --device cpu
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml_experiments._harness import (RUNS_DIR, _forward_loss, default_lr, is_lm_model,
                                     make_data, make_model, make_optimizer)


def parse_opt_tag(tag: str):
    """'soap@0.5' -> ('soap', 0.5); 'stable_evo' -> ('stable_evo', None); 'sgd'/'adamw' -> (tag, None)."""
    if tag.startswith("soap@"):
        return "soap", float(tag.split("@")[1])
    return tag, None


def _opt_kwargs(name):
    """Disable the trust region during the test so high-lr divergence shows."""
    if name in ("soap", "stable_evo"):
        return dict(max_update_norm=0.0)
    return {}


def lr_range_test(model, optimizer, train_loader, device, *, lr_min, lr_max, n_iter,
                  accum_steps, amp, grad_clip=1.0, beta=0.98, diverge=4.0):
    """Exponentially ramp lr from lr_min to lr_max over n_iter steps; return per-step records."""
    is_lm = is_lm_model(model)
    use_cuda = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=(amp and use_cuda))
    model.to(device); model.train()

    mult = (lr_max / lr_min) ** (1.0 / max(n_iter - 1, 1))
    lr = lr_min
    for pg in optimizer.param_groups:
        pg["lr"] = lr

    def loader():
        while True:
            for b in train_loader:
                yield b
    it = loader()

    records, avg, best = [], 0.0, float("inf")
    optimizer.zero_grad(set_to_none=True)
    for i in range(1, n_iter + 1):
        micro = 0.0
        for _ in range(accum_steps):
            batch = next(it)
            with torch.amp.autocast("cuda", enabled=(amp and use_cuda)):
                loss, _, _ = _forward_loss(model, batch, device, is_lm)
                loss = loss / accum_steps
            scaler.scale(loss).backward()
            micro += float(loss.item())
        if grad_clip and grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        avg = beta * avg + (1.0 - beta) * micro
        smoothed = avg / (1.0 - beta ** i)
        records.append(dict(step=i, lr=lr, loss=micro, smoothed=smoothed))
        if not np.isfinite(micro) or (i > 8 and smoothed > diverge * best):
            break                                   # diverged -> stop the ramp
        best = min(best, smoothed)
        lr *= mult
        for pg in optimizer.param_groups:
            pg["lr"] = lr
    return records


def _choose_lr(steepest_lr, min_loss_lr, lr_min, fallback=None):
    """Robust hybrid pick. The steepest-descent point is trustworthy ONLY when it is clear of the
    ramp floor (it otherwise latches onto the early warmup transient) and at/below the loss
    minimum; otherwise use the classic conservative `min_loss/10` (one order below the minimum)."""
    if not (np.isfinite(min_loss_lr) and min_loss_lr > 0):
        return float(fallback) if fallback else float("nan")
    conservative = min_loss_lr / 10.0
    if (np.isfinite(steepest_lr) and steepest_lr > 3.0 * lr_min
            and steepest_lr <= min_loss_lr):
        return float(steepest_lr)
    return float(conservative)


def suggest_lr(records, fallback, lr_min=1e-6):
    """fastai-style LR suggestion via the robust hybrid (_choose_lr). Returns
    (suggested, steepest_lr, min_loss_lr)."""
    if len(records) < 12:
        return fallback, float("nan"), float("nan")
    lrs = np.array([r["lr"] for r in records])
    s = np.array([r["smoothed"] for r in records])
    lo, hi = 5, len(s) - 2                          # skip warmup + the very tail
    lrs, s = lrs[lo:hi], s[lo:hi]
    if len(s) < 6:
        return fallback, float("nan"), float("nan")
    grad = np.gradient(s, np.log(lrs))
    steepest_lr = float(lrs[int(np.argmin(grad))])  # most-negative slope
    min_loss_lr = float(lrs[int(np.argmin(s))])
    if np.min(grad) >= 0:                           # never descended -> fall back
        return fallback, steepest_lr, min_loss_lr
    return _choose_lr(steepest_lr, min_loss_lr, lr_min, fallback), steepest_lr, min_loss_lr


def recompute_suggested(csv_path, lr_min=1e-6):
    """Rewrite the suggested_lr column of a lr_finder suggested.csv from its recorded
    steepest_lr/min_loss_lr columns using the robust hybrid — repairs a CSV written by an older
    heuristic with NO GPU re-run."""
    import csv as _csv
    rows = list(_csv.DictReader(open(csv_path)))
    if not rows:
        return
    for r in rows:
        st = float(r.get("steepest_lr", "nan") or "nan")
        ml = float(r.get("min_loss_lr", "nan") or "nan")
        r["suggested_lr"] = f"{_choose_lr(st, ml, lr_min):g}"
    with open(csv_path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[lr_finder] recomputed suggested_lr (hybrid) in {csv_path}")


def run_cell(model_name, dataset, tag, batch_size, micro_batch, args, device):
    name, alpha = parse_opt_tag(tag)
    accum = max(1, batch_size // min(batch_size, micro_batch))
    micro = min(batch_size, micro_batch)
    gen = torch.Generator().manual_seed(args.seed)
    train_loader, _, meta = make_data(dataset, batch_size=micro, num_workers=args.num_workers,
                                      block_size=args.block_size, generator=gen,
                                      synthetic_n=args.synthetic_n)
    model = (make_model(model_name, vocab_size=meta.vocab_size) if meta.task == "lm"
             else make_model(model_name, num_classes=meta.num_classes))
    optimizer, _ = make_optimizer(name, model.parameters(), alpha=(alpha if alpha is not None else 0.5),
                                  lr=args.lr_min, weight_decay=args.weight_decay,
                                  **_opt_kwargs(name))
    recs = lr_range_test(model, optimizer, train_loader, device, lr_min=args.lr_min,
                         lr_max=args.lr_max, n_iter=args.n_iter, accum_steps=accum,
                         amp=args.amp, grad_clip=args.grad_clip)
    fallback = default_lr(name, alpha if alpha is not None else 0.5, args.base_lr)
    suggested, steepest, min_loss = suggest_lr(recs, fallback, lr_min=args.lr_min)

    out_dir = Path(args.out_dir) if args.out_dir else (RUNS_DIR / "lr_finder")
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = tag.replace("@", "")
    with open(out_dir / f"{model_name}__{dataset}__{safe}__bs{batch_size}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "lr", "loss", "smoothed"]); w.writeheader(); w.writerows(recs)
    print(f"[lr_finder] {tag:11s} bs={batch_size:<5} suggested={suggested:.2e} "
          f"(steepest={steepest:.2e} min_loss={min_loss:.2e})")
    return dict(model=model_name, dataset=dataset, optimizer=tag, eff_batch=batch_size,
                micro_batch=micro, accum_steps=accum, suggested_lr=suggested,
                steepest_lr=steepest, min_loss_lr=min_loss)


def run(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    rows = []
    for bs in args.batch_sizes:
        for tag in args.optimizers:
            rows.append(run_cell(args.model, args.dataset, tag, bs, args.micro_batch, args, device))
    out_dir = Path(args.out_dir) if args.out_dir else (RUNS_DIR / "lr_finder")
    sug = out_dir / "suggested.csv"
    write_header = not sug.exists()
    with open(sug, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            w.writeheader()
        w.writerows(rows)
    print(f"[lr_finder] appended {len(rows)} suggestions -> {sug}")
    return sug


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="vit_s")
    p.add_argument("--dataset", default="cifar100")
    p.add_argument("--optimizers", nargs="+",
                   default=["sgd", "adamw", "soap@0.5", "soap@1.0", "stable_evo"])
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[64, 256, 1024, 4096])
    p.add_argument("--micro-batch", type=int, default=256)
    p.add_argument("--n-iter", type=int, default=200, help="LR-ramp steps")
    p.add_argument("--lr-min", type=float, default=1e-6)
    p.add_argument("--lr-max", type=float, default=1.0)
    p.add_argument("--base-lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--synthetic-n", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="")
    p.add_argument("--out-dir", default="")
    return p


def main(argv=None):
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()

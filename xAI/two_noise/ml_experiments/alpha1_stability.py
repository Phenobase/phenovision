"""Does TRUE-FISHER curvature + damping stabilize the full-inverse step (α=1) at SMALL batch?

This is the "import biology's correctly-specified, low-noise curvature to ML, at scale" test
(notes_evolution_vs_ml.md §1, §4; notes_posterior_sampling.md §2.4). The α-vs-batch sweep showed
SOAPFullPower at precond_power=1.0 (full inverse / natural gradient) runs fine at batch 256 but
**NaN's at batch 16**: the empirical minibatch Fisher mis-estimates flat-direction curvature, so
1/denom blows up exactly where the estimate is noisiest (notes §2). §2.4 showed that feeding the
optimizer the TRUE (sampled-label) Fisher *eigenvalues* (`precond_eigvals_from_hook`) plus relative
LM damping fixed the analogous over-dispersion in the SAMPLER. Here we test whether the SAME import
fixes α=1 TRAINING STABILITY on a real ViT at small batch.

Three conditions, ViT-S / CIFAR-100, batch=16, AMP, precond_power=1.0:
  (A) empirical Fisher, low damping (damping=1e-4), no trust region   -> EXPECTED to diverge (NaN).
  (B) TRUE-Fisher eigenvalues (precond_eigvals_from_hook=True; sampled-label grad assigned to
      p._soap_precond_grad each step BEFORE opt.step()) + relative damping ~1e-2 -> EXPECTED stable.
  (C) empirical Fisher + relative damping ~1e-2 (no true-Fisher)  -> control: damping alone.

(B) vs (C) separates the true-Fisher effect from damping alone; (A) vs (C) isolates damping; (A) is
the documented failure. (B) costs an extra forward+backward per step (the sampled-label gradient).

This module is SELF-CONTAINED: it IMPORTS make_model/make_data/make_optimizer/evaluate from
ml_experiments._harness and sampled_label_gradient/assign_precond_grad from curvature.true_fisher,
but does NOT modify them (a benchmark SLURM array imports _harness concurrently). It runs its OWN
training loop (not _harness.train_eval) so it can CATCH non-finite losses, keep going, and record
the fraction of finite steps instead of crashing on the first NaN.

CPU smoke:
    python -m ml_experiments.alpha1_stability --device cpu --model tiny_vision \
        --dataset synthetic_vision --synthetic-n 64 --batch 8 --max-steps 6 \
        --conditions B --no-amp

GPU (real test, all three conditions sequentially):
    python -m ml_experiments.alpha1_stability --device cuda --batch 16 --max-steps 2000 --amp
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml_experiments._harness import (RUNS_DIR, evaluate, is_lm_model, make_data,
                                     make_model, make_optimizer)
from curvature.true_fisher import assign_precond_grad, sampled_label_gradient

CSV_COLUMNS = [
    "condition", "label", "model", "dataset", "alpha", "lr", "batch", "max_steps",
    "use_true_fisher", "damping", "relative_damping", "max_update_norm", "grad_clip", "amp",
    "diverged", "first_nan_step", "finite_fraction", "n_steps_run",
    "final_train_loss", "best_train_loss", "final_val_loss", "final_val_acc",
    "loss_ckpts", "mean_step_time_ms", "peak_mem_mb", "seed",
]

# The three conditions. `damping`/`relative_damping`/`use_true_fisher`/`max_update_norm` are the
# only things that differ; everything else (lr, betas, precond_freq) is held fixed across them so
# the comparison is clean. precond_power is forced to 1.0 (the full inverse) for all three.
CONDITIONS = {
    "A": dict(
        label="empirical_lowdamp",
        use_true_fisher=False,
        damping=1e-4,
        relative_damping=True,
        max_update_norm=0.0,   # no trust region: this is the documented failure case
    ),
    "B": dict(
        label="truefisher_eig_damp",
        use_true_fisher=True,
        damping=1e-2,
        relative_damping=True,
        max_update_norm=0.0,   # the fix must come from the curvature, not a trust region
    ),
    "C": dict(
        label="empirical_damp",
        use_true_fisher=False,
        damping=1e-2,
        relative_damping=True,
        max_update_norm=0.0,
    ),
}


def _forward_loss(model, batch, device, is_lm):
    """CE (vision) / LM loss for one batch. Returns (loss, inputs) — inputs reused for true-Fisher."""
    if is_lm:
        x, y = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        _, loss = model(x, y)
        return loss, x
    images, targets = batch
    images = images.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    logits = model(images)
    loss = F.cross_entropy(logits, targets)
    return loss, images


def _infinite(loader):
    while True:
        for b in loader:
            yield b


def run_condition(cond_key, args, device):
    """Train ViT at precond_power=1.0 under one condition; never crash on NaN — record it instead."""
    cfg = CONDITIONS[cond_key]
    use_tf = cfg["use_true_fisher"]

    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)              # data shuffle RNG
    fisher_gen = torch.Generator(device="cpu").manual_seed(args.seed + 1)  # label-sampling RNG

    train_loader, val_loader, meta = make_data(
        args.dataset, batch_size=args.batch, num_workers=args.num_workers,
        generator=gen, synthetic_n=args.synthetic_n,
    )
    is_lm = meta.task == "lm"
    if is_lm:
        model = make_model(args.model, vocab_size=meta.vocab_size)
    else:
        model = make_model(args.model, num_classes=meta.num_classes)
    model.to(device)
    is_lm = is_lm_model(model)

    # All conditions: SOAP full inverse (precond_power=1.0). The condition sets damping, the
    # true-Fisher eigenvalue hook flag, and the (absent) trust region.
    optimizer, lr = make_optimizer(
        "soap", model.parameters(), alpha=1.0, lr=args.lr,
        weight_decay=args.weight_decay, base_lr=args.base_lr,
        damping=cfg["damping"], relative_damping=cfg["relative_damping"],
        max_update_norm=cfg["max_update_norm"],
        precond_eigvals_from_hook=use_tf,
    )

    use_cuda = device.type == "cuda"
    use_amp = args.amp and use_cuda
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    # true-Fisher label sampling needs a generator on the model's device.
    if use_tf and use_cuda:
        fisher_gen = torch.Generator(device=device).manual_seed(args.seed + 1)

    batch_iter = _infinite(train_loader)
    n_finite = 0
    diverged = False
    first_nan_step = -1
    last_finite_loss = float("nan")
    best_train = float("inf")
    loss_ckpts = []        # (step, loss) snapshots
    ckpt_every = max(1, args.max_steps // 8)
    step_times = []

    model.train()
    t0 = time.time()
    last_t = t0
    for step in range(1, args.max_steps + 1):
        batch = next(batch_iter)

        # --- TRUE-FISHER eigenvalue hook: sampled-label gradient assigned BEFORE opt.step() ---
        # Extra forward+backward (full precision; it only feeds the preconditioner statistics, not
        # the descent direction). Must run before backward of the CE loss so p._soap_precond_grad is
        # in place when SOAPFullPower reads it in step().
        if use_tf and not is_lm:
            images, _ = batch
            images = images.to(device, non_blocking=True)
            grad_dict = sampled_label_gradient(model, images, generator=fisher_gen,
                                               loss_kind="classification")
            assign_precond_grad(model, grad_dict)
            model.zero_grad(set_to_none=True)   # clear the autograd.grad side effects on .grad

        # --- descent gradient: ordinary CE loss backward (empirical) ---
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            loss, _ = _forward_loss(model, batch, device, is_lm)
        scaler.scale(loss).backward()

        if args.grad_clip and args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if use_tf:
            assign_precond_grad(model, None)    # clear hook so it can't leak into the next step

        if use_cuda:
            torch.cuda.synchronize()
        now = time.time()
        step_times.append((now - last_t) * 1000.0)
        last_t = now

        lv = float(loss.item())
        finite = math.isfinite(lv)
        if finite:
            n_finite += 1
            last_finite_loss = lv
            best_train = min(best_train, lv)
        elif not diverged:
            diverged = True
            first_nan_step = step

        if step == 1 or step % ckpt_every == 0 or step == args.max_steps:
            loss_ckpts.append((step, lv))
            print(f"[alpha1][{cond_key}:{cfg['label']}] step {step}/{args.max_steps} "
                  f"loss={lv:.4f} finite={finite} diverged={diverged}")

        # Once parameters are NaN, every subsequent loss is NaN — stop early to save compute, but
        # still report it as diverged with the finite fraction up to the blow-up.
        if diverged and not finite and (step - first_nan_step) >= max(20, ckpt_every):
            print(f"[alpha1][{cond_key}] persistent NaN since step {first_nan_step}; stopping early")
            break

    n_steps = step
    # Final val accuracy: only meaningful if the model is finite. evaluate() handles vision/LM.
    params_finite = all(torch.isfinite(p).all().item() for p in model.parameters())
    if params_finite:
        val_metric, val_loss, _ = evaluate(model, val_loader, device, is_lm,
                                           max_batches=args.eval_max_batches)
    else:
        val_metric, val_loss = float("nan"), float("nan")

    peak_mb = (torch.cuda.max_memory_allocated(device) / 1e6) if use_cuda else 0.0
    mean_step_ms = float(sum(step_times) / len(step_times)) if step_times else 0.0

    row = dict(
        condition=cond_key, label=cfg["label"], model=args.model, dataset=args.dataset,
        alpha=1.0, lr=lr, batch=args.batch, max_steps=args.max_steps,
        use_true_fisher=use_tf, damping=cfg["damping"],
        relative_damping=cfg["relative_damping"], max_update_norm=cfg["max_update_norm"],
        grad_clip=args.grad_clip, amp=use_amp,
        diverged=diverged, first_nan_step=first_nan_step,
        finite_fraction=n_finite / max(n_steps, 1), n_steps_run=n_steps,
        final_train_loss=last_finite_loss, best_train_loss=best_train,
        final_val_loss=val_loss, final_val_acc=(val_metric if not is_lm else float("nan")),
        loss_ckpts=";".join(f"{s}:{l:.4f}" for s, l in loss_ckpts),
        mean_step_time_ms=mean_step_ms, peak_mem_mb=peak_mb, seed=args.seed,
    )
    print(f"[alpha1][{cond_key}:{cfg['label']}] DONE diverged={diverged} "
          f"finite_frac={row['finite_fraction']:.3f} final_train={last_finite_loss:.4f} "
          f"val_acc={row['final_val_acc']} step_ms={mean_step_ms:.1f}")
    return row


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_COLUMNS})


def run(args):
    device = torch.device(args.device if args.device
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[alpha1] device={device} conditions={args.conditions} batch={args.batch} "
          f"max_steps={args.max_steps} amp={args.amp}")
    rows = []
    for cond in args.conditions:
        cond = cond.upper()
        if cond not in CONDITIONS:
            raise ValueError(f"unknown condition {cond!r}; choose from {list(CONDITIONS)}")
        rows.append(run_condition(cond, args, device))

    out_dir = Path(args.out_dir) if args.out_dir else (RUNS_DIR / "alpha1_stability")
    csv_path = out_dir / "results.csv"
    write_csv(csv_path, rows)
    print(f"[alpha1] wrote {csv_path} ({len(rows)} conditions)")
    # Human-readable verdict.
    by = {r["condition"]: r for r in rows}
    if "A" in by and ("B" in by):
        a, b = by["A"], by["B"]
        print(f"[alpha1] VERDICT: A(empirical,lowdamp) diverged={a['diverged']} "
              f"(finite_frac={a['finite_fraction']:.2f}); "
              f"B(true-Fisher+damp) diverged={b['diverged']} "
              f"(finite_frac={b['finite_fraction']:.2f}, val_acc={b['final_val_acc']})")
    return csv_path


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="vit_s", help="vit_s|vit_b|tiny_vision")
    p.add_argument("--dataset", default="cifar100",
                   help="cifar100|synthetic_vision (smoke)")
    p.add_argument("--conditions", nargs="+", default=["A", "B", "C"],
                   help="any of A (empirical lowdamp), B (true-Fisher+damp), C (empirical+damp)")
    p.add_argument("--batch", type=int, default=16, help="SMALL batch (the regime α=1 fails in)")
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=None,
                   help="explicit lr; default_lr(soap, 1.0) ~ base_lr/10 if unset")
    p.add_argument("--base-lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-max-batches", type=int, default=50)
    amp = p.add_mutually_exclusive_group()
    amp.add_argument("--amp", dest="amp", action="store_true", default=True)
    amp.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--synthetic-n", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="")
    p.add_argument("--out-dir", default="")
    return p


def main(argv=None):
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()

"""O2 — does the cheap rho-SCHEDULE (1x cost) recover most of TRUE-FISHER's (2x cost) gain?

The practical payoff of the Riccati arm. Three conditions train the same model/data with the
matrix-free RiccatiPrecond, logging loss-vs-step AND loss-vs-wallclock:

  whiten          : precond='whiten' (alpha=1/2), shrink=0      -- 1x cost, the stable baseline.
  inverse_fisher  : precond='inverse' (alpha=1) + TRUE-FISHER curvature (sampled-label gradient
                    fed to the preconditioner via precond_stats_from_hook) -- 2x cost (extra
                    fwd/bwd), the expensive "clean curvature makes the full inverse safe" fix.
  schedule        : precond='inverse' + shrink = shrink_from_batch(B) (the noise-dependent rho
                    schedule) -- 1x cost, the cheap effective-exponent reduction.

Headline number (computed in summarize / figures): the FRACTION OF TRUE-FISHER GAIN RECOVERED
at 1x cost = (L_whiten - L_schedule) / (L_whiten - L_inverse_fisher), at matched wallclock and
at matched steps. If ~1, the noise-dependent exponent is the cheap substitute for clean Fisher.

Uses bf16 autocast (NO GradScaler) so the loss closure passes to step() and fp16 overflow can't
silently apply bad gradients (the CUDA smoke caught that). Runs its OWN NaN-surviving loop (like
alpha1_stability) so a diverging condition records instead of crashing.

CPU smoke:
    python -m ml_experiments.riccati_schedule_vs_fisher --device cpu --model tiny_vision \
        --dataset synthetic_vision --synthetic-n 64 --batch 16 --max-steps 6 --no-amp
GPU:
    python -m ml_experiments.riccati_schedule_vs_fisher --device cuda --batch 64 --max-steps 4000 --amp
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
from optim.riccati_precond import RiccatiPrecond
from curvature.true_fisher import assign_precond_grad, sampled_label_gradient

CSV_COLUMNS = [
    "condition", "label", "model", "dataset", "precond_mode", "alpha", "lr", "shrink",
    "use_true_fisher", "batch", "max_steps", "amp", "diverged", "finite_fraction",
    "n_steps_run", "step", "train_loss", "val_loss", "val_metric", "wallclock_s",
    "step_time_ms", "peak_mem_mb", "seed",
]


def _forward_loss(model, batch, device, is_lm):
    if is_lm:
        x, y = batch
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        _, loss = model(x, y)
        return loss, x
    images, targets = batch
    images = images.to(device, non_blocking=True); targets = targets.to(device, non_blocking=True)
    loss = F.cross_entropy(model(images), targets)
    return loss, images


def _infinite(loader):
    while True:
        for b in loader:
            yield b


def _make_condition(cond, args):
    """Return (kwargs for make_optimizer, use_true_fisher, label)."""
    if cond == "whiten":
        return dict(precond_mode="whiten", shrink=0.0), False, "whiten_a05"
    if cond == "inverse_fisher":
        return (dict(precond_mode="inverse", shrink=0.0, damping=1e-2,
                     precond_stats_from_hook=True),
                True, "inverse_truefisher")
    if cond == "schedule":
        rho = RiccatiPrecond.shrink_from_batch(args.batch, ref_batch=args.ref_batch,
                                               rho_max=args.rho_max)
        return (dict(precond_mode="inverse", shrink=rho, damping=1e-2, inner_steps=args.inner_steps),
                False, f"inverse_shrink{rho:.2f}")
    raise ValueError(f"unknown condition {cond!r}")


def run_condition(cond, args, device):
    opt_kw, use_tf, label = _make_condition(cond, args)
    alpha = 0.5 if opt_kw["precond_mode"] == "whiten" else 1.0

    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    train_loader, val_loader, meta = make_data(
        args.dataset, batch_size=args.batch, num_workers=args.num_workers,
        generator=gen, synthetic_n=args.synthetic_n,
    )
    is_lm = meta.task == "lm"
    model = (make_model(args.model, vocab_size=meta.vocab_size) if is_lm
             else make_model(args.model, num_classes=meta.num_classes)).to(device)
    is_lm = is_lm_model(model)

    optimizer, lr = make_optimizer("riccati", model.parameters(), alpha=alpha, lr=args.lr,
                                   weight_decay=args.weight_decay, base_lr=args.base_lr, **opt_kw)

    use_cuda = device.type == "cuda"
    use_amp = args.amp and use_cuda
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)
    fisher_gen = torch.Generator(device=device if use_cuda else "cpu").manual_seed(args.seed + 1)

    batch_iter = _infinite(train_loader)
    records, step_times = [], []
    n_finite, diverged = 0, False
    last_finite = float("nan")
    ckpt_every = max(1, args.max_steps // 40)
    model.train()
    t0 = time.time(); last_t = t0
    for step in range(1, args.max_steps + 1):
        batch = next(batch_iter)
        if use_tf and not is_lm:
            images, _ = batch
            images = images.to(device, non_blocking=True)
            grad_dict = sampled_label_gradient(model, images, generator=fisher_gen,
                                               loss_kind="classification")
            assign_precond_grad(model, grad_dict)
            model.zero_grad(set_to_none=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            loss, _ = _forward_loss(model, batch, device, is_lm)
        loss.backward()                         # fp32 grads under bf16 autocast
        if args.grad_clip and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step(closure=lambda: loss.detach())
        if use_tf:
            assign_precond_grad(model, None)

        if use_cuda:
            torch.cuda.synchronize()
        now = time.time(); step_times.append((now - last_t) * 1000.0); last_t = now

        lv = float(loss.item())
        finite = math.isfinite(lv)
        if finite:
            n_finite += 1; last_finite = lv
        elif not diverged:
            diverged = True

        if step == 1 or step % ckpt_every == 0 or step == args.max_steps:
            val_metric = val_loss = float("nan")
            params_finite = all(torch.isfinite(p).all().item() for p in model.parameters())
            if params_finite:
                val_metric, val_loss, _ = evaluate(model, val_loader, device, is_lm,
                                                   max_batches=args.eval_max_batches)
                model.train()
            records.append(dict(
                condition=cond, label=label, model=args.model, dataset=args.dataset,
                precond_mode=opt_kw["precond_mode"], alpha=alpha, lr=lr,
                shrink=opt_kw.get("shrink", 0.0), use_true_fisher=use_tf,
                batch=args.batch, max_steps=args.max_steps, amp=use_amp,
                diverged=diverged, finite_fraction=n_finite / step, n_steps_run=step,
                step=step, train_loss=lv, val_loss=val_loss, val_metric=val_metric,
                wallclock_s=now - t0, step_time_ms=step_times[-1],
                peak_mem_mb=(torch.cuda.max_memory_allocated(device) / 1e6) if use_cuda else 0.0,
                seed=args.seed,
            ))
            print(f"[o2][{cond}:{label}] step {step}/{args.max_steps} loss={lv:.4f} "
                  f"val={val_loss:.4f} finite={finite} wall={now-t0:.1f}s")
        if diverged and not finite and step > args.max_steps // 5:
            print(f"[o2][{cond}] persistent NaN; stopping early at step {step}")
            break
    print(f"[o2][{cond}:{label}] DONE diverged={diverged} final_train={last_finite:.4f} "
          f"final_val={records[-1]['val_loss']:.4f}")
    return records


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_COLUMNS})


def summarize(by_cond):
    """Fraction of true-Fisher gain recovered at 1x cost, at the final logged step (matched steps)."""
    def final_val(c):
        recs = by_cond.get(c, [])
        return recs[-1]["val_loss"] if recs else float("nan")
    Lw, Lf, Ls = final_val("whiten"), final_val("inverse_fisher"), final_val("schedule")
    frac = float("nan")
    if all(math.isfinite(x) for x in (Lw, Lf, Ls)) and abs(Lw - Lf) > 1e-9:
        frac = (Lw - Ls) / (Lw - Lf)
    print(f"[o2] final val_loss  whiten={Lw:.4f}  inverse_fisher={Lf:.4f}  schedule={Ls:.4f}")
    print(f"[o2] fraction of true-Fisher gain recovered by the 1x schedule = {frac:.3f}")
    return frac


def run(args):
    device = torch.device(args.device if args.device
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[o2] device={device} conditions={args.conditions} batch={args.batch} "
          f"max_steps={args.max_steps} amp={args.amp}")
    all_rows, by_cond = [], {}
    for cond in args.conditions:
        recs = run_condition(cond, args, device)
        by_cond[cond] = recs
        all_rows.extend(recs)
    out_dir = Path(args.out_dir) if args.out_dir else (RUNS_DIR / "riccati_schedule_vs_fisher")
    tag = f"{args.model}_{args.dataset}_b{args.batch}"
    csv_path = out_dir / f"{tag}.csv"
    write_csv(csv_path, all_rows)
    print(f"[o2] wrote {csv_path} ({len(all_rows)} rows)")
    summarize(by_cond)
    return csv_path


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="vit_s")
    p.add_argument("--dataset", default="cifar100")
    p.add_argument("--conditions", nargs="+",
                   default=["whiten", "inverse_fisher", "schedule"])
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--ref-batch", type=int, default=512, help="shrink_from_batch reference batch")
    p.add_argument("--rho-max", type=float, default=0.6, help="shrink_from_batch ceiling")
    p.add_argument("--inner-steps", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=4000)
    p.add_argument("--lr", type=float, default=None)
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

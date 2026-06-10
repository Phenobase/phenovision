"""O4 — the evolving-M meta-loop (the headline; the optimizer-side analog of biological M-evolution).

RiccatiPrecond with evolve_M=True learns the preconditioner's TARGET source M by CMA-ES-style
rank-mu accumulation of PRODUCTIVE steps (loss-weighted). The claim: the meta-loop reaches stable
approximate-INVERSE-curvature structure in the WELL-SAMPLED (top-k) subspace -- where naive alpha=1
diverges -- while genuinely flat directions stay near alpha=1/2, beating the alpha=1/2 whitening
baseline on convergence at alpha=1/2-level stability.

The headline DIAGNOSTIC is the realized OPERATIVE EXPONENT over time: for each Kronecker factor,
the applied preconditioner G solves G C G = M, so the eigenvalue-vs-curvature log-log slope of
(eig G vs eig C) is -operative_exponent. We track it in the top-k (steep) and bottom (flat)
curvature subspaces separately: the meta-loop should drive op_exponent_topk from ~1/2 toward ~1
while op_exponent_flat stays ~1/2.

Comparators (same model/data/budget): whiten (alpha=1/2 baseline), inverse (naive alpha=1, no
evolve -- expected to struggle/diverge), evolve (the meta-loop) over a small eta_M x meta_every grid.

bf16 autocast (no GradScaler -> the loss closure reaches step(), needed by the loss-weighted
meta-loop). NaN-surviving loop. Writes runs/riccati_evolve_m/<model>_<dataset>.csv.

CPU smoke:
    python -m ml_experiments.riccati_evolve_m --device cpu --model tiny_vision \
        --dataset synthetic_vision --synthetic-n 64 --batch 16 --max-steps 8 --no-amp
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml_experiments._harness import (RUNS_DIR, evaluate, is_lm_model, make_data,
                                     make_model, make_optimizer)
from curvature.operative_exponent import operative_exponent_factors, operative_exponent_lanczos

CSV_COLUMNS = [
    "condition", "label", "model", "dataset", "eta_M", "meta_every", "lr",
    "batch", "max_steps", "amp", "diverged", "finite_fraction", "n_steps_run",
    "step", "train_loss", "val_loss", "val_metric", "wallclock_s",
    "op_exponent_overall", "op_exponent_topk", "op_exponent_flat",
    "op_exponent_true_hessian", "seed",
]


def _forward_loss(model, batch, device, is_lm):
    if is_lm:
        x, y = batch
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        _, loss = model(x, y)
        return loss
    images, targets = batch
    images = images.to(device, non_blocking=True); targets = targets.to(device, non_blocking=True)
    return F.cross_entropy(model(images), targets)


def _infinite(loader):
    while True:
        for b in loader:
            yield b


def _loglog_slope(c, g):
    c = np.asarray(c, float); g = np.asarray(g, float)
    ok = (c > 1e-12) & (g > 1e-12)
    if ok.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(c[ok]), np.log(g[ok]), 1)[0])


# The operative-exponent diagnostics live in curvature/operative_exponent.py (validated):
#   (i)  operative_exponent_factors  -- realized exponent vs the optimizer's own curvature C
#        (the DESIGN exponent: whiten->0.5, inverse->1.0). Measured every checkpoint (cheap).
#        MUST be read mid-training: at convergence the gradient->0, C drops below the damping
#        floor, and the exponent degenerates to 0.
#   (ii) operative_exponent_lanczos  -- exponent vs the TRUE loss Hessian (HVP/Lanczos). Differs
#        from (i) when C != H (empirical vs true Fisher); computed once at the final step (costly).


def _make_opt(cond, eta_M, meta_every, model, lr, wd, base_lr):
    if cond == "whiten":
        return make_optimizer("riccati", model.parameters(), alpha=0.5, lr=lr,
                              weight_decay=wd, base_lr=base_lr, precond_mode="whiten", shrink=0.0)
    if cond == "inverse":   # naive full inverse, no evolve (expected to struggle/diverge)
        return make_optimizer("riccati", model.parameters(), alpha=1.0, lr=lr,
                              weight_decay=wd, base_lr=base_lr, precond_mode="inverse",
                              shrink=0.0, damping=1e-2)
    if cond == "evolve":
        return make_optimizer("riccati", model.parameters(), alpha=1.0, lr=lr,
                              weight_decay=wd, base_lr=base_lr, precond_mode="inverse",
                              shrink=0.2, damping=1e-2, evolve_m=True, eta_m=eta_M,
                              meta_every=meta_every)
    raise ValueError(cond)


def run_condition(cond, args, device, eta_M=1e-3, meta_every=20):
    label = cond if cond != "evolve" else f"evolve_etaM{eta_M:g}_m{meta_every}"
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    train_loader, val_loader, meta = make_data(
        args.dataset, batch_size=args.batch, num_workers=args.num_workers,
        generator=gen, synthetic_n=args.synthetic_n)
    is_lm = meta.task == "lm"
    model = (make_model(args.model, vocab_size=meta.vocab_size) if is_lm
             else make_model(args.model, num_classes=meta.num_classes)).to(device)
    is_lm = is_lm_model(model)
    optimizer, lr = _make_opt(cond, eta_M, meta_every, model, args.lr,
                              args.weight_decay, args.base_lr)

    use_cuda = device.type == "cuda"; use_amp = args.amp and use_cuda
    batch_iter = _infinite(train_loader)
    records = []; n_finite = 0; diverged = False; last = float("nan")
    ckpt_every = max(1, args.max_steps // 30)
    model.train(); t0 = time.time()
    for step in range(1, args.max_steps + 1):
        batch = next(batch_iter)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            loss = _forward_loss(model, batch, device, is_lm)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step(closure=lambda: loss.detach())
        lv = float(loss.item()); finite = math.isfinite(lv)
        if finite:
            n_finite += 1; last = lv
        elif not diverged:
            diverged = True
        if step == 1 or step % ckpt_every == 0 or step == args.max_steps:
            params_finite = all(torch.isfinite(p).all().item() for p in model.parameters())
            # (i) DESIGN exponent vs the optimizer's own curvature C (cheap, every checkpoint)
            op_over, op_top, op_flat = (operative_exponent_factors(optimizer)
                                        if params_finite else (float("nan"),) * 3)
            # (ii) TRUE-Hessian exponent at the final step only (HVP/Lanczos; costly)
            op_true = float("nan")
            if params_finite and step == args.max_steps and args.lanczos:
                fb = next(batch_iter)
                def _lf():
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                        return _forward_loss(model, fb, device, is_lm)
                op_true = operative_exponent_lanczos(
                    _lf, optimizer, [p for p in model.parameters() if p.dim() == 2],
                    k=args.lanczos_k, generator=torch.Generator(device="cpu").manual_seed(0))[0]
            val_metric = val_loss = float("nan")
            if params_finite:
                val_metric, val_loss, _ = evaluate(model, val_loader, device, is_lm,
                                                   max_batches=args.eval_max_batches)
                model.train()
            records.append(dict(
                condition=cond, label=label, model=args.model, dataset=args.dataset,
                eta_M=eta_M if cond == "evolve" else "", meta_every=meta_every if cond == "evolve" else "",
                lr=lr, batch=args.batch, max_steps=args.max_steps, amp=use_amp,
                diverged=diverged, finite_fraction=n_finite / step, n_steps_run=step,
                step=step, train_loss=lv, val_loss=val_loss, val_metric=val_metric,
                wallclock_s=time.time() - t0, op_exponent_overall=op_over,
                op_exponent_topk=op_top, op_exponent_flat=op_flat,
                op_exponent_true_hessian=op_true, seed=args.seed))
            print(f"[o4][{label}] step {step}/{args.max_steps} loss={lv:.4f} val={val_loss:.4f} "
                  f"op(i)=[{op_over:.2f},top{op_top:.2f},flat{op_flat:.2f}] op(ii)H={op_true:.2f} fin={finite}")
        if diverged and not finite and step > args.max_steps // 5:
            break
    print(f"[o4][{label}] DONE diverged={diverged} final_train={last:.4f}")
    return records


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS); w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_COLUMNS})


def run(args):
    device = torch.device(args.device if args.device
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[o4] device={device} batch={args.batch} max_steps={args.max_steps} amp={args.amp}")
    rows = []
    rows += run_condition("whiten", args, device)
    rows += run_condition("inverse", args, device)
    grid = [(em, me) for em in args.eta_m_grid for me in args.meta_every_grid]
    for em, me in grid:
        rows += run_condition("evolve", args, device, eta_M=em, meta_every=me)
    out_dir = Path(args.out_dir) if args.out_dir else (RUNS_DIR / "riccati_evolve_m")
    csv_path = out_dir / f"{args.model}_{args.dataset}.csv"
    write_csv(csv_path, rows)
    print(f"[o4] wrote {csv_path} ({len(rows)} rows)")
    return csv_path


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="vit_s")
    p.add_argument("--dataset", default="cifar100")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--max-steps", type=int, default=4000)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--base-lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--lanczos", action="store_true", help="(ii) true-Hessian operative exponent at the final step")
    p.add_argument("--lanczos-k", type=int, default=16)
    p.add_argument("--eta-m-grid", type=float, nargs="+", default=[3e-4, 1e-3, 3e-3])
    p.add_argument("--meta-every-grid", type=int, nargs="+", default=[20])
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

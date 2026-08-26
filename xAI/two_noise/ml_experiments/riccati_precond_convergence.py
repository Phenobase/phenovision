"""DIAGNOSTIC -- does the matrix-free Newton-Schulz factor actually reach its target on a real ViT?

O4 surfaced a red flag: every riccati condition read operative exponent ~0 (should be ~0.5 whiten
/ ~1.0 inverse) and riccati-whiten barely learned (train ~4.5) while eigh-based SOAP whitening
reached ~0.3 on the SAME alpha=1/2 target. An exponent of 0 means G ~ c*I -- the NS factor never
left its identity init. Hypothesis: under-converged NS (the harness default inner_steps=2,
precond_every=5) on the real ViT's curvature.

This sweeps inner_steps x precond_every (x mode) and measures, mid-training:
  (a) realized operative exponent   -- eig(G) vs eig(C) log-log slope (curvature/operative_exponent)
  (b) NS fixed-point residual       -- whiten: ||G C G - I|| ; inverse: ||G C - I||  (Frobenius,
      normalized), averaged over the live Kronecker factors. This is the DIRECT convergence check,
      independent of the slope fit.
plus train/val loss. If exponent->0.5 and residual->0 only at larger inner_steps / precond_every=1,
the matrix-free path just needs more NS budget on real models.

bf16 autocast (no GradScaler). Writes runs/riccati_precond_convergence/<model>_<dataset>.csv.

CPU smoke:
    python -m ml_experiments.riccati_precond_convergence --device cpu --model tiny_vision \
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
from curvature.operative_exponent import operative_exponent_factors

CSV_COLUMNS = [
    "label", "mode", "inner_steps", "precond_every", "lr", "model", "dataset",
    "batch", "max_steps", "step", "train_loss", "val_loss", "val_metric",
    "op_exponent_overall", "op_exponent_topk", "op_exponent_flat",
    "ns_residual", "wallclock_s", "seed",
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


@torch.no_grad()
def ns_residual(optimizer, mode):
    """Direct NS fixed-point residual on the live Kronecker factors, on CPU (robust). Returns the
    spread-weighted mean over factors of  whiten: ||G C G - I||_F/sqrt(d)  ;  inverse: ||G C - I||_F/sqrt(d).
    NaN if no factors are present yet."""
    num = den = 0.0
    for group in optimizer.param_groups:
        for p in group["params"]:
            st = optimizer.state.get(p, {})
            if not st.get("use_kron", False) or "CL" not in st:
                continue
            for Ck, Gk in (("CL", "GL"), ("CR", "GR")):
                C = st[Ck].float().cpu(); G = st[Gk].float().cpu()
                d = C.shape[0]
                eye = torch.eye(d)
                if mode == "whiten":
                    R = G @ C @ G - eye
                else:                                  # inverse: G -> C^{-1}, fixed point G C = I
                    R = G @ C - eye
                r = float(R.norm() / math.sqrt(d))
                num += r; den += 1.0
    return num / den if den > 0 else float("nan")


def run_config(cfg, args, device):
    mode, inner_steps, precond_every, lr = cfg["mode"], cfg["inner_steps"], cfg["precond_every"], cfg["lr"]
    label = f"{mode}_k{inner_steps}_pe{precond_every}_lr{lr:g}"
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    train_loader, val_loader, meta = make_data(
        args.dataset, batch_size=args.batch, num_workers=args.num_workers,
        generator=gen, synthetic_n=args.synthetic_n)
    is_lm = meta.task == "lm"
    model = (make_model(args.model, vocab_size=meta.vocab_size) if is_lm
             else make_model(args.model, num_classes=meta.num_classes)).to(device)
    is_lm = is_lm_model(model)
    alpha = 0.5 if mode == "whiten" else 1.0
    damping = 1e-6 if mode == "whiten" else 1e-2          # match O4: alpha=1 needs the floor
    optimizer, lr = make_optimizer("riccati", model.parameters(), alpha=alpha, lr=lr,
                                   weight_decay=args.weight_decay, base_lr=args.base_lr,
                                   precond_mode=mode, shrink=0.0, inner_steps=inner_steps,
                                   precond_every=precond_every, damping=damping)
    use_cuda = device.type == "cuda"; use_amp = args.amp and use_cuda
    batch_iter = _infinite(train_loader)
    records = []; last = float("nan"); ckpt_every = max(1, args.max_steps // 12)
    model.train(); t0 = time.time()
    for step in range(1, args.max_steps + 1):
        batch = next(batch_iter)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            loss = _forward_loss(model, batch, device, is_lm)
        loss.backward()
        optimizer.step(closure=lambda: loss.detach())
        lv = float(loss.item())
        if math.isfinite(lv):
            last = lv
        if step == 1 or step % ckpt_every == 0 or step == args.max_steps:
            pf = all(torch.isfinite(p).all().item() for p in model.parameters())
            op_over, op_top, op_flat = (operative_exponent_factors(optimizer)
                                        if pf else (float("nan"),) * 3)
            res = ns_residual(optimizer, mode) if pf else float("nan")
            vm = vl = float("nan")
            if pf:
                vm, vl, _ = evaluate(model, val_loader, device, is_lm,
                                     max_batches=args.eval_max_batches)
                model.train()
            records.append(dict(
                label=label, mode=mode, inner_steps=inner_steps, precond_every=precond_every,
                lr=lr, model=args.model, dataset=args.dataset, batch=args.batch,
                max_steps=args.max_steps, step=step, train_loss=lv, val_loss=vl, val_metric=vm,
                op_exponent_overall=op_over, op_exponent_topk=op_top, op_exponent_flat=op_flat,
                ns_residual=res, wallclock_s=time.time() - t0, seed=args.seed))
            print(f"[conv][{label}] step {step}/{args.max_steps} loss={lv:.4f} val={vl:.4f} "
                  f"op=[{op_over:.2f},top{op_top:.2f},flat{op_flat:.2f}] NSres={res:.3e}", flush=True)
    print(f"[conv][{label}] DONE final_train={last:.4f}", flush=True)
    return records


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS); w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_COLUMNS})


def build_configs(args):
    cfgs = []
    for mode in args.modes:
        lr = args.whiten_lr if mode == "whiten" else args.inverse_lr
        for k in args.inner_steps_grid:
            for pe in args.precond_every_grid:
                cfgs.append(dict(mode=mode, inner_steps=k, precond_every=pe, lr=lr))
    return cfgs


def run(args):
    device = torch.device(args.device if args.device
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[conv] device={device} batch={args.batch} max_steps={args.max_steps}", flush=True)
    out_dir = Path(args.out_dir) if args.out_dir else (RUNS_DIR / "riccati_precond_convergence")
    csv_path = out_dir / f"{args.model}_{args.dataset}.csv"
    rows = []
    for cfg in build_configs(args):
        try:
            rows += run_config(cfg, args, device)
        except Exception as e:
            print(f"[conv] config {cfg} FAILED: {type(e).__name__}: {e}", flush=True)
        write_csv(csv_path, rows)            # incremental
    print(f"[conv] wrote {csv_path} ({len(rows)} rows)", flush=True)
    return csv_path


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="vit_s")
    p.add_argument("--dataset", default="cifar100")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--max-steps", type=int, default=1200)
    p.add_argument("--modes", nargs="+", default=["whiten", "inverse"])
    p.add_argument("--inner-steps-grid", type=int, nargs="+", default=[2, 5, 10, 20])
    p.add_argument("--precond-every-grid", type=int, nargs="+", default=[1, 5])
    p.add_argument("--whiten-lr", type=float, default=3e-4)
    p.add_argument("--inverse-lr", type=float, default=1e-4)
    p.add_argument("--base-lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
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

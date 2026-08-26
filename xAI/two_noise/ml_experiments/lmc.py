#!/usr/bin/env python
"""Linear mode connectivity (LMC) between two trained checkpoints — the decisive test of
"same basin" vs "different basin".

Interpolate the weights linearly, θ(λ) = (1−λ)·θ_A + λ·θ_B, and evaluate train+val loss/acc
along λ ∈ [0,1]. If the endpoints sit in the SAME loss basin the interpolated path stays low
(no barrier); if they are in DIFFERENT basins a loss barrier rises in the middle.

Why it's valid here WITHOUT permutation alignment: both runs share the SAME initialization and
SAME data order (same --seed); they diverge only by the injected demographic noise after the
warmup. That is exactly the "spawn from a shared prefix" setting (Frankle et al. 2020) in which
naive LMC is meaningful — a barrier means the noise pushed the two solutions into genuinely
distinct basins, not merely a permutation of the same one.

Endpoints also directly answer "is the cooled constant-T run actually interpolating?" — λ=1 vs
λ=0 train accuracy. LayerNorm net → no BatchNorm running-stats to recalibrate, so interpolated
weights evaluate directly.

Usage:
  mamba run -n two_noise python -m ml_experiments.lmc \
    --model vit_s --dataset cifar100 \
    --ckpt-a runs/.../<base>.pt --label-a base \
    --ckpt-b runs/.../<constT_cooled>.pt --label-b constT_cooled \
    --steps 11 --eval-max-batches 40 --out runs/lmc/base_vs_constT.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from ml_experiments._harness import (make_data, make_model, make_train_eval_loader, evaluate)


def _load_state(path, device):
    sd = torch.load(path, map_location=device)
    # tolerate a wrapped checkpoint {"state_dict": ...} as well as a bare state_dict
    return sd.get("state_dict", sd) if isinstance(sd, dict) and "state_dict" in sd else sd


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="cifar100")
    ap.add_argument("--ckpt-a", required=True, help="endpoint A weights (.pt state_dict)")
    ap.add_argument("--ckpt-b", required=True, help="endpoint B weights (.pt state_dict)")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--steps", type=int, default=11, help="number of lambda points in [0,1]")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--eval-max-batches", type=int, default=0,
                    help="cap eval batches per split (0 = full; e.g. 40 for a fast estimate)")
    ap.add_argument("--device", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    device = torch.device(args.device if args.device
                          else ("cuda" if torch.cuda.is_available() else "cpu"))

    _, val_loader, meta = make_data(args.dataset, batch_size=args.batch_size,
                                    num_workers=args.num_workers)
    train_loader = make_train_eval_loader(args.dataset, batch_size=args.batch_size,
                                          num_workers=args.num_workers)
    model = make_model(args.model, num_classes=meta.num_classes).to(device)

    sd_a = _load_state(args.ckpt_a, device)
    sd_b = _load_state(args.ckpt_b, device)
    keys = list(sd_a.keys())
    assert set(keys) == set(sd_b.keys()), "checkpoints have mismatched parameter keys"

    rows = []
    lambdas = [i / (args.steps - 1) for i in range(args.steps)]
    for lam in lambdas:
        # θ(λ) = (1−λ)·A + λ·B for every tensor (params + any buffers carried in the state_dict)
        interp = {}
        for k in keys:
            a, b = sd_a[k].to(device), sd_b[k].to(device)
            if a.is_floating_point():
                interp[k] = (1.0 - lam) * a + lam * b
            else:
                interp[k] = a if lam < 0.5 else b   # integer buffers: nearest endpoint
        model.load_state_dict(interp, strict=True)
        va, vl, _ = evaluate(model, val_loader, device, is_lm=False,
                             max_batches=args.eval_max_batches)
        if train_loader is not None:
            ta, tl, _ = evaluate(model, train_loader, device, is_lm=False,
                                 max_batches=args.eval_max_batches)
        else:
            ta, tl = float("nan"), float("nan")
        rows.append(dict(lam=lam, train_acc=ta, train_loss=tl, val_acc=va, val_loss=vl))
        print(f"[lmc] λ={lam:.3f}  train_acc={ta:.4f} train_loss={tl:.4f}  "
              f"val_acc={va:.4f} val_loss={vl:.4f}", flush=True)

    # barrier = max interpolated loss minus the higher endpoint (positive => distinct basins)
    def barrier(metric, worse=max):
        endpoints = worse(rows[0][metric], rows[-1][metric])
        extreme = worse(r[metric] for r in rows)
        return extreme - endpoints
    tl_barrier = barrier("train_loss", max)
    vl_barrier = barrier("val_loss", max)
    ta_drop = max(rows[0]["train_acc"], rows[-1]["train_acc"]) - min(r["train_acc"] for r in rows)
    va_drop = max(rows[0]["val_acc"], rows[-1]["val_acc"]) - min(r["val_acc"] for r in rows)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["label_a", "label_b", "lam",
                                          "train_acc", "train_loss", "val_acc", "val_loss"])
        w.writeheader()
        for r in rows:
            w.writerow(dict(label_a=args.label_a, label_b=args.label_b, **r))
    print(f"[lmc] wrote {out}")
    print(f"[lmc] {args.label_a}(λ=0): train_acc={rows[0]['train_acc']:.4f} "
          f"val_acc={rows[0]['val_acc']:.4f}  |  "
          f"{args.label_b}(λ=1): train_acc={rows[-1]['train_acc']:.4f} "
          f"val_acc={rows[-1]['val_acc']:.4f}")
    print(f"[lmc] BARRIER train_loss=+{tl_barrier:.4f}  val_loss=+{vl_barrier:.4f}  "
          f"(train_acc dip {ta_drop:.4f}, val_acc dip {va_drop:.4f})")
    # The decisive signal is whether the interpolated path's ACCURACY collapses, not the raw loss
    # bump: a small loss barrier with preserved accuracy (e.g. +0.17 train_loss, train_acc still
    # >0.9 mid-path) is still the SAME basin. Independent (different-init) nets collapse to chance
    # via permutation symmetry; a connected pair keeps accuracy along the whole path.
    distinct = ta_drop > 0.25
    print(f"[lmc] VERDICT: {'DISTINCT BASINS (path accuracy collapses)' if distinct else 'SAME BASIN (linearly connected — path accuracy preserved)'}")
    return out


if __name__ == "__main__":
    main()

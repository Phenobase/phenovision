#!/usr/bin/env python3
"""
landscape_slices.py — filter-normalized random-direction loss-landscape slices
(Li et al. 2018, "Visualizing the Loss Landscape of Neural Nets"; plan component C7,
briefing Part II §9.2). Post-hoc / ladder-only.

Given ONE PhenoVision preadapt checkpoint θ*, draw 1 or 2 random directions, FILTER-NORMALIZE
each per-filter so its per-filter norm equals θ*'s corresponding per-filter norm (this is the
whole point of Li et al.: it removes the scale-invariance of ReLU/LayerNorm nets so the sharpness
you see is comparable across checkpoints), then sweep

  1-D:  θ(α) = θ* + α·d1                       over an α grid
  2-D:  θ(α,β) = θ* + α·d1 + β·d2               over an (α,β) grid

evaluating the loss on a FIXED probe batch/set at each grid point, and write a grid CSV
(1-D: ``alpha,loss[,auc_pr,auc_roc]``; 2-D: ``alpha,beta,loss[,auc_pr,auc_roc]``).

The FROZEN INPUT STAGE (``patch_embed.*`` + ``pos_embed``) is EXCLUDED from the perturbation:
directions are drawn and normalized only over :func:`preadapt_common.trainable_named_parameters`
(briefing "Do not include the frozen patch-embedding / positional-embedding parameters in any
displacement, projection, or rank statistic"). Those tensors are left at θ* at every grid point.

Filter normalization (Li et al. §4):
  For a 2-D+ weight tensor we treat dim-0 as the "filters/output units" and normalize EACH
  filter slice of the random direction to the L2 norm of the SAME filter slice of θ*:
      d_{i} <- d_{i} / ||d_{i}|| * ||θ*_{i}||      (per output unit i)
  For 1-D params (LayerNorm weights, biases) and the CLS token there is no filter axis, so we
  normalize the WHOLE tensor's direction to the whole tensor's θ* norm (the standard treatment of
  bias/BN/LN terms; a zero-norm θ* tensor keeps a zero direction so it never moves).

Cost: O(grid_points) (1-D) or O(grid_points²) (2-D) full forward passes over the probe set per
checkpoint — many forward passes. This is why it is post-hoc and run only on retained-ladder
checkpoints (briefing §9.2). Use ``--eval-max-batches`` and a modest ``--grid-points`` to bound it.

Every random draw threads an explicit ``torch.Generator`` (CONVENTIONS.md §5); the slice is fully
reproducible from ``--seed``.

Usage:
  mamba run -n reticulate-gpu2 python -m posthoc.landscape_slices \
    --ckpt xAI/output/preadapt/mae__stable_evo__s42/checkpoints/step00001234.pt \
    --val-csv data/inat/val_v1.1.0.csv \
    --n-dirs 2 --grid-points 21 --alpha-range 1.0 --eval-max-batches 20 \
    --out xAI/output/posthoc/landscape/mae_stable_evo_step1234_2d.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch

# --- project paths (mirror preadapt_train.py) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_PY_DIR = os.path.dirname(_HERE)                                  # .../xAI/py
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_PY_DIR))         # .../phenovision
_PLANTCLEF = os.path.join(_PROJECT_ROOT, "PlantCLEF2022")
_TWO_NOISE = os.path.join(_PROJECT_ROOT, "xAI", "two_noise")
for _p in (_PY_DIR, _PROJECT_ROOT, _PLANTCLEF, _TWO_NOISE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from preadapt_common import trainable_named_parameters  # noqa: E402

EvalFn = Callable[[torch.nn.Module, object, torch.nn.Module, torch.device], Dict[str, float]]


# =============================================================================
# Checkpoint state loading (shared convention with mode_connectivity)
# =============================================================================

def load_model_state(path: str, device: torch.device) -> Dict[str, torch.Tensor]:
    """Extract the model state_dict from a preadapt checkpoint (``model_state_dict`` key, or a
    nested/bare state_dict)."""
    sd = torch.load(path, map_location=device, weights_only=False)
    if isinstance(sd, dict):
        if "model_state_dict" in sd:
            return sd["model_state_dict"]
        if "state_dict" in sd:
            return sd["state_dict"]
    return sd


# =============================================================================
# Filter-normalized random directions (Li et al. 2018)
# =============================================================================

@torch.no_grad()
def filter_normalized_direction(
    model: torch.nn.Module,
    generator: torch.Generator,
    device: torch.device,
) -> "OrderedDict[str, torch.Tensor]":
    """Draw ONE filter-normalized random direction over the TRAINABLE params (frozen input stage
    excluded), keyed by parameter name.

    For each trainable parameter θ*_p the direction tensor d_p is Gaussian noise (same shape),
    then normalized:
      * ndim >= 2 : per-filter — slice along dim 0; each slice d_p[i] is rescaled to
        ``||d_p[i]|| -> ||θ*_p[i]||``.
      * ndim <= 1 (LN weights, biases) or ``cls_token`` : whole-tensor — ``||d_p|| -> ||θ*_p||``.
    A zero-norm reference slice yields a zero direction slice (the param never moves), matching
    Li et al.'s handling of zero-norm filters.

    The generator is threaded into every ``randn`` so the direction is reproducible.
    """
    from collections import OrderedDict
    direction: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for name, p in trainable_named_parameters(model, include_requires_grad_false=True):
        ref = p.detach().to(device, torch.float32)
        d = torch.randn(ref.shape, generator=generator, device=generator.device).to(device)
        if ref.ndim >= 2:
            # Per-filter normalization along dim 0 (output units / filters).
            d_flat = d.reshape(ref.shape[0], -1)
            r_flat = ref.reshape(ref.shape[0], -1)
            d_norm = d_flat.norm(dim=1, keepdim=True)               # [F,1]
            r_norm = r_flat.norm(dim=1, keepdim=True)               # [F,1]
            scale = torch.where(d_norm > 0, r_norm / d_norm, torch.zeros_like(d_norm))
            d = (d_flat * scale).reshape(ref.shape)
        else:
            # Whole-tensor normalization for 1-D (LN/bias) and cls_token.
            d_norm = d.norm()
            r_norm = ref.norm()
            d = d * (r_norm / d_norm) if float(d_norm) > 0 else torch.zeros_like(d)
        direction[name] = d
    return direction


@torch.no_grad()
def apply_perturbation(
    model: torch.nn.Module,
    theta_star: "OrderedDict[str, torch.Tensor]",
    d1: "OrderedDict[str, torch.Tensor]",
    alpha: float,
    d2: "Optional[OrderedDict[str, torch.Tensor]]" = None,
    beta: float = 0.0,
) -> None:
    """Set the model's trainable params to ``θ* + α·d1 (+ β·d2)`` IN PLACE.

    Only the trainable params (keys of ``theta_star`` / ``d1``) are written; the frozen input
    stage is never touched. Reads each param from the model by name so the model's parameter
    objects stay the same (no state_dict reload needed)."""
    named = dict(trainable_named_parameters(model, include_requires_grad_false=True))
    for name, base in theta_star.items():
        new = base + alpha * d1[name]
        if d2 is not None and beta != 0.0:
            new = new + beta * d2[name]
        named[name].data.copy_(new.to(named[name].dtype))


@torch.no_grad()
def snapshot_trainable(model: torch.nn.Module, device: torch.device) -> "OrderedDict[str, torch.Tensor]":
    """θ* snapshot of the trainable params (float32, on ``device``), keyed by name."""
    from collections import OrderedDict
    snap: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for name, p in trainable_named_parameters(model, include_requires_grad_false=True):
        snap[name] = p.detach().to(device, torch.float32).clone()
    return snap


# =============================================================================
# Eval helpers (PhenoVision multi-label probe; injectable for the smoke test)
# =============================================================================

def _capped_loader(loader, eval_max_batches: int):
    if not eval_max_batches:
        return loader

    class _Capped:
        def __init__(self, base, cap):
            self.base, self.cap = base, cap

        def __iter__(self):
            for i, b in enumerate(self.base):
                if i >= self.cap:
                    break
                yield b

    return _Capped(loader, eval_max_batches)


def make_phenovision_eval(eval_max_batches: int) -> EvalFn:
    """Return an :data:`EvalFn` wrapping ``xai_train.validate`` over a batch-capped FIXED probe.
    Lazy import so the module is importable on CPU without a ViT-L forward pass."""
    from xai_train import validate  # noqa: E402

    def _eval(model, loader, criterion, device) -> Dict[str, float]:
        return validate(model, _capped_loader(loader, eval_max_batches), criterion, device)

    return _eval


def build_phenovision_model(num_classes: int = 2):
    from models_vit import vit_large_patch16  # noqa: E402
    return vit_large_patch16(num_classes=num_classes)


def build_phenovision_probe(val_csv: str, batch_size: int, seed: int, num_workers: int,
                            val_subset_size: int):
    """Build the FIXED val probe loader + criterion via ``xai_train.create_data_loaders``."""
    import torch.nn as nn  # noqa: E402
    from xai_train import create_data_loaders  # noqa: E402

    args = argparse.Namespace(
        train_csv=val_csv,   # unused for the slice; create_data_loaders needs a path
        val_csv=val_csv,
        batch_size=batch_size,
        test_batch_size=None,
        num_workers=num_workers,
        val_subset_size=val_subset_size,
        repr_subset_size=min(1000, val_subset_size),
        max_train_samples=None,
        seed=seed,
        output_dir=os.path.join(_PROJECT_ROOT, "xAI", "output", "posthoc", "_slice_probe_indices"),
    )
    _train_dl, val_dl, _repr_dl = create_data_loaders(args)
    criterion = nn.BCEWithLogitsLoss().to(
        torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return val_dl, criterion


# =============================================================================
# Core slice sweep (model + eval injected; testable without ViT-L)
# =============================================================================

def slice_sweep(
    model: torch.nn.Module,
    eval_fn: EvalFn,
    loader,
    criterion,
    device: torch.device,
    n_dirs: int,
    grid_points: int,
    alpha_range: float,
    generator: torch.Generator,
    verbose: bool = True,
) -> Tuple[List[Dict[str, float]], "OrderedDict[str, torch.Tensor]"]:
    """Run the 1-D or 2-D filter-normalized landscape slice. ``model`` is assumed to already hold
    θ* (the checkpoint weights). Returns ``(rows, theta_star)``; ``model`` is restored to θ* at
    the end so the caller can reuse it.

    1-D (n_dirs=1): grid of ``grid_points`` α in [-alpha_range, +alpha_range]; rows ``{alpha,
    loss, auc_pr, auc_roc}``.
    2-D (n_dirs=2): grid² over (α,β) in the same range; rows ``{alpha, beta, loss, auc_pr,
    auc_roc}``.
    """
    if n_dirs not in (1, 2):
        raise ValueError(f"n_dirs must be 1 or 2, got {n_dirs}")

    theta_star = snapshot_trainable(model, device)
    d1 = filter_normalized_direction(model, generator, device)
    d2 = filter_normalized_direction(model, generator, device) if n_dirs == 2 else None

    grid = _linspace(-alpha_range, alpha_range, grid_points)
    rows: List[Dict[str, float]] = []

    if n_dirs == 1:
        for a in grid:
            apply_perturbation(model, theta_star, d1, a)
            vm = eval_fn(model, loader, criterion, device)
            row = dict(alpha=a, loss=float(vm.get("val_loss", float("nan"))),
                       auc_pr=float(vm.get("auc_pr_mean", float("nan"))),
                       auc_roc=float(vm.get("auc_roc_mean", float("nan"))))
            rows.append(row)
            if verbose:
                print(f"[slice] α={a:+.4f}  loss={row['loss']:.4f} "
                      f"auc_pr={row['auc_pr']:.4f}", flush=True)
    else:
        for a in grid:
            for b in grid:
                apply_perturbation(model, theta_star, d1, a, d2=d2, beta=b)
                vm = eval_fn(model, loader, criterion, device)
                row = dict(alpha=a, beta=b, loss=float(vm.get("val_loss", float("nan"))),
                           auc_pr=float(vm.get("auc_pr_mean", float("nan"))),
                           auc_roc=float(vm.get("auc_roc_mean", float("nan"))))
                rows.append(row)
            if verbose:
                print(f"[slice] α={a:+.4f} row done ({grid_points} β pts)", flush=True)

    # Restore θ* so the model is clean for any reuse.
    apply_perturbation(model, theta_star, d1, 0.0, d2=d2, beta=0.0)
    return rows, theta_star


def _linspace(lo: float, hi: float, n: int) -> List[float]:
    if n <= 1:
        return [0.0]
    return [lo + (hi - lo) * i / (n - 1) for i in range(n)]


def write_grid_csv(path: str, rows: List[Dict[str, float]], n_dirs: int) -> Path:
    """Write the slice grid to CSV. 1-D: ``alpha,loss,auc_pr,auc_roc``; 2-D adds ``beta``."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if n_dirs == 1:
        fieldnames = ["alpha", "loss", "auc_pr", "auc_roc"]
    else:
        fieldnames = ["alpha", "beta", "loss", "auc_pr", "auc_roc"]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})
    return out


# =============================================================================
# CLI
# =============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="preadapt checkpoint (.pt) — θ* center of the slice")
    p.add_argument("--val-csv", required=True,
                   help="held-out probe CSV (file_name, fruiting, flowering) — the FIXED probe set")
    p.add_argument("--n-dirs", type=int, default=2, choices=[1, 2],
                   help="1 = 1-D slice (α only); 2 = 2-D slice (α,β)")
    p.add_argument("--grid-points", type=int, default=21,
                   help="points per axis (2-D total forward passes = grid-points²)")
    p.add_argument("--alpha-range", type=float, default=1.0,
                   help="sweep α (and β) over [-alpha_range, +alpha_range]")
    p.add_argument("--num-classes", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--eval-max-batches", type=int, default=0,
                   help="cap eval batches (0 = full probe set; bounds the many forward passes)")
    p.add_argument("--val-subset-size", type=int, default=10000)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42, help="seeds the random directions + probe subset")
    p.add_argument("--device", default="")
    p.add_argument("--out", required=True, help="output grid CSV path")
    return p


def main(argv: Optional[List[str]] = None) -> Path:
    args = build_arg_parser().parse_args(argv)

    device = torch.device(args.device if args.device
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    n_fwd = args.grid_points if args.n_dirs == 1 else args.grid_points ** 2
    print(f"[slice] device={device}  n_dirs={args.n_dirs}  grid_points={args.grid_points}  "
          f"=> {n_fwd} forward passes over the probe set (post-hoc cost)")

    loader, criterion = build_phenovision_probe(
        val_csv=args.val_csv, batch_size=args.batch_size, seed=args.seed,
        num_workers=args.num_workers, val_subset_size=args.val_subset_size)

    model = build_phenovision_model(num_classes=args.num_classes).to(device)
    sd = load_model_state(args.ckpt, device)
    model.load_state_dict(sd, strict=True)

    # Directions live on the eval device so randn + the param tensors share it.
    generator = torch.Generator(device=device if device.type == "cuda" else "cpu")
    generator.manual_seed(args.seed)

    eval_fn = make_phenovision_eval(args.eval_max_batches)
    rows, _theta = slice_sweep(model, eval_fn, loader, criterion, device,
                               n_dirs=args.n_dirs, grid_points=args.grid_points,
                               alpha_range=args.alpha_range, generator=generator)

    out = write_grid_csv(args.out, rows, args.n_dirs)
    losses = [r["loss"] for r in rows if r["loss"] == r["loss"]]
    if losses:
        center = next((r for r in rows
                       if r["alpha"] == 0.0 and (args.n_dirs == 1 or r.get("beta") == 0.0)), None)
        c = center["loss"] if center else float("nan")
        print(f"[slice] wrote {out}  (center loss={c:.4f}, min={min(losses):.4f}, "
              f"max={max(losses):.4f})")
    else:
        print(f"[slice] wrote {out}")
    return out


if __name__ == "__main__":
    main()

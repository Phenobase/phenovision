#!/usr/bin/env python3
"""
mode_connectivity.py — Linear mode connectivity (LMC) between two PhenoVision preadapt
checkpoints (plan component C7; briefing Part II §9.2). The decisive post-hoc test of
"same basin" vs "distinct basin" over the retention ladder.

PORTED from xAI/two_noise/ml_experiments/lmc.py (the CIFAR/_harness version) to PhenoVision.
We do NOT import that module — it is bound to the two_noise `_harness` (CIFAR data, ViT-S/GPT
models, an accuracy/perplexity `evaluate`). Here we rebuild the ViT-L/16 of the experiment and
evaluate with the validated multi-label probe (xai_train.validate -> val_loss + AUC-PR/ROC).

Idea (unchanged from the port): interpolate the weights linearly,
``θ(λ) = (1−λ)·θ_A + λ·θ_B`` for λ ∈ [0,1], and evaluate train+val loss/AUC along the path.
If the endpoints sit in the SAME loss basin the interpolated path stays low (no barrier); if
they are in DIFFERENT basins a loss barrier rises in the middle and the AUC dips.

------------------------------------------------------------------------------------------------
WHEN IS THIS VALID? (read before interpreting a barrier)
Naive LMC (no permutation alignment) is only meaningful between endpoints that SHARE the same
initialization AND the same data order (Frankle et al. 2020, "spawn from a shared prefix"):

  * VALID — same condition, adamw-vs-stable_evo at the same seed: both share the Phase-1 frozen
    head, the Phase-2 init θ₀ (`init_model.pt`), and the data order (same --seed); they diverge
    only by the optimizer variant (and the demographic noise after warmup). A barrier here means
    the variant pushed the two solutions into genuinely distinct basins, not a permutation of one.
  * VALID — along ONE run's trajectory (two ladder checkpoints of the same run_id): they are on
    one continuous optimization path, so a low barrier is expected and a high one flags a basin
    hop within the run.
  * MEANINGFUL-DIFFERENTLY — cross-condition (e.g. mae vs plantclef): the endpoints do NOT share
    an init (different pretrained starts) and may not share data order, so permutation symmetry
    makes a raw barrier EXPECTED and large. Such a barrier does NOT imply "the two conditions are
    in different basins of one shared landscape" — it is the generic independent-solutions
    barrier. We still compute it (it is the operational "is there a low-loss path between the
    endpoints" question of §9.2), but the verdict line is annotated ``cross-init`` and must not
    be read as the within-init same/distinct-basin test. The decisive within-init signal is
    whether the path's AUC COLLAPSES (mid-path AUC near chance ~0.5), not the raw loss bump:
    a small loss barrier with preserved AUC is still the same basin.

LayerNorm net (no BatchNorm running-stats to recalibrate), so interpolated weights evaluate
directly — no BN-recalibration pass is needed.
------------------------------------------------------------------------------------------------

Usage (mirrors lmc.py CLI):
  mamba run -n reticulate-gpu2 python -m posthoc.mode_connectivity \
    --ckpt-a xAI/output/preadapt/mae__adamw__s42/checkpoints/step00001234.pt --label-a mae_adamw \
    --ckpt-b xAI/output/preadapt/mae__stable_evo__s42/checkpoints/step00001234.pt \
    --label-b mae_stable_evo \
    --steps 11 --batch-size 64 --eval-max-batches 40 \
    --val-csv data/inat/val_v1.1.0.csv \
    --out xAI/output/posthoc/lmc/mae_adamw_vs_stable_evo.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch

# --- project paths (mirror preadapt_train.py so PlantCLEF2022 + py modules import cleanly) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_PY_DIR = os.path.dirname(_HERE)                                  # .../xAI/py
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_PY_DIR))         # .../phenovision
_PLANTCLEF = os.path.join(_PROJECT_ROOT, "PlantCLEF2022")
_TWO_NOISE = os.path.join(_PROJECT_ROOT, "xAI", "two_noise")
for _p in (_PY_DIR, _PROJECT_ROOT, _PLANTCLEF, _TWO_NOISE):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# =============================================================================
# Checkpoint state loading
# =============================================================================

def load_model_state(path: str, device: torch.device) -> Dict[str, torch.Tensor]:
    """Extract the model state_dict from a preadapt checkpoint.

    The trainer writes a :func:`preadapt_common.build_train_ckpt` dict whose model weights live
    under ``model_state_dict`` (alongside ``optimizer_state_dict`` + metadata). For robustness we
    also tolerate a checkpoint that nests under ``state_dict`` or is a bare state_dict (mirrors
    lmc.py's ``_load_state``).
    """
    sd = torch.load(path, map_location=device, weights_only=False)
    if isinstance(sd, dict):
        if "model_state_dict" in sd:
            return sd["model_state_dict"]
        if "state_dict" in sd:
            return sd["state_dict"]
    return sd


def interpolate_state(
    sd_a: Dict[str, torch.Tensor],
    sd_b: Dict[str, torch.Tensor],
    lam: float,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """θ(λ) = (1−λ)·A + λ·B for every tensor.

    Floating-point tensors (params + float buffers) are linearly interpolated. Integer / bool
    buffers (e.g. ``num_batches_tracked``, any index buffer) cannot be averaged; they take the
    NEAREST endpoint (A if λ<0.5 else B) — exactly the lmc.py convention.
    """
    keys = list(sd_a.keys())
    interp: Dict[str, torch.Tensor] = {}
    for k in keys:
        a = sd_a[k].to(device)
        b = sd_b[k].to(device)
        if a.is_floating_point():
            interp[k] = (1.0 - lam) * a + lam * b
        else:
            interp[k] = a if lam < 0.5 else b   # integer / bool buffers: nearest endpoint
    return interp


# =============================================================================
# Evaluation (PhenoVision multi-label probe)
# =============================================================================

# An eval fn maps (model, loader, criterion, device) -> {"val_loss", "auc_pr_mean",
# "auc_roc_mean", ...}. The real one is xai_train.validate; tests inject a trivial stub so the
# ViT-L code path stays importable without paying for a ViT-L forward pass on CPU.
EvalFn = Callable[[torch.nn.Module, object, torch.nn.Module, torch.device], Dict[str, float]]


def _capped_loader(loader, eval_max_batches: int):
    """Yield at most ``eval_max_batches`` batches from ``loader`` (0 = no cap). Keeps a fixed
    probe set comparable across all λ by always taking the same leading batches."""
    if not eval_max_batches:
        return loader

    class _Capped:
        def __init__(self, base, cap):
            self.base = base
            self.cap = cap

        def __iter__(self):
            for i, b in enumerate(self.base):
                if i >= self.cap:
                    break
                yield b

    return _Capped(loader, eval_max_batches)


def make_phenovision_eval(eval_max_batches: int) -> EvalFn:
    """Return an :data:`EvalFn` that runs ``xai_train.validate`` over a batch-capped loader.

    ``validate`` returns ``{val_loss, auc_roc_{fruit,flower}, auc_pr_{fruit,flower},
    auc_roc_mean, auc_pr_mean}`` — we keep loss + the two means for the curve. Imported lazily so
    the smoke test can import this module (and the real PhenoVision code path) on a CPU box that
    cannot afford a ViT-L forward pass; tests pass their own stub instead.
    """
    from xai_train import validate  # noqa: E402 (lazy: heavy import chain)

    def _eval(model, loader, criterion, device) -> Dict[str, float]:
        capped = _capped_loader(loader, eval_max_batches)
        return validate(model, capped, criterion, device)

    return _eval


def build_phenovision_model(num_classes: int = 2):
    """Build the experiment's ViT-L/16 (no pretrained weights — we overwrite with interpolated
    state). Imported lazily for the same CPU-smoke reason as :func:`make_phenovision_eval`."""
    from models_vit import vit_large_patch16  # noqa: E402
    return vit_large_patch16(num_classes=num_classes)


def build_phenovision_loaders(val_csv: str, batch_size: int, seed: int, num_workers: int,
                              val_subset_size: int, train_csv: Optional[str]):
    """Build a FIXED val probe loader (and an optional train probe loader) via
    ``xai_train.create_data_loaders``. The probe set is fixed by ``seed`` so every λ — and every
    invocation comparing the same endpoints — sees the same images (briefing §4 invariant 1).

    Returns ``(val_loader, train_loader_or_None, criterion)``. If ``train_csv`` is None the train
    split is skipped (val-only LMC; cheaper). create_data_loaders uses ``args.val_subset_size``
    for the fixed val subset; the returned train loader is the FULL shuffled train loader, which
    we treat as a train-side probe under ``--eval-max-batches`` (so it is bounded).
    """
    import torch.nn as nn  # noqa: E402
    from xai_train import create_data_loaders  # noqa: E402

    args = argparse.Namespace(
        train_csv=train_csv or val_csv,   # create_data_loaders always needs a train csv to read
        val_csv=val_csv,
        batch_size=batch_size,
        test_batch_size=None,
        num_workers=num_workers,
        val_subset_size=val_subset_size,
        repr_subset_size=min(1000, val_subset_size),
        max_train_samples=None,
        seed=seed,
        output_dir=os.path.join(_PROJECT_ROOT, "xAI", "output", "posthoc", "_probe_indices"),
    )
    train_dl, val_dl, _repr_dl = create_data_loaders(args)
    criterion = nn.BCEWithLogitsLoss().to(
        torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    train_probe = train_dl if train_csv is not None else None
    return val_dl, train_probe, criterion


# =============================================================================
# Core LMC sweep (model + eval injected; testable without ViT-L)
# =============================================================================

def lmc_sweep(
    model: torch.nn.Module,
    sd_a: Dict[str, torch.Tensor],
    sd_b: Dict[str, torch.Tensor],
    eval_fn: EvalFn,
    val_loader,
    criterion,
    device: torch.device,
    steps: int,
    train_loader=None,
    verbose: bool = True,
) -> List[Dict[str, float]]:
    """Interpolate θ(λ) over ``steps`` λ-points in [0,1], load each into ``model``, evaluate
    train+val with ``eval_fn``, and return one row dict per λ.

    Each row: ``{lam, train_loss, train_auc_pr, train_auc_roc, val_loss, val_auc_pr,
    val_auc_roc}``. Train metrics are NaN when ``train_loader`` is None. ``model`` is mutated in
    place (its state_dict is overwritten each λ); the caller owns it.
    """
    keys_a, keys_b = set(sd_a.keys()), set(sd_b.keys())
    if keys_a != keys_b:
        only_a = sorted(keys_a - keys_b)[:5]
        only_b = sorted(keys_b - keys_a)[:5]
        raise ValueError(
            f"checkpoints have mismatched parameter keys "
            f"(A-only e.g. {only_a}, B-only e.g. {only_b})")

    rows: List[Dict[str, float]] = []
    lambdas = [i / (steps - 1) for i in range(steps)] if steps > 1 else [0.0]
    for lam in lambdas:
        interp = interpolate_state(sd_a, sd_b, lam, device)
        model.load_state_dict(interp, strict=True)
        vm = eval_fn(model, val_loader, criterion, device)
        if train_loader is not None:
            tm = eval_fn(model, train_loader, criterion, device)
        else:
            tm = {"val_loss": float("nan"), "auc_pr_mean": float("nan"),
                  "auc_roc_mean": float("nan")}
        row = dict(
            lam=lam,
            train_loss=float(tm.get("val_loss", float("nan"))),
            train_auc_pr=float(tm.get("auc_pr_mean", float("nan"))),
            train_auc_roc=float(tm.get("auc_roc_mean", float("nan"))),
            val_loss=float(vm.get("val_loss", float("nan"))),
            val_auc_pr=float(vm.get("auc_pr_mean", float("nan"))),
            val_auc_roc=float(vm.get("auc_roc_mean", float("nan"))),
        )
        rows.append(row)
        if verbose:
            print(f"[lmc] λ={lam:.3f}  train_loss={row['train_loss']:.4f} "
                  f"train_auc_pr={row['train_auc_pr']:.4f}  "
                  f"val_loss={row['val_loss']:.4f} val_auc_pr={row['val_auc_pr']:.4f}",
                  flush=True)
    return rows


def loss_barrier(rows: List[Dict[str, float]], metric: str) -> float:
    """Loss barrier = max interpolated value minus the WORSE (higher-loss) endpoint.

    Positive => the interpolated path rises above both endpoints (a barrier => distinct basins);
    ~0 / negative => the path stays at or below the endpoints (linearly connected). NaN-safe:
    rows whose ``metric`` is NaN are skipped.
    """
    vals = [r[metric] for r in rows if r[metric] == r[metric]]  # drop NaN
    if not vals:
        return float("nan")
    ends = [rows[0][metric], rows[-1][metric]]
    ends = [v for v in ends if v == v]
    if not ends:
        return float("nan")
    worse_endpoint = max(ends)          # higher loss = worse
    return max(vals) - worse_endpoint


def auc_dip(rows: List[Dict[str, float]], metric: str) -> float:
    """AUC dip = the BETTER (higher) endpoint minus the MINIMUM interpolated value.

    Positive => AUC collapses somewhere on the path (distinct basins / the path leaves the basin);
    ~0 => AUC is preserved along the whole path (same basin). NaN-safe.
    """
    vals = [r[metric] for r in rows if r[metric] == r[metric]]
    if not vals:
        return float("nan")
    ends = [rows[0][metric], rows[-1][metric]]
    ends = [v for v in ends if v == v]
    if not ends:
        return float("nan")
    better_endpoint = max(ends)         # higher AUC = better
    return better_endpoint - min(vals)


def verdict(rows: List[Dict[str, float]], cross_init: bool,
            auc_dip_threshold: float = 0.15) -> str:
    """SAME-BASIN vs DISTINCT-BASIN verdict from the train-AUC-PR dip (the within-init signal).

    The decisive signal is whether the path's AUC COLLAPSES, not the raw loss bump (lmc.py's
    note). For multi-label PhenoVision, AUC-PR near its endpoint along the whole path => the
    interpolated networks still classify => SAME basin; a large dip => the path leaves the basin.
    When ``cross_init`` (endpoints did not share an init), the result is annotated as the generic
    independent-solutions barrier, NOT the within-init same/distinct test.
    """
    dip = auc_dip(rows, "train_auc_pr")
    if dip != dip:  # NaN (no train metrics) -> fall back to val AUC-PR
        dip = auc_dip(rows, "val_auc_pr")
    distinct = (dip == dip) and (dip > auc_dip_threshold)
    base = ("DISTINCT BASINS (path AUC-PR collapses)" if distinct
            else "SAME BASIN (linearly connected — path AUC-PR preserved)")
    if cross_init:
        return (f"CROSS-INIT endpoints: a barrier is EXPECTED (generic independent-solutions "
                f"barrier, not the within-init same/distinct-basin test) — {base}")
    return base


# =============================================================================
# CLI
# =============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt-a", required=True, help="endpoint A preadapt checkpoint (.pt)")
    p.add_argument("--ckpt-b", required=True, help="endpoint B preadapt checkpoint (.pt)")
    p.add_argument("--label-a", default="A")
    p.add_argument("--label-b", default="B")
    p.add_argument("--steps", type=int, default=11, help="number of λ points in [0,1]")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--eval-max-batches", type=int, default=0,
                   help="cap eval batches per split (0 = full; e.g. 40 for a fast estimate)")
    p.add_argument("--val-csv", required=True,
                   help="held-out probe CSV (file_name, fruiting, flowering) — the fixed probe set")
    p.add_argument("--train-csv", default=None,
                   help="optional train CSV; if set, also evaluate a train-side probe along λ")
    p.add_argument("--num-classes", type=int, default=2)
    p.add_argument("--val-subset-size", type=int, default=10000,
                   help="fixed val-probe subset size (seed-fixed, comparable across λ)")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42,
                   help="fixes the probe subset (NOT a training seed)")
    p.add_argument("--cross-init", action="store_true",
                   help="endpoints do NOT share an init (cross-condition): annotate the verdict as "
                        "the generic independent-solutions barrier rather than the within-init test")
    p.add_argument("--auc-dip-threshold", type=float, default=0.15,
                   help="train-AUC-PR dip above which the path is called DISTINCT BASINS")
    p.add_argument("--device", default="")
    p.add_argument("--out", required=True, help="output λ-curve CSV path")
    return p


def main(argv: Optional[List[str]] = None) -> Path:
    args = build_arg_parser().parse_args(argv)

    device = torch.device(args.device if args.device
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[lmc] device={device}  steps={args.steps}  eval_max_batches={args.eval_max_batches}")

    # Build the FIXED probe loaders (seed-fixed) and the ViT-L/16.
    val_loader, train_loader, criterion = build_phenovision_loaders(
        val_csv=args.val_csv, batch_size=args.batch_size, seed=args.seed,
        num_workers=args.num_workers, val_subset_size=args.val_subset_size,
        train_csv=args.train_csv)
    model = build_phenovision_model(num_classes=args.num_classes).to(device)
    eval_fn = make_phenovision_eval(args.eval_max_batches)

    sd_a = load_model_state(args.ckpt_a, device)
    sd_b = load_model_state(args.ckpt_b, device)

    rows = lmc_sweep(model, sd_a, sd_b, eval_fn, val_loader, criterion, device,
                     steps=args.steps, train_loader=train_loader)

    out = write_curve_csv(args.out, rows, args.label_a, args.label_b)

    tl_bar = loss_barrier(rows, "train_loss")
    vl_bar = loss_barrier(rows, "val_loss")
    tpr_dip = auc_dip(rows, "train_auc_pr")
    vpr_dip = auc_dip(rows, "val_auc_pr")
    verd = verdict(rows, cross_init=args.cross_init, auc_dip_threshold=args.auc_dip_threshold)

    print(f"[lmc] wrote {out}")
    print(f"[lmc] {args.label_a}(λ=0): val_loss={rows[0]['val_loss']:.4f} "
          f"val_auc_pr={rows[0]['val_auc_pr']:.4f}  |  "
          f"{args.label_b}(λ=1): val_loss={rows[-1]['val_loss']:.4f} "
          f"val_auc_pr={rows[-1]['val_auc_pr']:.4f}")
    print(f"[lmc] BARRIER train_loss=+{tl_bar:.4f}  val_loss=+{vl_bar:.4f}  "
          f"(train_auc_pr dip {tpr_dip:.4f}, val_auc_pr dip {vpr_dip:.4f})")
    print(f"[lmc] VERDICT: {verd}")
    return out


def write_curve_csv(path: str, rows: List[Dict[str, float]],
                    label_a: str, label_b: str) -> Path:
    """Write the λ-curve to CSV (one row per λ), prefixed with the endpoint labels."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["label_a", "label_b", "lam",
                  "train_loss", "train_auc_pr", "train_auc_roc",
                  "val_loss", "val_auc_pr", "val_auc_roc"]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(dict(label_a=label_a, label_b=label_b, **r))
    return out


if __name__ == "__main__":
    main()

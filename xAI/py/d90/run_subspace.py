#!/usr/bin/env python3
"""
d90/run_subspace.py — ONE independent intrinsic-dimension subspace run (briefing Part II §9.1,
plan component C6).

Per condition, we measure the intrinsic dimension of fine-tuning *from that condition's init*:
the smallest random-subspace dimension ``d`` at which fine-tuning reaches 90% of the *gain*
of full fine-tuning. The framework prediction is that more task-aligned pretraining lowers
d90 (expected order plantclef(VT) < mae < naive). Because it depends only on the three init
checkpoints (which exist before any fine-tuning starts), each run here is embarrassingly
parallel and has NO data dependency on the live main block — it runs as a separate batch
before and/or after the main runs (never concurrent), see ``scripts/submit_d90.sh``.

It is a *downstream* d90: the input-encoding stage (patch_embed tokenizer + positional
embeddings) is the SAME frozen MAE stage for all conditions (``--shared-tokenizer mae``), so
the only thing the subspace confines is the *trainable* processing+readout weights.

Reparameterization (Li et al. 2018; Aghajanyan et al. 2021)
-----------------------------------------------------------
    theta = theta0 + P v
where
  * ``theta0`` is the condition's init over **trainable parameters only** (frozen input stage
    EXCLUDED — exactly as everywhere else in the experiment, via
    ``preadapt_common.trainable_named_parameters`` / ``trainable_layout``),
  * ``P`` is a fixed ``D_trainable x d`` projection drawn once per (condition, d, seed) from the
    seeded, matrix-free :class:`preadapt_common.SparseRandomProjection` (NEVER dense; the SAME
    machinery as extractor §6.2), and
  * ``v in R^d`` is trained from zero.

We train ONLY ``v``. Because ``SparseRandomProjection.matvec`` / ``rmatvec`` are matrix-free
operators (NOT autograd ops over ``v``), the gradient w.r.t. ``v`` is supplied MANUALLY:

    forward:   set the model's trainable params to  theta0 + P.matvec(v)   (no_grad write-back)
    backward:  loss.backward()  populates each TRAINABLE model param's ``.grad``
    chain rule: dL/dv = P^T (dL/dtheta) = P.rmatvec( concat(trainable grads in layout order) )
                => assign  v.grad = P.rmatvec(flat_trainable_grad)   then  opt.step()

This is the standard intrinsic-dimension reparam. It is exact: with theta(v) = theta0 + P v,
dtheta/dv = P, so dL/dv = (dtheta/dv)^T dL/dtheta = P^T g = rmatvec(g). The module's tiny
self-test (``--selftest``) asserts this manual gradient matches a finite-difference estimate
on a trivial MLP, and that the loss decreases.

Optimizer for ``v`` (briefing §9.1)
-----------------------------------
ONE fixed optimizer across ALL d90 runs and all conditions — that invariant matters more than
the choice. Adam is the literature default for this method (``v`` is a low-dimensional generic
vector where the main optimizer's curvature-adaptive machinery buys little). We use
``torch.optim.Adam([v], lr=...)`` with the SAME ``--lr`` for every run; do not vary it.

Stopping rule (briefing §9.1)
-----------------------------
    target = perf_init + 0.9 * (perf_full - perf_init)        # 90% of the GAIN, not absolute
Stop on whichever comes first:
  (a) held-out metric (``xai_train.validate -> auc_pr_mean``) >= target  -> ``criterion_met``;
  (b) plateau: no improvement > ``eps`` over ``patience`` consecutive evals;
  (c) step cap matched to the main run's budget (``--step_cap``).
``perf_full`` / ``perf_init`` are PER-CONDITION and come from that condition's **stable_evo
main run** (final / initial held-out val); pass them in via ``--perf_full`` / ``--perf_init``
(``gen_d90_grid.py`` injects them from a per-condition --perf-table).

Output
------
Appends one row ``(condition, d, seed, perf, steps, criterion_met, target, perf_init,
perf_full, variant, shared_tokenizer, stop_reason, wall_seconds)`` to a d90 table at ``--out``
(.parquet or .csv; both supported, append-safe).

Precision: bf16 autocast, NO GradScaler — consistent with the main runs (the fp16 GradScaler
corrupts StableEvo's generative preconditioner and would also be a precision confound here).

Conventions (xAI/two_noise/CONVENTIONS.md §5): every stochastic op threads an explicit
``torch.Generator``. The projection seed is the run's ``--seed`` (so replicates differ in P).
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# --- project paths (mirror preadapt_train.py / xai_train.py so PlantCLEF2022 + py modules
# import cleanly, and StableEvo's two_noise dir is importable for build_model's MAE helper). ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_PY_DIR = os.path.dirname(_HERE)                                  # .../xAI/py
_PROJECT_ROOT = os.path.join(_HERE, "..", "..", "..")             # .../phenovision
_PLANTCLEF = os.path.join(_PROJECT_ROOT, "PlantCLEF2022")
_TWO_NOISE = "/blue/guralnick/share/r.dinnage/Projects/phenovision/xAI/two_noise"
for _p in (_PY_DIR, _PLANTCLEF, _PROJECT_ROOT, _TWO_NOISE):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# =============================================================================
# CLI
# =============================================================================

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="One independent init-d90 intrinsic-dimension subspace run "
                    "(theta = theta0 + P v; train only v). Briefing Part II §9.1.")
    # --- identity / subspace ---
    # NOTE: the experiment-required args below are NOT argparse-``required`` so that
    # ``--selftest`` (CPU, no ViT-L / no data) can run standalone; ``run_subspace`` validates
    # their presence at call time instead.
    p.add_argument("--condition", default=None,
                   choices=["mae", "plantclef", "naive"],
                   help="starting 'evolutionary history': mae | plantclef (VT) | naive. "
                        "ImageNet is DROPPED from this experiment.")
    p.add_argument("--d", type=int, default=None,
                   help="intrinsic subspace dimension (P maps R^d -> R^D_trainable).")
    p.add_argument("--seed", type=int, default=42,
                   help="replicate seed; seeds the projection P (replicates differ in P).")
    p.add_argument("--variant", default="stable_evo", choices=["adamw", "stable_evo"],
                   help="which MAIN run the perf_full/perf_init targets came from (recorded "
                        "in the output row; the d90 'v'-optimizer is ALWAYS Adam regardless).")
    p.add_argument("--shared-tokenizer", dest="shared_tokenizer", default="mae",
                   choices=["imagenet", "mae", "none"],
                   help="frozen input stage shared across conditions (LOCKED to 'mae' for this "
                        "experiment; matches the main runs). Threaded into build_model.")
    p.add_argument("--init_ckpt", type=str, default=None,
                   help="optional path to this condition's init checkpoint (model_state_dict). "
                        "If unset, build a fresh init via preadapt_train.build_model (the "
                        "condition's pretrained start + MAE shared tokenizer).")

    # --- 90%-of-gain targets (PER CONDITION; from the condition's main run) ---
    p.add_argument("--perf_full", type=float, default=None,
                   help="converged held-out metric (auc_pr_mean) of this condition's MAIN run.")
    p.add_argument("--perf_init", type=float, default=None,
                   help="held-out metric (auc_pr_mean) at this condition's init (pre-finetune).")

    # --- backfill mode (idle-GPU work; run-to-convergence, compute d90 POST-HOC) -------------
    # The hybrid collector workers run d90 configs whenever live extraction is idle. At that
    # point the condition's MAIN run has usually NOT finished, so perf_full is unknown and the
    # 90%-of-gain target cannot be formed. --backfill therefore: (1) requires only condition/
    # d/out, (2) MEASURES perf_init by validating the init model at step 0, (3) skips the
    # criterion-a (perf>=target) early-stop, running purely to plateau/step_cap, and (4) records
    # the full (perf, steps, max-perf, plateau curve) so d90 thresholds are computed POST-HOC
    # once perf_full is known (target left null in the row). Convergence == the run's own
    # plateau, which is exactly what d90 needs to localize the intrinsic dimension.
    p.add_argument("--backfill", action="store_true",
                   help="idle-GPU mode: require only --condition/--d/--out; measure perf_init by "
                        "validating the init model; skip the 90%%-gain early-stop and run to "
                        "plateau/step_cap; compute d90 thresholds post-hoc once perf_full known.")

    # --- stopping rule ---
    p.add_argument("--step_cap", type=int, default=4000,
                   help="(c) hard step cap matched to the main run's budget.")
    p.add_argument("--patience", type=int, default=8,
                   help="(b) plateau patience: stop after this many evals with no >eps gain.")
    p.add_argument("--eps", type=float, default=1e-4,
                   help="(b) minimum held-out improvement counted as progress.")
    p.add_argument("--eval_every", type=int, default=50,
                   help="evaluate the held-out metric every N optimization steps.")

    # --- data ---
    p.add_argument("--train_csv", type=str, default="data/inat/train_v1.1.0.csv")
    p.add_argument("--val_csv", type=str, default="data/inat/val_v1.1.0.csv")
    p.add_argument("--batch_size", type=int, default=384,
                   help="train batch size (also the held-out eval batch size).")
    p.add_argument("--val_subset_size", type=int, default=10000,
                   help="held-out eval subset size (fixed by seed, like the main runs).")
    p.add_argument("--max_train_samples", type=int, default=None,
                   help="limit training data (short pilots / smoke).")
    p.add_argument("--num_workers", type=int, default=None)

    # --- v-optimizer (ONE fixed choice for ALL d90 runs) ---
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Adam lr for v; the SAME for every d90 run (do not vary).")

    # --- time-box + resume (HYBRID-worker backfill: yield the GPU back to collection) ----------
    # A d90 config can take 25 min-2 h on an L4 (step_cap x ~seconds/step). When run as idle-GPU
    # backfill by the hybrid collector workers, a single long blocking d90 call would hold a worker
    # off collection for that whole time — so if Phase 2 starts emitting while both workers are deep
    # in d90 runs, the watch dir fills to the backpressure high-water and the B200 TRAINER STALLS
    # until a d90 run finishes. --max_wall_seconds bounds ONE claim's training time: on hitting it
    # before convergence the run SAVES its state (v + optimizer + counters) under --state_dir and
    # exits with code 2 (INCOMPLETE), so the worker re-queues it to pending/ and returns to the
    # collect-first loop. The NEXT claim of that config RESUMES from the saved state (no perf_init
    # re-measure, no v reset), so progress accumulates across short boxes. Both default to off
    # (the array-based live d90 sweep runs each config to completion in one shot).
    p.add_argument("--max_wall_seconds", type=float, default=None,
                   help="per-claim training wall-time budget (s); on timeout SAVE state to "
                        "--state_dir and exit 2 (INCOMPLETE, resumable). Default: no limit.")
    p.add_argument("--state_dir", type=str, default=None,
                   help="dir for resumable per-config state (v/opt/counters), keyed "
                        "(condition,d,seed). Enables resume across time-boxed --max_wall_seconds "
                        "claims; deleted on completion. Default: no resume (single-shot).")

    # --- output ---
    p.add_argument("--out", type=str, default=None,
                   help="d90 table to append one row to (.parquet or .csv).")
    p.add_argument("--device", type=str, default=None,
                   help="torch device (default: cuda if available else cpu).")

    # --- self-test (no ViT-L / no data) ---
    p.add_argument("--selftest", action="store_true",
                   help="run the tiny intrinsic-dim manual-gradient sanity on an MLP and exit.")
    return p.parse_args(argv)


# =============================================================================
# Subspace plumbing (the manual theta = theta0 + P v reparam)
# =============================================================================

@torch.no_grad()
def write_subspace_params(
    model: torch.nn.Module,
    theta0: torch.Tensor,
    P,
    v: torch.Tensor,
    layout: List[Tuple[str, int, int]],
    param_by_name: Dict[str, torch.nn.Parameter],
) -> None:
    """Set the model's TRAINABLE params to ``theta0 + P.matvec(v)`` in-place (no autograd).

    ``layout`` is ``preadapt_common.trainable_layout(model)`` = ``(name, offset, numel)`` in
    the same concatenation order ``flatten_trainable`` uses, which is the coordinate
    convention ``P``'s ``D_trainable`` axis indexes. We expand the subspace coords once
    (``P.matvec(v)``, matrix-free), add ``theta0``, and copy each layout block into its
    parameter tensor. Done under ``no_grad`` so this assignment is NOT part of the autograd
    graph — the gradient path to ``v`` is supplied manually in the training loop.
    """
    pv = P.matvec(v)                          # R^d -> R^D_trainable (matrix-free)
    theta = theta0 + pv                        # full trainable vector
    for name, off, n in layout:
        p = param_by_name[name]
        p.copy_(theta[off:off + n].view_as(p))


def flat_trainable_grad(
    layout: List[Tuple[str, int, int]],
    param_by_name: Dict[str, torch.nn.Parameter],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Concatenate the TRAINABLE params' ``.grad`` in ``layout`` order into one
    ``D_trainable`` vector (the ``dL/dtheta`` that ``P.rmatvec`` maps to ``dL/dv``).

    A param with no grad this step contributes zeros (so the layout/offsets stay aligned).
    """
    total = layout[-1][1] + layout[-1][2] if layout else 0
    g = torch.zeros(total, device=device, dtype=dtype)
    for name, off, n in layout:
        p = param_by_name[name]
        if p.grad is not None:
            g[off:off + n] = p.grad.detach().reshape(-1).to(device=device, dtype=dtype)
    return g


# =============================================================================
# Model / data builders (lazy heavy imports kept out of module import)
# =============================================================================

def build_init_model(args: argparse.Namespace, device: torch.device):
    """Build the condition's init model EXACTLY like the main runs and freeze its input stage.

    Reuses ``preadapt_train.build_model`` (which itself reuses ``preadapt_models.init_model``
    + the MAE shared-tokenizer overwrite) so the start weights, the exact-control input stage,
    and the head-init RNG are identical to the main run. Then ``freeze_input_stage`` flips
    ``requires_grad=False`` on patch_embed.* + pos_embed so they are excluded from theta0 /
    the projection (the frozen stage contributes zero and must never enter any displacement /
    projection stat — briefing "Do not").

    If ``--init_ckpt`` is given, we load that model_state_dict on top of the freshly built
    model (same arch), so a saved Phase-2 init can be reused verbatim.
    """
    import preadapt_train as pt           # noqa: E402  (lazy; pulls torch+timm+PlantCLEF)
    import preadapt_models as pm          # noqa: E402

    # build_model reads args.condition / args.shared_tokenizer / args.seed.
    model = pt.build_model(args, device)
    if args.init_ckpt:
        ckpt = torch.load(args.init_ckpt, map_location=device, weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[d90] loaded init_ckpt={args.init_ckpt}: "
              f"{len(missing)} missing, {len(unexpected)} unexpected.")
    pm.freeze_input_stage(model)
    return model


def build_loaders(args: argparse.Namespace):
    """Build a train loader and a fixed held-out eval loader, reusing xai_train's data path
    (``load_csv_data`` + the project ``PhenoDataset`` + timm transforms). We build directly
    (not via ``create_data_loaders``) to avoid its subset-index file writes / output_dir
    dependency; the held-out subset is fixed by ``--seed`` exactly as the main runs do."""
    import timm                                          # noqa: E402
    from xai_train import load_csv_data                  # noqa: E402
    from py.custom_dataset import PhenoDataset           # noqa: E402

    train_files, train_labels = load_csv_data(args.train_csv)
    val_files, val_labels = load_csv_data(args.val_csv)

    if args.max_train_samples and args.max_train_samples < len(train_files):
        rng_train = np.random.RandomState(args.seed + 1)
        idx = rng_train.choice(len(train_files), args.max_train_samples, replace=False)
        train_files = [train_files[i] for i in idx]
        train_labels = train_labels[idx]

    rng = np.random.RandomState(args.seed)
    val_idx = rng.choice(len(val_files),
                         min(args.val_subset_size, len(val_files)), replace=False)
    val_sub_files = [val_files[i] for i in val_idx]
    val_sub_labels = val_labels[val_idx]

    nw = args.num_workers
    if nw is None:
        nw = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))

    transform_train = timm.data.create_transform(
        input_size=224, is_training=True,
        auto_augment="rand-m9-mstd0.5-inc1", re_prob=0.25, re_mode="pixel", re_count=1,
    )
    transform_val = timm.data.create_transform(input_size=224, is_training=False)

    train_ds = PhenoDataset(train_files, train_labels); train_ds.transform = transform_train
    val_ds = PhenoDataset(val_sub_files, val_sub_labels); val_ds.transform = transform_val

    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=nw, pin_memory=True, drop_last=True)
    val_dl = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=nw, pin_memory=True, drop_last=False)
    return train_dl, val_dl


# =============================================================================
# The subspace run
# =============================================================================

def _state_path(args: argparse.Namespace) -> Optional[str]:
    """Resumable-state file path for this (condition,d,seed), or None if --state_dir unset."""
    if not args.state_dir:
        return None
    os.makedirs(args.state_dir, exist_ok=True)
    return os.path.join(args.state_dir, f"{args.condition}_d{args.d}_s{args.seed}.pt")


def run_subspace(args: argparse.Namespace) -> Tuple[Dict[str, object], bool]:
    """Run one (condition, d, seed) subspace fine-tune; return ``(row, complete)``.

    Implements the briefing §9.1 runner: theta = theta0 + P v, train only v with a single fixed
    Adam, stop on target / plateau / step-cap. With --max_wall_seconds set, a claim that hits the
    wall before convergence SAVES resumable state (to --state_dir) and returns ``complete=False``
    (the caller re-queues it; the next claim resumes). ``complete=True`` means a terminal stop
    (criterion_met / plateau / step_cap / nonfinite); the row should be appended and state cleared.
    """
    import preadapt_common as pc                          # noqa: E402
    import torch.nn as nn                                 # noqa: E402
    from xai_train import validate                        # noqa: E402

    # Validate the experiment-required args here (they are not argparse-``required`` so that
    # ``--selftest`` can run standalone). In --backfill mode perf_full/perf_init are UNKNOWN at
    # launch (the main run hasn't converged), so only condition/d/out are required; perf_init is
    # measured below and the 90%-gain target is skipped.
    required = ("condition", "d", "out") if args.backfill \
        else ("condition", "d", "perf_full", "perf_init", "out")
    missing = [k for k in required if getattr(args, k) is None]
    if missing:
        hint = ("Generate a backfill config via xAI/scripts/gen_d90_backfill_queue.py."
                if args.backfill
                else "Generate a grid line via xAI/py/d90/gen_d90_grid.py.")
        raise SystemExit(
            f"run_subspace requires {missing} (omit only with --selftest). {hint}")

    device = torch.device(
        args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    print("=" * 60)
    print("d90 subspace run (theta = theta0 + P v)" + (" [BACKFILL]" if args.backfill else ""))
    print(f"  condition        : {args.condition}")
    print(f"  d                : {args.d}")
    print(f"  seed             : {args.seed}")
    print(f"  shared-tokenizer : {args.shared_tokenizer}")
    _pi = f"{args.perf_init:.4f}" if args.perf_init is not None else "<measure>"
    _pf = f"{args.perf_full:.4f}" if args.perf_full is not None else "<post-hoc>"
    print(f"  perf_init/full   : {_pi} / {_pf}")
    print(f"  device           : {device}")
    print("=" * 60)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # --- model (init + frozen input stage), data ---
    model = build_init_model(args, device)
    train_dl, val_dl = build_loaders(args)
    criterion = nn.BCEWithLogitsLoss().to(device)

    # --- theta0 over TRAINABLE params (frozen input stage EXCLUDED) + the projection P ---
    layout = pc.trainable_layout(model)                          # (name, offset, numel)
    param_by_name = {name: p for name, p in pc.trainable_named_parameters(model)}
    theta0 = pc.flatten_trainable(model).detach().to(device, torch.float32)
    D_trainable = theta0.numel()
    P = pc.SparseRandomProjection(
        D_trainable=D_trainable, d=args.d, seed=args.seed, method="achlioptas")
    print(f"  D_trainable      : {D_trainable:,}  ->  d={args.d}  (P: {P.to_meta()})")

    # --- v in R^d trained from zero; ONE fixed optimizer (Adam) for all d90 runs ---
    v = torch.zeros(args.d, device=device, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([v], lr=args.lr)

    # --- RESUME a time-boxed claim, if state exists (skips the perf_init re-measure + v reset) ----
    state_path = _state_path(args)
    resumed = False
    if state_path and os.path.exists(state_path):
        try:
            st = torch.load(state_path, map_location=device, weights_only=False)
            with torch.no_grad():
                v.copy_(st["v"].to(device=device, dtype=torch.float32))
            opt.load_state_dict(st["opt"])
            perf_init = st["perf_init"]; best = st["best"]; since_improve = st["since_improve"]
            perf = st["perf"]; step = st["step"]
            resumed = True
            print(f"  [resume] state from {state_path}: step={step} best={best:.4f} "
                  f"since_improve={since_improve} perf_init={perf_init:.4f}")
        except Exception as e:  # corrupt/incompatible state -> start fresh (do not lose the config)
            print(f"  [resume] WARNING: could not load {state_path} ({e}); starting fresh.")

    # --- perf_init: in --backfill MEASURE it (validate init model at v=0); else use the arg.
    #     On resume it is already loaded from state, so skip the (expensive) measurement. ---
    if not resumed:
        if args.backfill and args.perf_init is None:
            write_subspace_params(model, theta0, P, v, layout, param_by_name)  # v==0 -> theta0
            model.eval()
            perf_init = float(validate(model, val_dl, criterion, device)["auc_pr_mean"])
            model.train()
            print(f"  perf_init (measured at v=0): {perf_init:.4f}")
        else:
            perf_init = args.perf_init
        best = perf_init
        since_improve = 0
        perf = perf_init
        step = 0

    # --- target: only formed when perf_full is known. In --backfill it is computed POST-HOC, so
    #     target=None disables criterion-a (perf>=target) and the run goes purely to plateau/cap. ---
    if args.backfill or args.perf_full is None:
        target = None
        print("  target (90%-gain): <post-hoc; running to plateau/step_cap>")
    else:
        target = perf_init + 0.9 * (args.perf_full - perf_init)
        print(f"  target (90%-gain): {target:.4f}")

    criterion_met = False
    stop_reason = "step_cap"
    # Per-claim training wall-time budget (s); measured from NOW (build/load/perf_init excluded).
    t0 = time.time()
    deadline = (t0 + args.max_wall_seconds) if args.max_wall_seconds else None

    train_iter = iter(train_dl)
    model.train()
    while step < args.step_cap:
        # Per-claim time-box: yield the GPU back to collection (resumable). Checked at the top of
        # the step so overshoot is <= one step; perf/best/step carry the last-eval state we save.
        if deadline is not None and time.time() >= deadline:
            stop_reason = "wall_timeout"
            break
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dl)
            batch = next(train_iter)
        samples = batch[0].to(device, non_blocking=True)
        targets = batch[-1].to(device, non_blocking=True)

        # forward at theta = theta0 + P v (param write-back is no_grad; grad path is manual).
        write_subspace_params(model, theta0, P, v, layout, param_by_name)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            outputs = model(samples)
            loss = criterion(outputs, targets)
        loss_value = float(loss.item())
        if not math.isfinite(loss_value):
            print(f"[d90] non-finite loss {loss_value} at step {step}; stopping.")
            stop_reason = "nonfinite"
            break

        # backward populates model param .grad; manually chain to v: dL/dv = P^T (dL/dtheta).
        model.zero_grad(set_to_none=True)
        if v.grad is not None:
            v.grad = None
        loss.backward()
        g = flat_trainable_grad(layout, param_by_name, device, torch.float32)
        v.grad = P.rmatvec(g)                              # R^D_trainable -> R^d (matrix-free)
        opt.step()
        step += 1

        if step % args.eval_every == 0:
            # eval at the just-updated v.
            write_subspace_params(model, theta0, P, v, layout, param_by_name)
            vm = validate(model, val_dl, criterion, device)
            perf = float(vm["auc_pr_mean"])
            model.train()
            elapsed = time.time() - t0
            _tgt = f"{target:.4f}" if target is not None else "<post-hoc>"
            print(f"  step={step} loss={loss_value:.4f} auc_pr_mean={perf:.4f} "
                  f"best={best:.4f} target={_tgt} ({elapsed/60:.1f} min)")
            if target is not None and perf >= target:
                criterion_met = True
                stop_reason = "criterion_met"
                break
            if perf > best + args.eps:
                best, since_improve = perf, 0
            else:
                since_improve += 1
            if since_improve >= args.patience:
                stop_reason = "plateau"
                break

    final_perf = max(perf, best)
    # criterion_met only meaningful when a target exists (live grid). In --backfill it stays
    # False here and is decided POST-HOC once perf_full is known.
    criterion_met = bool(criterion_met or (target is not None and final_perf >= target))
    wall = time.time() - t0
    row = {
        "condition": args.condition,
        "d": int(args.d),
        "seed": int(args.seed),
        "perf": float(final_perf),
        "steps": int(step),
        "criterion_met": bool(criterion_met),
        "target": float(target) if target is not None else None,
        "perf_init": float(perf_init) if perf_init is not None else None,
        "perf_full": float(args.perf_full) if args.perf_full is not None else None,
        "variant": args.variant,
        "shared_tokenizer": args.shared_tokenizer,
        "stop_reason": stop_reason,
        "wall_seconds": float(wall),
    }
    # complete == terminal stop (criterion_met / plateau / step_cap / nonfinite). wall_timeout is
    # the ONLY non-terminal stop: save resumable state and report incomplete so the caller re-queues.
    complete = stop_reason != "wall_timeout"
    if not complete:
        if state_path is None:
            # No --state_dir but a wall budget hit: cannot resume -> treat as terminal (append row).
            print("[d90] wall_timeout but no --state_dir: cannot resume; recording as terminal.")
            complete = True
        else:
            torch.save(
                {"v": v.detach().cpu(), "opt": opt.state_dict(), "step": int(step),
                 "best": float(best), "since_improve": int(since_improve), "perf": float(perf),
                 "perf_init": float(perf_init)},
                state_path)
            print(f"[d90] YIELD condition={args.condition} d={args.d} seed={args.seed} "
                  f"step={step} best={best:.4f} -> state saved {state_path} ({wall/60:.1f} min "
                  f"this claim); re-queue to resume.")
            return row, False

    # Terminal: clear any resume state so a re-run starts fresh, then report DONE.
    if state_path and os.path.exists(state_path):
        try:
            os.remove(state_path)
        except OSError:
            pass
    print(f"[d90] DONE condition={args.condition} d={args.d} seed={args.seed} "
          f"perf={final_perf:.4f} criterion_met={criterion_met} steps={step} "
          f"reason={stop_reason} ({wall/60:.1f} min)")
    return row, True


# =============================================================================
# Append-safe d90 table writer
# =============================================================================

#: Column order for the d90 table (briefing §9.1 logging: condition,d,seed,perf,steps,
#: criterion_met + the targets / provenance that make each row self-describing).
D90_COLUMNS: Tuple[str, ...] = (
    "condition", "d", "seed", "perf", "steps", "criterion_met", "target",
    "perf_init", "perf_full", "variant", "shared_tokenizer", "stop_reason", "wall_seconds",
)


def append_row(out_path: str, row: Dict[str, object]) -> None:
    """Append one result ``row`` to the d90 table at ``out_path`` (.parquet or .csv).

    Append-safe and concurrency-tolerant for many independent array tasks: each call reads
    any existing rows, concatenates, and rewrites via a temp file + atomic rename. (At the
    d90 sweep's scale — O(120) rows — the read-modify-write cost is negligible, and parquet/
    csv have no native append; the atomic rename avoids a torn file. If two array tasks race
    the rewrite, the SLURM ``%3`` throttle plus distinct ``(condition,d,seed)`` keys make a
    lost row unlikely; a post-hoc de-dup on the key column is the safety net.)"""
    import pandas as pd                                   # noqa: E402

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    new = pd.DataFrame([{c: row.get(c) for c in D90_COLUMNS}])
    is_parquet = out_path.endswith(".parquet")
    if os.path.exists(out_path):
        try:
            old = pd.read_parquet(out_path) if is_parquet else pd.read_csv(out_path)
            df = pd.concat([old, new], ignore_index=True)
        except Exception:
            df = new
    else:
        df = new
    tmp = out_path + ".tmp"
    if is_parquet:
        df.to_parquet(tmp, index=False)
    else:
        df.to_csv(tmp, index=False)
    os.replace(tmp, out_path)
    print(f"[d90] appended row -> {out_path} (now {len(df)} rows)")


# =============================================================================
# Tiny intrinsic-dim self-test (CPU; no ViT-L, no data) — proves the manual grad path
# =============================================================================

def _selftest() -> None:
    """Construct a 2-layer MLP with a fake frozen 'input stage' the helpers skip, build
    theta0 + P (d=8), train v ~20 steps of the manual matvec/rmatvec loop on a trivial
    regression loss, and ASSERT (1) the loss decreases and (2) v.grad == P.rmatvec(param_grad)
    numerically (direct check + a finite-difference check of dL/dv against P).
    """
    import torch.nn as nn
    import preadapt_common as pc

    torch.manual_seed(0)

    class _Toy(nn.Module):
        """A tiny model whose 'input stage' (patch_embed.* / pos_embed) is excluded by
        ``trainable_named_parameters`` exactly as for the real ViT-L."""

        def __init__(self):
            super().__init__()
            # Fake frozen input stage — must be SKIPPED by the trainable selection.
            self.patch_embed = nn.Module()
            self.patch_embed.proj = nn.Linear(4, 6)
            self.pos_embed = nn.Parameter(torch.randn(1, 3, 6) * 0.1)
            # Trainable processing + readout.
            self.cls_token = nn.Parameter(torch.zeros(1, 1, 6))
            self.fc1 = nn.Linear(6, 6)
            self.head = nn.Linear(6, 2)

        def forward(self, x):
            h = torch.tanh(self.fc1(self.patch_embed.proj(x)) + self.cls_token.view(1, 6))
            return self.head(h)

    model = _Toy()
    # Freeze the fake input stage so trainable_named_parameters' requires_grad filter matches
    # the real freeze discipline (it also excludes by NAME, but freeze for full fidelity).
    for n, p in model.named_parameters():
        if pc.is_frozen_input_stage(n):
            p.requires_grad_(False)

    device = torch.device("cpu")
    layout = pc.trainable_layout(model)
    param_by_name = {name: p for name, p in pc.trainable_named_parameters(model)}
    names = [n for n, _ in pc.trainable_named_parameters(model)]
    assert not any(n.startswith("patch_embed.") for n in names), names
    assert "pos_embed" not in names, names
    assert "cls_token" in names and "head.weight" in names and "fc1.weight" in names, names

    theta0 = pc.flatten_trainable(model).detach().to(device, torch.float32)
    D = theta0.numel()
    d = 8
    P = pc.SparseRandomProjection(D_trainable=D, d=d, seed=123, method="achlioptas")

    # Trivial fixed regression problem.
    X = torch.randn(16, 4)
    Y = torch.randn(16, 2)
    crit = nn.MSELoss()

    v = torch.zeros(d, requires_grad=True)
    opt = torch.optim.Adam([v], lr=0.05)

    def forward_loss():
        write_subspace_params(model, theta0, P, v, layout, param_by_name)
        return crit(model(X), Y)

    losses = []
    for step in range(20):
        model.zero_grad(set_to_none=True)
        v.grad = None
        loss = forward_loss()
        loss.backward()
        g = flat_trainable_grad(layout, param_by_name, device, torch.float32)
        manual_v_grad = P.rmatvec(g)
        # (1) the manual v.grad IS exactly P.rmatvec(flat param grad) — that is how we set it.
        v.grad = manual_v_grad.clone()

        if step == 0:
            # (2a) DIRECT identity: v.grad == P^T g (definitional, but assert it holds bytewise).
            assert torch.allclose(v.grad, P.rmatvec(g), rtol=1e-5, atol=1e-6), \
                "v.grad != P.rmatvec(param_grad)"
            # (2b) FINITE-DIFFERENCE check that P^T g really is dL/dv (chain rule correctness).
            #      Perturb v along a few random directions; compare central difference to
            #      <manual_v_grad, direction>.
            with torch.no_grad():
                base = float(forward_loss().item())
                gen = torch.Generator().manual_seed(7)
                ok = 0
                for _ in range(4):
                    e = torch.randn(d, generator=gen)
                    e = e / e.norm()
                    h = 1e-3
                    v.add_(h * e)
                    lp = float(forward_loss().item())
                    v.add_(-2 * h * e)
                    lm = float(forward_loss().item())
                    v.add_(h * e)  # restore
                    fd = (lp - lm) / (2 * h)
                    analytic = float(torch.dot(manual_v_grad, e).item())
                    rel = abs(fd - analytic) / (abs(fd) + abs(analytic) + 1e-8)
                    assert rel < 1e-2, f"FD mismatch: fd={fd:.6g} analytic={analytic:.6g} rel={rel:.3g}"
                    ok += 1
                # restore the post-step-0 loss path (re-eval to set params back to current v).
                _ = forward_loss()
                print(f"  [selftest] finite-difference chain-rule check passed ({ok}/4 dirs).")

        opt.step()
        losses.append(float(loss.item()))

    assert losses[-1] < losses[0], f"loss did not decrease: {losses[0]:.4g} -> {losses[-1]:.4g}"
    print(f"  [selftest] loss decreased {losses[0]:.4g} -> {losses[-1]:.4g} over 20 steps.")
    print("  [selftest] manual matvec/rmatvec intrinsic-dim gradient path verified.")
    print("All run_subspace self-tests passed.")


# =============================================================================
# Main
# =============================================================================

def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    if args.selftest:
        _selftest()
        return
    row, complete = run_subspace(args)
    if complete:
        append_row(args.out, row)
        sys.exit(0)
    # Time-boxed (wall_timeout): state saved, no row appended. Exit 2 so the hybrid worker
    # re-queues this config to pending/ (resume on the next claim) rather than marking it done.
    print("[d90] INCOMPLETE (wall_timeout): state saved; exit 2 -> caller should re-queue.")
    sys.exit(2)


if __name__ == "__main__":
    main()

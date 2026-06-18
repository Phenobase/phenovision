#!/usr/bin/env python3
"""
block_probes.py — extractor block §6.7: per-patch + per-layer linear probes.

Per the briefing (``xAI/notes/phenovision_full_briefing.md`` Part II §6.7). Two related
linear-decodability probes on the **fixed probe set** with a **fixed probe-train / probe-eval
split** (§4 invariant 1 — same images, same order, every checkpoint, every condition; the
split never leaks into the fine-tuning data). They answer "what does each patch mean and
where does task content live", which the attention heatmap (block_interp §6.6) does not.

  1. **Per-layer image-level linear probe.** The CLS token at block ``l`` -> the image class
     set (multilabel {fruit, flower}). Closed-form ridge regression fit on probe-train CLS
     features, evaluated on probe-eval. Reported as ``probe_acc`` (per layer ``l``, head=None,
     mean over the two binary traits) plus the full accuracy-vs-depth curve to the array store
     (``probe_acc_vs_depth``). Shows where task content lives and how it migrates / sharpens
     over training.

  2. **Per-patch -> image-level class probe.** Each patch token at block ``l`` -> the image
     class set, giving (a) a per-patch class map = coarse 14x14 segmentation at patch
     resolution showing which patches carry the class signal (for phenology, whether
     flower/fruit patches light up — a biologically meaningful localization readout), stored
     for a TINY fixed image subset to the array store (``patch_class_maps``; heavy, so
     sparse); and (b) the **emergence depth** = the first block at which patch tokens become
     linearly class-predictive (per-patch eval accuracy crosses a threshold above chance),
     and how that depth shifts over training as the localization computation forms. Scalar
     ``emergence_depth`` (global) plus per-layer ``patch_probe_acc`` and the
     ``patch_acc_vs_depth`` array.

CAVEATS (do not overclaim — briefing §6.7). **Linear decodability is representational
*content*, not proof the network causally uses it (content, not mechanism).** A class being
linearly readable off a token says only that the information is present in that
representation, not that any downstream computation reads it that way. **Patch-resolution
segmentation is coarse** (14x14 for ViT-L/16 @ 224 — 16px patches), so the per-patch class
maps are a coarse localization, not fine segmentation. Both caveats are stated here and must
be carried into any figure caption.

Multilabel handling: labels are multilabel {fruit, flower}; we treat each as an independent
binary probe and report the mean (and store per-class where useful), which is multilabel-safe
(never a softmax over mutually-exclusive classes).

Feature capture: per-layer token features ``[B, N, D]`` (token 0 = CLS, 1..196 = patches on a
14x14 grid) are captured by forward hooks on each ``blocks.{i}`` output in a single no-grad
forward pass over the probe loaders. The fine-tuned classifier head is never touched; the
probes are independent linear maps fit on the captured features.

Robustness / RNG: this block reads model weights and runs no-grad forward passes only; it
does not depend on the optimizer variant (``ctx.opt`` is untouched), so it is identical for
``adamw`` and ``stable_evo`` (nothing optimizer-specific to degrade). The ridge fit is
deterministic (closed form). Any subsampling (the tiny per-patch-map subset, the probe
example cap) is deterministic; we still resolve and thread a per-checkpoint
``torch.Generator`` from ``ctx`` / ``(seed, step)`` for consistency and never touch global
RNG (CONVENTIONS.md §5).
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# Project paths so preadapt_common / xai_train import cleanly when run from anywhere.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PY = os.path.dirname(_HERE)                       # xAI/py
_PROJECT_ROOT = os.path.join(_PY, "..", "..")      # repo root
_PLANTCLEF = os.path.join(_PROJECT_ROOT, "PlantCLEF2022")
for _p in (_PY, _PROJECT_ROOT, _PLANTCLEF):
    _ap = os.path.abspath(_p)
    if _ap not in sys.path:
        sys.path.insert(0, _ap)

if TYPE_CHECKING:  # the dataclass is defined by the collector/driver; import lazily.
    from extractor._ctx import ExtractCtx  # noqa: F401


# ViT-L/16 architecture facts (verified against the checkpoints).
N_BLOCKS = 24
D_MODEL = 1024
GRID = 14              # 14x14 patch grid (224 / 16)
N_PATCH = GRID * GRID  # 196
# Token layout: index 0 = CLS, indices 1..196 = patches in row-major grid order.

# Reproductive head order matches xai_train.validate: index 0 = fruit, index 1 = flower.
_CLASS_NAMES = ("fruit", "flower")
N_CLASSES = len(_CLASS_NAMES)

# Cost knobs (tunable by the collector for the §2.3 utilization budget) ------------------
PROBE_MAX_TRAIN = 2048   # cap probe-train examples feeding the ridge fit (closed form is
                         #   cheap, but bound the feature memory: 24 layers x N x 1024)
PROBE_MAX_EVAL = 2048    # cap probe-eval examples
RIDGE_LAMBDA = 1.0       # ridge regularization (added to the normalized Gram diagonal)
PATCH_MAP_SUBSET = 8     # tiny fixed image subset for stored per-patch class maps (heavy)
# Per-patch probe trains on a subsample of patches per image to bound cost; eval uses all.
PATCH_TRAIN_SUBSAMPLE = 16   # patches sampled per image for the per-patch ridge fit
# Emergence-depth threshold: a layer counts as "class-predictive" once per-patch eval
# accuracy exceeds chance by this absolute margin. Chance for a balanced binary is 0.5; we
# use the per-class positive rate as the empirical chance baseline (multilabel-safe).
EMERGENCE_MARGIN = 0.05


# =============================================================================
# Per-layer token-feature capture (forward hooks on blocks.{i} outputs)
# =============================================================================

class _BlockTokenCapture:
    """Registers a forward hook on every ``blocks.{i}`` so a single forward pass yields,
    per block index, the full token sequence ``[B, N, D]`` (token 0 = CLS, 1..196 = patches).

    Used as a context manager; :attr:`feats` holds ``{block_idx: tensor[B, N, D]}`` after a
    forward pass. Hooks capture the block OUTPUT (the residual-stream state after block i),
    which is exactly the per-layer token representation the briefing probes.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.feats: Dict[int, torch.Tensor] = {}
        for i, blk in enumerate(model.blocks):
            self.handles.append(blk.register_forward_hook(self._make_hook(i)))

    def _make_hook(self, block_idx: int):
        def hook(module: nn.Module, inputs, output) -> None:
            # block output is the token sequence [B, N, D]; detach + cpu/float to free GPU.
            out = output[0] if isinstance(output, (tuple, list)) else output
            self.feats[block_idx] = out.detach().float().cpu()
        return hook

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def __enter__(self) -> "_BlockTokenCapture":
        return self

    def __exit__(self, *exc) -> None:
        self.remove()


@torch.no_grad()
def _collect_layer_features(model: nn.Module, loader, device, max_examples: int
                            ) -> Tuple[Dict[int, torch.Tensor], torch.Tensor]:
    """Run the probe ``loader`` (capped at ``max_examples``) and collect, per block, the
    token features ``[n, N, D]`` plus the multilabel targets ``[n, C]``.

    Returns ``(layer_feats, targets)`` where ``layer_feats[l]`` is ``[n, N, D]`` (CPU float32)
    and ``targets`` is ``[n, C]`` (CPU float32). One forward pass per batch; hooks capture all
    24 layers simultaneously (no per-layer recompute)."""
    model.eval()
    per_layer: Dict[int, List[torch.Tensor]] = {i: [] for i in range(len(model.blocks))}
    tgt_list: List[torch.Tensor] = []
    n_seen = 0
    with _BlockTokenCapture(model) as cap:
        for batch in loader:
            images = batch[0].to(device, non_blocking=True)
            targets = batch[-1]
            take = images.shape[0]
            if n_seen + take > max_examples:
                take = max_examples - n_seen
                images = images[:take]
                targets = targets[:take]
            cap.feats.clear()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _ = model(images)
            for l, t in cap.feats.items():
                per_layer[l].append(t[:take])
            tgt_list.append(targets.detach().float().cpu()[:take])
            n_seen += take
            if n_seen >= max_examples:
                break
    layer_feats = {l: (torch.cat(v, dim=0) if v else torch.empty(0, N_PATCH + 1, D_MODEL))
                   for l, v in per_layer.items()}
    targets = torch.cat(tgt_list, dim=0) if tgt_list else torch.empty(0, N_CLASSES)
    return layer_feats, targets


# =============================================================================
# Closed-form ridge linear probe (multilabel-safe: independent per-class regression)
# =============================================================================

def _ridge_fit(X: torch.Tensor, Y: torch.Tensor, lam: float) -> torch.Tensor:
    """Closed-form ridge regression with a bias column.

    ``X`` is ``[n, d]`` (features), ``Y`` is ``[n, C]`` (multilabel targets in {0,1}). Returns
    weights ``W`` of shape ``[d+1, C]`` mapping ``[X | 1] -> Y``. Each of the C columns is an
    independent binary ridge regressor (multilabel-safe — no softmax coupling). Solved on
    CPU/float64 for numerical safety on the small (d=1024) Gram system.
    """
    Xd = X.to(torch.float64)
    Yd = Y.to(torch.float64)
    n, d = Xd.shape
    ones = torch.ones(n, 1, dtype=torch.float64)
    Xa = torch.cat([Xd, ones], dim=1)          # [n, d+1]
    G = Xa.t() @ Xa                            # [d+1, d+1]
    # Regularize the feature dims (not the bias) — scale lambda by the mean Gram diagonal so
    # it is invariant to feature magnitude across layers/checkpoints.
    diag_mean = float(torch.diagonal(G)[:d].mean().item()) if d > 0 else 1.0
    reg = lam * max(diag_mean, 1e-8)
    eye = torch.eye(d + 1, dtype=torch.float64)
    eye[d, d] = 0.0                            # do not regularize the bias term
    A = G + reg * eye
    b = Xa.t() @ Yd                            # [d+1, C]
    try:
        W = torch.linalg.solve(A, b)
    except Exception:
        W = torch.linalg.lstsq(A, b).solution
    return W


def _ridge_predict(X: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """Apply a ridge probe ``W`` ([d+1, C]) to features ``X`` ([n, d]) -> scores ``[n, C]``."""
    n = X.shape[0]
    Xa = torch.cat([X.to(torch.float64), torch.ones(n, 1, dtype=torch.float64)], dim=1)
    return Xa @ W


def _multilabel_accuracy(scores: torch.Tensor, Y: torch.Tensor) -> Tuple[float, np.ndarray]:
    """Per-class binary accuracy at threshold 0.5 on ridge scores, and their mean.

    Returns ``(mean_acc, per_class_acc[C])``. Multilabel-safe (each class thresholded
    independently). Empty inputs -> ``(nan, [])``."""
    if scores.numel() == 0 or Y.numel() == 0:
        return float("nan"), np.zeros(0, dtype=np.float64)
    pred = (scores >= 0.5).to(torch.float64)
    correct = (pred == Y.to(torch.float64)).to(torch.float64)
    per_class = correct.mean(dim=0).cpu().numpy()      # [C]
    return float(per_class.mean()), per_class


# =============================================================================
# Main entry point
# =============================================================================

def _resolve_generator(ctx, device: torch.device) -> torch.Generator:
    """Return an explicit ``torch.Generator`` on ``device`` (never global RNG). Prefer a
    driver-supplied ``ctx.gen_probes``; otherwise derive deterministically from
    ``(ctx.seed, ctx.step)`` so any subsampling is reproducible per checkpoint
    (CONVENTIONS.md §5)."""
    g = getattr(ctx, "gen_probes", None)
    if isinstance(g, torch.Generator):
        if g.device == device:
            return g
        seed = int(torch.randint(0, 2 ** 31 - 1, (1,), generator=g).item())
        dg = torch.Generator(device=device)
        dg.manual_seed(seed)
        return dg
    seed = (int(getattr(ctx, "seed", 0)) * 1_000_003 + int(getattr(ctx, "step", 0))) & 0x7FFFFFFF
    dg = torch.Generator(device=device)
    dg.manual_seed(seed)
    return dg


def extract(ctx: "ExtractCtx") -> Dict[str, object]:
    """Run §6.7 per-layer + per-patch linear-probe extraction on the fixed probe split.

    Writes scalars: ``probe_acc`` (per layer, head=None) — per-layer CLS image-probe
    accuracy; ``patch_probe_acc`` (per layer, head=None) — per-layer per-patch probe
    accuracy; per-class ``probe_acc_<class>`` / ``patch_probe_acc_<class>`` (per layer);
    ``emergence_depth`` (global) — first block where patch tokens become class-predictive.
    Writes arrays: ``probe_acc_vs_depth`` / ``patch_acc_vs_depth`` (the [N_BLOCKS] curves),
    and ``patch_class_maps`` (the tiny fixed-subset 14x14 per-patch score maps, per layer).

    Returns a compact summary dict for the collector log. Robust to missing probe loaders
    (returns an empty summary); identical for both optimizer variants (no optimizer state is
    read). See module docstring for the linear-decodability and patch-resolution caveats."""
    model = ctx.model
    device = ctx.device
    c, r, step, wt = ctx.condition, ctx.run_id, ctx.step, ctx.wall_time

    train_loader = getattr(ctx, "probe_train_loader", None)
    eval_loader = getattr(ctx, "probe_eval_loader", None)
    if train_loader is None or eval_loader is None:
        return {"block": "probes", "skipped": "missing probe_train/eval loader"}

    gen = _resolve_generator(ctx, torch.device("cpu"))
    model.eval()
    summary: Dict[str, object] = {"block": "probes"}

    # ---------------------------------------------------------------------
    # (0) Capture per-layer token features on the fixed probe-train/eval splits.
    # ---------------------------------------------------------------------
    try:
        tr_feats, tr_y = _collect_layer_features(model, train_loader, device, PROBE_MAX_TRAIN)
        ev_feats, ev_y = _collect_layer_features(model, eval_loader, device, PROBE_MAX_EVAL)
    except Exception as e:
        print(f"[block_probes] feature capture failed at run_id={r} step={step}: {e!r}",
              file=sys.stderr, flush=True)
        return {"block": "probes", "error": repr(e)}

    if tr_y.numel() == 0 or ev_y.numel() == 0:
        return {"block": "probes", "skipped": "empty probe features"}

    n_cls = int(tr_y.shape[1])
    cls_names = _CLASS_NAMES if n_cls == N_CLASSES else tuple(f"c{i}" for i in range(n_cls))

    # Empirical per-class chance baseline (positive rate) on eval — the multilabel-safe
    # "chance" the emergence-depth margin is measured against.
    chance = ev_y.to(torch.float64).mean(dim=0).cpu().numpy()           # [C] positive rate
    base_acc = np.maximum(chance, 1.0 - chance)                        # majority-class acc

    # ---------------------------------------------------------------------
    # (1) Per-layer CLS image-level linear probe (accuracy vs depth).
    # ---------------------------------------------------------------------
    probe_acc_curve = np.full((N_BLOCKS,), np.nan, dtype=np.float32)
    for l in range(N_BLOCKS):
        Xtr = tr_feats[l][:, 0, :]   # CLS token [n, D]
        Xev = ev_feats[l][:, 0, :]
        if Xtr.numel() == 0 or Xev.numel() == 0:
            continue
        W = _ridge_fit(Xtr, tr_y, RIDGE_LAMBDA)
        scores = _ridge_predict(Xev, W)
        mean_acc, per_class = _multilabel_accuracy(scores, ev_y)
        if np.isfinite(mean_acc):
            ctx.scalar.add(c, r, step, wt, quantity="probe_acc", value=float(mean_acc),
                           layer=f"blocks.{l}", head=None)
            probe_acc_curve[l] = mean_acc
            for i in range(min(n_cls, len(per_class))):
                ctx.scalar.add(c, r, step, wt, quantity=f"probe_acc_{cls_names[i]}",
                               value=float(per_class[i]), layer=f"blocks.{l}", head=None)
    ctx.array.put(group="probe_acc_vs_depth", step=step, array=probe_acc_curve, dtype=np.float16)
    finite_probe = probe_acc_curve[np.isfinite(probe_acc_curve)]
    summary["probe_acc_max"] = float(finite_probe.max()) if finite_probe.size else float("nan")

    # ---------------------------------------------------------------------
    # (2) Per-patch -> image-level class probe (per-patch class maps + emergence depth).
    #     Each patch token inherits the IMAGE label; we fit one ridge probe per layer on a
    #     subsample of patch tokens (bounding cost) and evaluate on all patch tokens.
    # ---------------------------------------------------------------------
    patch_acc_curve = np.full((N_BLOCKS,), np.nan, dtype=np.float32)
    emergence_depth = -1
    n_tr = tr_feats[0].shape[0] if tr_feats else 0

    # Pre-sample which patches feed the per-patch ridge fit (deterministic via gen).
    if n_tr > 0:
        sub = min(PATCH_TRAIN_SUBSAMPLE, N_PATCH)
        patch_idx = torch.randperm(N_PATCH, generator=gen)[:sub]      # 0..195 (patch space)
    else:
        patch_idx = torch.arange(0, min(PATCH_TRAIN_SUBSAMPLE, N_PATCH))

    for l in range(N_BLOCKS):
        Ptr = tr_feats[l][:, 1:, :]   # patch tokens [n, P, D]
        Pev = ev_feats[l][:, 1:, :]
        if Ptr.numel() == 0 or Pev.numel() == 0:
            continue
        n_tr_l, P, D = Ptr.shape
        # Build per-patch training matrix from the sampled patches; each patch token gets the
        # image's multilabel target broadcast across the sampled patches.
        Xtr = Ptr[:, patch_idx, :].reshape(-1, D)                    # [n*sub, D]
        Ytr = tr_y.unsqueeze(1).expand(-1, patch_idx.numel(), -1).reshape(-1, n_cls)
        W = _ridge_fit(Xtr, Ytr, RIDGE_LAMBDA)
        # Evaluate per-patch accuracy: every patch token in eval gets the image label.
        n_ev_l = Pev.shape[0]
        Xev = Pev.reshape(-1, D)                                      # [n_ev*P, D]
        Yev = ev_y.unsqueeze(1).expand(-1, P, -1).reshape(-1, n_cls)
        scores = _ridge_predict(Xev, W)
        mean_acc, per_class = _multilabel_accuracy(scores, Yev)
        if np.isfinite(mean_acc):
            ctx.scalar.add(c, r, step, wt, quantity="patch_probe_acc", value=float(mean_acc),
                           layer=f"blocks.{l}", head=None)
            patch_acc_curve[l] = mean_acc
            for i in range(min(n_cls, len(per_class))):
                ctx.scalar.add(c, r, step, wt, quantity=f"patch_probe_acc_{cls_names[i]}",
                               value=float(per_class[i]), layer=f"blocks.{l}", head=None)
            # emergence depth: first layer whose per-patch accuracy clears the chance margin
            if emergence_depth < 0 and mean_acc >= float(base_acc.mean()) + EMERGENCE_MARGIN:
                emergence_depth = l

    ctx.array.put(group="patch_acc_vs_depth", step=step, array=patch_acc_curve, dtype=np.float16)
    if emergence_depth >= 0:
        ctx.scalar.add(c, r, step, wt, quantity="emergence_depth",
                       value=float(emergence_depth))
        summary["emergence_depth"] = int(emergence_depth)
    else:
        # No layer became class-predictive on patches — record N_BLOCKS as a "never" sentinel
        # so the time series has a value at every checkpoint (the trajectory of this scalar is
        # the §6.7 "how emergence-depth shifts over training" signal).
        ctx.scalar.add(c, r, step, wt, quantity="emergence_depth", value=float(N_BLOCKS))
        summary["emergence_depth"] = N_BLOCKS

    # ---------------------------------------------------------------------
    # (3) Sparse per-patch class maps on a TINY fixed image subset (heavy -> store sparsely).
    #     For each subset image and each layer, the per-patch probe SCORE reshaped to 14x14
    #     per class -> array [PATCH_MAP_SUBSET, C, GRID, GRID], one slice per layer. We refit a
    #     full-patch ridge per layer on the train patches (same closed form) to score the
    #     subset. The slice shape is FIXED at [PATCH_MAP_SUBSET, n_cls, GRID, GRID] every
    #     checkpoint (NaN-padded when fewer eval images are available) so the Zarr step axis
    #     appends cleanly across checkpoints — the array store requires a constant slice shape
    #     per (group, layer) key (mirrors block_optim's fixed-width _topk_array convention).
    # ---------------------------------------------------------------------
    n_avail = ev_feats[0].shape[0] if ev_feats else 0
    n_sub = min(PATCH_MAP_SUBSET, n_avail)
    if n_sub > 0:
        for l in range(N_BLOCKS):
            Ptr = tr_feats[l][:, 1:, :]
            if Ptr.numel() == 0:
                continue
            n_tr_l, P, D = Ptr.shape
            Xtr = Ptr[:, patch_idx, :].reshape(-1, D)
            Ytr = tr_y.unsqueeze(1).expand(-1, patch_idx.numel(), -1).reshape(-1, n_cls)
            W = _ridge_fit(Xtr, Ytr, RIDGE_LAMBDA)
            sub_patches = ev_feats[l][:n_sub, 1:, :]                  # [n_sub, P, D]
            sc = _ridge_predict(sub_patches.reshape(-1, D), W)        # [n_sub*P, C]
            sc = sc.reshape(n_sub, P, n_cls).permute(0, 2, 1)         # [n_sub, C, P]
            grid = sc.reshape(n_sub, n_cls, GRID, GRID).to(torch.float32).cpu().numpy()
            # Pad to the fixed [PATCH_MAP_SUBSET, n_cls, GRID, GRID] slice shape with NaN.
            fixed = np.full((PATCH_MAP_SUBSET, n_cls, GRID, GRID), np.nan, dtype=np.float32)
            fixed[:n_sub] = grid
            ctx.array.put(group="patch_class_maps", step=step, array=fixed,
                          layer=f"blocks.{l}", dtype=np.float16)
        summary["patch_map_subset"] = int(n_sub)

    finite_patch = patch_acc_curve[np.isfinite(patch_acc_curve)]
    summary["patch_probe_acc_max"] = (float(finite_patch.max())
                                      if finite_patch.size else float("nan"))
    summary["n_probe_train"] = int(tr_y.shape[0])
    summary["n_probe_eval"] = int(ev_y.shape[0])
    return summary

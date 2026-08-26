#!/usr/bin/env python3
"""
block_fitness.py — extractor block §6.8: Fitness on held-out data.

Per the briefing (``xAI/notes/phenovision_full_briefing.md`` Part II §6.8). Held-out loss
and the phenology metric on the **fixed probe-eval set** (``ctx.probe_eval_loader``; §4
invariant 1 — same images, same order, every checkpoint, every condition). In the
framework's language this is the run's *fitness*: pre-update loss is ecological fitting,
early slope is short-term evolvability, and final loss is long-term evolvability / basin
depth. The trajectory of initial -> early-slope -> final is the spine of the comparison.

Division of labour (do not duplicate work):
  * The *trajectory spine* (initial / early-slope / final) is logged inline by the trainer
    along the run (briefing §5 / §6.8); this block does **not** try to reconstruct slopes —
    it records the per-checkpoint held-out fitness so the spine can be assembled post-hoc
    from the scalar store across checkpoints (one row per checkpoint).
  * ``block_interp`` (§6.6) and ``block_probes`` (§6.7) handle representational geometry /
    decodability; this block is purely the held-out predictive fitness on the eval split.

Implementation: reuse ``xai_train.validate(model, val_loader, criterion, device)`` verbatim —
it is the single source of truth for the phenology metric (AUC-PR / AUC-ROC per class +
mean, plus held-out BCE loss). That keeps this block cheap (one forward pass over the fixed
probe-eval set, no backward, no double-backprop) and identical to what the trainer reports.

Robustness / RNG:
  This block reads model weights and runs one no-grad forward pass; it does not depend on
  the optimizer variant (``ctx.opt`` is untouched), so it is identical for ``adamw`` and
  ``stable_evo`` (graceful degradation is trivial here — there is nothing optimizer-specific
  to degrade). ``validate`` is deterministic (no sampling), so no generator is threaded; we
  still resolve and consume a per-checkpoint ``torch.Generator`` from ``ctx`` / ``(seed,
  step)`` for consistency with the other blocks and to keep this future-proof if a sampled
  metric is added (CONVENTIONS.md §5 — never touch global RNG).
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING, Dict, Optional

import numpy as np
import torch

# Project paths so xai_train / preadapt_common import cleanly when run from anywhere.
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

# Reproductive head order matches xai_train.validate: index 0 = fruit, index 1 = flower.
_CLASS_NAMES = ("fruit", "flower")

#: Natural validation-set marginal prevalence per class (val_v1.1.0.csv): flowering 63.8%,
#: fruiting 18.0%. The PROBE-EVAL set is balanced (stratified 4-cell), so its raw AUC is the
#: BALANCED cross-condition discriminator. The PREV-WEIGHTED reading reweights each class's
#: eval samples so the positive fraction matches this deployment prevalence — the
#: "deployment-prevalence" number. Indexed by head order: [fruit, flower].
_NATURAL_PREVALENCE = {"fruit": 0.180, "flower": 0.638}


def _prevalence_weights(y_class: np.ndarray, target_pos_rate: float) -> Optional[np.ndarray]:
    """Per-sample weights that reweight a single class's eval samples so the positive fraction
    equals ``target_pos_rate`` (the natural deployment prevalence), with weights summing to N.

    The balanced probe-eval set has some empirical positive rate ``p_hat`` for this class;
    weighting positives by ``target/p_hat`` and negatives by ``(1-target)/(1-p_hat)`` makes the
    weighted positive mass equal ``target_pos_rate``. This yields a prevalence-weighted AUC-PR
    / AUC-ROC that reads as if the eval set had the deployment class balance, WITHOUT assuming
    anything about the joint distribution (it is a per-class marginal reweighting). Returns
    ``None`` if the class is degenerate (all-positive or all-negative), in which case AUC is
    undefined anyway."""
    y = y_class > 0.5
    n = y.shape[0]
    n_pos = int(y.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    p_hat = n_pos / n
    w = np.empty(n, dtype=np.float64)
    w[y] = target_pos_rate / p_hat
    w[~y] = (1.0 - target_pos_rate) / (1.0 - p_hat)
    return w


def _resolve_generator(ctx, device: torch.device) -> torch.Generator:
    """Return an explicit ``torch.Generator`` on ``device`` (never global RNG).

    Prefer a driver-supplied ``ctx.gen_fitness``; otherwise derive a deterministic one from
    ``(ctx.seed, ctx.step)`` so any future stochastic metric is reproducible per checkpoint
    (CONVENTIONS.md §5). ``validate`` itself is deterministic, so this is currently unused
    beyond being threaded for consistency with the other blocks."""
    g = getattr(ctx, "gen_fitness", None)
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


def _collect_probs_targets(model, eval_loader, device):
    """One no-grad forward pass over the fixed probe-eval set -> ``(probs[N,2], targets[N,2])``
    as float64 numpy arrays (col 0 = fruit, col 1 = flower). Mirrors ``xai_train.validate``'s
    forward (sigmoid of logits under autocast) so the BALANCED numbers reconstructed here match
    ``validate`` exactly, while also giving us the raw arrays needed to reweight to deployment
    prevalence. The probe-eval set is tiny (~512 images at the balanced default), so the cost
    of this single pass is negligible."""
    all_probs, all_targets = [], []
    with torch.no_grad():
        for batch in eval_loader:
            images = batch[0].to(device, non_blocking=True)
            targets = batch[-1].to(device, non_blocking=True)
            with torch.amp.autocast("cuda"):
                outputs = model(images)
            probs = torch.sigmoid(outputs)
            all_probs.append(probs.float().cpu().numpy())
            all_targets.append(targets.float().cpu().numpy())
    probs = np.concatenate(all_probs, axis=0).astype(np.float64)
    targets = np.concatenate(all_targets, axis=0).astype(np.float64)
    return probs, targets


def extract(ctx: "ExtractCtx") -> Dict[str, object]:
    """§6.8 held-out fitness on the fixed (balanced, 4-cell-stratified) probe-eval set.

    Writes scalars (all global; ``layer=None``, ``head=None``):
      * ``val_loss``            — held-out BCE-with-logits loss (ecological-fitting axis).
      * ``val_auc_pr_mean``     — mean BALANCED held-out AUC-PR over {fruit, flower} (the
                                  headline phenology fitness; PR is the threshold-free analog
                                  of the PPV-calibrated production metric).
      * ``val_auc_roc_mean``    — mean BALANCED held-out AUC-ROC over {fruit, flower}.
      * ``val_auc_pr_<class>``  — per-class BALANCED AUC-PR (fruit, flower).
      * ``val_auc_roc_<class>`` — per-class BALANCED AUC-ROC (fruit, flower).
      * ``val_auc_pr_mean_prevweighted`` / ``val_auc_roc_mean_prevweighted`` — the same two
                                  means but PREVALENCE-WEIGHTED to the natural deployment class
                                  balance (flowering 63.8%, fruiting 18.0%).
      * ``val_auc_pr_<class>_prevweighted`` / ``val_auc_roc_<class>_prevweighted`` — per-class
                                  prevalence-weighted variants.

    PRIMARY DISCRIMINATOR: the BALANCED numbers (computed on the 4-cell-stratified probe-eval
    set, so no class dominates) are the primary cross-condition comparison — they isolate
    learned discriminability from the eval-set base rate. The PREVALENCE-WEIGHTED numbers are
    the secondary, deployment-prevalence reading: what the held-out fitness looks like at the
    natural val class balance. Both are logged so either lens is available downstream.

    Returns a compact summary dict for the collector log. Robust to a missing probe-eval
    loader (returns a skipped summary rather than raising); identical for both optimizer
    variants (no optimizer state is read)."""
    from sklearn.metrics import roc_auc_score, average_precision_score

    eval_loader = getattr(ctx, "probe_eval_loader", None)
    if eval_loader is None:
        return {"block": "fitness", "skipped": "no probe_eval_loader"}

    model = ctx.model
    criterion = ctx.criterion
    device = ctx.device
    c, r, step, wt = ctx.condition, ctx.run_id, ctx.step, ctx.wall_time

    # Thread a generator for consistency with the other blocks (the forward is deterministic).
    _ = _resolve_generator(ctx, torch.device(device))

    model.eval()
    summary: Dict[str, object] = {"block": "fitness"}

    # --- held-out loss (ecological-fitting / basin-depth spine) via validate -------------
    # validate() remains the single source of truth for the held-out BCE loss; the balanced
    # AUCs are recomputed below from the raw probs so the prev-weighted variants share an
    # identical forward (and the balanced numbers match validate by construction).
    try:
        from xai_train import validate
        metrics = validate(model, eval_loader, criterion, device)
        val_loss = metrics.get("val_loss")
        if val_loss is not None and np.isfinite(val_loss):
            ctx.scalar.add(c, r, step, wt, quantity="val_loss", value=float(val_loss))
            summary["val_loss"] = float(val_loss)
    except Exception as e:
        print(f"[block_fitness] validate(loss) failed at run_id={r} step={step}: {e!r}",
              file=sys.stderr, flush=True)

    # --- one forward pass -> raw probs/targets for balanced + prev-weighted metrics ------
    try:
        probs, targets = _collect_probs_targets(model, eval_loader, device)
    except Exception as e:  # never kill the collector on a single bad checkpoint
        print(f"[block_fitness] forward pass failed at run_id={r} step={step}: {e!r}",
              file=sys.stderr, flush=True)
        return {"block": "fitness", "error": repr(e)}

    def _safe(fn, *a, **kw):
        try:
            v = float(fn(*a, **kw))
            return v if np.isfinite(v) else None
        except ValueError:
            return None

    # --- per-class metrics: BALANCED (no weights) + PREV-WEIGHTED (deployment balance) ---
    bal_pr, bal_roc, pw_pr, pw_roc = {}, {}, {}, {}
    for i, cls in enumerate(_CLASS_NAMES):
        y = targets[:, i]
        p = probs[:, i]
        # BALANCED (raw) — matches validate's per-class numbers on the balanced eval set.
        v_pr = _safe(average_precision_score, y, p)
        v_roc = _safe(roc_auc_score, y, p)
        if v_pr is not None:
            ctx.scalar.add(c, r, step, wt, quantity=f"val_auc_pr_{cls}", value=v_pr)
            summary[f"val_auc_pr_{cls}"] = v_pr
            bal_pr[cls] = v_pr
        if v_roc is not None:
            ctx.scalar.add(c, r, step, wt, quantity=f"val_auc_roc_{cls}", value=v_roc)
            summary[f"val_auc_roc_{cls}"] = v_roc
            bal_roc[cls] = v_roc
        # PREV-WEIGHTED — reweight this class to its natural marginal prevalence.
        w = _prevalence_weights(y, _NATURAL_PREVALENCE[cls])
        if w is not None:
            v_pr_w = _safe(average_precision_score, y, p, sample_weight=w)
            v_roc_w = _safe(roc_auc_score, y, p, sample_weight=w)
            if v_pr_w is not None:
                ctx.scalar.add(c, r, step, wt,
                               quantity=f"val_auc_pr_{cls}_prevweighted", value=v_pr_w)
                summary[f"val_auc_pr_{cls}_prevweighted"] = v_pr_w
                pw_pr[cls] = v_pr_w
            if v_roc_w is not None:
                ctx.scalar.add(c, r, step, wt,
                               quantity=f"val_auc_roc_{cls}_prevweighted", value=v_roc_w)
                summary[f"val_auc_roc_{cls}_prevweighted"] = v_roc_w
                pw_roc[cls] = v_roc_w

    # --- means over {fruit, flower} (the headline phenology fitness) ---------------------
    def _emit_mean(quantity: str, per_class: Dict[str, float]) -> None:
        vals = [per_class[cls] for cls in _CLASS_NAMES if cls in per_class]
        if vals:
            m = float(np.mean(vals))
            ctx.scalar.add(c, r, step, wt, quantity=quantity, value=m)
            summary[quantity] = m

    _emit_mean("val_auc_pr_mean", bal_pr)                       # PRIMARY discriminator
    _emit_mean("val_auc_roc_mean", bal_roc)                     # PRIMARY discriminator
    _emit_mean("val_auc_pr_mean_prevweighted", pw_pr)           # deployment-prevalence reading
    _emit_mean("val_auc_roc_mean_prevweighted", pw_roc)         # deployment-prevalence reading

    return summary

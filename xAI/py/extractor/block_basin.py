#!/usr/bin/env python3
"""
block_basin.py — CONVERGENCE-ONLY basin-geometry characterization (plan component C7;
briefing Part II §9.2 "is the converged solution a wide/narrow basin, and how isotropic?").

This is NOT a live per-checkpoint block. It is DELIBERATELY excluded from
``extract.BLOCK_ORDER`` so the streaming collectors never touch it — it is a heavy POST-HOC
block run ONCE per run on the FINAL + last-few kept ladder checkpoints (the converged region of
the trajectory), where a careful basin profile is worth many forward passes. It still exposes the
FROZEN block interface ``extract(ctx) -> dict`` (same surface as ``block_curvature.extract``) so a
small runner (the ``__main__`` CLI below, or a future post-hoc driver) can drive it through the
exact same ctx/store conventions every §6 block uses.

What it measures, per checkpoint, on the FIXED probe/Hessian batch (``ctx.hessian_batch``) and over
filter-normalized random directions in the TRAINABLE-param subspace (frozen input stage EXCLUDED via
:func:`preadapt_common.trainable_named_parameters`, exactly like every other block):

  1. BASIN RADIUS + ANISOTROPY (Li et al. 2018 filter-normalization). Draw ``R`` filter-normalized
     random directions ``d`` (same per-filter-norm rescaling as ``posthoc.landscape_slices``), walk
     ``θ ± α·d`` over an α-grid, eval the BCE loss on the fixed batch, and per direction record the
     RADIUS to a loss rise of ``epsilon`` above the basin floor (linear interpolation between the two
     bracketing grid points; the smaller of the +α and −α radius — the basin half-width along that
     direction). The DISTRIBUTION of radii over the R directions IS the basin geometry: its
     location (mean/median/quantiles) is basin SIZE, its spread (CV) is basin ANISOTROPY. Per-direction
     radii go to the ARRAY store (``basin_radii``); summary scalars
     (``basin_radius_mean/median/p10/p90``, ``basin_radius_cv``) to the SCALAR store.

  2. FLATNESS / VOLUME proxy from the Hessian spectrum (reuses ``curvature.lanczos`` — the SAME HVP
     machinery ``block_curvature`` uses): one Lanczos pass -> top-k Ritz values; Hutchinson trace.
     From these: ``basin_lambda_max`` (sharpness), ``basin_hessian_trace`` (mean curvature),
     ``basin_eff_rank`` (participation ratio of the top-k spectrum = effective # of curved
     directions), ``basin_logdet_proxy`` (Σ log(max(λ,eps)) over top-k — a log-volume / flatness
     proxy: smaller => flatter/wider basin). Scalars; the top-k spectrum array goes to
     ``basin_hessian_spectrum``.

  3. (optional) SAM-style WORST-CASE sharpness in an ε-ball: one ascent step to the worst-case
     point ``θ + ρ·ĝ`` (ĝ = normalized gradient on the fixed batch over the trainable params) and
     the loss rise there — the adversarial/flat-minima sharpness number (Foret et al. 2021).
     Scalar ``basin_sam_sharpness`` (gated by ``SAM_ENABLED``).

The loss is BCE-with-logits on the ONE fixed batch so every number is comparable across the
last-few checkpoints of a run and across runs. All stochastic ops thread an explicit
``torch.Generator`` (CONVENTIONS.md §5): the basin directions use a generator derived from
``(ctx.seed, ctx.step)`` (or a driver-supplied ``ctx.gen_basin`` if present), and the Lanczos /
Hutchinson probes reuse ``block_curvature``'s ``ctx.gen_curvature`` resolution so the spectrum is
byte-reproducible.

Cost knobs (module constants): ``BASIN_R_DIRECTIONS`` (random directions), ``BASIN_ALPHA_MAX`` +
``BASIN_N_ALPHA`` (the half-grid resolution; total loss evals per direction = ``2*BASIN_N_ALPHA``),
``BASIN_RISE_EPS`` (the loss-rise threshold defining the radius), plus the Lanczos knobs reused
from the spectrum section. Because this is convergence-only, these default GENEROUS relative to
``block_curvature``'s collector-speed cuts.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# --- project paths: mirror _ctx.py / extract.py so PlantCLEF2022 + xAI/py + two_noise import. ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_PY_DIR = os.path.dirname(_HERE)                                  # .../xAI/py
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_PY_DIR))         # .../phenovision
_PLANTCLEF = os.path.join(_PROJECT_ROOT, "PlantCLEF2022")
_TWO_NOISE = os.path.join(_PROJECT_ROOT, "xAI", "two_noise")
for _p in (_PY_DIR, _PROJECT_ROOT, _PLANTCLEF, _TWO_NOISE):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# =============================================================================
# §2.3 cost knobs (generous — convergence-only, NOT on the live collector path)
# =============================================================================
# Basin radius / anisotropy: R filter-normalized directions, each walked over an α half-grid
# in BOTH directions. Total loss evals per checkpoint ~= R * (2 * BASIN_N_ALPHA) full forward
# passes on the fixed batch. Defaults: 24 directions * 2 * 16 = 768 forward passes / checkpoint —
# fine post-hoc on a handful of converged checkpoints; never on the streaming path.
BASIN_R_DIRECTIONS: int = 24       # number of filter-normalized random directions (anisotropy sample)
BASIN_ALPHA_MAX: float = 1.0       # walk θ ± α·d for α in (0, BASIN_ALPHA_MAX]
BASIN_N_ALPHA: int = 16            # α grid points on EACH side of the minimum (per direction)
BASIN_RISE_EPS: float = 0.05       # loss rise (above the basin floor) defining the basin radius

# Flatness / volume proxy (Hessian spectrum) — reuse curvature.lanczos. Generous vs block_curvature.
BASIN_LANCZOS_K: int = 20          # top-k Ritz values for the spectrum
BASIN_LANCZOS_N_ITER: int = 20     # Lanczos steps (>= K); the single pass's HVP budget
BASIN_HUTCHINSON_PROBES: int = 8   # Rademacher probes for the stochastic trace

# Optional SAM-style worst-case sharpness in an ε-ball (Foret et al. 2021). One gradient + one
# perturbed forward pass. Cheap; on by default.
SAM_ENABLED: bool = True
SAM_RHO: float = 0.05              # ε-ball radius for the single ascent step

_EPS: float = 1e-12


# =============================================================================
# helpers (shared style with block_curvature / posthoc.landscape_slices)
# =============================================================================

def _resolve_generator(ctx, device: torch.device, attr: str) -> torch.Generator:
    """Return an explicit ``torch.Generator`` on ``device``. Prefer a driver-supplied
    ``ctx.<attr>`` (e.g. ``gen_basin`` / ``gen_curvature``); otherwise derive a deterministic one
    from ``(ctx.seed, ctx.step, attr)`` so probe / direction noise is reproducible per checkpoint
    and never touches the global RNG (CONVENTIONS.md §5)."""
    g = getattr(ctx, attr, None)
    if isinstance(g, torch.Generator):
        if g.device == device:
            return g
        seed = int(torch.randint(0, 2 ** 31 - 1, (1,), generator=g).item())
        dg = torch.Generator(device=device)
        dg.manual_seed(seed)
        return dg
    salt = abs(hash(attr)) & 0xFFFF
    seed = ((int(getattr(ctx, "seed", 0)) * 1_000_003
             + int(getattr(ctx, "step", 0)) * 131 + salt) & 0x7FFFFFFF)
    dg = torch.Generator(device=device)
    dg.manual_seed(seed)
    return dg


def _participation_ratio(values: np.ndarray) -> float:
    """PR = (Σ λ)² / Σ λ² of a non-negative spectrum (effective # of directions). Mirrors
    ``block_curvature._participation_ratio``."""
    if values is None or values.size == 0:
        return float("nan")
    s = np.clip(values.astype(np.float64), 0.0, None)
    s2 = float(np.sum(s * s))
    if s2 <= 0.0:
        return float("nan")
    return float((float(np.sum(s)) ** 2) / s2)


def _move_batch(hessian_batch, device: torch.device):
    """Unpack ``ctx.hessian_batch`` (an ``(images, targets)`` pair) onto ``device``."""
    images, targets = hessian_batch
    if isinstance(images, torch.Tensor):
        images = images.to(device, non_blocking=True)
    if isinstance(targets, torch.Tensor):
        targets = targets.to(device, non_blocking=True)
    return images, targets


@torch.no_grad()
def _bce_loss_on_batch(model: nn.Module, criterion, images: torch.Tensor,
                       targets: torch.Tensor) -> float:
    """Scalar BCE-with-logits loss on the fixed batch (no grad; for the α-grid walk)."""
    logits = model(images)
    return float(criterion(logits, targets.to(logits.dtype)).item())


@torch.no_grad()
def _snapshot_trainable(model: nn.Module, device: torch.device) -> "OrderedDict[str, torch.Tensor]":
    """θ* snapshot of the trainable params (float32, on ``device``), keyed by name.
    Frozen input stage excluded. Mirrors ``posthoc.landscape_slices.snapshot_trainable``."""
    from preadapt_common import trainable_named_parameters
    snap: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for name, p in trainable_named_parameters(model, include_requires_grad_false=True):
        snap[name] = p.detach().to(device, torch.float32).clone()
    return snap


@torch.no_grad()
def _filter_normalized_direction(model: nn.Module, generator: torch.Generator,
                                 device: torch.device) -> "OrderedDict[str, torch.Tensor]":
    """Draw ONE filter-normalized random direction over the TRAINABLE params (Li et al. 2018).

    Identical convention to ``posthoc.landscape_slices.filter_normalized_direction``:
      * ndim >= 2 : per-filter (slice along dim 0); each slice rescaled ``||d_i|| -> ||θ*_i||``.
      * ndim <= 1 (LN/bias) and cls_token : whole-tensor ``||d|| -> ||θ*||``.
    A zero-norm reference slice yields a zero direction slice (that param never moves)."""
    from preadapt_common import trainable_named_parameters
    direction: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for name, p in trainable_named_parameters(model, include_requires_grad_false=True):
        ref = p.detach().to(device, torch.float32)
        d = torch.randn(ref.shape, generator=generator, device=generator.device).to(device)
        if ref.ndim >= 2:
            d_flat = d.reshape(ref.shape[0], -1)
            r_flat = ref.reshape(ref.shape[0], -1)
            d_norm = d_flat.norm(dim=1, keepdim=True)
            r_norm = r_flat.norm(dim=1, keepdim=True)
            scale = torch.where(d_norm > 0, r_norm / d_norm, torch.zeros_like(d_norm))
            d = (d_flat * scale).reshape(ref.shape)
        else:
            d_norm = d.norm()
            r_norm = ref.norm()
            d = d * (r_norm / d_norm) if float(d_norm) > 0 else torch.zeros_like(d)
        direction[name] = d
    return direction


@torch.no_grad()
def _apply_perturbation(model: nn.Module, theta_star: "OrderedDict[str, torch.Tensor]",
                        d: "OrderedDict[str, torch.Tensor]", alpha: float,
                        named: Dict[str, nn.Parameter]) -> None:
    """Set the model's trainable params to ``θ* + α·d`` IN PLACE (frozen stage untouched).
    ``named`` is the cached name->Parameter map so we don't rebuild it per grid point."""
    for name, base in theta_star.items():
        new = base if alpha == 0.0 else (base + alpha * d[name])
        named[name].data.copy_(new.to(named[name].dtype))


def _radius_to_rise(alphas: List[float], losses: List[float], base_loss: float,
                    eps: float, alpha_max: float) -> float:
    """Radius (in α) at which the loss first rises ``eps`` ABOVE ``base_loss``, by linear
    interpolation between the two bracketing grid points. ``alphas`` are positive, ascending;
    ``losses`` the matching loss values along ONE side of the direction. If the loss never rises
    by ``eps`` within the grid, returns ``alpha_max`` (a censored / right-bounded radius — the
    basin is at least this wide along that direction)."""
    target = base_loss + eps
    prev_a, prev_l = 0.0, base_loss
    for a, l in zip(alphas, losses):
        if l >= target:
            if l == prev_l:           # flat segment at/above target: take the bracket's near edge
                return float(prev_a)
            frac = (target - prev_l) / (l - prev_l)
            frac = min(max(frac, 0.0), 1.0)
            return float(prev_a + frac * (a - prev_a))
        prev_a, prev_l = a, l
    return float(alpha_max)            # never crossed -> censored at the grid edge


# =============================================================================
# section 1 — basin radius + anisotropy (filter-normalized random directions)
# =============================================================================

def _basin_radius_anisotropy(ctx, model, criterion, images, targets,
                             device: torch.device, summary: Dict[str, Any]) -> None:
    """Sample R filter-normalized directions, walk θ ± α·d on the fixed batch, record the
    per-direction basin radius (min of the +α and −α half-width to a loss rise of EPS), and
    summarize the distribution (size = location, anisotropy = spread). Writes per-direction
    radii to the array store and summary scalars to the scalar store."""
    from preadapt_common import trainable_named_parameters

    gen = _resolve_generator(ctx, device, "gen_basin")
    named = dict(trainable_named_parameters(model, include_requires_grad_false=True))
    theta_star = _snapshot_trainable(model, device)

    # positive α grid (excludes 0); we walk both +α and −α per direction.
    n_a = max(int(BASIN_N_ALPHA), 1)
    pos_alphas = [BASIN_ALPHA_MAX * (i + 1) / n_a for i in range(n_a)]

    base_loss = _bce_loss_on_batch(model, criterion, images, targets)
    summary["basin_base_loss"] = float(base_loss)

    radii: List[float] = []
    radii_pos: List[float] = []
    radii_neg: List[float] = []
    try:
        for _r in range(int(BASIN_R_DIRECTIONS)):
            d = _filter_normalized_direction(model, gen, device)
            # +α side
            losses_p = []
            for a in pos_alphas:
                _apply_perturbation(model, theta_star, d, a, named)
                losses_p.append(_bce_loss_on_batch(model, criterion, images, targets))
            r_pos = _radius_to_rise(pos_alphas, losses_p, base_loss, BASIN_RISE_EPS, BASIN_ALPHA_MAX)
            # −α side
            losses_n = []
            for a in pos_alphas:
                _apply_perturbation(model, theta_star, d, -a, named)
                losses_n.append(_bce_loss_on_batch(model, criterion, images, targets))
            r_neg = _radius_to_rise(pos_alphas, losses_n, base_loss, BASIN_RISE_EPS, BASIN_ALPHA_MAX)
            # the basin half-width along this direction is the SMALLER (tighter) side.
            radii.append(float(min(r_pos, r_neg)))
            radii_pos.append(float(r_pos))
            radii_neg.append(float(r_neg))
    finally:
        # ALWAYS restore θ* so the model is clean for the spectrum / SAM sections and any reuse.
        with torch.no_grad():
            for name, base in theta_star.items():
                named[name].data.copy_(base.to(named[name].dtype))

    if not radii:
        summary["basin_radius_skipped"] = "no directions"
        return

    arr = np.asarray(radii, dtype=np.float32)
    c, r, step, wt = ctx.condition, ctx.run_id, ctx.step, ctx.wall_time

    # per-direction radii -> array store (the full distribution for post-hoc anisotropy analysis)
    if ctx.array is not None:
        ctx.array.put(group="basin_radii", step=step, array=arr)
        ctx.array.put(group="basin_radii_pos", step=step, array=np.asarray(radii_pos, dtype=np.float32))
        ctx.array.put(group="basin_radii_neg", step=step, array=np.asarray(radii_neg, dtype=np.float32))

    a64 = arr.astype(np.float64)
    mean = float(np.mean(a64))
    median = float(np.median(a64))
    p10 = float(np.quantile(a64, 0.10))
    p90 = float(np.quantile(a64, 0.90))
    std = float(np.std(a64))
    cv = float(std / mean) if mean > _EPS else float("nan")   # coefficient of variation = anisotropy

    if ctx.scalar is not None:
        ctx.scalar.add(c, r, step, wt, quantity="basin_radius_mean", value=mean)
        ctx.scalar.add(c, r, step, wt, quantity="basin_radius_median", value=median)
        ctx.scalar.add(c, r, step, wt, quantity="basin_radius_p10", value=p10)
        ctx.scalar.add(c, r, step, wt, quantity="basin_radius_p90", value=p90)
        ctx.scalar.add(c, r, step, wt, quantity="basin_radius_cv", value=cv)
        ctx.scalar.add(c, r, step, wt, quantity="basin_n_directions", value=float(arr.size))

    summary.update({
        "basin_radius_mean": mean,
        "basin_radius_median": median,
        "basin_radius_p10": p10,
        "basin_radius_p90": p90,
        "basin_radius_cv": cv,
        "basin_n_directions": int(arr.size),
        "basin_alpha_max": float(BASIN_ALPHA_MAX),
        "basin_rise_eps": float(BASIN_RISE_EPS),
    })


# =============================================================================
# section 2 — flatness / volume proxy from the Hessian spectrum (curvature.lanczos)
# =============================================================================

def _build_loss_closure(model: nn.Module, criterion, images: torch.Tensor, targets: torch.Tensor):
    """No-arg ``loss_fn()`` that recomputes the forward pass and returns the scalar BCE loss WITH
    the autograd graph attached (so ``curvature.lanczos`` can take double-backprop HVPs). No
    autocast — HVPs need a clean float32 graph. Mirrors ``block_curvature._build_loss_closure``."""
    def loss_fn() -> torch.Tensor:
        logits = model(images)
        return criterion(logits, targets.to(logits.dtype))
    return loss_fn


def _basin_spectrum(ctx, model, criterion, images, targets,
                    device: torch.device, summary: Dict[str, Any]) -> None:
    """Hessian spectrum flatness/volume proxy on the fixed batch: one Lanczos pass -> top-k Ritz
    values; Hutchinson trace. Derives λ_max, trace, effective rank, log-det proxy. Reuses the SAME
    HVP machinery as block_curvature; the model is put in train-grad mode for the HVPs and restored."""
    from curvature.lanczos import hutchinson_trace, lanczos_tridiag
    from preadapt_common import trainable_named_parameters

    gen = _resolve_generator(ctx, device, "gen_curvature")
    named = trainable_named_parameters(model, include_requires_grad_false=True)
    params = [p for _, p in named]
    prev_rg = [p.requires_grad for p in params]
    prev_training = model.training
    for p in params:
        p.requires_grad_(True)
    model.train()

    c, r, step, wt = ctx.condition, ctx.run_id, ctx.step, ctx.wall_time
    try:
        loss_fn = _build_loss_closure(model, criterion, images, targets)

        # --- ONE Lanczos pass: top-k Ritz values from its tridiagonal (no second pass). -------
        ev = np.array([], dtype=np.float64)
        try:
            n_iter = max(int(BASIN_LANCZOS_N_ITER), int(BASIN_LANCZOS_K))
            alphas, betas, _ = lanczos_tridiag(loss_fn, params, n_iter, generator=gen)
            af = alphas.detach().to("cpu", dtype=torch.float64)
            bf = betas.detach().to("cpu", dtype=torch.float64)
            T = torch.diag(af)
            if bf.numel():
                idx = torch.arange(bf.numel())
                T[idx, idx + 1] = bf
                T[idx + 1, idx] = bf
            ritz = torch.linalg.eigvalsh(T)
            ritz = torch.sort(ritz, descending=True).values[:BASIN_LANCZOS_K]
            ev = ritz.to(dtype=torch.float64).numpy()
        except Exception as e:
            summary["basin_lanczos_error"] = repr(e)

        if ev.size > 0:
            if ctx.array is not None:
                ctx.array.put(group="basin_hessian_spectrum", step=step, array=ev.astype(np.float32))
            lam_max = float(ev[0])
            eff_rank = _participation_ratio(ev)                    # effective # of curved directions
            pos = np.clip(ev, 0.0, None)
            logdet = float(np.sum(np.log(np.clip(pos, _EPS, None))))  # Σ log(max(λ,eps)) over top-k
            if ctx.scalar is not None:
                ctx.scalar.add(c, r, step, wt, quantity="basin_lambda_max", value=lam_max)
                ctx.scalar.add(c, r, step, wt, quantity="basin_eff_rank", value=eff_rank)
                ctx.scalar.add(c, r, step, wt, quantity="basin_logdet_proxy", value=logdet)
            summary.update({
                "basin_lambda_max": lam_max,
                "basin_eff_rank": eff_rank,
                "basin_logdet_proxy": logdet,
                "basin_n_eigs": int(ev.size),
            })

        # --- Hutchinson trace (mean curvature) ------------------------------------------------
        try:
            tr = hutchinson_trace(loss_fn, params, n_probes=BASIN_HUTCHINSON_PROBES, generator=gen)
            if ctx.scalar is not None:
                ctx.scalar.add(c, r, step, wt, quantity="basin_hessian_trace", value=float(tr))
            summary["basin_hessian_trace"] = float(tr)
        except Exception as e:
            summary["basin_hessian_trace_error"] = repr(e)

    finally:
        for p, rg in zip(params, prev_rg):
            p.requires_grad_(rg)
        model.zero_grad(set_to_none=True)
        if not prev_training:
            model.eval()


# =============================================================================
# section 3 — optional SAM-style worst-case sharpness in an ε-ball (Foret et al. 2021)
# =============================================================================

def _basin_sam_sharpness(ctx, model, criterion, images, targets,
                         device: torch.device, summary: Dict[str, Any]) -> None:
    """One ascent step to the worst-case point ``θ + ρ·ĝ`` (ĝ = unit gradient over the trainable
    params on the fixed batch) and the loss RISE there vs the basin floor — the SAM/flat-minima
    sharpness number. One backward + one perturbed forward. Restores θ* afterward."""
    from preadapt_common import trainable_named_parameters

    named_pairs = trainable_named_parameters(model, include_requires_grad_false=True)
    named = {n: p for n, p in named_pairs}
    params = [p for _, p in named_pairs]
    prev_rg = [p.requires_grad for p in params]
    prev_training = model.training
    for p in params:
        p.requires_grad_(True)
    model.train()

    c, r, step, wt = ctx.condition, ctx.run_id, ctx.step, ctx.wall_time
    snapshot = {n: p.detach().clone() for n, p in named.items()}
    try:
        model.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets.to(logits.dtype))
        base_loss = float(loss.detach().item())
        grads = torch.autograd.grad(loss, params, retain_graph=False, allow_unused=True)

        # global unit gradient direction (Foret et al. dual-norm ascent, p=2)
        flat = [g.detach().reshape(-1) for g in grads if g is not None]
        if not flat:
            summary["basin_sam_skipped"] = "no grad"
            return
        gnorm = float(torch.cat(flat).norm().item())
        if gnorm <= _EPS:
            summary["basin_sam_sharpness"] = 0.0
            if ctx.scalar is not None:
                ctx.scalar.add(c, r, step, wt, quantity="basin_sam_sharpness", value=0.0)
            return

        scale = SAM_RHO / gnorm
        with torch.no_grad():
            for (name, p), g in zip(named_pairs, grads):
                if g is not None:
                    p.data.add_(g.detach(), alpha=scale)    # θ + ρ·(g/||g||)
        adv_loss = _bce_loss_on_batch(model, criterion, images, targets)
        sharp = float(adv_loss - base_loss)
        if ctx.scalar is not None:
            ctx.scalar.add(c, r, step, wt, quantity="basin_sam_sharpness", value=sharp)
        summary["basin_sam_sharpness"] = sharp
        summary["basin_sam_rho"] = float(SAM_RHO)
    except Exception as e:
        summary["basin_sam_error"] = repr(e)
    finally:
        with torch.no_grad():
            for n, p in named.items():
                p.data.copy_(snapshot[n])
        model.zero_grad(set_to_none=True)
        for p, rg in zip(params, prev_rg):
            p.requires_grad_(rg)
        if not prev_training:
            model.eval()


# =============================================================================
# entry point (FROZEN block interface — same signature as block_curvature.extract)
# =============================================================================

def extract(ctx) -> dict:
    """CONVERGENCE-ONLY basin-geometry block on the fixed batch. See module docstring.

    Writes scalars: ``basin_radius_{mean,median,p10,p90,cv}``, ``basin_n_directions``,
    ``basin_lambda_max``, ``basin_hessian_trace``, ``basin_eff_rank``, ``basin_logdet_proxy``,
    ``basin_sam_sharpness`` (optional). Writes arrays: ``basin_radii`` (per-direction radii),
    ``basin_radii_pos`` / ``basin_radii_neg``, ``basin_hessian_spectrum`` (top-k Ritz values).

    Returns a compact summary dict. Robust to a missing Hessian batch (returns an empty summary
    rather than raising) and isolates each section (one failing section never kills the others)."""
    if getattr(ctx, "hessian_batch", None) is None:
        return {"block": "basin", "skipped": "no hessian_batch"}

    model = ctx.model
    criterion = ctx.criterion
    try:
        param_device = next(model.parameters()).device
    except StopIteration:
        param_device = torch.device(ctx.device)
    images, targets = _move_batch(ctx.hessian_batch, param_device)

    summary: Dict[str, Any] = {"block": "basin"}
    summary["basin_r_directions"] = int(BASIN_R_DIRECTIONS)
    summary["basin_n_alpha"] = int(BASIN_N_ALPHA)
    summary["basin_lanczos_k"] = int(BASIN_LANCZOS_K)

    # 1. basin radius + anisotropy (no-grad α-walk over filter-normalized directions)
    try:
        _basin_radius_anisotropy(ctx, model, criterion, images, targets, param_device, summary)
    except Exception as e:
        summary["basin_radius_error"] = repr(e)

    # 2. flatness / volume proxy from the Hessian spectrum (double-backprop HVPs)
    try:
        _basin_spectrum(ctx, model, criterion, images, targets, param_device, summary)
    except Exception as e:
        summary["basin_spectrum_error"] = repr(e)

    # 3. optional SAM-style worst-case sharpness
    if SAM_ENABLED:
        try:
            _basin_sam_sharpness(ctx, model, criterion, images, targets, param_device, summary)
        except Exception as e:
            summary["basin_sam_error"] = repr(e)

    return summary


# =============================================================================
# Standalone runner: load final + last-k kept checkpoints and run the block on each.
# =============================================================================

class _MiniCtx:
    """Minimal ExtractCtx-compatible duck type for the standalone runner. Carries exactly the
    fields ``block_basin.extract`` reads — identity columns, model/criterion/device, the fixed
    hessian_batch, and the scalar/array stores — without the full per-checkpoint resource machinery
    that the live collector builds. (The live path uses the real ``ExtractCtx`` from ``_ctx.py``.)"""

    def __init__(self, *, condition, variant, run_id, seed, step, wall_time,
                 device, model, criterion, hessian_batch, scalar, array):
        self.condition = condition
        self.variant = variant
        self.run_id = run_id
        self.seed = seed
        self.step = step
        self.wall_time = wall_time
        self.device = device
        self.model = model
        self.criterion = criterion
        self.hessian_batch = hessian_batch
        self.scalar = scalar
        self.array = array


def _discover_converged_checkpoints(run_dir: str, last_k: int) -> List[str]:
    """Return the FINAL + last-``last_k`` kept ladder checkpoints of a run (highest steps first;
    de-duplicated; existing files only). Looks in ``<run>/kept/`` (kept-storage redesign location),
    then the run dir + ``<run>/checkpoints``. Recognizes ``stepNNNNNNNN.pt`` by step number; the
    ``init_model.pt`` / ``phase1_final.pt`` references are NOT included (they are the START, not the
    converged region)."""
    run_dir = os.path.abspath(run_dir.rstrip("/"))
    cands_dirs = [
        os.path.join(run_dir, "kept"),
        run_dir,
        os.path.join(run_dir, "checkpoints"),
    ]
    found: Dict[int, str] = {}
    for d in cands_dirs:
        if not os.path.isdir(d):
            continue
        for entry in sorted(os.listdir(d)):
            if not (entry.startswith("step") and entry.endswith(".pt")):
                continue
            digits = entry[len("step"):-len(".pt")]
            if not digits.isdigit():
                continue
            stepnum = int(digits)
            path = os.path.join(d, entry)
            if os.path.exists(path):
                found.setdefault(stepnum, path)   # first dir (kept/) wins for a given step
    if not found:
        return []
    ordered_steps = sorted(found.keys(), reverse=True)   # final first
    n_take = max(int(last_k) + 1, 1)                      # final + last_k
    return [found[s] for s in ordered_steps[:n_take]]


def _load_model_from_ckpt(ckpt_path: str, num_classes: int, device: torch.device):
    """Build the ViT-L/16, load this checkpoint's model weights, eval mode. Returns
    ``(model, meta)`` where meta carries condition/variant/seed/run_id/step/wall_time as best they
    can be recovered from the checkpoint dict (falls back to manifest / dirname)."""
    from models_vit import vit_large_patch16
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = vit_large_patch16(num_classes=num_classes)
    sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()
    meta: Dict[str, Any] = {}
    if isinstance(ckpt, dict):
        for k in ("condition", "variant", "seed", "run_id", "step", "wall_time"):
            if k in ckpt and ckpt[k] is not None:
                meta[k] = ckpt[k]
    del ckpt
    return model, meta


def _build_fixed_batch(val_csv: str, batch_size: int, seed: int, num_workers: int,
                       val_subset_size: int, device: torch.device):
    """Draw ONE fixed (images, targets) batch + the BCE criterion from the val probe loader
    (seed-fixed -> the same images every checkpoint / invocation; briefing §4 invariant 1). Reuses
    ``posthoc.landscape_slices.build_phenovision_probe`` so the loader convention is identical."""
    from posthoc.landscape_slices import build_phenovision_probe
    loader, criterion = build_phenovision_probe(
        val_csv=val_csv, batch_size=batch_size, seed=seed,
        num_workers=num_workers, val_subset_size=val_subset_size)
    images, targets = next(iter(loader))
    images = images.to(device)
    targets = targets.to(device)
    return (images, targets), criterion


def _run_id_from_dir(run_dir: str) -> str:
    return os.path.basename(os.path.abspath(run_dir.rstrip("/")))


def _parse_run_id(run_id: str) -> Tuple[str, str, int]:
    """``{condition}__{variant}__s{seed}`` -> (condition, variant, seed). Tolerant of extra parts."""
    parts = run_id.split("__")
    condition = parts[0] if parts else "unknown"
    variant = parts[1] if len(parts) > 1 else "unknown"
    seed = 0
    if len(parts) > 2 and parts[2].startswith("s") and parts[2][1:].isdigit():
        seed = int(parts[2][1:])
    return condition, variant, seed


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True,
                   help="run dir (e.g. xAI/output/preadapt_v2/<run_id>); kept/ ladder is searched")
    p.add_argument("--out-store", required=True,
                   help="output store base (scalar parts dir + arrays.zarr created under it)")
    p.add_argument("--val-csv", required=True,
                   help="held-out probe CSV (file_name, fruiting, flowering) — the FIXED batch source")
    p.add_argument("--last-k", type=int, default=3,
                   help="run on the FINAL + this many preceding kept checkpoints (default 3)")
    p.add_argument("--num-classes", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=64,
                   help="size of the FIXED basin/Hessian batch")
    p.add_argument("--val-subset-size", type=int, default=10000)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42,
                   help="fixes the basin batch subset (NOT a training seed)")
    p.add_argument("--device", default="")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    from preadapt_common import ScalarStore, ArrayStore

    args = build_arg_parser().parse_args(argv)
    device = torch.device(args.device if args.device
                          else ("cuda" if torch.cuda.is_available() else "cpu"))

    run_id = _run_id_from_dir(args.run_dir)
    cond, variant, seed = _parse_run_id(run_id)

    ckpts = _discover_converged_checkpoints(args.run_dir, args.last_k)
    if not ckpts:
        print(f"[basin] no kept step checkpoints found under {args.run_dir!r}; nothing to do.",
              file=sys.stderr, flush=True)
        return 0
    print(f"[basin] run_id={run_id} condition={cond} variant={variant} seed={seed} "
          f"device={device}", flush=True)
    print(f"[basin] {len(ckpts)} converged checkpoint(s) (final + last-{args.last_k}):", flush=True)
    for cp in ckpts:
        print(f"[basin]   {cp}", flush=True)

    # FIXED batch + criterion (seed-fixed; identical for every checkpoint).
    (images, targets), criterion = _build_fixed_batch(
        args.val_csv, args.batch_size, args.seed, args.num_workers,
        args.val_subset_size, device)
    criterion = criterion.to(device)

    # Stores under the out-store base (idempotent: ScalarStore appends a new part per run;
    # ArrayStore opens in append mode — re-running overwrites a (group,step) slice cleanly).
    os.makedirs(args.out_store, exist_ok=True)
    scalar = ScalarStore(os.path.join(args.out_store, "basin_scalars.parquet"))
    array = ArrayStore(os.path.join(args.out_store, "basin_arrays.zarr"))

    n_ok = 0
    for cp in ckpts:
        model, meta = _load_model_from_ckpt(cp, args.num_classes, device)
        step = int(meta.get("step", 0) or 0)
        if step == 0:
            # recover the step from the filename when the ckpt dict lacks it
            base = os.path.basename(cp)
            if base.startswith("step") and base.endswith(".pt"):
                digits = base[len("step"):-len(".pt")]
                if digits.isdigit():
                    step = int(digits)
        ctx = _MiniCtx(
            condition=meta.get("condition", cond),
            variant=meta.get("variant", variant),
            run_id=meta.get("run_id", run_id),
            seed=int(meta.get("seed", seed) or seed),
            step=step,
            wall_time=float(meta.get("wall_time", time.time())),
            device=device,
            model=model,
            criterion=criterion,
            hessian_batch=(images, targets),
            scalar=scalar,
            array=array,
        )
        t0 = time.time()
        summary = extract(ctx)
        dt = time.time() - t0
        print(f"[basin] step={step} ({dt:.1f}s) {summary}", flush=True)
        scalar.flush()
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        n_ok += 1

    scalar.close()
    print(f"[basin] done: {n_ok} checkpoint(s) -> {args.out_store}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
block_curvature.py — §6.4 Curvature / loss-landscape.

Beyond the free curvature estimate the optimizer hands us (§6.1's rotated ``exp_avg_sq``),
this block measures the loss-landscape geometry **explicitly** on the fixed Hessian batch
(briefing Part II §4 invariant 3, §6.4):

  * **Top-k Hessian eigenvalues (Ritz values)** from a single Lanczos pass
    (``curvature.lanczos.lanczos_tridiag`` -> ``eigvalsh`` of the tridiagonal): the top
    eigenvalue is sharpness; the spectrum is the basin's spectral fingerprint — short-term
    evolvability in the local geometry, long-term evolvability in basin sharpness.
  * **Hutchinson trace** (``curvature.lanczos.hutchinson_trace``): mean curvature.
  * **Lanczos spectral density** (the tridiagonal ``alphas`` / ``betas`` from
    ``lanczos_tridiag``): a stochastic-Lanczos-quadrature fingerprint of the whole
    eigenvalue density, stored to the array store so the density can be reconstructed
    post-hoc without re-running the forward/backward passes.
  * **Participation ratio of the curvature spectrum, per layer** (a cheap diagonal-Fisher
    proxy from the per-layer gradient on the same fixed batch — the briefing's "cheap
    stand-in for the full intrinsic dimension d₉₀", which is its own §9 protocol).

The loss is BCE-with-logits (the multi-label phenology head; ``ctx.criterion``) on the one
fixed batch ``ctx.hessian_batch`` so sharpness numbers are comparable over time and across
conditions. Hessian-vector products use double-backprop through a loss CLOSURE that
recomputes the forward pass each call (required by ``curvature.lanczos``); we therefore put
``ctx.model`` in train-grad mode for the duration and restore its mode afterward.

Only ``trainable_named_parameters`` enter the Hessian — the frozen input stage
(``patch_embed.*`` / ``pos_embed``) is excluded everywhere, as in every other block.

Cost knobs (module constants, so the collector can cut Lanczos cost per the §2.3
utilization check): ``LANCZOS_K`` (eigenpairs), ``LANCZOS_N_ITER`` (Lanczos steps),
``HUTCHINSON_PROBES``. The top-k Ritz values AND the spectral density share ONE Lanczos
pass (the tridiagonal is reused — no second pass), so the dominant per-checkpoint cost is
``LANCZOS_N_ITER + HUTCHINSON_PROBES`` HVPs total, not ``2*LANCZOS_N_ITER + ...``. All
stochastic ops thread an explicit ``torch.Generator`` (``ctx.gen_curvature`` if the driver
supplies one, else a fresh seeded generator on the batch device) per CONVENTIONS.md §5.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# --- §2.3 cost knobs (tunable by the collector for the per-checkpoint utilization budget) -
# Each Lanczos step and each Hutchinson probe is ONE double-backprop HVP on ViT-L (a full
# forward + two backward passes through ~302M trainable params), so the per-checkpoint cost
# is dominated by BOTH the *total HVP count* ~= LANCZOS_N_ITER + HUTCHINSON_PROBES AND the
# *Hessian-batch size* (each HVP recomputes the forward+double-backward on that many images).
# We run ONE Lanczos pass and reuse its tridiagonal for BOTH the top-k Ritz values and the
# spectral density (no second Lanczos), and the HVP frees its autograd graph every iteration
# (the loss closure recomputes the forward each call and the inner ``torch.autograd.grad``
# uses ``retain_graph=False`` on its second backward — no graph accumulation across steps).
#
# COLLECTOR SPEED CUT (§6.4): halved HVP count (20->10 Lanczos, 8->4 Hutchinson) AND halved
# the effective Hessian-batch size (64->~32 images via the stratified-stride subsample below),
# so the per-checkpoint curvature cost drops ~4x (HVP count x batch size). All three are
# deterministic, so sharpness numbers stay comparable across checkpoints/conditions.
LANCZOS_K: int = 10          # number of top Hessian eigenpairs (briefing §6.4 "k=10")
LANCZOS_N_ITER: int = 10     # Lanczos steps (>= LANCZOS_K); the single pass's HVP budget
HUTCHINSON_PROBES: int = 4   # Rademacher probes for the stochastic trace (modest)

# Hessian-batch subsample cap (collector speed cut). ``ctx.hessian_batch`` is the fixed
# 64-image curvature batch, STRATIFIED as 16 images per joint label-cell in contiguous
# 16-blocks (order: neither[0:16], flower_only[16:32], fruit_only[32:48], both[48:64]). To
# cut HVP forward cost while STAYING balanced we take a DETERMINISTIC stride-2 slice
# (indices 0,2,4,...) of the batch: stride-2 over contiguous 16-blocks lands exactly 8 even
# indices in each 16-block, i.e. ~8 images per cell and ~32 total — NOT the first 32 rows
# (which would be only the neither+flower_only cells). The subsample is recorded in the
# returned summary (``hessian_n_images`` / ``hessian_subsample``) so it is reproducible and
# comparable across checkpoints/runs. Set HESSIAN_MAX_IMAGES >= batch size to disable.
HESSIAN_MAX_IMAGES: int = 32   # cap the curvature batch to ~this many images (stratified-stride)

_EPS: float = 1e-12


# =============================================================================
# helpers
# =============================================================================

def _resolve_generator(ctx, device: torch.device) -> torch.Generator:
    """Return an explicit ``torch.Generator`` on ``device`` for the Lanczos / Hutchinson
    probes. Prefer a driver-supplied ``ctx.gen_curvature``; otherwise derive a deterministic
    one from ``(ctx.seed, ctx.step)`` so probe noise is reproducible per checkpoint and
    never touches global RNG (CONVENTIONS.md §5)."""
    g = getattr(ctx, "gen_curvature", None)
    if isinstance(g, torch.Generator):
        if g.device == device:
            return g
        # re-seed a device-matched generator deterministically from the supplied one
        seed = int(torch.randint(0, 2 ** 31 - 1, (1,), generator=g).item())
        dg = torch.Generator(device=device)
        dg.manual_seed(seed)
        return dg
    seed = (int(getattr(ctx, "seed", 0)) * 1_000_003 + int(getattr(ctx, "step", 0))) & 0x7FFFFFFF
    dg = torch.Generator(device=device)
    dg.manual_seed(seed)
    return dg


def _participation_ratio(values: np.ndarray) -> float:
    """PR = (Σ λ)² / Σ λ² of a non-negative spectrum (effective # of directions)."""
    if values is None or values.size == 0:
        return float("nan")
    s = np.clip(values.astype(np.float64), 0.0, None)
    s2 = float(np.sum(s * s))
    if s2 <= 0.0:
        return float("nan")
    return float((float(np.sum(s)) ** 2) / s2)


def _move_batch(hessian_batch, device: torch.device):
    """Unpack ``ctx.hessian_batch`` (an ``(images, targets)`` pair) onto ``device``.
    Tolerates tensors already on-device."""
    images, targets = hessian_batch
    images = images.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    return images, targets


def _stratified_stride_subsample(images: torch.Tensor, targets: torch.Tensor,
                                 max_images: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """DETERMINISTICALLY subsample the fixed Hessian batch to <= ``max_images`` rows while
    PRESERVING the joint-cell balance, and report what was kept (CONVENTIONS.md §5: any
    subsampling is deterministic + recorded).

    ``ctx.hessian_batch`` is stratified as equal contiguous label-cell blocks (16 each:
    neither[0:16], flower_only[16:32], fruit_only[32:48], both[48:64]). A naive ``[:max_images]``
    head would only cover the first cells; instead we take a fixed **stride-2** slice
    ``images[0::stride]`` (indices 0,2,4,...). Stride-2 over contiguous 16-blocks lands exactly
    8 even indices in each 16-block, so the subsample keeps ~8 images per cell and ~32 total —
    the balance is preserved. The stride is the smallest integer that brings the row count to
    <= ``max_images`` (stride 1 = no subsample when the batch already fits).

    Returns ``(images_sub, targets_sub, info)`` where ``info`` records the stride, the kept
    index list, and the original/kept row counts so the choice is reproducible.
    """
    n = int(images.shape[0])
    info: Dict[str, Any] = {"n_original": n, "max_images": int(max_images)}
    if max_images is None or max_images <= 0 or n <= max_images:
        info.update({"stride": 1, "n_kept": n, "subsampled": False})
        return images, targets, info
    # smallest stride s>=2 with ceil(n/s) <= max_images  =>  s = ceil(n / max_images)
    stride = int(-(-n // int(max_images)))           # ceil division, >= 2 here since n>max_images
    idx = torch.arange(0, n, stride, device=images.device)
    images_sub = images.index_select(0, idx)
    targets_sub = targets.index_select(0, idx)
    info.update({
        "stride": stride,
        "n_kept": int(idx.numel()),
        "subsampled": True,
        "kept_indices": [int(i) for i in idx.detach().cpu().tolist()],
    })
    return images_sub, targets_sub, info


def _build_loss_closure(model: nn.Module, criterion, images: torch.Tensor,
                        targets: torch.Tensor):
    """Return a no-arg ``loss_fn()`` that recomputes the forward pass and returns the scalar
    BCE-with-logits loss WITH the autograd graph attached (so ``curvature.lanczos`` can take
    double-backprop HVPs). Targets are cast to the logits' float dtype for BCE.

    NOTE: no autocast here — Hessian-vector products need a clean float32 graph; bf16
    autocast inside double-backprop is numerically unreliable and unnecessary for the small
    fixed Hessian batch."""
    def loss_fn() -> torch.Tensor:
        logits = model(images)
        return criterion(logits, targets.to(logits.dtype))
    return loss_fn


# =============================================================================
# per-layer cheap curvature proxy (diagonal-Fisher participation ratio)
# =============================================================================

def _per_layer_gradfisher_participation(ctx, model, criterion,
                                        images, targets) -> int:
    """Cheap per-layer curvature proxy (briefing §6.4 "effective-dimensionality proxy:
    participation ratio of the curvature or gradient-covariance spectrum, per layer").

    A single backward pass on the fixed batch gives each trainable layer its gradient
    ``g_l``; the squared-gradient values ``g_l**2`` are a diagonal Fisher / curvature proxy,
    and their participation ratio is a cheap stand-in for that layer's effective curvature
    dimension. One backward, no double-backprop — far cheaper than a per-layer Hessian.

    Returns the number of layers recorded. Frozen input-stage params are excluded by
    construction (``trainable_named_parameters``)."""
    from preadapt_common import trainable_named_parameters

    named = trainable_named_parameters(model)
    params = [p for _, p in named]
    model.zero_grad(set_to_none=True)
    logits = model(images)
    loss = criterion(logits, targets.to(logits.dtype))
    grads = torch.autograd.grad(loss, params, retain_graph=False, allow_unused=True)

    c, r, step, wt = ctx.condition, ctx.run_id, ctx.step, ctx.wall_time
    n_recorded = 0
    for (name, _), g in zip(named, grads):
        if g is None:
            continue
        g2 = (g.detach().to(dtype=torch.float64).reshape(-1) ** 2)
        if g2.numel() == 0:
            continue
        pr = _participation_ratio(g2.cpu().numpy())
        ctx.scalar.add(c, r, step, wt, quantity="gradfisher_participation",
                       value=pr, layer=name)
        # also the per-layer gradient-Fisher trace (Σ g²) — mean curvature proxy per layer
        ctx.scalar.add(c, r, step, wt, quantity="gradfisher_trace",
                       value=float(g2.sum().item()), layer=name)
        n_recorded += 1
    model.zero_grad(set_to_none=True)
    return n_recorded


# =============================================================================
# entry point
# =============================================================================

def extract(ctx) -> dict:
    """§6.4 curvature / loss-landscape on the fixed Hessian batch. See module docstring.

    Writes scalars: ``hessian_top_eig``, ``hessian_trace``, plus per-layer
    ``gradfisher_participation`` / ``gradfisher_trace``. Writes arrays: the top-k Hessian
    eigenvalue spectrum (``hessian_eig_spectrum``), the Lanczos tridiagonal density
    (``lanczos_alphas`` / ``lanczos_betas``), and the curvature-spectrum participation
    ratio over the top-k Ritz values (``hessian_eig_participation`` scalar).

    Returns a compact summary dict for the collector log. Robust to a missing Hessian batch
    (returns an empty summary rather than raising)."""
    from curvature.lanczos import hutchinson_trace, lanczos_tridiag
    from preadapt_common import trainable_named_parameters

    if getattr(ctx, "hessian_batch", None) is None:
        return {"block": "curvature", "skipped": "no hessian_batch"}

    model = ctx.model
    criterion = ctx.criterion
    device = ctx.device

    # Determine batch device from the model's parameters (model is already loaded/eval).
    try:
        param_device = next(model.parameters()).device
    except StopIteration:
        param_device = torch.device(device)
    images, targets = _move_batch(ctx.hessian_batch, param_device)
    # Collector speed cut: deterministically halve the Hessian batch (64->~32) while keeping the
    # joint-cell balance, so every HVP recomputes the forward+double-backward on fewer images.
    images, targets, _sub_info = _stratified_stride_subsample(images, targets, HESSIAN_MAX_IMAGES)

    c, r, step, wt = ctx.condition, ctx.run_id, ctx.step, ctx.wall_time
    gen = _resolve_generator(ctx, param_device)

    # The HVP closure needs grads to flow through the trainable params. Snapshot and set
    # requires_grad, run train() (so dropout/etc are consistent — phenology ViT has none of
    # significance, but be explicit), then restore both afterward.
    named = trainable_named_parameters(model, include_requires_grad_false=True)
    params = [p for _, p in named]
    prev_rg = [p.requires_grad for p in params]
    prev_training = model.training
    for p in params:
        p.requires_grad_(True)
    model.train()

    summary: Dict[str, Any] = {"block": "curvature"}
    # Record the deterministic subsample + cost knobs so the curvature numbers are reproducible
    # and comparable across checkpoints/runs (CONVENTIONS.md §5).
    summary["hessian_n_images"] = int(_sub_info.get("n_kept", images.shape[0]))
    summary["hessian_subsample"] = _sub_info
    summary["lanczos_n_iter"] = int(LANCZOS_N_ITER)
    summary["hutchinson_probes"] = int(HUTCHINSON_PROBES)
    try:
        loss_fn = _build_loss_closure(model, criterion, images, targets)

        # --- ONE Lanczos pass: drives BOTH the top-k Ritz values AND the spectral density.
        # Running ``top_k_eigenpairs`` and ``lanczos_tridiag`` separately would double the HVP
        # count; instead we take the single tridiagonal (alphas/betas) here and diagonalize
        # the small (n_iter x n_iter) ``T`` ourselves for the Ritz values. n_iter is clamped
        # to be at least LANCZOS_K so eigh(T) yields k Ritz values.
        ev = np.array([], dtype=np.float64)
        a = np.array([], dtype=np.float32)
        b = np.array([], dtype=np.float32)
        try:
            n_iter = max(int(LANCZOS_N_ITER), int(LANCZOS_K))
            alphas, betas, _ = lanczos_tridiag(loss_fn, params, n_iter, generator=gen)
            a = alphas.detach().to("cpu", dtype=torch.float32).numpy()
            b = betas.detach().to("cpu", dtype=torch.float32).numpy()
            # Build the tridiagonal T from (alphas, betas) and take its eigenvalues = Ritz vals.
            af = alphas.detach().to("cpu", dtype=torch.float64)
            bf = betas.detach().to("cpu", dtype=torch.float64)
            T = torch.diag(af)
            if bf.numel():
                idx = torch.arange(bf.numel())
                T[idx, idx + 1] = bf
                T[idx + 1, idx] = bf
            ritz = torch.linalg.eigvalsh(T)
            ritz = torch.sort(ritz, descending=True).values[:LANCZOS_K]
            ev = ritz.to(dtype=torch.float64).numpy()
        except Exception as e:  # never kill the collector on a single bad checkpoint
            summary["lanczos_error"] = repr(e)

        # top-k Hessian eigenpairs (Ritz values)
        if ev.size > 0:
            ctx.array.put(group="hessian_eig_spectrum", step=step,
                          array=ev.astype(np.float32))
            ctx.scalar.add(c, r, step, wt, quantity="hessian_top_eig", value=float(ev[0]))
            # participation ratio over the top-k Ritz values (clamped non-negative)
            ctx.scalar.add(c, r, step, wt, quantity="hessian_eig_participation",
                           value=_participation_ratio(ev))
            summary["hessian_top_eig"] = float(ev[0])
            summary["n_eigs"] = int(ev.size)

        # Lanczos spectral density (the tridiagonal alphas/betas from the SAME pass)
        if a.size > 0:
            ctx.array.put(group="lanczos_alphas", step=step, array=a)
        if b.size > 0:
            ctx.array.put(group="lanczos_betas", step=step, array=b)
        summary["lanczos_n_alphas"] = int(a.size)

        # --- Hutchinson trace -----------------------------------------------------------
        try:
            tr = hutchinson_trace(loss_fn, params, n_probes=HUTCHINSON_PROBES, generator=gen)
            ctx.scalar.add(c, r, step, wt, quantity="hessian_trace", value=float(tr))
            summary["hessian_trace"] = float(tr)
        except Exception as e:
            summary["hessian_trace_error"] = repr(e)

    finally:
        # restore grad flags + training mode regardless of any failure above
        for p, rg in zip(params, prev_rg):
            p.requires_grad_(rg)
        model.zero_grad(set_to_none=True)
        if not prev_training:
            model.eval()

    # --- per-layer cheap curvature proxy (separate single backward, no double-backprop) --
    # Done after restoring grad flags: it uses trainable_named_parameters (requires_grad
    # True) directly. Run in no-autocast float for a clean gradient.
    try:
        n_layers = _per_layer_gradfisher_participation(ctx, model, criterion, images, targets)
        summary["n_layer_proxy"] = n_layers
    except Exception as e:
        summary["gradfisher_error"] = repr(e)
    finally:
        model.zero_grad(set_to_none=True)
        if not prev_training:
            model.eval()

    return summary

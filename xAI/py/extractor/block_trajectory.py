#!/usr/bin/env python3
"""
block_trajectory.py — extractor block §6.3: Trajectory geometry (the *type* of movement).

STATELESS REDESIGN 2026-06-16 (CONTRACT (B)/(C)/(D)). §6.3 trajectory geometry is INHERENTLY
CROSS-CHECKPOINT — every one of its diagnostics needs more than the single checkpoint in hand:

  * **(a) Straightness** ``R = net_displacement / path_length`` needs the trainer's inline
    accumulated ``path_length`` history (a cross-checkpoint time series).
  * **(b) Mean-squared-displacement scaling** ``MSD(tau)`` needs the WHOLE accumulated trajectory
    of projected coordinates (every prior checkpoint's ``proj_coords``).
  * **(c) Velocity autocorrelation** ``C(tau)`` is an inline cross-step time series.
  * **(d) Spectral flat/steep split of MOTION** needs the displacement-since-the-PREVIOUS
    checkpoint and the init curvature basis ``Q0`` — both cross-checkpoint.

Because per-checkpoint extraction is now STATELESS and order-independent (two collectors may run
``extract`` on different checkpoints concurrently and out of order — CONTRACT (B)), NONE of these
can be computed correctly on the live path. They are ALL MOVED to the **POST-HOC pass**
(CONTRACT (D)), which runs ONCE per run after the trainer + both collectors finish and reads:

  * the per-step projected coordinates (the §6.2 ``proj_coords`` / ``proj_coords_layer`` arrays
    block_weights stored) -> §6.3(b) MSD(tau) + slope + plateau, computed across all steps;
  * the trainer's inline ``path_length`` / ``velocity_norm`` / ``velocity_autocorr`` /
    ``train_loss`` log -> §6.3(a) straightness and §6.3(c) velocity autocorrelation;
  * the per-step stored top-k QL/QR eigvecs (``ql_eigvecs`` / ``qr_eigvecs`` from block_optim) as
    the curvature basis, and consecutive-step displacements -> §6.3(d) motion flat/steep split.

What this LIVE block does now
-----------------------------
``extract(ctx) -> dict`` stays importable and runs on every checkpoint, but only records the ONE
trajectory quantity that is genuinely per-checkpoint independent: the global **net displacement
from the FIXED INIT reference** ``||theta(t) - theta(0)||`` over the trainable params (frozen
stage excluded). ``theta(0)`` is the run's fixed on-disk INIT reference threaded in as
``ctx.init_model_sd`` (CONTRACT (C)) — it does NOT depend on any previously-processed checkpoint,
so this is order-independent. It is the same scalar §6.2 (block_weights) reports as its primary
vs-init displacement; we re-emit it here under ``net_disp_vs_init`` so the trajectory timeline
carries a position-magnitude anchor for the post-hoc pass to normalize against. Everything else
(``straightness`` / ``msd_slope`` / ``msd_plateau`` / ``velocity_autocorr`` /
``motion_frac_flat`` / ``motion_frac_steep`` / ``loss_per_motion_*`` / ``net_disp_since_prev``)
is NO LONGER written here — it is produced by the POST-HOC §6.3 pass.

Robustness / RNG
----------------
Deterministic (no stochastic ops), so no ``torch.Generator`` is threaded (mirrors ``block_optim``).
Robust to a missing INIT reference (``ctx.init_model_sd`` None -> the one scalar is NaN/skipped)
and NEVER raises out of ``extract``.
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import numpy as np
import torch

# Project paths so preadapt_common imports cleanly when run from anywhere.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PY = os.path.dirname(_HERE)  # xAI/py
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_PY))  # repo root
_PLANTCLEF = os.path.join(_PROJECT_ROOT, "PlantCLEF2022")
_TWO_NOISE = "/blue/guralnick/share/r.dinnage/Projects/phenovision/xAI/two_noise"
for _p in (_PY, _PROJECT_ROOT, _PLANTCLEF, _TWO_NOISE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if TYPE_CHECKING:  # the dataclass is defined by the collector/driver; import lazily.
    from extractor._ctx import ExtractCtx  # noqa: F401

_EPS = 1e-12


# =============================================================================
# trainable-parameter displacement helper (per-checkpoint independent: vs the FIXED init)
# =============================================================================

def _net_displacement(model, ref_sd: Optional[Dict[str, Any]]) -> float:
    """Global ``||theta(t) - theta_ref||`` over trainable params (frozen stage excluded).

    ``ref_sd`` is the FIXED INIT (E(0)) reference (``ctx.init_model_sd``, loaded from disk), so
    this is a single-checkpoint quantity that does not depend on processing order. Returns NaN
    if ``ref_sd`` is None. Per-layer values are intentionally NOT computed here (the per-layer
    travel signal is a §6.2 / post-hoc concern)."""
    from preadapt_common import trainable_named_parameters
    if ref_sd is None:
        return float("nan")
    total_sq = 0.0
    for name, p in trainable_named_parameters(model, include_requires_grad_false=True):
        if name not in ref_sd:
            continue
        ref = ref_sd[name]
        if tuple(ref.shape) != tuple(p.shape):
            continue
        d = (p.detach().to("cpu", dtype=torch.float64).reshape(-1)
             - ref.detach().to("cpu", dtype=torch.float64).reshape(-1))
        total_sq += float(d.norm().item()) ** 2
    return float(total_sq ** 0.5)


# =============================================================================
# entry point (LIVE; near-no-op — §6.3 is POST-HOC, CONTRACT (D))
# =============================================================================

def extract(ctx: "ExtractCtx") -> Dict[str, object]:
    """§6.3 trajectory geometry — LIVE per-checkpoint stub. See module docstring.

    Writes only ``net_disp_vs_init`` (global) — the one per-checkpoint-independent trajectory
    quantity (displacement from the FIXED on-disk INIT reference). All cross-checkpoint §6.3
    diagnostics (straightness / MSD / velocity-autocorr / motion flat-steep split) are computed
    in the POST-HOC pass. Robust to a missing INIT reference; NEVER raises.

    Returns a compact summary dict for the collector log.
    """
    summary: Dict[str, object] = {
        "block": "block_trajectory",
        "live": "net_disp_vs_init_only",
        "trajectory_geometry": "post_hoc",  # §6.3(a-d) are computed post-hoc (CONTRACT (D))
    }

    init_sd = getattr(ctx, "init_model_sd", None)
    summary["init_reference_available"] = bool(init_sd is not None)

    net_init = float("nan")
    try:
        net_init = _net_displacement(ctx.model, init_sd)
        if np.isfinite(net_init) and ctx.scalar is not None:
            ctx.scalar.add(ctx.condition, ctx.run_id, ctx.step, ctx.wall_time,
                           quantity="net_disp_vs_init", value=float(net_init))
    except Exception as e:  # never kill extraction
        print(f"[block_trajectory] net_disp_vs_init skipped: {e}", file=sys.stderr)
    summary["net_disp_vs_init"] = net_init

    return summary

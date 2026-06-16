#!/usr/bin/env python3
"""
block_optim.py — §6.1 Optimizer geometric state (metric evolution; the headline).

This is the most theory-loaded block in the extractor. Training uses a
curvature-adaptive SOAP-family optimizer (``StableEvolutionSOAP``), so the optimizer's
own slow state is already a running estimate of the loss-landscape geometry — the moving
Riemannian metric on the loss landscape, the analog of the mutational/genetic covariance
(M/G). Harvesting it turns each checkpoint into a direct measurement of *metric*
evolution, not just *position* evolution (briefing Part II §6.1 / §6.0). It is nearly
free because the optimizer computes it anyway, and standard fine-tuning logging discards
it.

What we read (per the §8 reconciliation against the real ``StableEvolutionSOAP`` class):
per-preconditioned-2D-parameter state =
  ``QL, QR``  Kronecker eigenbases — eigenvectors of ``L = E[G Gᵀ]`` (m×m) and
              ``R = E[Gᵀ G]`` (n×n), i.e. the slow metric directions.
  ``L, R``    the Kronecker accumulators themselves (square, symmetric PSD).
  ``exp_avg_sq``  rotated second moment — per-coordinate curvature-eigenvalue estimate.
  ``precond`` the GENERATED preconditioner eigenvalues P (StableEvo-specific; the
              multiplicative Riccati output, NOT a direct power of the second moment).
  ``alpha_last``  realized per-coordinate exponent α ∈ [alpha_min, alpha_max] (size-
              weighted mean = the optimizer-side "operative exponent" the cross-substrate
              law is about).
The param group carries the canalization-floor scalars ``alpha_min, alpha_max, kappa,
damping``. There is **no** ``precond_power`` (dynamic exponent) and **no** evolving-``M``
target buffer, so ``detect_evolving_M -> False``: the constant-isotropic canalization
floor case (briefing §6.1 "if M is absent"). We record those scalars and skip the
matrix-valued M extraction.

For the **AdamW** baseline variant only diagonal ``exp_avg`` / ``exp_avg_sq`` exist (no
``QL/QR/L/R/precond/alpha_last``); we record the per-layer diagonal-``exp_avg_sq``
spectrum / trace / participation ratio and skip everything Kronecker gracefully. Both
paths produce well-formed records that differ only in which quantities exist.

We work directly off the **saved tensors** in ``ctx.opt['state']`` (and the previous
checkpoint's ``ctx.prev_opt`` and the init basis ``ctx.Q0``); we never reconstruct the
live optimizer. Heavy reductions (eigh) run on CPU/float64 — the same defensive choice as
``curvature/operative_exponent.py``, because cuSOLVER's GPU ``eigh`` raises on the
ill-conditioned Kronecker factors that show up mid-training, and these factors are small
(Kronecker dims ≤ ``max_precond_dim``).

Conventions (xAI/two_noise/CONVENTIONS.md §5): no global RNG; this block is deterministic
(eigendecompositions only), so no generator is threaded here — the curvature block, which
does use Lanczos probes, threads ``ctx`` generators instead.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch

# Number of leading EIGENVALUES summarized per Kronecker factor / per diagonal spectrum
# (the compact ``*_topk`` vector written to the scalar store). This is purely a spectrum
# summary width and does NOT drive any matrix work, so it stays at 32. Module constant so
# the collector can dial cost without touching code.
TOP_K: int = 32

# Number of leading EIGENVECTORS (QL/QR columns) retained for the principal-angle rotation
# diagnostics (vs previous checkpoint and vs init Q0). Halved from TOP_K (32 -> 16) as the
# approved §6.1 collector speed cut: the principal-angle SVD then runs on 16-dim subspaces,
# roughly quartering that work, while every eigenVALUE spectrum / trace / participation /
# precond / realized-exponent / canalization-floor quantity is untouched (none depend on
# this count). Briefing §6.1: "k of order 16-64 is enough" — 16 is the low end of that band.
TOPK_EIGVECS: int = 16

# Floors / guards for log-space and division stability.
_EPS: float = 1e-12

# Above this factor dimension we do the (CPU) symmetric eigendecomposition in float32 rather
# than float64: a float64 eigh of a large [>=_FLOAT64_EIGH_MAX] matrix transiently allocates
# several copies (the matrix + workspace) at 8 bytes/elem, which is wasteful host RAM for the
# big Kronecker factors of an over-cap run. With the trainer's --max_precond_dim 2048 cap the
# factors are <= 2048 and stay float64; this is a defensive guard for over-cap checkpoints.
_FLOAT64_EIGH_MAX: int = 2048


# =============================================================================
# small numeric helpers (CPU/float64; robust to ill-conditioned factors)
# =============================================================================

def _sym_eigvals_desc(mat: torch.Tensor) -> Optional[np.ndarray]:
    """Descending eigenvalues of a (assumed symmetric PSD) square matrix, computed on CPU for
    numerical safety. Returns a 1-D numpy array (descending), or None if the decomposition fails
    entirely (never kills the run — mirrors operative_exponent.py).

    Factors up to ``_FLOAT64_EIGH_MAX`` use float64 (the original numerically-safe path; the
    trainer's --max_precond_dim 2048 cap keeps every real factor here). LARGER factors — only
    seen on an over-cap run — use float32 to avoid the several transient 8-byte/elem copies a
    big-matrix float64 eigh allocates in host RAM (defensive: bounds the extractor footprint)."""
    if mat is None:
        return None
    n = mat.shape[0] if mat.dim() == 2 else 0
    dtype = torch.float64 if n <= _FLOAT64_EIGH_MAX else torch.float32
    m = mat.detach().to("cpu", dtype=dtype)
    if m.dim() != 2 or m.shape[0] != m.shape[1]:
        return None
    m = 0.5 * (m + m.t())  # symmetrize defensively
    try:
        w = torch.linalg.eigvalsh(m)
    except Exception:
        # Last-ditch fallback: retry in float64 (small/ill-conditioned factors only).
        try:
            w = torch.linalg.eigvalsh(m.double())
        except Exception:
            return None
    w = torch.flip(w, dims=[0])  # descending
    return w.to(torch.float64).numpy()


def _participation_ratio(spectrum: np.ndarray) -> float:
    """Participation ratio PR = (Σ λ)² / Σ λ² of a non-negative spectrum — the effective
    number of significant directions (briefing §6.4 "effective-dimensionality proxy").
    Computed on the clamped-non-negative spectrum so a tiny negative Ritz/eigh artifact
    cannot blow it up. Returns NaN for an empty / all-zero spectrum."""
    if spectrum is None or spectrum.size == 0:
        return float("nan")
    s = np.clip(spectrum.astype(np.float64), 0.0, None)
    s2 = float(np.sum(s * s))
    if s2 <= 0.0:
        return float("nan")
    return float((float(np.sum(s)) ** 2) / s2)


def _top_columns(Q: Optional[torch.Tensor], k: int) -> Optional[torch.Tensor]:
    """Return the leading ``k`` columns of an eigenbasis ``Q`` (the basis stores
    eigenvectors in descending-eigenvalue order — ``_eigh_basis`` in the optimizer flips
    them so column 0 is the top direction). CPU/float32 for the principal-angle SVD."""
    if Q is None:
        return None
    Qc = Q.detach().to("cpu", dtype=torch.float32)
    if Qc.dim() != 2 or Qc.shape[1] == 0:
        return None
    return Qc[:, : min(k, Qc.shape[1])]


def _principal_angle_summary(U: Optional[torch.Tensor], V: Optional[torch.Tensor]):
    """Mean cosine of the principal angles between the column spaces of ``U`` and ``V``
    (1 = perfectly aligned, 0 = orthogonal), plus the full angle vector in radians.

    Reuses ``curvature.lanczos.principal_angles`` (which QR-orthonormalizes internally, so
    passing the top-k columns of the already-orthonormal bases is fine). Returns
    ``(mean_cos, angles_rad_ndarray)`` or ``(nan, None)`` if either basis is missing /
    the dimensions are incompatible."""
    if U is None or V is None:
        return float("nan"), None
    if U.shape[0] != V.shape[0]:
        return float("nan"), None
    try:
        from curvature.lanczos import principal_angles
    except Exception:
        return float("nan"), None
    try:
        ang = principal_angles(U, V)  # ascending radians, length min(k1,k2)
    except Exception:
        return float("nan"), None
    ang_np = ang.detach().to("cpu", dtype=torch.float64).numpy()
    if ang_np.size == 0:
        return float("nan"), None
    mean_cos = float(np.mean(np.cos(ang_np)))
    return mean_cos, ang_np


def _diag_spectrum_desc(t: Optional[torch.Tensor]) -> Optional[np.ndarray]:
    """Flatten a per-coordinate tensor (``exp_avg_sq`` / ``precond``) to a 1-D descending
    spectrum on CPU/float64. ``exp_avg_sq`` lives in the rotated eigenbasis, so its sorted
    values are the per-layer curvature-eigenvalue estimate (briefing §6.1)."""
    if t is None:
        return None
    v = t.detach().to("cpu", dtype=torch.float64).reshape(-1)
    if v.numel() == 0:
        return None
    v = torch.sort(v, descending=True).values
    return v.numpy()


def _topk_array(spectrum: np.ndarray, k: int) -> np.ndarray:
    """First ``k`` entries of a descending spectrum, right-padded with NaN to length ``k``
    so the array store has a fixed per-layer slice width across checkpoints."""
    out = np.full((k,), np.nan, dtype=np.float32)
    n = min(k, spectrum.size)
    out[:n] = spectrum[:n].astype(np.float32)
    return out


# =============================================================================
# per-layer extraction
# =============================================================================

def _extract_preconditioned_layer(ctx, name: str, st: Dict[str, Any],
                                   prev_st: Optional[Dict[str, Any]],
                                   q0: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """Extract the §6.1 quantities for one preconditioned 2-D layer (StableEvo path).

    Writes scalars via ``ctx.scalar.add(layer=name)`` and full spectra / angle vectors via
    ``ctx.array.put(layer=name)``. Returns a small per-layer summary dict."""
    c, r, step, wt = ctx.condition, ctx.run_id, ctx.step, ctx.wall_time
    summary: Dict[str, float] = {}

    def s(quantity: str, value: float, head: Optional[int] = None) -> None:
        ctx.scalar.add(c, r, step, wt, quantity=quantity, value=float(value),
                       layer=name, head=head)

    # --- Kronecker-factor spectra L, R (top-k eigvals, trace, participation) ------------
    for fac in ("L", "R"):
        spec = _sym_eigvals_desc(st.get(fac))
        if spec is None:
            continue
        ctx.array.put(group=f"kron_{fac}_spectrum", step=step, array=spec.astype(np.float32),
                      layer=name)
        s(f"kron_{fac}_top_eig", float(spec[0]))
        s(f"kron_{fac}_trace", float(np.sum(np.clip(spec, 0.0, None))))
        s(f"kron_{fac}_participation", _participation_ratio(spec))
        summary[f"{fac}_top_eig"] = float(spec[0])

    # --- basis rotation rate: vs previous checkpoint and vs init Q0 ----------------------
    # Rotation diagnostics use only the leading TOPK_EIGVECS (16) columns of each basis, so
    # the principal-angle SVD runs on 16-dim subspaces (the §6.1 cost cut). Eigenvalue
    # spectra above/below use TOP_K independently.
    QL_now = _top_columns(st.get("QL"), TOPK_EIGVECS)
    QR_now = _top_columns(st.get("QR"), TOPK_EIGVECS)

    prev_QL = _top_columns(prev_st.get("QL"), TOPK_EIGVECS) if prev_st else None
    prev_QR = _top_columns(prev_st.get("QR"), TOPK_EIGVECS) if prev_st else None
    q0_QL = _top_columns(q0.get("QL"), TOPK_EIGVECS) if q0 else None
    q0_QR = _top_columns(q0.get("QR"), TOPK_EIGVECS) if q0 else None

    # The scalar "rotation" we log is mean principal-angle COSINE (alignment): 1 = the
    # metric basis has not moved, lower = it has rotated. We log it for QL (and QR) against
    # both the previous checkpoint and the ancestral init basis (the phylogenetic
    # "metric divergence from ancestor" signal). Full angle vectors go to the array store.
    for tag, U_now, U_prev, U_init in (("ql", QL_now, prev_QL, q0_QL),
                                       ("qr", QR_now, prev_QR, q0_QR)):
        mc_prev, ang_prev = _principal_angle_summary(U_now, U_prev)
        mc_init, ang_init = _principal_angle_summary(U_now, U_init)
        if not np.isnan(mc_prev):
            s(f"{tag}_rotation_vs_prev", mc_prev)
            summary[f"{tag}_rotation_vs_prev"] = mc_prev
        if ang_prev is not None:
            ctx.array.put(group=f"{tag}_principal_angles_vs_prev", step=step,
                          array=ang_prev.astype(np.float32), layer=name)
        if not np.isnan(mc_init):
            s(f"{tag}_rotation_vs_init", mc_init)
            summary[f"{tag}_rotation_vs_init"] = mc_init
        if ang_init is not None:
            ctx.array.put(group=f"{tag}_principal_angles_vs_init", step=step,
                          array=ang_init.astype(np.float32), layer=name)

    # --- curvature-eigenvalue spectrum: rotated exp_avg_sq (free Hessian/Fisher est.) ----
    eas = _diag_spectrum_desc(st.get("exp_avg_sq"))
    if eas is not None:
        ctx.array.put(group="exp_avg_sq_spectrum", step=step, array=eas.astype(np.float32),
                      layer=name)
        s("curvature_eig_top", float(eas[0]))
        s("curvature_eig_trace", float(np.sum(eas)))
        s("curvature_eig_participation", _participation_ratio(eas))
        # compact top-k summary as a small vector too (queryable from the scalar store)
        ctx.scalar.add_vector(c, r, step, wt, quantity="curvature_eig_topk",
                              values=_topk_array(eas, TOP_K), layer=name)
        summary["curvature_eig_top"] = float(eas[0])

    # --- generated preconditioner P spectrum (StableEvo-specific) ------------------------
    pspec = _diag_spectrum_desc(st.get("precond"))
    if pspec is not None:
        ctx.array.put(group="precond_spectrum", step=step, array=pspec.astype(np.float32),
                      layer=name)
        s("precond_top", float(pspec[0]))
        s("precond_trace", float(np.sum(np.clip(pspec, 0.0, None))))
        s("precond_participation", _participation_ratio(pspec))

    # --- realized exponent from alpha_last (per-layer size-weighted mean) -----------------
    al = st.get("alpha_last")
    if al is not None:
        a = al.detach().to("cpu", dtype=torch.float64).reshape(-1)
        if a.numel() > 0:
            mean_a = float(a.mean())
            s("realized_exponent_mean", mean_a)
            s("realized_exponent_std", float(a.std(unbiased=False)))
            s("realized_exponent_max", float(a.max()))
            summary["realized_exponent_mean"] = mean_a
            summary["_alpha_sum"] = float(a.sum())
            summary["_alpha_n"] = float(a.numel())

    return summary


def _extract_diagonal_layer(ctx, name: str, st: Dict[str, Any]) -> Dict[str, float]:
    """AdamW path (and any param that fell back to the diagonal Adam branch in StableEvo):
    only the diagonal ``exp_avg_sq`` exists. Record its per-layer spectrum / trace /
    participation ratio and nothing Kronecker."""
    c, r, step, wt = ctx.condition, ctx.run_id, ctx.step, ctx.wall_time
    eas = _diag_spectrum_desc(st.get("exp_avg_sq"))
    if eas is None:
        return {}
    ctx.array.put(group="exp_avg_sq_spectrum", step=step, array=eas.astype(np.float32),
                  layer=name)
    ctx.scalar.add(c, r, step, wt, quantity="curvature_eig_top", value=float(eas[0]),
                   layer=name)
    ctx.scalar.add(c, r, step, wt, quantity="curvature_eig_trace", value=float(np.sum(eas)),
                   layer=name)
    ctx.scalar.add(c, r, step, wt, quantity="curvature_eig_participation",
                   value=_participation_ratio(eas), layer=name)
    ctx.scalar.add_vector(c, r, step, wt, quantity="curvature_eig_topk",
                          values=_topk_array(eas, TOP_K), layer=name)
    return {"curvature_eig_top": float(eas[0])}


# =============================================================================
# canalization-floor (M absent) — global scalars from the param group
# =============================================================================

def _record_canalization_floor(ctx) -> Dict[str, float]:
    """``detect_evolving_M -> False`` (StableEvo and AdamW): the optimizer's "target" is a
    constant isotropic floor set by ``damping`` / ``kappa`` / ``alpha_min`` / ``alpha_max``
    (the canalization floor), not an evolving M-matrix (briefing §6.1 "if M is absent").
    Record those group scalars and skip the matrix-valued M block. Robust to AdamW, where
    only a subset (or none) of these keys exists in the group."""
    c, r, step, wt = ctx.condition, ctx.run_id, ctx.step, ctx.wall_time
    grp = ctx.opt.get("group", {}) or {}
    recorded: Dict[str, float] = {}
    for key in ("damping", "kappa", "alpha_min", "alpha_max"):
        if key in grp and grp[key] is not None:
            try:
                val = float(grp[key])
            except (TypeError, ValueError):
                continue
            ctx.scalar.add(c, r, step, wt, quantity=f"canalization_floor_{key}", value=val)
            recorded[key] = val
    return recorded


# =============================================================================
# entry point
# =============================================================================

def extract(ctx) -> dict:
    """§6.1 optimizer geometric state. See module docstring.

    Iterates the saved per-parameter optimizer state in ``ctx.opt['state']`` (keyed by
    parameter name). For each preconditioned 2-D layer (those carrying ``QL`` / ``QR``)
    runs the full StableEvo extraction; for diagonal-only params records the diagonal
    ``exp_avg_sq`` spectrum. Skips the frozen input stage (it never enters the optimizer,
    so it is absent from ``ctx.opt['state']`` by construction, but we guard anyway).
    Gates the matrix-valued M block on ``ctx.opt['M_present']`` (False here) and records
    the canalization-floor scalars instead.

    Returns a compact summary dict for the collector log.
    """
    # Lazy import so the module is importable before the driver lands _ctx.py.
    from preadapt_common import is_frozen_input_stage

    variant = (ctx.opt or {}).get("variant", "unknown")
    state = (ctx.opt or {}).get("state", {}) or {}
    prev_state = (ctx.prev_opt or {}).get("state", {}) if ctx.prev_opt else {}
    q0 = ctx.Q0 or {}

    n_precond = 0
    n_diag = 0
    alpha_sum = 0.0
    alpha_n = 0.0
    layer_summaries: Dict[str, Dict[str, float]] = {}

    for name, st in state.items():
        if not isinstance(st, dict) or not st:
            continue
        if is_frozen_input_stage(name):
            continue  # defensive: frozen stage is excluded from the optimizer anyway

        has_kron = (st.get("QL") is not None) or (st.get("QR") is not None)
        if has_kron:
            prev_st = prev_state.get(name) if prev_state else None
            q0_layer = q0.get(name) if q0 else None
            summ = _extract_preconditioned_layer(ctx, name, st, prev_st, q0_layer)
            n_precond += 1
            layer_summaries[name] = summ
            alpha_sum += summ.get("_alpha_sum", 0.0)
            alpha_n += summ.get("_alpha_n", 0.0)
        else:
            # Diagonal Adam fallback (AdamW variant, or 1-D / over-threshold params under
            # StableEvo). Record what exists; skip QL/QR/precond/alpha_last gracefully.
            if st.get("exp_avg_sq") is not None:
                _extract_diagonal_layer(ctx, name, st)
                n_diag += 1

    # --- global realized-exponent: size-weighted mean across all preconditioned layers ---
    if alpha_n > 0:
        global_alpha = alpha_sum / alpha_n
        ctx.scalar.add(ctx.condition, ctx.run_id, ctx.step, ctx.wall_time,
                       quantity="realized_exponent_mean", value=float(global_alpha))
    else:
        global_alpha = float("nan")

    # --- M block (gated) -----------------------------------------------------------------
    m_present = bool((ctx.opt or {}).get("M_present", False))
    floor = {}
    if not m_present:
        # The expected, fully-valid non-evolving case for StableEvo/AdamW (briefing §8).
        floor = _record_canalization_floor(ctx)
    # (If a future evolving-M optimizer sets M_present=True, the matrix-valued M extraction
    #  would go here; StableEvo/AdamW never reach it.)

    return {
        "block": "optim",
        "variant": variant,
        "M_present": m_present,
        "n_preconditioned_layers": n_precond,
        "n_diagonal_layers": n_diag,
        "realized_exponent_mean": global_alpha,
        "canalization_floor": floor,
        # Diagnostic widths recorded for reproducibility / cross-checkpoint comparability:
        # TOP_K = eigenVALUE spectrum summary width; TOPK_EIGVECS = #eigvecs (QL/QR columns)
        # used for the principal-angle rotation diagnostics (the halved §6.1 cost cut).
        "topk_eigvals": TOP_K,
        "topk_eigvecs": TOPK_EIGVECS,
    }

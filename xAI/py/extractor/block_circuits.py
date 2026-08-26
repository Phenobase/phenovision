#!/usr/bin/env python3
"""
block_circuits.py — extractor block §6.5: QK / OV attention circuits (WEIGHT-ONLY).

Per the briefing (``xAI/notes/phenovision_full_briefing.md`` Part II §6.5) each attention
head factors into two circuits (Elhage et al., transformer-circuits framework):

  * the **QK circuit** ``W_QK = W_Q^T W_K`` sets the attention *pattern* (which patches
    read from which); and
  * the **OV circuit** ``W_OV = W_O W_V`` sets *what* is written to the destination patch
    when attention is paid.

Both are functions of the weights alone, so this block forms them directly from the model
state dict with **NO forward pass** (the forward/backward interpretability metrics live in
``block_interp.py`` §6.6).

Architecture (ViT-L/16, verified against the running checkpoints):
  D = 1024, 24 blocks, 16 heads, head_dim d_h = 64. The timm ``Attention`` module uses a
  fused ``qkv`` Linear with weight ``[3D, D] = [3072, 1024]`` and a ``proj`` Linear
  ``[D, D] = [1024, 1024]``. The fused-qkv layout (matching timm's
  ``qkv(x).reshape(B, N, 3, num_heads, head_dim)``) is, per head ``h`` (rows
  ``h*d_h:(h+1)*d_h``):

      Wq_h = qkv.weight[0:D]      [h*d_h:(h+1)*d_h, :]      # (d_h, D)
      Wk_h = qkv.weight[D:2D]     [h*d_h:(h+1)*d_h, :]      # (d_h, D)
      Wv_h = qkv.weight[2D:3D]    [h*d_h:(h+1)*d_h, :]      # (d_h, D)
      Wo_h = proj.weight[:, h*d_h:(h+1)*d_h]                # (D, d_h)
      W_QK_h = Wq_h.T @ Wk_h                                 # (D, D), rank <= d_h
      W_OV_h = Wo_h   @ Wv_h                                 # (D, D), rank <= d_h

Per head / layer we store:
  * **Singular spectra** of ``W_OV`` and ``W_QK`` (effective rank; how the read and write
    transforms reshape) -> arrays ``ov_singvals`` / ``qk_singvals``; scalar effective-rank
    (participation-ratio) summaries ``ov_eff_rank`` / ``qk_eff_rank``.
  * **OV eigenvalue sign structure** — positive eigenvalues are copying/reinforcing,
    negative are anti-copying/suppressing. ``W_OV`` is generally non-symmetric, so we take
    the eigenvalues of the square matrix and report the fraction of *real* eigenvalue
    magnitude that is positive -> scalar ``ov_eig_sign_pos_frac`` (and the full complex
    spectrum, as a ``[k, 2]`` real/imag array ``ov_eigvals``). This operationalizes the
    signed, competition/mutualism-style patch interactions a plain attention heatmap
    discards.
  * **Circuit drift from init** — ``||W_OV(t) - W_OV(0)||`` and per-head principal angles
    between the current and init OV (and QK) top subspaces -> scalars ``ov_circuit_drift``
    / ``qk_circuit_drift`` and ``ov_principal_angle_mean`` / ``qk_principal_angle_mean``.

CAVEATS (do not overclaim — briefing §6.5).
  The clean QK/OV factorization is **exact only for attention-only transformers**. A real
  ViT block interleaves an MLP and LayerNorms, so for ViT-L this is an *idealization*: the
  MLP is a third computational element and LayerNorm should be folded into the adjacent
  weights for a faithful circuit (this first pass uses *raw* weights and notes the
  approximation). Cross-layer composition (induction-head path tracing through the residual
  stream) is a research thread, not a per-checkpoint metric — out of scope here.

Robustness: this block reads only model weights, so it is identical for the ``adamw`` and
``stable_evo`` optimizer variants (``ctx.opt`` is not touched). All stochastic ops (none are
strictly needed here, but SVD/eig on GPU can be flaky on ill-conditioned matrices) fall back
to CPU; no global RNG is used.

PERFORMANCE (the heavy collector's #1 cost, ~472 s on B200 for all 384 head-circuits).
Two approved cuts make this cheap and keep the result reproducible/comparable:

  1. **Deterministic head subsampling** (``CIRCUIT_HEAD_STRIDE``, default 4). Instead of all
     16 heads we process a *fixed* stride-spaced subset ``range(0, 16, stride)`` — heads
     ``0, 4, 8, 12`` at stride 4 (96 head-circuits total = 4 heads x 24 layers). The same set
     is used at every checkpoint and for both optimizer variants, so trajectories stay
     comparable. ``CIRCUIT_LAYER_STRIDE`` (default 1 = all layers) gives an optional second
     axis. The exact sampled set and strides are recorded in the returned summary.
     **The ``*_layer_mean`` head-averaged scalars are now means over the *sampled* heads**
     (documented; identical aggregation, fewer terms).

  2. **Small-matrix spectra** (no D x D dense SVD/eig). ``W_QK = Wq^T Wk`` and
     ``W_OV = Wo Wv`` are ``[D, D]`` but rank <= d_h = 64. We never materialize/decompose the
     dense ``[1024, 1024]``: singular values are obtained from a thin QR of the two ``[D, 64]``
     factors followed by an SVD of the ``[64, 64]`` core (exact, not approximate), and the OV
     eigenvalues use the ``eig(AB)=eig(BA)`` identity on the ``[64, 64]`` product ``Wv Wo``
     (the nonzero spectrum of ``Wo Wv`` is exactly that of ``Wv Wo``). Drift subspaces come
     from the same thin-QR core SVD. This replaces ~five ``[1024, 1024]`` decompositions per
     head with ``[64, 64]`` ones (orders of magnitude cheaper) AND removes the ~858 GB-class
     host-RAM pressure of dense per-head matrices.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import torch

# Reuse the project's principal-angle implementation (subspace alignment).
import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_TWO_NOISE = "/blue/guralnick/share/r.dinnage/Projects/phenovision/xAI/two_noise"
if _TWO_NOISE not in sys.path:
    sys.path.insert(0, _TWO_NOISE)

try:  # principal_angles is a pure-torch helper; degrade gracefully if the path moves.
    from curvature.lanczos import principal_angles as _principal_angles
    _HAVE_PA = True
except Exception:  # pragma: no cover - environment guard
    _HAVE_PA = False

if TYPE_CHECKING:  # the dataclass is defined by the collector/driver; import lazily.
    from extractor._ctx import ExtractCtx  # noqa: F401


# ViT-L/16 architecture facts (verified against the checkpoints).
N_BLOCKS = 24
N_HEADS = 16
HEAD_DIM = 64
D_MODEL = 1024

# How many top singular values / eigenvalues to keep per head (full rank is d_h=64).
TOPK_SPECTRUM = HEAD_DIM  # keep the whole per-head spectrum (cheap: 64 values)
# Subspace dimension for principal-angle drift (leading directions of the circuit).
DRIFT_SUBSPACE_K = 8

# --- Deterministic subsampling (speed cut, reproducible & comparable across checkpoints) ---
# Process a fixed stride-spaced subset of heads/layers rather than all 384 head-circuits.
# Stride 4 over 16 heads -> heads {0, 4, 8, 12} (4 per layer); stride 1 over layers -> all 24.
# => 4 x 24 = 96 head-circuits. The chosen set is identical every checkpoint / variant, and is
# recorded in the summary dict. Override via the same-named env vars for ablations.
CIRCUIT_HEAD_STRIDE = int(os.environ.get("CIRCUIT_HEAD_STRIDE", "4"))
CIRCUIT_LAYER_STRIDE = int(os.environ.get("CIRCUIT_LAYER_STRIDE", "1"))


def _sampled_heads() -> List[int]:
    """Deterministic head subset: ``range(0, N_HEADS, stride)`` (e.g. [0,4,8,12] at stride 4)."""
    stride = max(1, CIRCUIT_HEAD_STRIDE)
    return list(range(0, N_HEADS, stride))


def _sampled_layers() -> List[int]:
    """Deterministic layer subset: ``range(0, N_BLOCKS, stride)`` (default stride 1 = all 24)."""
    stride = max(1, CIRCUIT_LAYER_STRIDE)
    return list(range(0, N_BLOCKS, stride))


# =============================================================================
# Weight extraction helpers (pure, weight-only)
# =============================================================================

def _attn_weights(state_dict: Dict[str, torch.Tensor], block: int
                  ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Return ``(qkv_weight, proj_weight)`` for ``blocks.{block}.attn`` as float32 CPU
    tensors, or ``None`` if the keys are absent (defensive against arch changes)."""
    qk = f"blocks.{block}.attn.qkv.weight"
    pk = f"blocks.{block}.attn.proj.weight"
    if qk not in state_dict or pk not in state_dict:
        return None
    qkv = state_dict[qk].detach().to("cpu", dtype=torch.float32)
    proj = state_dict[pk].detach().to("cpu", dtype=torch.float32)
    return qkv, proj


def _head_factors(qkv: torch.Tensor, proj: torch.Tensor, head: int
                  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the *thin* per-head factors ``(Wq, Wk, Wv, Wo)`` — never the dense ``[D, D]``.

    ``qkv`` is ``[3D, D]`` (q rows ``[0:D]``, k rows ``[D:2D]``, v rows ``[2D:3D]``);
    ``proj`` is ``[D, D]``. Shapes: ``Wq, Wk, Wv`` are ``[d_h, D]`` and ``Wo`` is ``[D, d_h]``.

    The full circuits are ``W_QK = Wq^T Wk`` and ``W_OV = Wo Wv`` (both ``[D, D]``, rank
    <= d_h = 64). We deliberately keep the factored form so every spectral quantity can be
    computed on ``[D, 64]`` / ``[64, 64]`` matrices instead of dense ``[1024, 1024]`` ones —
    this both removes the host-RAM spike of materializing 384 dense maps and is far faster.
    """
    D = D_MODEL
    dh = HEAD_DIM
    r0, r1 = head * dh, (head + 1) * dh
    Wq = qkv[0:D][r0:r1, :]          # (d_h, D)
    Wk = qkv[D:2 * D][r0:r1, :]      # (d_h, D)
    Wv = qkv[2 * D:3 * D][r0:r1, :]  # (d_h, D)
    Wo = proj[:, r0:r1]              # (D, d_h)
    return Wq, Wk, Wv, Wo


def _factored_svdvals(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Singular values (descending) of the product ``A @ B`` from its thin factors, **exact**.

    For ``A`` ``[D, r]`` and ``B`` ``[r, D]`` (here ``r = d_h = 64``), the rank of ``A @ B`` is
    <= ``r``, so its <= r nonzero singular values are those of the ``[r, r]`` core ``Ra @ Rb^T``
    where ``A = Qa Ra`` and ``B^T = Qb Rb`` are thin QRs. We never form ``A @ B`` (``[D, D]``).
    Returns at most ``r`` values, sorted descending.
    """
    try:
        Qa, Ra = torch.linalg.qr(A, mode="reduced")          # A = Qa Ra,  Ra [r, r]
        Qb, Rb = torch.linalg.qr(B.t(), mode="reduced")      # B^T = Qb Rb, Rb [r, r]
        core = Ra @ Rb.t()                                   # [r, r]; same singular values as A@B
        s = torch.linalg.svdvals(core)
    except Exception:  # pragma: no cover - numerical guard
        # Robust fallback: eigvalsh on the (small) Gram of the core via the factors.
        M = (A @ B)                                          # last resort: dense (rare)
        g = 0.5 * (M.t() @ M + (M.t() @ M).t())
        s = torch.linalg.eigvalsh(g).clamp_min(0.0).sqrt()
    return torch.sort(s, descending=True).values


def _factored_ov_eigvals(Wv: torch.Tensor, Wo: torch.Tensor) -> torch.Tensor:
    """Nonzero eigenvalues of ``W_OV = Wo @ Wv`` (``[D, D]``) via ``eig(AB) = eig(BA)``.

    ``Wo`` is ``[D, d_h]`` and ``Wv`` is ``[d_h, D]``; the nonzero spectrum of ``Wo @ Wv``
    equals that of the ``[d_h, d_h]`` product ``Wv @ Wo``. So we eig the small ``[64, 64]``
    matrix instead of the dense ``[1024, 1024]`` one (same nonzero eigenvalues; the rest are
    exact zeros). Returns the complex spectrum of the small product.
    """
    small = Wv @ Wo                                          # [d_h, d_h]
    return _safe_eigvals(small)


def _factored_top_subspace(A: torch.Tensor, B: torch.Tensor, k: int) -> torch.Tensor:
    """Top-``k`` left singular vectors of ``A @ B`` (``[D, k]``) from the thin factors.

    ``A = Qa Ra`` (thin QR); the left singular vectors of ``A @ B`` lie in ``span(Qa)``, and
    are ``Qa @ Uc`` where ``Uc`` are the left singular vectors of the ``[r, r]`` core
    ``Ra @ Rb^T`` (``B^T = Qb Rb``). Computed on ``[r, r]``, never ``[D, D]``.
    """
    try:
        Qa, Ra = torch.linalg.qr(A, mode="reduced")          # [D, r], [r, r]
        Qb, Rb = torch.linalg.qr(B.t(), mode="reduced")      # [D, r], [r, r]
        core = Ra @ Rb.t()                                   # [r, r]
        Uc, _, _ = torch.linalg.svd(core, full_matrices=False)
        return (Qa @ Uc)[:, :k]                              # [D, k]
    except Exception:  # pragma: no cover - numerical guard
        return _top_subspace(A @ B, k)


def _safe_svdvals(M: torch.Tensor) -> torch.Tensor:
    """Singular values (descending), robust to GPU/cuSOLVER flakiness (always on CPU here)."""
    try:
        s = torch.linalg.svdvals(M)
    except Exception:  # pragma: no cover - numerical guard
        # Fall back to eigenvalues of M^T M (always succeeds for finite matrices).
        evals = torch.linalg.eigvalsh(0.5 * (M.t() @ M + (M.t() @ M).t())).clamp_min(0.0)
        s = evals.sqrt()
    return torch.sort(s, descending=True).values


def _safe_eigvals(M: torch.Tensor) -> torch.Tensor:
    """Complex eigenvalues of a (generally non-symmetric) square matrix, robust."""
    try:
        return torch.linalg.eigvals(M)
    except Exception:  # pragma: no cover - numerical guard
        # Symmetrize as a last resort (loses sign info on the antisymmetric part).
        return torch.linalg.eigvalsh(0.5 * (M + M.t())).to(torch.complex64)


def _participation_ratio(spectrum: torch.Tensor) -> float:
    """Effective rank as participation ratio of the (nonneg) spectrum:
    ``(sum s)^2 / sum s^2``. Equals the true rank for a flat spectrum, ~1 for a spike."""
    s = spectrum.clamp_min(0.0)
    num = float(s.sum().item()) ** 2
    den = float((s * s).sum().item())
    return num / den if den > 0 else 0.0


def _top_subspace(M: torch.Tensor, k: int) -> torch.Tensor:
    """Top-``k`` left singular vectors of ``M`` (column space directions), ``[D, k]``.

    Used for principal-angle drift: the leading output directions the circuit reads/writes.
    """
    try:
        U, _, _ = torch.linalg.svd(M, full_matrices=False)
    except Exception:  # pragma: no cover - numerical guard
        # Eigenvectors of M M^T as a fallback.
        MMt = M @ M.t()
        _, U = torch.linalg.eigh(0.5 * (MMt + MMt.t()))
        U = U.flip(1)  # ascending -> descending
    return U[:, :k]


def _factored_diff_fro(A1: torch.Tensor, B1: torch.Tensor,
                       A0: torch.Tensor, B0: torch.Tensor) -> float:
    """Exact Frobenius norm of ``A1 B1 - A0 B0`` from thin factors, no ``[D, D]`` matrix.

    With ``E = A1 B1 - A0 B0``, ``||E||_F^2 = tr(E^T E)``. Stacking the rank-r terms,
    ``E = [A1, -A0] @ [B1; B0]``, so the squared norm reduces to traces of products of the
    small ``[r, *]`` factors:

        ||E||_F^2 = <A1^T A1, B1 B1^T> - 2 <A1^T A0, B1 B0^T> + <A0^T A0, B0 B0^T>

    where ``<X, Y> = sum(X * Y)`` and every matrix is at most ``[r, r]`` (here r <= 2*d_h).
    """
    # A* are [D, r], B* are [r, D].
    a11 = A1.t() @ A1            # [r, r]
    a00 = A0.t() @ A0            # [r, r]
    a10 = A1.t() @ A0            # [r, r]
    b11 = B1 @ B1.t()            # [r, r]
    b00 = B0 @ B0.t()            # [r, r]
    b10 = B1 @ B0.t()            # [r, r]
    sq = (a11 * b11).sum() - 2.0 * (a10 * b10).sum() + (a00 * b00).sum()
    return float(sq.clamp_min(0.0).sqrt().item())


def _ov_sign_pos_frac(eigvals: torch.Tensor) -> float:
    """Fraction of OV eigenvalue 'mass' that is positive (copying) vs negative
    (anti-copying). ``W_OV`` is non-symmetric, so eigenvalues are complex; we weight by the
    real part (the copying/anti-copying axis) and report
    ``sum(max(Re λ, 0)) / sum |Re λ|``. Returns 0.5 when there is no real signal."""
    re = eigvals.real
    pos = re.clamp_min(0.0).sum()
    mag = re.abs().sum()
    if float(mag.item()) <= 0.0:
        return 0.5
    return float((pos / mag).item())


# =============================================================================
# Main entry point
# =============================================================================

def extract(ctx: "ExtractCtx") -> Dict[str, object]:
    """Extract §6.5 QK/OV circuit statistics from ``ctx.model``'s current weights.

    Writes per-head/per-layer scalars via ``ctx.scalar.add`` and full spectra / drift arrays
    via ``ctx.array.put``. Weight-only: no forward pass, ``ctx.opt`` untouched, identical for
    both optimizer variants. Returns a small summary dict for the collector log.

    SAMPLING: heads/layers are subsampled deterministically (``CIRCUIT_HEAD_STRIDE`` /
    ``CIRCUIT_LAYER_STRIDE``). At the defaults (head stride 4, layer stride 1) the sampled set
    is heads ``{0, 4, 8, 12}`` over all 24 layers = 96 head-circuits, identical every
    checkpoint and across variants. The exact set + strides are returned in the summary. The
    ``*_layer_mean`` scalars (``head=None``) are means over the **sampled** heads in that layer.

    Quantities written (scalars):
      * ``ov_eff_rank`` / ``qk_eff_rank``           (per layer, head=h)
      * ``ov_eig_sign_pos_frac``                    (per layer, head=h)  — copying fraction
      * ``ov_circuit_drift`` / ``qk_circuit_drift`` (per layer, head=h)  — ||W(t)-W(0)||
      * ``ov_principal_angle_mean`` / ``qk_principal_angle_mean`` (per layer, head=h, rad)
      * ``ov_eig_sign_pos_frac_layer_mean`` (per layer, head=None) — sampled-head-averaged copying
      * ``ov_eff_rank_layer_mean`` / ``qk_eff_rank_layer_mean`` (per layer, head=None)
    Arrays:
      * group ``ov_singvals`` / ``qk_singvals`` (layer=block, head=h) — ``[d_h]`` spectrum
      * group ``ov_eigvals``                    (layer=block, head=h) — ``[d_h, 2]`` (re, im)

    ``variant == 'adamw'`` degrades gracefully — this block never touches ``ctx.opt``, so it
    behaves identically; the variant only affects which optimizer the run used, not the weights.
    """
    model = ctx.model
    cond = ctx.condition
    run_id = ctx.run_id
    step = ctx.step
    wt = ctx.wall_time

    cur_sd = model.state_dict()
    init_sd = getattr(ctx, "init_model_sd", None)

    sampled_heads = _sampled_heads()                 # deterministic, e.g. [0, 4, 8, 12]
    sampled_layers = _sampled_layers()               # deterministic, e.g. range(24)

    n_heads_done = 0
    drift_sum = 0.0
    pos_frac_sum = 0.0
    pos_frac_n = 0
    ov_rank_sum = 0.0
    sampled_pairs: List[Tuple[int, int]] = []        # (layer, head) actually processed

    for blk in sampled_layers:
        layer = f"blocks.{blk}.attn"
        cur = _attn_weights(cur_sd, blk)
        if cur is None:
            continue
        qkv_t, proj_t = cur
        init = _attn_weights(init_sd, blk) if init_sd is not None else None

        layer_pos_fracs: List[float] = []
        layer_ov_ranks: List[float] = []
        layer_qk_ranks: List[float] = []

        for h in sampled_heads:
            # Thin per-head factors only — no dense [D, D] circuit is ever materialized.
            Wq_t, Wk_t, Wv_t, Wo_t = _head_factors(qkv_t, proj_t, h)

            # --- spectra (effective rank) from the small [64,64] core (exact) ---
            # W_OV = Wo @ Wv  -> svdvals(Wo, Wv);  W_QK = Wq^T @ Wk -> svdvals(Wq^T, Wk)
            ov_s = _factored_svdvals(Wo_t, Wv_t)[:TOPK_SPECTRUM]
            qk_s = _factored_svdvals(Wq_t.t(), Wk_t)[:TOPK_SPECTRUM]
            ov_rank = _participation_ratio(ov_s)
            qk_rank = _participation_ratio(qk_s)
            ctx.scalar.add(cond, run_id, step, wt, quantity="ov_eff_rank",
                           value=ov_rank, layer=layer, head=h)
            ctx.scalar.add(cond, run_id, step, wt, quantity="qk_eff_rank",
                           value=qk_rank, layer=layer, head=h)
            ctx.array.put(group="ov_singvals", step=step,
                          array=ov_s.numpy().astype("float32"), layer=layer, head=h)
            ctx.array.put(group="qk_singvals", step=step,
                          array=qk_s.numpy().astype("float32"), layer=layer, head=h)

            # --- OV eigenvalue sign structure (copying vs anti-copying) ---
            # eig(Wo @ Wv) nonzero spectrum == eig(Wv @ Wo) on the small [64,64] product.
            ov_eig = _factored_ov_eigvals(Wv_t, Wo_t)
            pos_frac = _ov_sign_pos_frac(ov_eig)
            ctx.scalar.add(cond, run_id, step, wt, quantity="ov_eig_sign_pos_frac",
                           value=pos_frac, layer=layer, head=h)
            # store the complex spectrum as [k, 2] (re, im) for post-hoc analysis
            eig_re = ov_eig.real[:TOPK_SPECTRUM].numpy().astype("float32")
            eig_im = ov_eig.imag[:TOPK_SPECTRUM].numpy().astype("float32")
            ctx.array.put(group="ov_eigvals", step=step,
                          array=np.stack([eig_re, eig_im], axis=-1), layer=layer, head=h)

            layer_pos_fracs.append(pos_frac)
            layer_ov_ranks.append(ov_rank)
            layer_qk_ranks.append(qk_rank)
            pos_frac_sum += pos_frac
            pos_frac_n += 1
            ov_rank_sum += ov_rank
            n_heads_done += 1
            sampled_pairs.append((blk, h))

            # --- circuit drift from init (factored, no dense difference) ---
            if init is not None:
                qkv_0, proj_0 = init
                Wq_0, Wk_0, Wv_0, Wo_0 = _head_factors(qkv_0, proj_0, h)
                ov_drift = _factored_diff_fro(Wo_t, Wv_t, Wo_0, Wv_0)
                qk_drift = _factored_diff_fro(Wq_t.t(), Wk_t, Wq_0.t(), Wk_0)
                ctx.scalar.add(cond, run_id, step, wt, quantity="ov_circuit_drift",
                               value=ov_drift, layer=layer, head=h)
                ctx.scalar.add(cond, run_id, step, wt, quantity="qk_circuit_drift",
                               value=qk_drift, layer=layer, head=h)
                drift_sum += ov_drift
                if _HAVE_PA:
                    try:
                        ov_ang = _principal_angles(
                            _factored_top_subspace(Wo_t, Wv_t, DRIFT_SUBSPACE_K),
                            _factored_top_subspace(Wo_0, Wv_0, DRIFT_SUBSPACE_K))
                        qk_ang = _principal_angles(
                            _factored_top_subspace(Wq_t.t(), Wk_t, DRIFT_SUBSPACE_K),
                            _factored_top_subspace(Wq_0.t(), Wk_0, DRIFT_SUBSPACE_K))
                        ctx.scalar.add(cond, run_id, step, wt,
                                       quantity="ov_principal_angle_mean",
                                       value=float(ov_ang.mean().item()), layer=layer, head=h)
                        ctx.scalar.add(cond, run_id, step, wt,
                                       quantity="qk_principal_angle_mean",
                                       value=float(qk_ang.mean().item()), layer=layer, head=h)
                    except Exception:
                        pass  # never kill extraction on a numerical hiccup

        # --- layer-level sampled-head-averaged summaries (head=None) ---
        if layer_pos_fracs:
            ctx.scalar.add(cond, run_id, step, wt,
                           quantity="ov_eig_sign_pos_frac_layer_mean",
                           value=float(np.mean(layer_pos_fracs)), layer=layer, head=None)
            ctx.scalar.add(cond, run_id, step, wt, quantity="ov_eff_rank_layer_mean",
                           value=float(np.mean(layer_ov_ranks)), layer=layer, head=None)
            ctx.scalar.add(cond, run_id, step, wt, quantity="qk_eff_rank_layer_mean",
                           value=float(np.mean(layer_qk_ranks)), layer=layer, head=None)

    summary = {
        "block": "block_circuits",
        "n_heads": n_heads_done,
        # --- reproducibility: exactly what was sampled this run ---
        "circuit_head_stride": CIRCUIT_HEAD_STRIDE,
        "circuit_layer_stride": CIRCUIT_LAYER_STRIDE,
        "sampled_heads": sampled_heads,
        "n_sampled_heads_per_layer": len(sampled_heads),
        "sampled_layers": sampled_layers,
        "n_sampled_layers": len(sampled_layers),
        "n_head_circuits_target": len(sampled_heads) * len(sampled_layers),
        "n_head_circuits_done": len(sampled_pairs),
        "ov_eig_sign_pos_frac_mean": (pos_frac_sum / pos_frac_n) if pos_frac_n else float("nan"),
        "ov_eff_rank_mean": (ov_rank_sum / pos_frac_n) if pos_frac_n else float("nan"),
        "ov_circuit_drift_total": drift_sum if init_sd is not None else None,
        "principal_angles_available": _HAVE_PA,
    }
    return summary

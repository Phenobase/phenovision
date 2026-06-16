#!/usr/bin/env python3
"""
block_weights.py — extractor block §6.2: Dimension-reduced weights (position evolution).

Per the briefing (``xAI/notes/phenovision_full_briefing.md`` Part II §6.2). This block answers
"where in weight space is the model, and where is adaptation concentrating", in three pieces:

  * **Primary representation — the frozen shared random projection.** The trainable parameter
    vector ``theta(t)`` (frozen input stage EXCLUDED, ``trainable_named_parameters``) is reduced to
    ``d`` coordinates with ``ctx.projection.rmatvec`` (the dimension-REDUCING, L2-preserving
    direction; see ``preadapt_common.SparseRandomProjection``). The same seeded projection is used
    for every checkpoint and every condition (§4 invariant 2), so the three lineages share ONE
    low-dim coordinate system — their trajectories can be plotted/compared in the same space, and
    §6.3's trajectory geometry can be computed on these cheap coordinates faithfully. We also store
    per-layer projected coordinates (sliced from the same global projection via
    ``trainable_layout``) so per-module trajectories are available post-hoc. Stored to the array
    store under ``proj_coords`` (global) and ``proj_coords`` with ``layer=`` (per-layer).

  * **Per-layer change profile vs init** (``ctx.init_model_sd`` = E(0), the ancestral weights):
    for each trainable layer the displacement ``||theta_l(t) - theta_l(0)||`` (scalar
    ``layer_displacement``), the leading singular-value spectrum / participation-ratio effective
    rank of each 2-D weight (array ``weight_svd`` + scalar ``effective_rank``), and the cosine
    between the cumulative change direction ``theta_l(t) - theta_l(0)`` and the init weights
    ``theta_l(0)`` (scalar ``change_cos_init``). This is the per-module "how much and in what
    direction has this changed" profile that feeds the modularity reading. COLLECTOR-SPEED (CUT C):
    the per-layer effective_rank SVD is cost-classed by the weight's smaller dimension — small
    weights get an exact full SVD, the big ViT-L MLP/attn weights (min-dim 1024) get a cheap
    rank-``SVD_TOPK`` truncated SVD proxy, and pathologically large weights are skipped; the
    per-layer full/proxy/skip class is recorded in the summary for reproducibility.

  * **Displacement on the curvature basis (DIRECT TEST; needs ``ctx.Q0``).** Project the per-layer
    displacement ``theta_l(t) - theta_l(0)`` onto the pretrained-init Kronecker eigenbasis
    ``Q0[name] = {QL, QR}`` and report the fraction of displacement energy in low-curvature (flat)
    vs high-curvature (steep) directions: scalars ``disp_frac_flat`` / ``disp_frac_steep``. This
    asks whether adaptation moves along productive (steep, loss-reducing) directions or piles into
    flat directions — the geometric signature of short-term evolvability and the spectral
    counterpart of the efficiency ratio. Skipped cleanly when ``ctx.Q0`` is None (AdamW variant, or
    no preconditioned basis at init) or the specific layer has no Q0 entry.

Robustness / RNG
----------------
This block reads model weights + the (deterministic, seed-reproducible) projection; the projection
is matrix-free and regenerated from its seed, so no global RNG is touched and no per-call generator
is needed (mirrors ``block_optim`` which is likewise deterministic — only the curvature block,
which runs stochastic Lanczos probes, threads a ``torch.Generator``). The block degrades gracefully
for the ``adamw`` variant: ``ctx.Q0`` is None there, so the displacement-on-curvature piece is
skipped while the projection and per-layer change profile (which need no optimizer state) are
unaffected. Every step is wrapped so a single numerical hiccup never raises out of ``extract``.
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

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


# ViT-L/16 architecture facts (verified against the checkpoints).
N_BLOCKS = 24
N_HEADS = 16
HEAD_DIM = 64
D_MODEL = 1024

# How many top singular values to keep per 2-D weight. The leading spectrum + participation ratio
# capture the effective rank cheaply, so a modest top-k suffices; this also caps the rank ``q`` of
# the truncated/randomized SVD on the big weights below. COLLECTOR-SPEED (CUT C): lowered 64 -> 32
# to roughly halve both the truncated-SVD rank and the per-layer array-store slice width.
SVD_TOPK = 32

# Cost-class thresholds for the per-layer effective_rank SVD, by the weight's SMALLER dimension
# ``min(m, n)`` (a full ``svdvals`` of ``[m, n]`` is O(min^2 * max)):
#
#   * min-dim <= ``SVD_FULL_MAX_DIM``       -> EXACT full ``svdvals`` (cheap small weights, e.g. the
#                                              [2,1024]/[3,1024] head and [1024,1024] attn proj are
#                                              fine; see note below — the big ViT-L weights are NOT
#                                              in this class with the tuned threshold).
#   * ``SVD_PROXY_MIN_DIM`` <= min-dim      -> CHEAP PROXY: a truncated/randomized low-rank SVD
#                                              (``torch.svd_lowrank``, rank ``SVD_TOPK``) gives the
#                                              leading spectrum, from which the participation-ratio
#                                              effective rank is computed. O(m*n*q), q=SVD_TOPK <<
#                                              the O(min^2*max) full SVD.
#   * min-dim > ``SVD_SKIP_MIN_DIM``        -> SKIP the SVD entirely (effective_rank = NaN, no
#                                              weight_svd array) — for pathologically large weights
#                                              where even the truncated SVD is not worth it.
#
# COLLECTOR-SPEED (CUT C): the dominant §6.2 SVD cost was the ~2*N_BLOCKS ViT-L MLP/attn weights
# ([4096,1024], [1024,4096], [3072,1024], [1024,1024]) — all min-dim 1024 — each taking a full dense
# CPU ``svdvals``. ``SVD_PROXY_MIN_DIM`` is tuned BELOW 1024 so every one of those big weights now
# takes the cheap truncated-SVD proxy instead of the full SVD, collapsing the per-layer cost. Small
# weights (head, biases-as-vectors are 1-D and skipped) keep the exact path. ``SVD_FULL_MAX_DIM``
# is retained (and is the lower bound of the proxy band) so its name/role is unchanged for callers.
SVD_FULL_MAX_DIM = 768       # min-dim <= this -> exact full SVD
SVD_PROXY_MIN_DIM = 768      # min-dim >= this -> cheap truncated/randomized SVD proxy
SVD_SKIP_MIN_DIM = 8192      # min-dim >  this -> skip the SVD entirely (no big weight hits this)

# Whether to store the per-layer projected coordinates too. The per-layer projection uses the same
# global projection sliced to the layer's coordinate range, so per-layer coords sum (in the JL
# sense) to the global coords. This is cheap (one rmatvec over a zero-padded layer slice) and gives
# post-hoc per-module trajectories, so it is on by default; the collector can flip it off.
STORE_PER_LAYER_COORDS = True

_EPS = 1e-12


# =============================================================================
# helpers
# =============================================================================

def _participation_ratio(spectrum: np.ndarray) -> float:
    """Effective rank as participation ratio of a non-negative spectrum:
    ``(sum s)^2 / sum s^2`` (effective # of significant directions). NaN for empty/all-zero."""
    if spectrum is None or spectrum.size == 0:
        return float("nan")
    s = np.clip(spectrum.astype(np.float64), 0.0, None)
    s2 = float(np.sum(s * s))
    if s2 <= 0.0:
        return float("nan")
    return float((float(np.sum(s)) ** 2) / s2)


def _safe_svdvals(M: torch.Tensor) -> Tuple[Optional[np.ndarray], str]:
    """Descending singular values of a 2-D matrix + a COST-CLASS TAG, computed on CPU/float32,
    robust to cuSOLVER/GPU flakiness on ill-conditioned matrices.

    Returns ``(spectrum, tag)`` where ``spectrum`` is a 1-D descending numpy array (or None on a
    skip/failure) and ``tag`` is one of:

      * ``"full"``   — exact full ``svdvals`` (min-dim <= ``SVD_FULL_MAX_DIM``).
      * ``"proxy"``  — cheap truncated/randomized low-rank SVD of rank ``SVD_TOPK``
                       (``SVD_PROXY_MIN_DIM`` <= min-dim <= ``SVD_SKIP_MIN_DIM``). O(m*n*q) for
                       q=``SVD_TOPK`` << the O(min^2*max) full SVD; the participation-ratio
                       effective rank is dominated by the leading spectrum so the proxy is faithful.
      * ``"skip"``   — min-dim > ``SVD_SKIP_MIN_DIM`` (no SVD computed; spectrum=None).
      * ``"fallback"`` — the requested path failed and the exact dense path (or its eigvalsh
                       fallback) was used instead.
      * ``"error"``  — every path raised (spectrum=None).

    The tag is recorded per layer in the summary so the full/proxy/skipped split is reproducible
    and comparable across checkpoints/runs (the threshold constants are deterministic)."""
    if M is None or M.dim() != 2:
        return None, "skip"
    Mc = M.detach().to("cpu", dtype=torch.float32)
    min_dim = int(min(Mc.shape))

    # Pathologically large weights: skip the SVD entirely (deterministic, by shape).
    if min_dim > SVD_SKIP_MIN_DIM:
        return None, "skip"

    # Big weights: cheap truncated/randomized low-rank SVD (the dominant CUT-C saving).
    if min_dim >= SVD_PROXY_MIN_DIM:
        q = min(SVD_TOPK, min_dim)
        try:
            _, s_lr, _ = torch.svd_lowrank(Mc, q=q, niter=2)
            s = torch.sort(s_lr, descending=True).values
            return s.numpy(), "proxy"
        except Exception:
            pass  # fall through to the exact dense path below (rare; tagged "fallback")
        tag = "fallback"
    else:
        tag = "full"

    try:
        s = torch.linalg.svdvals(Mc)
    except Exception:
        try:
            # Fallback: eigenvalues of M^T M (always succeeds for finite matrices).
            G = Mc.t() @ Mc
            ev = torch.linalg.eigvalsh(0.5 * (G + G.t())).clamp_min(0.0)
            s = ev.sqrt()
            tag = "fallback"
        except Exception:
            return None, "error"
    s = torch.sort(s, descending=True).values
    return s.numpy(), tag


def _topk_padded(spectrum: np.ndarray, k: int) -> np.ndarray:
    """First ``k`` entries of a descending spectrum, right-padded with NaN to fixed width ``k``
    so the array store gets a constant per-layer slice shape across checkpoints."""
    out = np.full((k,), np.nan, dtype=np.float32)
    n = min(k, spectrum.size)
    out[:n] = spectrum[:n].astype(np.float32)
    return out


def _project_layer_slice(projection, theta: torch.Tensor, off: int, numel: int) -> torch.Tensor:
    """Project ONLY the layer's coordinate slice ``theta[off:off+numel]`` into ``R^d``, returning
    the same vector the global ``rmatvec`` would contribute from that range.

    Fast path: tile ``[off, off+numel)`` with the projection's own block size and scatter-add via
    the projection's per-coordinate maps (``_block_maps``, a pure function of
    ``(seed, absolute coordinate)``), so this is BIT-IDENTICAL to the global rmatvec restricted to
    the layer's coordinates — without allocating/scanning a full-D zero-padded vector. Total work
    over all layers is ~one full-D rmatvec instead of ``N_layers`` of them.

    Fallback (if the projection has no compatible ``_block_maps`` — e.g. a different projection
    class): the original zero-pad-then-full-rmatvec path, which is correct but O(D) per layer."""
    d = int(projection.d)
    device = theta.device
    dtype = theta.dtype
    block_maps = getattr(projection, "_block_maps", None)
    block = int(getattr(projection, "block", 0) or 0)
    if callable(block_maps) and block > 0:
        out = torch.zeros(d, device=device, dtype=dtype)
        start = off
        end = off + numel
        while start < end:
            n = min(block, end - start)
            idx, w = block_maps(start, n, device)
            out.index_add_(0, idx, w.to(dtype) * theta[start:start + n])
            start += n
        return out
    # Fallback: semantics-preserving zero-pad + full rmatvec.
    seg = torch.zeros_like(theta)
    seg[off:off + numel] = theta[off:off + numel]
    return projection.rmatvec(seg)


def _flatten_layer_disp(cur: torch.Tensor, init: torch.Tensor) -> Optional[torch.Tensor]:
    """``theta_l(t) - theta_l(0)`` flattened to 1-D float64 CPU, or None on a shape mismatch."""
    if cur is None or init is None:
        return None
    if tuple(cur.shape) != tuple(init.shape):
        return None
    c = cur.detach().to("cpu", dtype=torch.float64).reshape(-1)
    i = init.detach().to("cpu", dtype=torch.float64).reshape(-1)
    return c - i


def _disp_on_curvature_basis(cur_2d: torch.Tensor, init_2d: torch.Tensor,
                             q0: Dict[str, object]) -> Optional[Tuple[float, float]]:
    """Project the per-layer displacement ``D = theta_l(t) - theta_l(0)`` (a 2-D ``[m, n]`` weight)
    onto the init Kronecker eigenbasis ``Q0 = {QL (m x m), QR (n x n)}`` and return
    ``(frac_flat, frac_steep)`` — the fraction of displacement ENERGY (squared Frobenius norm) in
    the low-curvature (flat) vs high-curvature (steep) halves of the basis.

    The Kronecker eigenbasis stores eigenvectors in DESCENDING eigenvalue order (column 0 = the
    highest-curvature / steepest direction; ``block_optim._top_columns`` documents the same
    convention). The rotated displacement is ``QL^T D QR`` (coefficient ``[i, j]`` is the energy of
    ``D`` along steep-rank ``i`` of the left factor and steep-rank ``j`` of the right factor). We
    split the rotated coefficients by the COMBINED rank ``i + j``: the steep half is the
    top-curvature combined ranks, the flat half is the bottom-curvature combined ranks. The split
    point is the median combined rank so the two halves carry comparable direction counts. Returns
    None if neither QL nor QR is usable (then the caller skips this layer's direct test).
    """
    if cur_2d is None or init_2d is None or cur_2d.dim() != 2:
        return None
    if tuple(cur_2d.shape) != tuple(init_2d.shape):
        return None
    m, n = cur_2d.shape
    D = (cur_2d.detach().to("cpu", dtype=torch.float64)
         - init_2d.detach().to("cpu", dtype=torch.float64))

    QL = q0.get("QL") if isinstance(q0, dict) else None
    QR = q0.get("QR") if isinstance(q0, dict) else None
    QL = QL.detach().to("cpu", dtype=torch.float64) if isinstance(QL, torch.Tensor) else None
    QR = QR.detach().to("cpu", dtype=torch.float64) if isinstance(QR, torch.Tensor) else None

    # Rotate into the basis where it is available on a side; identity where it is not / mismatched.
    # The eigenbasis columns are the metric's principal directions in DESCENDING curvature order.
    if QL is not None and QL.dim() == 2 and QL.shape[0] == m and QL.shape[1] == m:
        left = QL.t()                      # rows = steep-ranked left directions
    elif QL is None and QR is not None:
        left = None                        # only the right factor carries a basis
    else:
        left = None
    if QR is not None and QR.dim() == 2 and QR.shape[0] == n and QR.shape[1] == n:
        right = QR                         # columns = steep-ranked right directions
    else:
        right = None

    if left is None and right is None:
        return None

    coeff = D
    if left is not None:
        coeff = left @ coeff               # [m, n], row i = left steep-rank i
    if right is not None:
        coeff = coeff @ right              # [m, n], col j = right steep-rank j

    energy = (coeff * coeff)               # [m, n] energy per (left-rank, right-rank)
    total = float(energy.sum().item())
    if total <= _EPS:
        return None

    # Combined-rank split: i + j small = steep (both factors high curvature), large = flat.
    ii = torch.arange(coeff.shape[0], dtype=torch.float64).reshape(-1, 1)
    jj = torch.arange(coeff.shape[1], dtype=torch.float64).reshape(1, -1)
    # Normalize each side's rank to [0, 1] so a wildly rectangular weight does not bias the split
    # toward the longer axis; combined "depth" in [0, 2].
    di = ii / max(coeff.shape[0] - 1, 1)
    dj = jj / max(coeff.shape[1] - 1, 1)
    depth = di + dj                         # 0 = steepest corner, 2 = flattest corner
    median_depth = float(torch.median(depth).item())
    steep_mask = (depth <= median_depth)
    steep_energy = float(energy[steep_mask].sum().item())
    frac_steep = steep_energy / total
    frac_flat = 1.0 - frac_steep
    return frac_flat, frac_steep


# =============================================================================
# entry point
# =============================================================================

def extract(ctx: "ExtractCtx") -> Dict[str, object]:
    """§6.2 dimension-reduced weights. See module docstring.

    Writes scalars: ``layer_displacement`` / ``effective_rank`` / ``change_cos_init`` (per layer);
    ``disp_frac_flat`` / ``disp_frac_steep`` (per layer, when ``ctx.Q0`` has the layer); plus the
    global ``proj_coords_norm`` summary. Writes arrays: ``proj_coords`` (global, and per-layer when
    enabled) and ``weight_svd`` (per 2-D layer). Returns a compact summary dict for the collector
    log. Robust to missing projection / init_sd / Q0 (returns partial, never raises).
    """
    from preadapt_common import (
        trainable_named_parameters, flatten_trainable, trainable_layout,
    )

    model = ctx.model
    cond = ctx.condition
    run_id = ctx.run_id
    step = ctx.step
    wt = ctx.wall_time

    summary: Dict[str, object] = {"block": "block_weights"}

    # ---------------------------------------------------------------------
    # (1) Primary representation: shared random projection coordinates.
    # ---------------------------------------------------------------------
    n_layers_proj = 0
    if ctx.projection is not None:
        try:
            theta = flatten_trainable(model)  # 1-D trainable vector (frozen stage excluded)
            if theta.numel() == ctx.projection.D_trainable:
                # --- GLOBAL projected coords ---
                # COLLECTOR-SPEED: if the trainer stored the global coords inline at emission
                # (ctx.inline_proj_coords), use them DIRECTLY and SKIP the dominant 302M-coord
                # rmatvec recompute. They were produced by the SAME run-shared projection over the
                # same flatten_trainable(model), so they are bit-identical to the recompute below.
                # Otherwise (old checkpoints / Phase-1 saves) fall back to recomputing the rmatvec.
                inline_coords = getattr(ctx, "inline_proj_coords", None)
                inline_meta = getattr(ctx, "inline_proj_meta", None)
                used_inline = False
                if inline_coords is not None:
                    coords_t = inline_coords.detach().to("cpu", dtype=torch.float32)
                    if coords_t.numel() == int(ctx.projection.d):
                        # Sanity-check the inline meta's seed/d match the collector's projection;
                        # log a warning on mismatch but PREFER the inline coords (the trainer is
                        # the authority on what coordinate frame the checkpoint was emitted in).
                        if isinstance(inline_meta, dict):
                            m_seed = inline_meta.get("seed")
                            m_d = inline_meta.get("d")
                            if (m_seed is not None and int(m_seed) != int(ctx.projection.seed)) or \
                               (m_d is not None and int(m_d) != int(ctx.projection.d)):
                                print(f"[block_weights] inline proj_meta mismatch "
                                      f"(inline seed={m_seed} d={m_d} vs projection "
                                      f"seed={ctx.projection.seed} d={ctx.projection.d}); "
                                      f"using inline coords.", file=sys.stderr)
                        coords = coords_t
                        used_inline = True
                    else:
                        print(f"[block_weights] inline proj_coords dim {coords_t.numel()} != "
                              f"projection d={ctx.projection.d}; recomputing.", file=sys.stderr)
                if not used_inline:
                    # The GLOBAL rmatvec is the matrix-free, O(D) vectorized path: it tiles the
                    # ~302M coords in blocks and does a torch ``index_add_`` scatter per block (no
                    # Python per-coordinate loop). This single full-D pass is the dominant
                    # per-checkpoint §6.2 cost — now skipped whenever inline coords are present.
                    coords = ctx.projection.rmatvec(theta)  # R^D -> R^d (L2-preserving)
                coords_np = coords.detach().to("cpu", dtype=torch.float32).numpy()
                ctx.array.put(group="proj_coords", step=step, array=coords_np)
                ctx.scalar.add(cond, run_id, step, wt, quantity="proj_coords_norm",
                               value=float(np.linalg.norm(coords_np)))
                summary["proj_coords_dim"] = int(coords_np.size)
                summary["proj_coords_inline"] = bool(used_inline)

                # Per-layer projected coords: each layer's coords are exactly the contribution of
                # its coordinate range ``[off, off+numel)`` to the global rmatvec scatter, so the
                # per-layer coords (in the JL sense) decompose the global coords. The previous
                # implementation zero-padded a full-D vector and ran a FULL O(D) rmatvec per layer
                # — ~N_layers full-D scatters. Instead we scatter ONLY each layer's slice (total
                # work ~= a single full-D rmatvec across all layers, an ~N_layers speedup). This is
                # bit-identical to the zero-pad path because the projection map is a pure function
                # of (seed, absolute coordinate) — independent of how the coords are tiled.
                #
                # IMPORTANT: a DISTINCT array-store group (``proj_coords_layer``) is used here, not
                # the global ``proj_coords`` group. The Zarr backend makes a key with a ``layer=``
                # child nest under its group node, and a node cannot be both an array and a parent
                # group — so reusing ``proj_coords`` for the (root) global array AND the per-layer
                # children raises ``ContainsArrayError``. Keeping per-layer coords in their own
                # group avoids that and keeps the global array a clean root array.
                if STORE_PER_LAYER_COORDS:
                    layout = trainable_layout(model)
                    for name, off, numel in layout:
                        try:
                            lc = _project_layer_slice(ctx.projection, theta, off, numel)
                            ctx.array.put(group="proj_coords_layer", step=step,
                                          array=lc.detach().to("cpu", dtype=torch.float32).numpy(),
                                          layer=name)
                            n_layers_proj += 1
                        except Exception:
                            continue
            else:
                summary["proj_skipped"] = (
                    f"D_trainable mismatch: model {theta.numel()} vs "
                    f"projection {ctx.projection.D_trainable}")
        except Exception as e:  # never kill extraction on a projection hiccup
            print(f"[block_weights] projection skipped: {e}", file=sys.stderr)
            summary["proj_error"] = repr(e)
    else:
        summary["proj_skipped"] = "no projection"
    summary["n_layers_proj"] = n_layers_proj

    # ---------------------------------------------------------------------
    # (2) Per-layer change profile vs init + (3) displacement on curvature basis.
    # ---------------------------------------------------------------------
    init_sd = getattr(ctx, "init_model_sd", None)
    q0 = ctx.Q0 or {}
    cur_named = dict(trainable_named_parameters(model, include_requires_grad_false=True))

    n_disp = 0
    n_svd = 0
    n_q0 = 0
    global_disp_sq = 0.0
    # Per-layer SVD cost-class accounting (CUT C): record which layers got an exact SVD vs the
    # cheap truncated proxy vs were skipped, so the speed/fidelity tradeoff is reproducible.
    svd_class_counts: Dict[str, int] = {"full": 0, "proxy": 0, "skip": 0, "fallback": 0, "error": 0}
    svd_layer_class: Dict[str, str] = {}
    for name, p in cur_named.items():
        cur = p.detach()

        # --- singular-value spectrum / effective rank of each 2-D weight ---
        if cur.dim() == 2:
            spec, svd_tag = _safe_svdvals(cur)
            svd_class_counts[svd_tag] = svd_class_counts.get(svd_tag, 0) + 1
            svd_layer_class[name] = svd_tag
            if spec is not None and spec.size:
                ctx.array.put(group="weight_svd", step=step,
                              array=_topk_padded(spec, SVD_TOPK), layer=name)
                ctx.scalar.add(cond, run_id, step, wt, quantity="effective_rank",
                               value=_participation_ratio(spec), layer=name)
                n_svd += 1

        # --- displacement vs init + cosine to init direction ---
        if init_sd is not None and name in init_sd:
            init_w = init_sd[name]
            disp = _flatten_layer_disp(cur, init_w)
            if disp is not None:
                disp_norm = float(disp.norm().item())
                ctx.scalar.add(cond, run_id, step, wt, quantity="layer_displacement",
                               value=disp_norm, layer=name)
                global_disp_sq += disp_norm * disp_norm
                n_disp += 1
                # cosine between cumulative change direction and the init weights
                init_flat = init_w.detach().to("cpu", dtype=torch.float64).reshape(-1)
                dn = disp.norm().item()
                inn = init_flat.norm().item()
                if dn > _EPS and inn > _EPS:
                    cos = float((disp @ init_flat).item() / (dn * inn))
                    ctx.scalar.add(cond, run_id, step, wt, quantity="change_cos_init",
                                   value=cos, layer=name)

                # --- DIRECT TEST: displacement on the init curvature basis Q0 ---
                if cur.dim() == 2 and name in q0 and isinstance(q0[name], dict):
                    try:
                        res = _disp_on_curvature_basis(cur, init_w, q0[name])
                    except Exception:
                        res = None
                    if res is not None:
                        frac_flat, frac_steep = res
                        ctx.scalar.add(cond, run_id, step, wt, quantity="disp_frac_flat",
                                       value=frac_flat, layer=name)
                        ctx.scalar.add(cond, run_id, step, wt, quantity="disp_frac_steep",
                                       value=frac_steep, layer=name)
                        n_q0 += 1

    if n_disp > 0:
        ctx.scalar.add(cond, run_id, step, wt, quantity="disp_from_init_global",
                       value=float(global_disp_sq ** 0.5))

    summary["n_layer_displacement"] = n_disp
    summary["n_weight_svd"] = n_svd
    summary["n_disp_on_curvature"] = n_q0
    summary["q0_available"] = bool(q0)
    summary["init_available"] = init_sd is not None
    # CUT C: SVD cost-class breakdown + the deterministic thresholds that produced it, so the
    # full/proxy/skipped split is reproducible and comparable across checkpoints/runs.
    summary["svd_class_counts"] = svd_class_counts
    summary["svd_layer_class"] = svd_layer_class
    summary["svd_topk"] = SVD_TOPK
    summary["svd_full_max_dim"] = SVD_FULL_MAX_DIM
    summary["svd_proxy_min_dim"] = SVD_PROXY_MIN_DIM
    summary["svd_skip_min_dim"] = SVD_SKIP_MIN_DIM
    return summary

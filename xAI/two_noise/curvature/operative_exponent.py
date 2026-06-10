"""Measuring the REALIZED operative exponent α of a preconditioner (the optimizer-arm headline
diagnostic): the applied preconditioner P acts as ≈ C^{-α} on the curvature C, and α is what the
evolve-M meta-loop is claimed to drive ½→1 in the well-sampled subspace.

The naive eig(G)-vs-eig(C) slope on a layer's Kronecker factor is unreliable on real models (the
gradient-EMA factors are noisy / near-isotropic). Two robust measurements:

  (i)  FACTOR PROBE  — operative_exponent_factors: per Kronecker factor, in C's eigenbasis fit
       log eig(G) = -α log eig(C); aggregate ACROSS factors/layers WEIGHTED by log-eigenvalue
       spread and SKIPPING near-isotropic factors (which carry no slope information).

  (ii) LANCZOS-HESSIAN PROBE — operative_exponent_lanczos: the ground truth. Take the top-k
       eigenpairs (λ_i, v_i) of the TRUE loss Hessian (HVP/Lanczos), measure the preconditioner's
       GAIN along each, gain_i = ⟨v_i, P v_i⟩, and fit log gain = -α log λ. Uses the real,
       genuinely ill-conditioned Hessian (not the optimizer's internal factors).
"""
from __future__ import annotations

import numpy as np
import torch

from curvature.lanczos import top_k_eigenpairs


# --------------------------------------------------------------------------- #
# (i) factor probe
# --------------------------------------------------------------------------- #

@torch.no_grad()
def operative_exponent_factors(optimizer, min_spread=0.7, top_frac=0.3):
    """Spread-weighted operative exponent from the optimizer's Kronecker factors. Returns
    (overall, top_k_subspace, flat_subspace): α fit over all eigendirections, over the steepest
    top_frac, and over the flattest. Factors with log-eigenvalue spread < min_spread are skipped."""
    over_n, over_d, top_n, top_d, flat_n, flat_d = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    for group in optimizer.param_groups:
        for p in group["params"]:
            st = optimizer.state.get(p, {})
            if not st.get("use_kron", False) or "CL" not in st:
                continue
            for Ck, Gk in (("CL", "GL"), ("CR", "GR")):
                C = st[Ck].float(); G = st[Gk].float()
                w, V = torch.linalg.eigh(0.5 * (C + C.t()))
                w = w.clamp_min(1e-12)
                g = torch.diagonal(V.t() @ G @ V).clamp_min(1e-12)
                lc = torch.log(w).cpu().numpy(); lg = torch.log(g).cpu().numpy()
                order = np.argsort(lc)
                lc, lg = lc[order], lg[order]
                spread = float(lc[-1] - lc[0])
                if spread < min_spread or len(lc) < 4:
                    continue
                k = max(2, int(top_frac * len(lc)))
                a_all = -np.polyfit(lc, lg, 1)[0]
                a_top = -np.polyfit(lc[-k:], lg[-k:], 1)[0]      # steepest (largest C) directions
                a_flat = -np.polyfit(lc[:k], lg[:k], 1)[0]       # flattest (smallest C) directions
                over_n += a_all * spread; over_d += spread
                top_n += a_top * spread; top_d += spread
                flat_n += a_flat * spread; flat_d += spread
    f = lambda n, d: float(n / d) if d > 0 else float("nan")
    return f(over_n, over_d), f(top_n, top_d), f(flat_n, flat_d)


# --------------------------------------------------------------------------- #
# (ii) Lanczos-Hessian probe (ground truth)
# --------------------------------------------------------------------------- #

@torch.no_grad()
def _apply_preconditioner(optimizer, vec_flat, params):
    """Map a flattened parameter-space vector through the optimizer's CURRENT preconditioner
    (per-layer Kronecker G_L · slice · G_R for 2D layers; diagonal h for the rest)."""
    out = []
    i = 0
    # build a param -> group lookup for the diag path's precond/damping
    grp_of = {}
    for group in optimizer.param_groups:
        for p in group["params"]:
            grp_of[id(p)] = group
    for p in params:
        n = p.numel()
        sl = vec_flat[i:i + n].reshape(p.shape); i += n
        st = optimizer.state.get(p, {})
        if st.get("use_kron", False) and "GL" in st:
            pv = (st["GL"].float() @ sl.float() @ st["GR"].float()).to(sl.dtype)
        elif "v" in st:                                  # diagonal path
            grp = grp_of[id(p)]
            alpha = 0.5 if grp["precond"] == "whiten" else 1.0
            h = (st["v"].float() + grp["damping"]) ** (-alpha)
            pv = (h * sl.float()).to(sl.dtype)
        else:
            pv = sl
        out.append(pv.reshape(-1))
    return torch.cat(out)


def operative_exponent_lanczos(loss_fn, optimizer, params, k=12, n_iter=None, generator=None,
                               top_frac=0.4):
    """Ground-truth operative exponent from the TRUE Hessian. For the top-k Hessian eigenpairs
    (λ_i, v_i), the preconditioner gain is gain_i = ⟨v_i, P v_i⟩; fit log gain = -α log λ over the
    positive-curvature directions. Returns (overall, top_k, flat) over the eigenvalue range."""
    params = [p for p in params if p.requires_grad]
    evals, evecs = top_k_eigenpairs(loss_fn, params, k=k, n_iter=n_iter, generator=generator)
    lam = evals.detach().cpu().numpy()
    gains = []
    for j in range(evecs.shape[1]):
        v = evecs[:, j]
        Pv = _apply_preconditioner(optimizer, v, params)
        gains.append(float(torch.dot(v, Pv)))           # ⟨v, P v⟩ (v is unit-norm)
    gains = np.array(gains)
    ok = (lam > 1e-8) & (gains > 1e-12)
    lam, gains = lam[ok], gains[ok]
    if len(lam) < 4:
        return float("nan"), float("nan"), float("nan")
    order = np.argsort(np.log(lam))
    ll, lg = np.log(lam)[order], np.log(gains)[order]
    kk = max(2, int(top_frac * len(ll)))
    a_all = -np.polyfit(ll, lg, 1)[0]
    a_top = -np.polyfit(ll[-kk:], lg[-kk:], 1)[0]        # steepest Hessian directions
    a_flat = -np.polyfit(ll[:kk], lg[:kk], 1)[0]
    return float(a_all), float(a_top), float(a_flat)

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
                # eigh on CPU (LAPACK): cuSOLVER's GPU eigh raises _LinAlgError on ill-conditioned
                # curvature factors mid-training. Factors are small (Kronecker dims) so CPU is cheap.
                C = st[Ck].float().cpu(); G = st[Gk].float().cpu()
                try:
                    w, V = torch.linalg.eigh(0.5 * (C + C.t()))
                except torch._C._LinAlgError:
                    continue                          # skip this factor; never kill the run
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


@torch.no_grad()
def operative_exponent_soap(optimizer, min_spread=0.7, top_frac=0.3):
    """Realized operative exponent for SOAPFullPower (the factor probe above reads RiccatiPrecond's
    CL/GL; SOAP keeps the in-basis second moment eas instead). The applied per-coordinate gain is
    h = sqrt(gL ⊗ gR) / shrink(v)^power, with v = exp_avg_sq (eigenvalue estimate λ_iμ_j) and
    gL,gR the unit-mean-normalized evolve-M gains (1 if evolve off). Realized exponent = -slope of
    log h vs log v, spread-weighted over 2D layers; returns (overall, top_k, flat). Whiten -> ~0.5,
    full inverse -> ~1.0; evolve-M raises it in the directions where its gain concentrates."""
    over_n, over_d, top_n, top_d, flat_n, flat_d = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    for group in optimizer.param_groups:
        power = group["precond_power"]; rho = group.get("shrink", 0.0); eps = group["eps"]
        for p in group["params"]:
            st = optimizer.state.get(p, {})
            if not st.get("use_precond", False) or "exp_avg_sq" not in st:
                continue
            v = st["exp_avg_sq"].float()
            if v.dim() != 2:
                continue
            vs = (1.0 - rho) * v + rho * v.mean() if rho > 0 else v
            denom = vs.pow(power)
            gain = torch.ones_like(v)
            if group.get("evolve_m", False) and "m_L" in st:
                gL = st["m_L"] / st["m_L"].mean().clamp_min(eps)
                gR = st["m_R"] / st["m_R"].mean().clamp_min(eps)
                gain = gL.sqrt().unsqueeze(1) * gR.sqrt().unsqueeze(0)
            h = gain / denom.clamp_min(eps)
            lv = torch.log(v.clamp_min(1e-12)).flatten().cpu().numpy()
            lh = torch.log(h.clamp_min(1e-30)).flatten().cpu().numpy()
            okm = np.isfinite(lv) & np.isfinite(lh)        # drop inf/nan (e.g. a diverged run)
            lv, lh = lv[okm], lh[okm]
            if len(lv) < 4:
                continue
            order = np.argsort(lv); lv, lh = lv[order], lh[order]
            spread = float(lv[-1] - lv[0])
            if spread < min_spread or len(lv) < 4:
                continue
            k = max(2, int(top_frac * len(lv)))
            a_all = -np.polyfit(lv, lh, 1)[0]
            a_top = -np.polyfit(lv[-k:], lh[-k:], 1)[0]
            a_flat = -np.polyfit(lv[:k], lh[:k], 1)[0]
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

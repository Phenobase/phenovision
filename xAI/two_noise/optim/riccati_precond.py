"""
riccati_precond.py
==================
A matrix-FREE preconditioned optimizer derived from the discrete G-equation
(matrix Riccati) of the quantitative-genetics / SGD correspondence.

Core update (per Kronecker factor, outer step t, inner relaxation j):

    G_t^(0) = G_{t-1}                                  # warm start
    G_t^(j+1) = G_t^(j) + eta_p ( M - G_t^(j) C_t G_t^(j) )    # Riccati / Newton-Schulz
    W_{t+1}  = W_t - eta * ( G_{L,t} g_t G_{R,t} )     # preconditioned step

Fixed point  G C G = M.  The *source* M sets the exponent:
    M = eps*I            -> G -> C^{-1/2}   (alpha = 1/2, whitening; ~ Muon/Adam regime)
    M = G                -> G -> C^{-1}      (alpha = 1,   natural-gradient/Newton regime)
The exponent is interpolated *in practice* by SHRINKAGE of C toward isotropy
(rho), which is also the stabilizer: update-noise variance ~ a_i^{1-2 alpha},
so alpha>1/2 amplifies flat directions (Part III of riccati_optimizer_and_plan.md);
shrinkage / lower effective alpha cure it.

This is a GENUINELY DIFFERENT algorithm from optim/soap_full_power.py: it uses
matmul-only Newton-Schulz iterations (NO eigendecomposition), Kronecker-factored,
warm-started, with two non-standard options: an evolving source M (slow CMA-ES-style
rank-mu accumulation -- the "evolving evolvability" meta-loop, the optimizer-side
analog of the biological M-evolution) and FDT-consistent Langevin noise.

PORTING NOTES (this file is the PyTorch port of xAI/notes/riccati_precond.py, which
was validated only at the numpy-core level):
  * Factor math (CL, CR, GL, GR, ML, MR) is forced to float32 and the descent
    direction is cast back to the parameter dtype. The matrix_norm(ord=2) SVD and
    the Newton-Schulz iterations are unstable in fp16/autocast; fp32 factors fix it.
  * `precond_stats_from_hook=True` accumulates the curvature factors C from
    `p._soap_precond_grad` (the true-Fisher gradient set by curvature/true_fisher.py)
    while the descent step keeps the empirical gradient -- mirrors SOAPFullPower's
    split so O2's `inverse_fisher` condition reuses the existing true-Fisher path.
  * `evolve_M_weighted=True` implements the doc's loss-weighted CMA-ES rank-mu form
    (w ~ max(0, dL) per step, self-normalized over each meta_every window); the
    unweighted skeleton is kept selectable for ablation.
Smoke-test (incl. CUDA/autocast) before serious use.
"""
import math
import warnings

import torch
from torch.optim.optimizer import Optimizer


# --------------------------------------------------------------------------- #
# matrix-free factor iterations (the Riccati core); all operate in float32
# --------------------------------------------------------------------------- #

def _shrink(C, rho):
    """Ledoit-Wolf-style shrinkage of an SPD matrix toward isotropy.
    rho is the *effective exponent reducer* and the primary stabilizer."""
    if rho <= 0.0:
        return C
    d = C.shape[0]
    tau = torch.diagonal(C).mean()
    eye = torch.eye(d, device=C.device, dtype=C.dtype)
    return (1.0 - rho) * C + rho * tau * eye


def _spec_norm(M, iters=3):
    """Cheap spectral-norm estimate via power iteration (deterministic init; matmuls only).
    Replaces torch.linalg.matrix_norm(ord=2) in the NS safeguard: ~30x faster than the SVD AND
    robust -- a large/non-finite entry makes the LAPACK/cuSOLVER SVD error or HANG, whereas pure
    matmuls degrade gracefully (the NaN-surviving training loop then catches it)."""
    n = M.shape[1]
    v = torch.ones(n, device=M.device, dtype=M.dtype) / (n ** 0.5)
    for _ in range(iters):
        u = M @ v
        v = M.t() @ u
        nv = torch.linalg.vector_norm(v)
        v = v / (nv + 1e-30)
    return torch.linalg.vector_norm(M @ v)


def _ns_inverse(C, G0, steps, eta_p, rho, safeguard):
    """Warm-started Newton-Schulz / Riccati toward C^{-1}  (alpha = 1).
    G_{j+1} = G_j + eta_p (G_j - G_j C G_j), i.e. Riccati with source M = G_j.
    Includes a spectral safeguard against the conditional-convergence overshoot
    (Newton-Schulz only converges from G0 with ||C G0|| < ~2)."""
    Cs = _shrink(C, rho)
    G = G0
    for _ in range(steps):
        CG = Cs @ G
        # rescale G into the convergence basin if it has drifted out
        nrm = _spec_norm(CG)
        if nrm > safeguard:
            G = G * (safeguard / nrm)
            CG = Cs @ G
        G = G + eta_p * (G - G @ CG)
    return G


def _ns_inv_sqrt(C, steps, eps):
    """Coupled Newton-Schulz toward C^{-1/2}  (alpha = 1/2, whitening).
    Higham coupled iteration; fast (quadratic) and well-conditioned, so it is
    used fresh each step rather than warm-started. Same fixed point as the
    eps*I-source Riccati, but far better numerics for the inverse-square-root."""
    d = C.shape[0]
    eye = torch.eye(d, device=C.device, dtype=C.dtype)
    c = torch.diagonal(C).sum() + eps          # safe upper bound on spectral norm
    Y = C / c
    Z = eye.clone()
    for _ in range(steps):
        T = 1.5 * eye - 0.5 * (Z @ Y)
        Y = Y @ T
        Z = T @ Z
    return Z / torch.sqrt(c)                    # Z -> sqrt(c) * C^{-1/2}


def _riccati_with_source(C, G0, M, steps, eta_p, rho, safeguard):
    """General Riccati G <- G + eta_p (M - G C G) for an arbitrary SPD source M
    (used by the evolving-M meta-loop). Fixed point G C G = M."""
    Cs = _shrink(C, rho)
    G = G0
    for _ in range(steps):
        CG = Cs @ G
        nrm = _spec_norm(CG)
        if nrm > safeguard:
            G = G * (safeguard / nrm)
            CG = Cs @ G
        G = G + eta_p * (M - G @ CG)
    return G


def _precond_factor(C, G_prev, mode, steps, eta_p, rho, safeguard, eps):
    if mode == "inverse":
        return _ns_inverse(C, G_prev, steps, eta_p, rho, safeguard)
    elif mode == "whiten":
        return _ns_inv_sqrt(C, steps, eps)
    else:
        raise ValueError(f"unknown precond mode {mode!r}")


# --------------------------------------------------------------------------- #
# optimizer
# --------------------------------------------------------------------------- #

class RiccatiPrecond(Optimizer):
    """Matrix-free Kronecker preconditioner via the Riccati / Newton-Schulz iteration.

    Args:
      lr (eta):            step size on the preconditioned gradient
      precond:             'whiten' (alpha=1/2, stable default) or 'inverse' (alpha=1)
      shrink (rho):        shrinkage of curvature toward isotropy in [0,1).
                           The cheap stabilizer / effective-exponent reducer.
                           Schedule UP at small batch (see shrink_from_batch).
      beta_c:              EMA decay for the Kronecker curvature factors C_L, C_R
      inner_steps (k):     warm-started Riccati/NS iterations per step (1-3 typical)
      eta_p:               Riccati inner step size (inverse mode)
      damping (eps):       additive floor on curvature (Levenberg-Marquardt / mutational floor)
      safeguard:           spectral-norm cap on ||C G|| guarding NS overshoot
      precond_every:       update preconditioner factors every N steps (amortize)
      weight_decay:        decoupled (AdamW-style)
      momentum:            heavy-ball on the *preconditioned* update (0 disables)
      precond_stats_from_hook:
                           if True, accumulate the curvature factors C from
                           p._soap_precond_grad (true-Fisher gradient) instead of the
                           empirical gradient; the descent direction stays empirical.
      # --- evolving-M meta-loop (off by default; the non-standard piece) ---
      evolve_M:            if True, source M_L,M_R become slow learned matrices
      evolve_M_weighted:   loss-weighted CMA-ES rank-mu form (w ~ max(0,dL)); else skeleton
      eta_M:               meta learning rate for M (eta_M << eta_p); slow timescale
      meta_every:          accumulate into M every this many steps
      # --- Langevin sampling (off by default) ---
      langevin:            if True, inject FDT-consistent noise for SGLD
      temperature:         Langevin temperature T
      langevin_fdt:        if True, shape noise covariance ~ G (FDT-correct) rather than
                           the ~G^2 'G^{1/2} proxy' of the reference (see O6 / posterior).
      langevin_generator:  torch.Generator for reproducible noise (RNG discipline)
    """

    def __init__(self, params, lr=3e-3, precond="whiten", shrink=0.0,
                 beta_c=0.95, inner_steps=2, eta_p=0.5, damping=1e-6,
                 safeguard=8.0, precond_every=1, weight_decay=0.0, momentum=0.0,
                 precond_stats_from_hook=False,
                 evolve_M=False, evolve_M_weighted=True, eta_M=1e-3, meta_every=20,
                 langevin=False, temperature=1.0, langevin_fdt=False,
                 langevin_generator=None, max_factor_dim=4096):
        if evolve_M and precond == "whiten":
            warnings.warn("evolve_M has no effect with precond='whiten' (the source M "
                          "only enters the inverse branch); use precond='inverse'.")
        defaults = dict(lr=lr, precond=precond, shrink=shrink, beta_c=beta_c,
                        inner_steps=inner_steps, eta_p=eta_p, damping=damping,
                        safeguard=safeguard, precond_every=precond_every,
                        weight_decay=weight_decay, momentum=momentum,
                        precond_stats_from_hook=precond_stats_from_hook,
                        evolve_M=evolve_M, evolve_M_weighted=evolve_M_weighted,
                        eta_M=eta_M, meta_every=meta_every,
                        langevin=langevin, temperature=temperature,
                        langevin_fdt=langevin_fdt, max_factor_dim=max_factor_dim)
        super().__init__(params, defaults)
        self._langevin_generator = langevin_generator
        self._prev_loss = None       # for loss-weighted evolve-M
        self._meta_weight = 0.0      # productivity weight w = max(0, dL) for the current step

    @staticmethod
    def shrink_from_batch(batch_size, ref_batch=512, rho_min=0.0, rho_max=0.6):
        """Suggested rho schedule: more shrinkage (-> effective alpha toward 1/2)
        at smaller batch, where gradient noise is larger. Linear in 1/B; tune
        ref_batch to your setup. This realizes the noise-dependent alpha* law."""
        rho = rho_max * max(0.0, 1.0 - batch_size / ref_batch)
        return float(min(rho_max, max(rho_min, rho)))

    @torch.no_grad()
    def step(self, closure=None):
        """One optimizer step.

        If `closure` is given it must return the current scalar training loss
        (it is NOT re-evaluated for gradients -- gradients must already be populated).
        The loss is used only by the loss-weighted evolving-M meta-loop to compute the
        per-step productivity weight w = max(0, prev_loss - loss).
        """
        loss = closure() if closure is not None else None
        cur = float(loss) if loss is not None else None
        if cur is not None and self._prev_loss is not None:
            self._meta_weight = max(0.0, self._prev_loss - cur)
        else:
            self._meta_weight = 0.0
        for group in self.param_groups:
            self._step_group(group)
        if cur is not None:
            self._prev_loss = cur
        return loss

    @torch.no_grad()
    def _step_group(self, group):
        eps = group["damping"]; bc = group["beta_c"]
        for p in group["params"]:
            if p.grad is None:
                continue
            g = p.grad
            # curvature-source gradient: true Fisher hook if requested and present,
            # else the empirical gradient. The descent direction always uses g.
            stat_g = g
            if group["precond_stats_from_hook"]:
                hg = getattr(p, "_soap_precond_grad", None)
                if hg is not None:
                    stat_g = hg
            st = self.state[p]
            if len(st) == 0:
                st["t"] = 0
                st["param_dtype"] = g.dtype
                st["use_kron"] = (g.dim() == 2 and
                                  max(g.shape) <= group["max_factor_dim"])
                dev = g.device
                if st["use_kron"]:
                    m, n = g.shape
                    st["CL"] = torch.eye(m, device=dev, dtype=torch.float32)
                    st["CR"] = torch.eye(n, device=dev, dtype=torch.float32)
                    st["GL"] = torch.eye(m, device=dev, dtype=torch.float32)
                    st["GR"] = torch.eye(n, device=dev, dtype=torch.float32)
                    if group["evolve_M"]:
                        st["ML"] = eps * torch.eye(m, device=dev, dtype=torch.float32)
                        st["MR"] = eps * torch.eye(n, device=dev, dtype=torch.float32)
                        st["M_accL"] = torch.zeros(m, m, device=dev, dtype=torch.float32)
                        st["M_accR"] = torch.zeros(n, n, device=dev, dtype=torch.float32)
                        st["M_accW"] = 0.0
                else:
                    st["v"] = torch.zeros_like(g, dtype=torch.float32)  # diagonal second moment
                if group["momentum"] > 0:
                    st["buf"] = torch.zeros_like(g)
            st["t"] += 1

            # ---- decoupled weight decay ----
            if group["weight_decay"] > 0:
                p.mul_(1.0 - group["lr"] * group["weight_decay"])

            if st["use_kron"]:
                upd = self._kron_update(g, stat_g, st, group, eps, bc)
            else:
                upd = self._diag_update(g, stat_g, st, group, eps, bc)

            # ---- momentum on the preconditioned update ----
            if group["momentum"] > 0:
                buf = st["buf"]; buf.mul_(group["momentum"]).add_(upd)
                upd = buf

            p.add_(upd, alpha=-group["lr"])

            # ---- Langevin noise (FDT-consistent) ----
            if group["langevin"]:
                self._inject_langevin(p, st, group)

    # ----- Kronecker path (factors in float32, update cast back) -----
    def _kron_update(self, g, stat_g, st, group, eps, bc):
        m, n = g.shape
        g32 = g.float()
        s32 = g32 if stat_g is g else stat_g.float()
        CL, CR = st["CL"], st["CR"]
        # curvature factors (Shampoo/SOAP): row- and column-space second moments
        CL.mul_(bc).add_((s32 @ s32.t()) / n, alpha=1 - bc)
        CR.mul_(bc).add_((s32.t() @ s32) / m, alpha=1 - bc)
        eyeL = torch.eye(m, device=g.device, dtype=torch.float32)
        eyeR = torch.eye(n, device=g.device, dtype=torch.float32)
        CLd = CL + eps * eyeL
        CRd = CR + eps * eyeR

        if (st["t"] - 1) % group["precond_every"] == 0:
            if group["evolve_M"] and group["precond"] == "inverse":
                # source-driven Riccati with the (slowly) learned M  (rare path):
                st["GL"] = _riccati_with_source(CLd, st["GL"], st["ML"],
                                                group["inner_steps"], group["eta_p"],
                                                group["shrink"], group["safeguard"])
                st["GR"] = _riccati_with_source(CRd, st["GR"], st["MR"],
                                                group["inner_steps"], group["eta_p"],
                                                group["shrink"], group["safeguard"])
            else:
                st["GL"] = _precond_factor(CLd, st["GL"], group["precond"],
                                           group["inner_steps"], group["eta_p"],
                                           group["shrink"], group["safeguard"], eps)
                st["GR"] = _precond_factor(CRd, st["GR"], group["precond"],
                                           group["inner_steps"], group["eta_p"],
                                           group["shrink"], group["safeguard"], eps)

        upd32 = st["GL"] @ g32 @ st["GR"]

        # ---- evolving-M meta-loop: slow rank-mu accumulation of productive steps ----
        if group["evolve_M"]:
            s = upd32
            if group["evolve_M_weighted"]:
                # weight this step's outer product by its productivity w = max(0, dL),
                # self-normalized over the meta window (CMA-ES rank-mu).
                w = self._meta_weight
                st["M_accL"].add_((s @ s.t()) / n, alpha=w)
                st["M_accR"].add_((s.t() @ s) / m, alpha=w)
                st["M_accW"] += w
                if st["t"] % group["meta_every"] == 0:
                    if st["M_accW"] > 0:
                        etaM = group["eta_M"]
                        st["ML"].mul_(1 - etaM).add_(st["M_accL"] / st["M_accW"], alpha=etaM)
                        st["MR"].mul_(1 - etaM).add_(st["M_accR"] / st["M_accW"], alpha=etaM)
                    st["M_accL"].zero_(); st["M_accR"].zero_(); st["M_accW"] = 0.0
            elif st["t"] % group["meta_every"] == 0:
                # unweighted skeleton (the reference form), kept for ablation
                etaM = group["eta_M"]
                st["ML"].mul_(1 - etaM).add_((s @ s.t()) / n, alpha=etaM)
                st["MR"].mul_(1 - etaM).add_((s.t() @ s) / m, alpha=etaM)

        return upd32.to(st["param_dtype"])

    # ----- diagonal path (1-D params, large factors); second moment in float32 -----
    def _diag_update(self, g, stat_g, st, group, eps, bc):
        v = st["v"]
        s32 = stat_g.float()
        v.mul_(bc).add_(s32 * s32, alpha=1 - bc)
        alpha = 0.5 if group["precond"] == "whiten" else 1.0
        rho = group["shrink"]
        vs = (1 - rho) * v + rho * v.mean()            # scalar shrinkage
        h = (vs + eps) ** (-alpha)
        return (h * g.float()).to(st["param_dtype"])

    # ----- Langevin -----
    def _inject_langevin(self, p, st, group):
        T, lr = group["temperature"], group["lr"]
        scale = math.sqrt(2.0 * lr * T)
        gen = self._langevin_generator
        if gen is not None:
            noise = torch.randn(p.shape, generator=gen, device=p.device, dtype=p.dtype)
        else:
            noise = torch.randn_like(p)
        if st["use_kron"]:
            n32 = noise.float()
            if group["langevin_fdt"] and "GL_half" in st:
                # FDT-correct: covariance ~ G via a one-time G^{1/2} factor (set at freeze).
                half = st["GL_half"] @ n32 @ st["GR_half"]
            else:
                # reference 'G^{1/2} proxy' (covariance ~ G^2); see docstring caveat.
                half = st["GL"] @ n32 @ st["GR"]
            p.add_(half.to(st["param_dtype"]), alpha=scale)
        else:
            alpha = 0.5 if group["precond"] == "whiten" else 1.0
            h = (st["v"] + group["damping"]) ** (-alpha / 2.0)
            p.add_((h * noise.float()).to(st["param_dtype"]), alpha=scale)

    @torch.no_grad()
    def freeze_langevin_factors(self):
        """Compute and cache G^{1/2} factors (GL_half, GR_half) for the FDT-correct
        Langevin path. Call AFTER the basis has stabilized (warm-up-then-freeze) and
        with langevin_fdt=True. Uses an eigh once per factor (allowed: basis is frozen,
        factors are small) so the injected noise covariance is exactly ~ G (not G^2)."""
        for group in self.param_groups:
            for p in group["params"]:
                st = self.state[p]
                if not st or not st.get("use_kron", False):
                    continue
                st["GL_half"] = _spd_sqrt(st["GL"])
                st["GR_half"] = _spd_sqrt(st["GR"])


def _spd_sqrt(G):
    """Symmetric PSD square root via eigh (factors are small and frozen)."""
    Gs = 0.5 * (G + G.t())
    w, V = torch.linalg.eigh(Gs)
    w = torch.clamp(w, min=0.0)
    return (V * w.sqrt()) @ V.t()

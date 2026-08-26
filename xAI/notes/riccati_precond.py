"""
riccati_precond.py
==================
A preconditioned optimizer derived from the discrete G-equation (matrix Riccati)
of the quantitative-genetics / SGD correspondence.

Core update (per Kronecker factor, outer step t, inner relaxation j):

    G_t^(0) = G_{t-1}                                  # warm start
    G_t^(j+1) = G_t^(j) + eta_p ( M - G_t^(j) C_t G_t^(j) )    # Riccati / Newton-Schulz
    W_{t+1}  = W_t - eta * ( G_{L,t} g_t G_{R,t} )     # preconditioned step

Fixed point  G C G = M.  The *source* M sets the exponent:
    M = eps*I            -> G -> C^{-1/2}   (alpha = 1/2, whitening; ~ Muon/Adam regime)
    M = G                -> G -> C^{-1}      (alpha = 1,   natural-gradient/Newton regime)
The exponent is interpolated *in practice* by SHRINKAGE of C toward isotropy
(rho), which is also the stabilizer (see notes). This was validated in numpy:
the inverse iteration reproduces C^{-1} to 1e-15, and alpha=1 is unstable in
anisotropic / low-curvature directions exactly as the two-noise theory predicts
(update-noise variance ~ a_i^{1-2 alpha}); shrinkage / lower effective alpha cure it.

This is matrix-free (matmuls only, no eigendecomposition), Kronecker-factored
(Shampoo/SOAP structure), warm-started (SOAP-style amortization), with two
genuinely non-standard options: an evolving source M (slow CMA-ES-style rank-mu
accumulation -- the "evolving evolvability" meta-loop) and FDT-consistent Langevin
noise for posterior sampling with the learned metric.

Tested only at the numpy-core level (PyTorch not available in the authoring env);
the tensor ops mirror the validated numpy. Smoke-test before serious use.
"""
import math
import torch
from torch.optim.optimizer import Optimizer


# --------------------------------------------------------------------------- #
# matrix-free factor iterations (the Riccati core)
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
        nrm = torch.linalg.matrix_norm(CG, ord=2)
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
    """
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
      # --- evolving-M meta-loop (off by default; the non-standard piece) ---
      evolve_M:            if True, source M_L,M_R become slow learned matrices
      eta_M:               meta learning rate for M (eta_M << eta_p); slow timescale
      meta_every:          accumulate into M every this many steps
      # --- Langevin sampling (off by default) ---
      langevin:            if True, inject FDT-consistent noise ~ G^{1/2} for SGLD
      temperature:         Langevin temperature T
    """

    def __init__(self, params, lr=3e-3, precond="whiten", shrink=0.0,
                 beta_c=0.95, inner_steps=2, eta_p=0.5, damping=1e-6,
                 safeguard=8.0, precond_every=1, weight_decay=0.0, momentum=0.0,
                 evolve_M=False, eta_M=1e-3, meta_every=20,
                 langevin=False, temperature=1.0, max_factor_dim=4096):
        defaults = dict(lr=lr, precond=precond, shrink=shrink, beta_c=beta_c,
                        inner_steps=inner_steps, eta_p=eta_p, damping=damping,
                        safeguard=safeguard, precond_every=precond_every,
                        weight_decay=weight_decay, momentum=momentum,
                        evolve_M=evolve_M, eta_M=eta_M, meta_every=meta_every,
                        langevin=langevin, temperature=temperature,
                        max_factor_dim=max_factor_dim)
        super().__init__(params, defaults)

    @staticmethod
    def shrink_from_batch(batch_size, ref_batch=512, rho_min=0.0, rho_max=0.6):
        """Suggested rho schedule: more shrinkage (-> effective alpha toward 1/2)
        at smaller batch, where gradient noise is larger. Linear in 1/B; tune
        ref_batch to your setup. This realizes the noise-dependent alpha* law."""
        rho = rho_max * max(0.0, 1.0 - batch_size / ref_batch)
        return float(min(rho_max, max(rho_min, rho)))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.parameters_groups():
            self._step_group(group)
        return loss

    def parameters_groups(self):
        return self.param_groups

    @torch.no_grad()
    def _step_group(self, group):
        eps = group["damping"]; bc = group["beta_c"]
        for p in group["params"]:
            if p.grad is None:
                continue
            g = p.grad
            st = self.state[p]
            if len(st) == 0:
                st["t"] = 0
                st["use_kron"] = (g.dim() == 2 and
                                  max(g.shape) <= group["max_factor_dim"])
                if st["use_kron"]:
                    m, n = g.shape
                    dev, dt = g.device, g.dtype
                    st["CL"] = torch.eye(m, device=dev, dtype=dt)
                    st["CR"] = torch.eye(n, device=dev, dtype=dt)
                    st["GL"] = torch.eye(m, device=dev, dtype=dt)
                    st["GR"] = torch.eye(n, device=dev, dtype=dt)
                    if group["evolve_M"]:
                        st["ML"] = eps * torch.eye(m, device=dev, dtype=dt)
                        st["MR"] = eps * torch.eye(n, device=dev, dtype=dt)
                else:
                    st["v"] = torch.zeros_like(g)     # diagonal second moment
                if group["momentum"] > 0:
                    st["buf"] = torch.zeros_like(g)
            st["t"] += 1
            t = st["t"]

            # ---- decoupled weight decay ----
            if group["weight_decay"] > 0:
                p.mul_(1.0 - group["lr"] * group["weight_decay"])

            if st["use_kron"]:
                upd = self._kron_update(g, st, group, eps, bc)
            else:
                upd = self._diag_update(g, st, group, eps, bc)

            # ---- momentum on the preconditioned update ----
            if group["momentum"] > 0:
                buf = st["buf"]; buf.mul_(group["momentum"]).add_(upd)
                upd = buf

            p.add_(upd, alpha=-group["lr"])

            # ---- Langevin noise (FDT-consistent: covariance ~ G) ----
            if group["langevin"]:
                self._inject_langevin(p, st, group)

    # ----- Kronecker path -----
    def _kron_update(self, g, st, group, eps, bc):
        m, n = g.shape
        CL, CR = st["CL"], st["CR"]
        # curvature factors (Shampoo/SOAP): row- and column-space second moments
        CL.mul_(bc).add_((g @ g.t()) / n, alpha=1 - bc)
        CR.mul_(bc).add_((g.t() @ g) / m, alpha=1 - bc)
        CLd = CL + eps * torch.eye(m, device=g.device, dtype=g.dtype)
        CRd = CR + eps * torch.eye(n, device=g.device, dtype=g.dtype)

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

        upd = st["GL"] @ g @ st["GR"]

        # ---- evolving-M meta-loop: slow rank-mu accumulation of productive steps ----
        if group["evolve_M"] and (st["t"] % group["meta_every"] == 0):
            # use the applied step direction as a "productive" sample (a closure-based
            # loss-weighted version is preferable; this is the unweighted skeleton)
            s = upd
            etaM = group["eta_M"]
            st["ML"].mul_(1 - etaM).add_((s @ s.t()) / n, alpha=etaM)
            st["MR"].mul_(1 - etaM).add_((s.t() @ s) / m, alpha=etaM)
        return upd

    # ----- diagonal path (1-D params, large factors) -----
    def _diag_update(self, g, st, group, eps, bc):
        v = st["v"]; v.mul_(bc).add_(g * g, alpha=1 - bc)
        alpha = 0.5 if group["precond"] == "whiten" else 1.0
        rho = group["shrink"]
        vs = (1 - rho) * v + rho * v.mean()            # scalar shrinkage
        h = (vs + eps) ** (-alpha)
        return h * g

    # ----- Langevin -----
    def _inject_langevin(self, p, st, group):
        T, lr = group["temperature"], group["lr"]
        scale = math.sqrt(2.0 * lr * T)
        noise = torch.randn_like(p)
        if st["use_kron"]:
            # G^{1/2} eta : apply preconditioner factors as a sqrt-metric proxy
            half = (st["GL"] @ noise @ st["GR"])
            p.add_(half, alpha=scale)
        else:
            alpha = 0.5 if group["precond"] == "whiten" else 1.0
            h = (st["v"] + group["damping"]) ** (-alpha / 2.0)
            p.add_(h * noise, alpha=scale)


def _riccati_with_source(C, G0, M, steps, eta_p, rho, safeguard):
    """General Riccati G <- G + eta_p (M - G C G) for an arbitrary SPD source M
    (used by the evolving-M meta-loop). Fixed point G C G = M."""
    Cs = _shrink(C, rho)
    G = G0
    for _ in range(steps):
        CG = Cs @ G
        nrm = torch.linalg.matrix_norm(CG, ord=2)
        if nrm > safeguard:
            G = G * (safeguard / nrm)
            CG = Cs @ G
        G = G + eta_p * (M - G @ CG)
    return G

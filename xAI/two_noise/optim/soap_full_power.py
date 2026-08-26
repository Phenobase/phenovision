"""
Full-power SOAP (a.k.a. SOAP-NG) — a PyTorch optimizer.

This is a modification of the SOAP algorithm (Vyas et al., ICLR 2025,
arXiv:2409.11321), which itself runs Adam in the eigenbasis of Shampoo's
Kronecker-factored preconditioner. SOAP applies the *inverse square root* of the
curvature estimate (gradient whitening); this variant exposes the exponent as a
free parameter so the optimizer can apply the *full inverse* (Newton /
natural-gradient-in-the-eigenbasis) while keeping SOAP's two-timescale cost.

Why this is free, structurally (Morwani et al., 2024, arXiv:2406.17748):
  - The slow component (eigenbasis Q_L, Q_R of L = E[G Gᵀ], R = E[Gᵀ G]) is the
    Kronecker estimate of the curvature matrix H = E[g gᵀ]. It does not depend
    on the exponent.
  - In that basis the per-coordinate second moment V_ij estimates the curvature
    eigenvalue λ_i μ_j. SOAP divides by sqrt(V) (power 1/2 -> whitening);
    dividing by V (power 1) gives the full inverse H^{-1}.
The efficiency over full Fisher scoring comes entirely from the Kronecker +
eigenbasis structure, never from the square root — so swapping the exponent
costs nothing in compute or memory.

Key differences from vanilla SOAP, all confined to the inner loop:
  1. precond_power: 0.5 recovers SOAP; 1.0 is full natural gradient / Newton.
  2. Levenberg–Marquardt damping (relative to the top eigenvalue by default),
     which is essential once the square root is removed, because flat directions
     would otherwise produce unbounded steps.
  3. Bias correction generalized to an arbitrary exponent.
  4. Optional global update-norm clipping (a cheap trust region) and an optional
     hook to feed *true-Fisher* (sampled-label) gradients to the preconditioner
     statistics instead of the empirical Fisher.

Scope: handles 2D parameters (the layers that matter for transformers / ViTs)
with full Kronecker preconditioning, and falls back to AdamW for 1D parameters,
exactly as SOAP does by default.
"""

from typing import Iterable, Tuple

import torch
from torch import Tensor

from optim.demographic_noise import langevin_noise_std


def _eigh_basis(mat: Tensor) -> Tensor:
    """Eigenvectors of a symmetric PSD matrix, columns sorted by descending eigenvalue."""
    # Tiny ridge for numerical robustness; fall back to float64 if eigh struggles.
    ridge = 1e-30 * torch.eye(mat.shape[0], device=mat.device, dtype=mat.dtype)
    try:
        evals, evecs = torch.linalg.eigh(mat + ridge)
    except Exception:
        evals, evecs = torch.linalg.eigh((mat + ridge).double())
        evals, evecs = evals.to(mat.dtype), evecs.to(mat.dtype)
    # eigh returns ascending; flip to descending so column 0 is the top eigenvector.
    return torch.flip(evecs, dims=[1])


def _qr_power_step(mat: Tensor, q_prev: Tensor) -> Tensor:
    """One step of subspace (power) iteration + QR — the cheap eigenbasis refresh.

    This is the Morwani et al. (2024) 'one round of power iteration from the
    current estimate' update. Much cheaper than a full eigh for large layers.
    """
    q, _ = torch.linalg.qr(mat @ q_prev)
    return q


class SOAPFullPower(torch.optim.Optimizer):
    """SOAP with a tunable preconditioner exponent (1.0 = full inverse-curvature).

    Args:
        params: parameters or param groups.
        lr: learning rate. NOTE: full power behaves like Newton, so the useful
            range is very different from SOAP/AdamW — start ~10x smaller and
            rely on damping + update clipping. Typical: 1e-3 to 3e-3 at power 1.0
            with relative damping ~1e-2, but tune.
        betas: (beta1, beta2) for the first/second moment EMAs.
        shampoo_beta: EMA for the L,R preconditioner statistics. If < 0, uses betas[1].
        precond_power: exponent applied to the curvature eigenvalues.
            0.5 -> vanilla SOAP (whitening). 1.0 -> Newton / natural gradient.
        shrink: Ledoit-Wolf shrinkage of the curvature spectrum toward isotropy in
            [0, 1), applied to the in-basis second moment BEFORE the power:
            v -> (1-shrink)*v + shrink*mean(v). Because the steep directions keep ~v
            while the flat ones are pulled toward mean(v), this yields a SPECTRALLY
            NON-UNIFORM effective exponent: ~precond_power where curvature is large,
            -> 0 where it is small. That is exactly the noise-aware preconditioner the
            two-noise SDE predicts (the alpha=1 update-noise ~ a^{1-2*power} blows up in
            the flat directions, which shrinkage selectively tames). shrink=0 disables.
            Same isotropy target as RiccatiPrecond's matrix-free _shrink, computed here
            exactly from the spectrum instead of via Newton-Schulz.
        evolve_m: the evolving-M meta-loop (the optimizer-side analog of M-evolution in
            biology), the SOAP counterpart of RiccatiPrecond.evolve_M. In the curvature
            eigenbasis the matrix source M collapses to a DIAGONAL, i.e. per-eigendirection
            row/column gain vectors m_L (len m), m_R (len n): the applied preconditioner
            becomes  h_ij = sqrt(m_L,i * m_R,j) * v_ij^{-power}  (whitening times a learned
            per-direction gain). The gains are a slow, loss-weighted CMA-ES rank-mu EMA of the
            PRODUCTIVE update's per-axis energy (only steps with dL>0 count). m is NORMALIZED to
            unit mean before use, so it reshapes the preconditioner's ANISOTROPY without changing
            the global step scale (that is the lr's job). Use with precond_power=0.5 (whitening
            base, mirroring Riccati's M=eps*I start). 2D Kronecker layers only. OFF by default.
        evolve_m_weighted: loss-weighted (w=max(0,dL), self-normalized per window) vs unweighted.
        eta_m: meta learning-rate for the m_L/m_R EMA (eta_m << lr; slow timescale).
        meta_every: accumulate productive-step energy into m every this many steps.
        damping: Levenberg–Marquardt term added to the (powered) denominator.
            With relative_damping=True it is scaled by the largest per-coordinate
            curvature, i.e. denom = V**p + damping * (V**p).amax(). This is the
            knob that keeps flat directions stable once the sqrt is gone.
        relative_damping: scale damping by the top eigenvalue (recommended at p=1).
        eps: absolute floor added to the denominator (kept for p=0.5 parity).
        weight_decay: decoupled (AdamW-style) weight decay.
        precondition_frequency: steps between eigenbasis refreshes (the slow timescale).
        max_precond_dim: skip Kronecker preconditioning on dims larger than this
            (e.g. vocab/embedding) and treat them Adam-style on that axis.
        precondition_1d: if True, also build a 1D preconditioner; else AdamW for 1D.
        use_qr_refresh: cheap QR power-iteration refresh (True) vs full eigh (False).
        max_update_norm: optional global-norm clip on the final update (<=0 disables).
        correct_bias: Adam bias correction.
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 3e-3,
        betas: Tuple[float, float] = (0.95, 0.95),
        shampoo_beta: float = -1.0,
        precond_power: float = 1.0,
        shrink: float = 0.0,
        evolve_m: bool = False,
        evolve_m_weighted: bool = True,
        eta_m: float = 1e-3,
        meta_every: int = 20,
        m_noise: float = 0.0,
        m_load: float = 0.0,
        m_nstar: float = 1e9,
        damping: float = 1e-2,
        relative_damping: bool = True,
        eps: float = 1e-12,
        weight_decay: float = 0.01,
        precondition_frequency: int = 10,
        max_precond_dim: int = 10000,
        precondition_1d: bool = False,
        use_qr_refresh: bool = True,
        max_update_norm: float = 0.0,
        correct_bias: bool = True,
        demographic_noise: bool = False,
        demographic_temperature: float = 0.0,
        demographic_generator: "torch.Generator | None" = None,
        demographic_precond_clamp: float = 0.0,
        demographic_warmup: int = 0,
        precond_eigvals_from_hook: bool = False,
        parity_mode: bool = False,
    ):
        defaults = dict(
            lr=lr, betas=betas, shampoo_beta=shampoo_beta,
            precond_power=precond_power, shrink=shrink,
            evolve_m=evolve_m, evolve_m_weighted=evolve_m_weighted,
            eta_m=eta_m, meta_every=meta_every, m_noise=m_noise, m_load=m_load,
            m_nstar=m_nstar, damping=damping,
            relative_damping=relative_damping, eps=eps,
            weight_decay=weight_decay,
            precondition_frequency=precondition_frequency,
            max_precond_dim=max_precond_dim, precondition_1d=precondition_1d,
            use_qr_refresh=use_qr_refresh, max_update_norm=max_update_norm,
            correct_bias=correct_bias,
            # --- §9.5 demographic-noise (pSGLD) injection; OFF by default ---
            demographic_noise=demographic_noise,
            demographic_temperature=demographic_temperature,
            demographic_generator=demographic_generator,
            demographic_precond_clamp=demographic_precond_clamp,
            demographic_warmup=demographic_warmup,
            # --- accumulate the in-basis second moment (eigenvalues) from the true-Fisher hook
            #     gradient instead of the empirical gradient (the descent direction stays empirical).
            #     Fixes flat-direction over-amplification when the empirical Fisher is mis-specified.
            precond_eigvals_from_hook=precond_eigvals_from_hook,
            # --- exact-parity routing with vanilla SOAP (test-only) ---
            parity_mode=parity_mode,
        )
        super().__init__(params, defaults)
        # per-device RNG cache so a user CPU generator can drive GPU param noise (M2 fix).
        self._demo_gen_cache: dict = {}
        # evolve-M meta-loop: productivity weight w = max(0, dL) from the step closure.
        self._prev_loss = None
        self._meta_weight = 1.0

    def _device_generator(self, user_gen, device):
        """Return a generator on `device`. If the user passed a generator on another device,
        lazily create a device-matched one seeded deterministically from it (so seeding is
        honored and CUDA params don't crash on a CPU generator)."""
        if user_gen is None:
            return None
        if user_gen.device == device:
            return user_gen
        if device not in self._demo_gen_cache:
            seed = int(torch.randint(0, 2 ** 31 - 1, (1,), generator=user_gen).item())
            g = torch.Generator(device=device)
            g.manual_seed(seed)
            self._demo_gen_cache[device] = g
        return self._demo_gen_cache[device]

    @classmethod
    def vanilla_soap_config(cls) -> dict:
        """Hyperparameters that make SOAPFullPower reproduce vanilla SOAP exactly.

        Use as ``SOAPFullPower(params, **SOAPFullPower.vanilla_soap_config(), lr=...,
        betas=..., precondition_frequency=...)`` in the parity test. ``parity_mode``
        routes the denominator as ``sqrt(eas).add_(eps)`` with bias correction folded
        into ``step_size`` (exactly as official SOAP), removing the residual
        ``eps*(1-sqrt(1-beta2**t))`` mismatch that the production tensor-folded path
        carries. Production defaults stay Newton-like (precond_power=1.0); this is opt-in.
        """
        return dict(
            precond_power=0.5,
            damping=0.0,
            relative_damping=False,
            max_update_norm=0.0,
            weight_decay=0.0,
            use_qr_refresh=False,   # full eigh refresh, matching official init basis
            correct_bias=True,
            parity_mode=True,
        )

    # ---- eigenbasis projection for 2D tensors: G' = Q_Lᵀ G Q_R --------------
    @staticmethod
    def _project(g: Tensor, qL, qR) -> Tensor:
        if qL is not None:
            g = qL.t() @ g
        if qR is not None:
            g = g @ qR
        return g

    @staticmethod
    def _project_back(g: Tensor, qL, qR) -> Tensor:
        if qL is not None:
            g = qL @ g
        if qR is not None:
            g = g @ qR.t()
        return g

    def _init_state(self, p, grad, group, state):
        state["step"] = 0
        state["exp_avg"] = torch.zeros_like(grad)      # first moment, rotated basis
        state["exp_avg_sq"] = torch.zeros_like(grad)   # second moment, rotated basis
        state["use_precond"] = False
        if grad.dim() == 2:
            m, n = grad.shape
            state["use_precond"] = True
            state["L"] = torch.zeros(m, m, device=grad.device, dtype=grad.dtype) if m <= group["max_precond_dim"] else None
            state["R"] = torch.zeros(n, n, device=grad.device, dtype=grad.dtype) if n <= group["max_precond_dim"] else None
            state["QL"] = None
            state["QR"] = None
            if group["evolve_m"]:
                # per-eigendirection learned gains (the diagonal source M) + windowed accumulators
                state["m_L"] = torch.ones(m, device=grad.device, dtype=torch.float32)
                state["m_R"] = torch.ones(n, device=grad.device, dtype=torch.float32)
                state["m_accL"] = torch.zeros(m, device=grad.device, dtype=torch.float32)
                state["m_accR"] = torch.zeros(n, device=grad.device, dtype=torch.float32)
                state["m_accW"] = 0.0
        elif grad.dim() == 1 and group["precondition_1d"] and grad.shape[0] <= group["max_precond_dim"]:
            state["use_precond"] = True
            d = grad.shape[0]
            state["L"] = torch.zeros(d, d, device=grad.device, dtype=grad.dtype)
            state["R"] = None
            state["QL"] = None
            state["QR"] = None

    def _accumulate_stats(self, grad2d, group, state):
        """EMA of the Kronecker factors L = E[GGᵀ], R = E[GᵀG]."""
        beta = group["shampoo_beta"] if group["shampoo_beta"] >= 0 else group["betas"][1]
        g = grad2d
        if state.get("L") is not None:
            state["L"].lerp_(g @ g.t(), 1.0 - beta)
        if state.get("R") is not None:
            state["R"].lerp_(g.t() @ g, 1.0 - beta)

    def _refresh_basis(self, group, state):
        """Recompute Q_L, Q_R (the slow timescale)."""
        use_qr = group["use_qr_refresh"] and state["QL"] is not None
        if state.get("L") is not None:
            if use_qr and state["QL"] is not None:
                state["QL"] = _qr_power_step(state["L"], state["QL"])
            else:
                state["QL"] = _eigh_basis(state["L"])
        if state.get("R") is not None:
            if use_qr and state["QR"] is not None:
                state["QR"] = _qr_power_step(state["R"], state["QR"])
            else:
                state["QR"] = _eigh_basis(state["R"])

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None

        # evolve-M productivity weight: w = max(0, prev_loss - loss) (loss decrease this step),
        # self-normalized per meta window below. No closure -> unweighted (w=1).
        if loss is not None:
            cur = float(loss)
            self._meta_weight = max(0.0, self._prev_loss - cur) if self._prev_loss is not None else 0.0
            self._prev_loss = cur
        else:
            self._meta_weight = 1.0

        for group in self.param_groups:
            p_clip = group["max_update_norm"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                if "step" not in state:
                    self._init_state(p, grad, group, state)
                    # Seed preconditioner stats and basis; skip the first update so we
                    # never precondition with the gradient that built the basis.
                    if state["use_precond"]:
                        g2d = grad if grad.dim() == 2 else grad.unsqueeze(1)
                        precond_grad = getattr(p, "_soap_precond_grad", None)
                        if precond_grad is not None:
                            g2d = precond_grad if precond_grad.dim() == 2 else precond_grad.unsqueeze(1)
                        self._accumulate_stats(g2d, group, state)
                        self._refresh_basis(group, state)
                    continue

                state["step"] += 1
                beta1, beta2 = group["betas"]
                qL, qR = state.get("QL"), state.get("QR")
                two_d = grad.dim() == 2
                g2d = grad if two_d else grad.unsqueeze(1)

                if state["use_precond"]:
                    g_rot = self._project(g2d, qL, qR)
                else:
                    g_rot = g2d  # AdamW path (1D etc.)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                ea = exp_avg if two_d else exp_avg.unsqueeze(1)
                eas = exp_avg_sq if two_d else exp_avg_sq.unsqueeze(1)

                ea.mul_(beta1).add_(g_rot, alpha=1.0 - beta1)
                # Second moment (the preconditioner eigenvalues): from the empirical projected
                # gradient by default, or from the projected TRUE-FISHER hook gradient when
                # precond_eigvals_from_hook is set (so the eigenvalue scaling, not just the basis,
                # reflects the true curvature). The first moment (descent direction) stays empirical.
                sq_src = g_rot
                if (group["precond_eigvals_from_hook"] and state["use_precond"]):
                    pg = getattr(p, "_soap_precond_grad", None)
                    if pg is not None:
                        pg2d = pg if pg.dim() == 2 else pg.unsqueeze(1)
                        sq_src = self._project(pg2d, qL, qR)
                eas.mul_(beta2).add_(sq_src.square(), alpha=1.0 - beta2)

                # --- THE MODIFICATION: tunable exponent + LM damping ---------
                # Vanilla SOAP: denom = eas.sqrt() + eps   (power = 0.5)
                # Full power:   denom = eas + damping        (power = 1.0)
                power = group["precond_power"]
                demo_noise_param = None
                if state["use_precond"]:
                    if group["parity_mode"]:
                        # Exact vanilla-SOAP arithmetic: sqrt-then-eps on the raw second
                        # moment, bias correction folded into step_size (not the tensors).
                        # Removes the eps*(1-sqrt(1-beta2**t)) residual the production
                        # tensor-folded path carries. Test-only; see vanilla_soap_config().
                        denom = eas.sqrt().add_(group["eps"])
                        update_rot = ea / denom
                        step_size = group["lr"]
                        if group["correct_bias"]:
                            bc1 = 1.0 - beta1 ** state["step"]
                            bc2 = 1.0 - beta2 ** state["step"]
                            step_size = step_size * (bc2 ** 0.5) / bc1
                    else:
                        if group["correct_bias"]:
                            v_hat = eas / (1.0 - beta2 ** state["step"])
                            m_hat = ea / (1.0 - beta1 ** state["step"])
                        else:
                            v_hat, m_hat = eas, ea
                        # Shrink the spectrum toward isotropy BEFORE the power (effective-exponent
                        # reduction; steep directions keep ~v, flat ones pulled to mean(v)).
                        rho = group["shrink"]
                        if rho > 0.0:
                            v_hat = (1.0 - rho) * v_hat + rho * v_hat.mean()
                        denom = v_hat.pow(power) if power != 1.0 else v_hat.clone()
                        if group["relative_damping"]:
                            denom = denom + group["damping"] * denom.amax().clamp_min(group["eps"])
                        else:
                            denom = denom + group["damping"]
                        denom = denom.add_(group["eps"])
                        update_rot = m_hat / denom
                        step_size = group["lr"]

                        # --- evolving-M meta-loop (the SOAP analog of RiccatiPrecond.evolve_M) ---
                        # In the eigenbasis M is DIAGONAL: per-axis gains m_L, m_R. Apply
                        # sqrt(m_L outer m_R) to the in-basis update (normalized to unit mean so
                        # only the ANISOTROPY acts, not the global scale), then accumulate the
                        # PRODUCTIVE move's per-axis energy (loss-weighted rank-mu, slow EMA).
                        if group["evolve_m"] and two_d and "m_L" in state:
                            mL, mR = state["m_L"], state["m_R"]
                            gL = (mL / mL.mean().clamp_min(group["eps"])).sqrt()
                            gR = (mR / mR.mean().clamp_min(group["eps"])).sqrt()
                            # exploration noise on M (single-trajectory analog of population diversity
                            # / the batch noise that lets C be estimated): perturb the per-direction
                            # gain log-normally each step, then accumulate the PERTURBED productive
                            # move -- so perturbations that reduced the loss (w>0) are reinforced.
                            mn = group["m_noise"]
                            if mn > 0.0:
                                gL = gL * torch.exp(mn * torch.randn_like(gL))
                                gR = gR * torch.exp(mn * torch.randn_like(gR))
                            update_rot = update_rot * (gL.unsqueeze(1) * gR.unsqueeze(0))
                            s = update_rot
                            srow = (s * s).sum(dim=1)          # diag(s sᵀ), len m
                            scol = (s * s).sum(dim=0)          # diag(sᵀ s), len n
                            mm, nn = s.shape
                            # accumulate the loss-weighted (w=max(0,dL)) productive energy over the window
                            w = self._meta_weight if group["evolve_m_weighted"] else 1.0
                            state["m_accL"].add_(srow, alpha=w / nn)
                            state["m_accR"].add_(scol, alpha=w / mm)
                            state["m_accW"] += w
                            if state["step"] % group["meta_every"] == 0 and state["m_accW"] > 0:
                                # LOG-SPACE M-update: evolve log(m) so m stays > 0 (SPD) under all
                                # three additive terms -- benefit, mutation-load COST, noise:
                                #   d log m_i = eta * ( benefit_i  -  kappa * load_i )
                                # benefit_i = centered log(productive energy)  (drives M -> A^-1 at
                                #   equilibrium, the climb); load_i = mutation-load gradient
                                #   d/dlog(m)[tr(A M)] = v_i * m_i, which BLOWS UP exactly where M runs
                                #   away (m huge in flat dirs) -> the restoring force that was missing.
                                em = group["eta_m"]; eps = group["eps"]
                                # theory's noise-aware load coefficient kappa(N*) = 0.5 + 1/(4 N*):
                                # heavier cost at small N* (more gradient noise) -> lower exponent.
                                kappa = group["m_load"] * (0.5 + 1.0 / (4.0 * group["m_nstar"]))
                                Wn = state["m_accW"]
                                vrow = v_hat.mean(dim=1).clamp_min(eps)   # per-row curvature scale (a_i)
                                vcol = v_hat.mean(dim=0).clamp_min(eps)
                                for key, acc, vsc in (("m_L", "m_accL", vrow), ("m_R", "m_accR", vcol)):
                                    m = state[key]
                                    b = (state[acc] / Wn).clamp_min(eps)
                                    benefit = torch.log(b); benefit = benefit - benefit.mean()
                                    # mutation-load gradient from L_eff = kappa*tr(A*Ghat): d/dlog(m) = (kappa/2)*sqrt(a_i*m_i)
                                    load = (vsc * m).sqrt(); load = load / load.mean().clamp_min(eps)
                                    ell = torch.log(m.clamp_min(eps)) + em * (benefit - kappa * (load - 1.0))
                                    ell = ell - ell.mean()                # only the SHAPE matters (gain unit-mean-normed)
                                    state[key] = torch.exp(ell)
                                state["m_accL"].zero_(); state["m_accR"].zero_(); state["m_accW"] = 0.0

                    # --- §9.5 demographic-noise (pSGLD) injection, in the eigenbasis ---
                    # precond_diag = 1/denom is the APPLIED preconditioner H per-coordinate.
                    # We add parameter-space noise ~ N(0, 2*T*lr*H) directly to p (NOT
                    # scaled by step_size), so the SDE is dθ = -lr*H∇L + sqrt(2*T*lr*H)dW
                    # whose stationary law is the Gibbs measure π ∝ exp(-L/T). Injecting
                    # in the rotated basis with per-coord std sqrt(2*T*lr/denom) and
                    # projecting back by orthogonal Q yields param-space cov Q diag Qᵀ = H.
                    inject = (group["demographic_noise"]
                              and group["demographic_temperature"] > 0.0
                              and state["step"] > group["demographic_warmup"])
                    if inject:
                        # Per-coordinate Langevin noise std sqrt(2*T*lr/denom) in the eigenbasis
                        # (single source of truth: optim.demographic_noise). The applied
                        # preconditioner is H_coord = 1/denom; for FDT the noise covariance must
                        # share it. CAVEAT: 1/denom is the Adam second-moment denominator, which
                        # → 0 on a *noise-free* gradient (init / a deterministic minimum), making
                        # the noise blow up. The pSGLD/Gibbs guarantee holds in the STOCHASTIC
                        # regime (denom → C/S floor) used in §2.4; `demographic_warmup` skips the
                        # warmup transient and `demographic_precond_clamp` caps 1/denom so a
                        # near-zero denom cannot inject unbounded noise.
                        std = langevin_noise_std(denom, group["demographic_temperature"], step_size)
                        clamp = group["demographic_precond_clamp"]
                        if clamp and clamp > 0:
                            std = std.clamp_max((2.0 * group["demographic_temperature"]
                                                 * step_size * clamp) ** 0.5)
                        gen = self._device_generator(group["demographic_generator"], update_rot.device)
                        z = torch.randn(update_rot.shape, generator=gen,
                                        device=update_rot.device, dtype=update_rot.dtype)
                        noise_rot = std * z
                        demo_noise_param = self._project_back(noise_rot, qL, qR)
                        demo_noise_param = demo_noise_param if two_d else demo_noise_param.squeeze(1)

                    update = self._project_back(update_rot, qL, qR)
                else:
                    # Standard AdamW inner loop (always whitening power for 1D fallback).
                    # Demographic noise is intentionally not injected on the 1D/AdamW path
                    # (the FDT story is about the 2D Kronecker layers).
                    denom = eas.sqrt().add_(group["eps"])
                    update = ea / denom
                    step_size = group["lr"]
                    if group["correct_bias"]:
                        bc1 = 1.0 - beta1 ** state["step"]
                        bc2 = 1.0 - beta2 ** state["step"]
                        step_size = step_size * (bc2 ** 0.5) / bc1

                update = update if two_d else update.squeeze(1)

                if p_clip and p_clip > 0:
                    un = update.norm()
                    if un > p_clip:
                        update = update * (p_clip / (un + 1e-12))

                p.add_(update, alpha=-step_size)
                if demo_noise_param is not None:
                    p.add_(demo_noise_param)
                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=-group["lr"] * group["weight_decay"])

                # --- preconditioner / basis update (slow timescale) ----------
                if state["use_precond"]:
                    precond_grad = getattr(p, "_soap_precond_grad", None)
                    stat_g = precond_grad if precond_grad is not None else grad
                    stat_g = stat_g if stat_g.dim() == 2 else stat_g.unsqueeze(1)
                    self._accumulate_stats(stat_g, group, state)
                    if state["step"] % group["precondition_frequency"] == 0:
                        # Rotate the first moment from the old basis into the new one
                        # so momentum stays consistent across a basis change.
                        ea_orig = self._project_back(ea, qL, qR)
                        self._refresh_basis(group, state)
                        ea_new = self._project(ea_orig, state.get("QL"), state.get("QR"))
                        if two_d:
                            state["exp_avg"] = ea_new
                        else:
                            state["exp_avg"] = ea_new.squeeze(1)
                        # exp_avg_sq is elementwise in the old basis; it is left to be
                        # re-learned by the EMA over the next ~1/(1-beta2) steps. Set
                        # reset below if you prefer to zero it on each refresh.
                        # state["exp_avg_sq"].zero_()

        return loss

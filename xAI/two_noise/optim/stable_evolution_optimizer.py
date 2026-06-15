"""
StableEvolutionSOAP — a dynamically-stable, evolution-like preconditioned optimizer.

This is a minimal, deliberate modification of `soap_full_power.py` (SOAPFullPower /
SOAP-NG). It keeps SOAP's structure exactly — run Adam in the eigenbasis of the
Kronecker-factored curvature estimate (Q_L, Q_R of L = E[GGᵀ], R = E[GᵀG]) — and
changes only how the curvature eigenvalues are turned into a preconditioner. The
design follows two principles that make biological evolution / CMA-ES dynamically
stable, where a fixed full-inverse (precond_power = 1) is not:

  (1) GENERATE, don't INVERT.  SOAP(-NG) forms the preconditioner by a direct power
      of the second moment, denom = v̂**power, i.e. it *inverts* a noisy curvature
      estimate. Here the per-coordinate preconditioner eigenvalue P is instead
      *generated* by a multiplicative (geometric / log-space) mutation–selection
      recursion toward its target,
            P  <-  P * (P_target / P) ** kappa ,   P_target = (v̂ + damping)**(-alpha),
      which stays strictly positive (on the SPD manifold) for any noisy v̂ and never
      divides by a small noisy curvature value. This is the discrete, diagonal-in-
      eigenbasis form of the breeder/Riccati flow  Ġ = M − G C G  whose fixed point is
      the matrix geometric mean A^{-1} # M; it is bounded under heavy curvature noise
      where the direct inverse has a divergent tail (verified in the NumPy reference).

  (2) SELECTION sets the per-coordinate exponent.  Rather than one global power, each
      rotated coordinate gets alpha_ij in [1/2, alpha_max], driven by the gradient
      *signal fraction*
            shrink_ij = m̂_ij² / (v̂_ij + eps)   ∈ [0, 1]
      (the consistent/total-energy ratio; ~1 when the accumulated gradient is reliable,
      ~0 when noise-dominated), via  alpha = 1/2 + 1/2 * shrink.  Coordinates with clean,
      consistent signal lean toward Newton (alpha → 1, fast); noise-dominated coordinates
      stay at whitening (alpha = 1/2 — the stationary-variance stability boundary, where
      update-noise amplification a^{1-2α} is flat). shrink rises with batch size (less
      per-step gradient noise → m̂² closer to v̂), so the *operative exponent increases
      with batch automatically* — the framework's cross-substrate prediction, for free.
      Near a minimum the gradient becomes noise (shrink → 0), so the optimizer retreats
      to whitening and converges; it cannot get stuck amplifying flat-direction noise.

  (3) BOUNDED RESPONSE.  A spectral floor on the curvature and an optional global
      update-norm clip cap the per-step change — the analog of evolution's bounded
      per-generation response Δz̄ = G β. On by default.

Relation to the framework's two framings:
  (i)  Working optimizer (this default): selection-weighted, sits at/near whitening,
       leans to Newton only where signal is clean and bounded below alpha_max < 1.
  (ii) Biological recapitulation: set selection_off=True and alpha_max=1.0 with the
       generative tracking — the *unweighted* accumulation drifts the operative exponent
       toward 1 (preconditioner → A^{-1}), reproducing the evolvability/exploration
       optimum that the evolution arm and CMA-ES reach. That is the divergent target for
       an optimizer, which is exactly why selection (framing i) is what makes it work.

All inner-loop logic mirrors the verified NumPy reference `stable_evolution_reference.py`.
Smoke-test in your env before a real run; the surrounding SOAP machinery is unchanged
from soap_full_power.py.
"""

from typing import Iterable, Tuple

import torch
from torch import Tensor

from optim.demographic_noise import langevin_noise_std


def _eigh_basis(mat: Tensor) -> Tensor:
    ridge = 1e-30 * torch.eye(mat.shape[0], device=mat.device, dtype=mat.dtype)
    try:
        _, evecs = torch.linalg.eigh(mat + ridge)
    except Exception:
        evals, evecs = torch.linalg.eigh((mat + ridge).double())
        evecs = evecs.to(mat.dtype)
    return torch.flip(evecs, dims=[1])


def _qr_power_step(mat: Tensor, q_prev: Tensor) -> Tensor:
    q, _ = torch.linalg.qr(mat @ q_prev)
    return q


class StableEvolutionSOAP(torch.optim.Optimizer):
    """SOAP with a selection-driven per-coordinate exponent and a generative preconditioner.

    Args (only the ones new or changed vs SOAPFullPower are commented):
        lr, betas, shampoo_beta, eps, weight_decay, precondition_frequency,
        max_precond_dim, precondition_1d, use_qr_refresh, correct_bias:
            identical to SOAPFullPower.
        alpha_max: ceiling on the per-coordinate exponent (1.0 = allow full Newton in
            clean directions; default 0.9 keeps a margin below the alpha=1 instability).
        alpha_min: floor (0.5 = whitening, the stability boundary). Keep at 0.5.
        kappa: step of the multiplicative generative recursion in (0, 0.5]. 0.5 reaches
            the target in one step (== direct power); smaller smooths/tracks. Default 0.4.
        damping: relative Levenberg–Marquardt term inside the target,
            (v̂ + damping*v̂.amax())**(-alpha). Keeps the target finite in flat directions.
        selection_off: if True, force alpha = alpha_max everywhere (no signal-fraction
            weighting). With alpha_max=1.0 this is the biological-recapitulation mode (ii).
        max_update_norm: global-norm trust region on the final update (<=0 disables).
            Default 1.0 (bounded response on).
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 3e-3,
        betas: Tuple[float, float] = (0.95, 0.95),
        shampoo_beta: float = -1.0,
        alpha_max: float = 0.9,
        alpha_min: float = 0.5,
        kappa: float = 0.4,
        damping: float = 1e-2,
        eps: float = 1e-12,
        weight_decay: float = 0.01,
        precondition_frequency: int = 10,
        max_precond_dim: int = 10000,
        precondition_1d: bool = False,
        use_qr_refresh: bool = True,
        selection_off: bool = False,
        max_update_norm: float = 1.0,
        correct_bias: bool = True,
        demographic_noise: bool = False,
        demographic_temperature: float = 0.0,
        demographic_generator: "torch.Generator | None" = None,
        demographic_warmup: int = 0,
        demographic_precond_clamp: float = 0.0,
        demographic_shape_exp: "float | None" = None,
        demographic_anneal: bool = False,
        demographic_anneal_loss: bool = False,
        demographic_match_grad: bool = False,
        demographic_kappa: float = 1.0,
        demographic_batch: int = 0,
        demographic_match_raw: bool = False,
    ):
        defaults = dict(
            lr=lr, betas=betas, shampoo_beta=shampoo_beta,
            alpha_max=alpha_max, alpha_min=alpha_min, kappa=kappa, damping=damping,
            eps=eps, weight_decay=weight_decay,
            precondition_frequency=precondition_frequency,
            max_precond_dim=max_precond_dim, precondition_1d=precondition_1d,
            use_qr_refresh=use_qr_refresh, selection_off=selection_off,
            max_update_norm=max_update_norm, correct_bias=correct_bias,
            # --- demographic-noise (pSGLD) injection; OFF by default. The applied per-coordinate
            #     preconditioner here is P (the GENERATED eigenvalue), so FDT-correct noise has
            #     covariance ∝ P: std = sqrt(2*T*lr*P) = langevin_noise_std(1/P, T, lr). Restoring
            #     this term turns convergence into posterior SAMPLING (framing ii / FDT). ---
            demographic_noise=demographic_noise,
            demographic_temperature=demographic_temperature,
            demographic_generator=demographic_generator,
            demographic_warmup=demographic_warmup,
            demographic_precond_clamp=demographic_precond_clamp,
            # noise SHAPE knob (experiment B): None => pSGLD/FDT (variance ∝ P). A finite β sets
            #     variance ∝ v̂**β, trace-matched to P so only the SHAPE varies at matched
            #     temperature: β=+1 ~ Fisher/curvature-aligned (like minibatch noise), β=0 isotropic.
            demographic_shape_exp=demographic_shape_exp,
            # self-annealing knob: scale injected std by sqrt(shrink) so noise fades near
            #     convergence (canalization) — separates ANNEALING from shape.
            demographic_anneal=demographic_anneal,
            # loss-scaled self-annealing (v2): scale injected VARIANCE by the current training
            #     loss (fed via set_loss_scale) — the true Sigma ~ (loss)*H law, genuinely
            #     high-early / declining, unlike the already-floored sqrt(shrink). The clean
            #     test of whether annealing (not shape) relieves the drift load.
            demographic_anneal_loss=demographic_anneal_loss,
            # gradient-noise-matched temperature (the principled, tuning-free setting): set the
            #     injected ∝P noise's trace to kappa × the minibatch-gradient-noise trace, using v̂
            #     (≈ per-example grad 2nd moment ≈ Σ) as the estimate and the batch as N_e.
            #     kappa=1 => N_e == sampling population. Auto-scales 1/batch AND self-anneals (v̂->0).
            demographic_match_grad=demographic_match_grad,
            demographic_kappa=demographic_kappa,
            demographic_batch=demographic_batch,
            # match the RAW grad-noise trace (tr v̂ ∝ loss, SELF-ANNEALING) instead of the
            #     preconditioned/whitened one (tr(P²v̂), ~constant/growing as P blows up).
            demographic_match_raw=demographic_match_raw,
        )
        super().__init__(params, defaults)
        self._demo_gen_cache = {}
        self._loss_scale = 1.0   # normalized current loss (1 early -> 0 at interpolation); see set_loss_scale

    @staticmethod
    def _project(g, qL, qR):
        if qL is not None:
            g = qL.t() @ g
        if qR is not None:
            g = g @ qR
        return g

    @staticmethod
    def _project_back(g, qL, qR):
        if qL is not None:
            g = qL @ g
        if qR is not None:
            g = g @ qR.t()
        return g

    def _init_state(self, p, grad, group, state):
        state["step"] = 0
        state["exp_avg"] = torch.zeros_like(grad)
        state["exp_avg_sq"] = torch.zeros_like(grad)
        state["precond"] = torch.ones_like(grad)     # generated preconditioner eigenvalues P
        state["use_precond"] = False
        if grad.dim() == 2:
            m, n = grad.shape
            state["use_precond"] = True
            state["L"] = torch.zeros(m, m, device=grad.device, dtype=grad.dtype) if m <= group["max_precond_dim"] else None
            state["R"] = torch.zeros(n, n, device=grad.device, dtype=grad.dtype) if n <= group["max_precond_dim"] else None
            state["QL"] = None
            state["QR"] = None
        elif grad.dim() == 1 and group["precondition_1d"] and grad.shape[0] <= group["max_precond_dim"]:
            state["use_precond"] = True
            d = grad.shape[0]
            state["L"] = torch.zeros(d, d, device=grad.device, dtype=grad.dtype)
            state["R"] = None
            state["QL"] = None
            state["QR"] = None

    def _accumulate_stats(self, grad2d, group, state):
        beta = group["shampoo_beta"] if group["shampoo_beta"] >= 0 else group["betas"][1]
        g = grad2d
        if state.get("L") is not None:
            state["L"].lerp_(g @ g.t(), 1.0 - beta)
        if state.get("R") is not None:
            state["R"].lerp_(g.t() @ g, 1.0 - beta)

    def _refresh_basis(self, group, state):
        use_qr = group["use_qr_refresh"] and state["QL"] is not None
        if state.get("L") is not None:
            state["QL"] = _qr_power_step(state["L"], state["QL"]) if (use_qr and state["QL"] is not None) else _eigh_basis(state["L"])
        if state.get("R") is not None:
            state["QR"] = _qr_power_step(state["R"], state["QR"]) if (use_qr and state["QR"] is not None) else _eigh_basis(state["R"])

    def _device_generator(self, user_gen, device):
        """Return a generator on `device`, lazily seeding a device-matched one from a CPU generator
        (so seeding is honored and CUDA params don't crash on a CPU generator)."""
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

    def exponent_stats(self):
        """Distributional summary of the realized per-coordinate exponent across all preconditioned
        2D params, computed at READ time (a single host sync). The MEAN alone hides that selection
        produces a SPREAD — noise-dominated directions sit at ~1/2 while clean directions lean
        toward Newton — so we also report the spread and the tails:

          mean       : size-weighted mean exponent
          std        : spread of exponents across coordinates
          max        : the most-Newton direction (the strongest exploit; <= alpha_max)
          frac_high  : fraction strongly leaning to Newton (normalized lean > 0.8 within
                       [alpha_min, alpha_max]) — "how much is being exploited"
          frac_floor : fraction pinned near whitening (normalized lean < 0.1) — "how much is
                       held at the stable boundary"

        All accumulators stay on-device; one stacked .tolist() does the only host sync."""
        n_tot = 0
        ssum = ssq = smax = nhi = nlo = None
        for group in self.param_groups:
            a_min, a_max = group["alpha_min"], group["alpha_max"]
            span = max(a_max - a_min, 1e-9)
            for p in group["params"]:
                al = self.state.get(p, {}).get("alpha_last")
                if al is None:
                    continue
                n_tot += al.numel()
                lean = (al - a_min) / span
                s_, q_, m_ = al.sum(), (al * al).sum(), al.max()
                hi, lo = (lean > 0.8).sum(), (lean < 0.1).sum()
                ssum = s_ if ssum is None else ssum + s_
                ssq = q_ if ssq is None else ssq + q_
                smax = m_ if smax is None else torch.maximum(smax, m_)
                nhi = hi if nhi is None else nhi + hi
                nlo = lo if nlo is None else nlo + lo
        if n_tot == 0 or ssum is None:
            return dict(mean=float("nan"), std=float("nan"), max=float("nan"),
                        frac_high=float("nan"), frac_floor=float("nan"))
        mean = ssum / n_tot
        var = (ssq / n_tot - mean * mean).clamp_min(0.0)
        vals = torch.stack([mean, var.sqrt(), smax.to(mean.dtype),
                            nhi.to(mean.dtype) / n_tot, nlo.to(mean.dtype) / n_tot]).tolist()
        return dict(mean=vals[0], std=vals[1], max=vals[2], frac_high=vals[3], frac_floor=vals[4])

    def mean_exponent(self):
        """Size-weighted mean realized exponent (NaN before the first preconditioned step)."""
        return self.exponent_stats()["mean"]

    def demo_trace(self):
        """Total injected demographic-noise variance (Σ varᵢ) from the last step — the actual
        AMOUNT of noise (vs loss_tax, which is its curvature-weighted loss cost)."""
        return float(getattr(self, "_demo_trace", 0.0))

    def demo_T(self):
        """The demographic temperature actually used last step (matched value if match_grad)."""
        return float(getattr(self, "_demo_T", 0.0))

    def loss_tax(self):
        """Diagonal loss-tax ½·tr(H·Σ_noise) ≈ ½Σ v̂·variance from the LAST step's injection
        (0 when not injecting). The rate the injected noise raises the loss — the transient
        observable whose equilibrium integral is trace-only but whose path is shape-dependent."""
        return float(getattr(self, "_loss_tax", 0.0))

    def set_loss_scale(self, s):
        """Feed the optimizer the current normalized training loss (≈ loss/loss_ref, ~1 early and
        →0 toward interpolation). Used only by the loss-scaled annealing variant
        (demographic_anneal_loss): the injected VARIANCE is multiplied by this, reproducing the
        Σ ~ (loss)·H self-annealing of minibatch noise. No-op otherwise."""
        self._loss_scale = float(s)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        self._loss_tax = 0.0      # diagonal noise loss-tax this step (0 unless injecting)
        self._demo_trace = 0.0    # TOTAL injected noise variance this step (Σ varᵢ — the real amount)
        self._demo_T = 0.0        # the demographic temperature actually used (matched or fixed)

        for group in self.param_groups:
            p_clip = group["max_update_norm"]
            a_min, a_max = group["alpha_min"], group["alpha_max"]
            kappa, damping, eps = group["kappa"], group["damping"], group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                if "step" not in state:
                    self._init_state(p, grad, group, state)
                    if state["use_precond"]:
                        g2d = grad if grad.dim() == 2 else grad.unsqueeze(1)
                        self._accumulate_stats(g2d, group, state)
                        self._refresh_basis(group, state)
                    continue

                state["step"] += 1
                beta1, beta2 = group["betas"]
                qL, qR = state.get("QL"), state.get("QR")
                two_d = grad.dim() == 2
                g2d = grad if two_d else grad.unsqueeze(1)

                g_rot = self._project(g2d, qL, qR) if state["use_precond"] else g2d

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                ea = exp_avg if two_d else exp_avg.unsqueeze(1)
                eas = exp_avg_sq if two_d else exp_avg_sq.unsqueeze(1)
                ea.mul_(beta1).add_(g_rot, alpha=1.0 - beta1)
                eas.mul_(beta2).add_(g_rot.square(), alpha=1.0 - beta2)

                demo_noise_param = None
                if state["use_precond"]:
                    if group["correct_bias"]:
                        v_hat = eas / (1.0 - beta2 ** state["step"])
                        m_hat = ea / (1.0 - beta1 ** state["step"])
                    else:
                        v_hat, m_hat = eas, ea

                    # (2) SELECTION: per-coordinate exponent from the gradient signal fraction.
                    #     shrink (signal fraction m̂²/v̂ ∈[0,1]) is always computed — it drives the
                    #     exponent AND the optional self-annealing injection (canalization).
                    shrink = (m_hat.square() / (v_hat + eps)).clamp_(0.0, 1.0)
                    if group["selection_off"]:
                        alpha = torch.full_like(v_hat, a_max)
                    else:
                        alpha = a_min + (a_max - a_min) * shrink      # in [alpha_min, alpha_max]

                    # surface the realized per-coordinate exponent (the optimizer's OWN-terms
                    # operative alpha) for cross-substrate logging. Store the FULL per-coordinate
                    # tensor as a detached reference (no compute, no host sync per step) — the
                    # distributional reductions happen only when exponent_stats()/mean_exponent()
                    # are read (every log_every steps), so step time stays competitive with SOAP.
                    state["alpha_last"] = alpha.detach()

                    # (3) BOUNDED RESPONSE: relative spectral floor / LM damping in the target.
                    v_damp = v_hat + damping * v_hat.amax().clamp_min(eps)

                    # (1) GENERATE: multiplicative Riccati toward P_target = v_damp**(-alpha).
                    #     P stays > 0; never divides by a small noisy v directly.
                    P = state["precond"]
                    P = P if two_d else P.unsqueeze(1)
                    P_target = v_damp.pow(-alpha)
                    ratio = (P_target / P.clamp_min(eps)).clamp_(1e-6, 1e6)
                    P = P * ratio.pow(kappa)
                    P = P.clamp_(1e-12, 1e12)
                    state["precond"] = P if two_d else P.squeeze(1)

                    update_rot = m_hat * P
                    step_size = group["lr"]

                    # demographic-noise (pSGLD) injection in the eigenbasis. Applied preconditioner
                    # H_coord = P, so FDT-correct param noise ~ N(0, 2*T*lr*P); std =
                    # langevin_noise_std(1/P, T, lr). Turns convergence into posterior sampling.
                    inject = (group["demographic_noise"]
                              and (group["demographic_temperature"] > 0.0
                                   or group["demographic_match_grad"])
                              and state["step"] > group["demographic_warmup"])
                    if inject:
                        T = group["demographic_temperature"]
                        if group["demographic_match_grad"]:
                            # GRADIENT-NOISE-MATCHED T (kappa=1 => N_e = batch). Match the injected
                            # (∝P) noise TRACE to the per-step minibatch-gradient-noise trace in
                            # update space: grad-noise update var_i ≈ lr²·P_i²·Σ_ii/B, Σ≈v̂. Setting
                            # the ∝P injection's trace equal gives, per parameter,
                            #   T = kappa·lr·tr(P²v̂)/(2·B·tr(P)).
                            # FDT shape (∝P) is kept; only the magnitude tracks the gradient noise.
                            B = max(1, group["demographic_batch"])
                            trP = P.sum().clamp_min(eps)
                            # numerator = the grad-noise trace to match. RAW (tr v̂ ∝ loss) self-
                            # anneals; preconditioned (tr P²v̂ ≈ const, whitened) does not.
                            num = (v_hat.sum() if group["demographic_match_raw"]
                                   else (P.square() * v_hat).sum())
                            T = float(group["demographic_kappa"] * step_size * num / (2.0 * B * trP))
                        # noise SHAPE (experiment B): per-coordinate variance ∝ `shape`.
                        #   beta is None  -> shape = P  (pSGLD/FDT; std = sqrt(2*T*lr*P), the
                        #                    langevin_noise_std(1/P,...) form — byte-identical).
                        #   beta finite   -> shape = v_damp**beta, renormalized to P's per-tensor
                        #                    mean so the TRACE (total injected temperature) matches
                        #                    and only the SHAPE differs. beta=+1 ~ Fisher/curvature-
                        #                    aligned (like minibatch noise); beta=0 isotropic.
                        beta = group["demographic_shape_exp"]
                        if beta is None:
                            std = langevin_noise_std(P.reciprocal(), T, step_size)
                        else:
                            shape = v_damp.pow(beta)
                            shape = shape * (P.mean() / shape.mean().clamp_min(eps))
                            std = (2.0 * T * step_size * shape).sqrt()
                        clamp = group["demographic_precond_clamp"]
                        if clamp and clamp > 0:
                            std = std.clamp_max((2.0 * T * step_size * clamp) ** 0.5)
                        if group["demographic_anneal"]:
                            # SELF-ANNEALING / canalization: scale the injected std by sqrt(shrink)
                            # so the noise FADES where/when the gradient becomes noise-dominated
                            # (shrink->0 near convergence) — mimicking minibatch noise vanishing
                            # toward interpolation. Tests whether ANNEALING, not shape, is what makes
                            # noise benign. (Not trace-matched to pSGLD by design — annealing is a
                            # deliberate reduction of the late-stage noise.)
                            std = std * shrink.sqrt()
                        if group["demographic_anneal_loss"]:
                            # LOSS-SCALED self-annealing (v2): variance *= loss_scale (set per step
                            # by the harness, ~1 early -> 0 at interpolation), i.e. std *= sqrt(loss
                            # _scale). The true Σ ~ (loss)·H law — high early, declining over training,
                            # unlike the already-floored sqrt(shrink). NOT batch-dependent.
                            std = std * (self._loss_scale ** 0.5)
                        # diagonal loss-tax: rate the injected noise raises the loss,
                        # ½ tr(H·Σ_noise) ≈ ½ Σ v_damp·variance (v_damp ~ diagonal curvature in the
                        # eigenbasis). Accumulated across param groups; read via loss_tax().
                        self._loss_tax += 0.5 * float((v_damp * std.square()).sum())
                        self._demo_trace += float(std.square().sum())   # total injected variance (amount)
                        self._demo_T = float(T)
                        gen = self._device_generator(group["demographic_generator"], update_rot.device)
                        z = torch.randn(update_rot.shape, generator=gen,
                                        device=update_rot.device, dtype=update_rot.dtype)
                        demo_noise_param = self._project_back(std * z, qL, qR)
                        demo_noise_param = demo_noise_param if two_d else demo_noise_param.squeeze(1)

                    update = self._project_back(update_rot, qL, qR)
                else:
                    denom = eas.sqrt().add_(eps)
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
                    p.add_(demo_noise_param)           # pSGLD noise (not scaled by step_size)
                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=-group["lr"] * group["weight_decay"])

                if state["use_precond"]:
                    stat_g = grad if grad.dim() == 2 else grad.unsqueeze(1)
                    self._accumulate_stats(stat_g, group, state)
                    if state["step"] % group["precondition_frequency"] == 0:
                        ea_orig = self._project_back(ea, qL, qR)
                        self._refresh_basis(group, state)
                        ea_new = self._project(ea_orig, state.get("QL"), state.get("QR"))
                        state["exp_avg"] = ea_new if two_d else ea_new.squeeze(1)
                        # exp_avg_sq and precond are elementwise in the rotated basis; they
                        # re-track over ~1/(1-beta2) steps after a basis refresh.

        return loss

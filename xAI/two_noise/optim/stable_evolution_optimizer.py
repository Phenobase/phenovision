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
    ):
        defaults = dict(
            lr=lr, betas=betas, shampoo_beta=shampoo_beta,
            alpha_max=alpha_max, alpha_min=alpha_min, kappa=kappa, damping=damping,
            eps=eps, weight_decay=weight_decay,
            precondition_frequency=precondition_frequency,
            max_precond_dim=max_precond_dim, precondition_1d=precondition_1d,
            use_qr_refresh=use_qr_refresh, selection_off=selection_off,
            max_update_norm=max_update_norm, correct_bias=correct_bias,
        )
        super().__init__(params, defaults)

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

    def mean_exponent(self):
        """Size-weighted mean of the last realized per-coordinate exponent across all
        preconditioned params (the optimizer's OWN-terms operative alpha). NaN before the
        first preconditioned step. Used by the cross-substrate alpha*(noise) overlay."""
        tot, w = 0.0, 0.0
        for group in self.param_groups:
            for p in group["params"]:
                st = self.state.get(p, {})
                if "alpha_mean" in st:
                    n = st.get("alpha_numel", 1)
                    tot += st["alpha_mean"] * n
                    w += n
        return tot / w if w > 0 else float("nan")

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None

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

                if state["use_precond"]:
                    if group["correct_bias"]:
                        v_hat = eas / (1.0 - beta2 ** state["step"])
                        m_hat = ea / (1.0 - beta1 ** state["step"])
                    else:
                        v_hat, m_hat = eas, ea

                    # (2) SELECTION: per-coordinate exponent from the gradient signal fraction.
                    if group["selection_off"]:
                        alpha = torch.full_like(v_hat, a_max)
                    else:
                        shrink = (m_hat.square() / (v_hat + eps)).clamp_(0.0, 1.0)
                        alpha = a_min + (a_max - a_min) * shrink      # in [alpha_min, alpha_max]

                    # surface the realized per-coordinate exponent (the optimizer's OWN-terms
                    # operative alpha) for cross-substrate logging; aggregate via mean_exponent().
                    state["alpha_mean"] = float(alpha.mean())
                    state["alpha_numel"] = int(alpha.numel())

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
                    update = self._project_back(update_rot, qL, qR)
                    step_size = group["lr"]
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

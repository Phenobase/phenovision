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
    ):
        defaults = dict(
            lr=lr, betas=betas, shampoo_beta=shampoo_beta,
            precond_power=precond_power, damping=damping,
            relative_damping=relative_damping, eps=eps,
            weight_decay=weight_decay,
            precondition_frequency=precondition_frequency,
            max_precond_dim=max_precond_dim, precondition_1d=precondition_1d,
            use_qr_refresh=use_qr_refresh, max_update_norm=max_update_norm,
            correct_bias=correct_bias,
        )
        super().__init__(params, defaults)

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
                eas.mul_(beta2).add_(g_rot.square(), alpha=1.0 - beta2)

                # --- THE MODIFICATION: tunable exponent + LM damping ---------
                # Vanilla SOAP: denom = eas.sqrt() + eps   (power = 0.5)
                # Full power:   denom = eas + damping        (power = 1.0)
                power = group["precond_power"]
                if state["use_precond"]:
                    if group["correct_bias"]:
                        v_hat = eas / (1.0 - beta2 ** state["step"])
                        m_hat = ea / (1.0 - beta1 ** state["step"])
                    else:
                        v_hat, m_hat = eas, ea
                    denom = v_hat.pow(power) if power != 1.0 else v_hat.clone()
                    if group["relative_damping"]:
                        denom = denom + group["damping"] * denom.amax().clamp_min(group["eps"])
                    else:
                        denom = denom + group["damping"]
                    denom = denom.add_(group["eps"])
                    update_rot = m_hat / denom
                    update = self._project_back(update_rot, qL, qR)
                    step_size = group["lr"]
                else:
                    # Standard AdamW inner loop (always whitening power for 1D fallback).
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

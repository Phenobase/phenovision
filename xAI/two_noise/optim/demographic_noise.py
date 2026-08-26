"""Demographic-noise (preconditioned SGLD) injection — standalone, unit-testable math.

The injection itself lives inside ``SOAPFullPower.step()`` (gated by ``demographic_noise``);
this module factors out the *math* so it can be tested in isolation and reused.

Theory (``notes/two_noise_framework_summary.md`` §9.5). Standard SGD carries only the
gradient-noise term (diffusion covariance ``HCH/S``) and therefore does not sample the
Bayesian posterior. Restoring biology's demographic term means running preconditioned SGLD,

    dθ = -H ∇L dt + sqrt(2 T H) dW,     stationary law  π(θ) ∝ exp(-L(θ)/T),

which is the Gibbs measure for *any* PD preconditioner ``H`` precisely because the ``H`` in the
drift and the ``H`` in the noise covariance match. The noise covariance is ``∝ H`` (shaping
matrix ``∝ H^{1/2}``); injecting covariance ``∝ sqrt(H)`` instead would break the Gibbs form.

In SOAP's eigenbasis the applied preconditioner is diagonal: ``H_coord = 1/denom`` where
``denom = V^α (+ damping + eps)`` is the per-coordinate denominator the deterministic update
divides by. With an Euler–Maruyama step of size ``lr`` the per-coordinate parameter-noise
variance is ``2 * T * lr * H_coord``; injecting it in the rotated basis and projecting back by
the orthogonal eigenbasis ``Q`` gives parameter-space covariance ``Q diag(2 T lr H_coord) Qᵀ =
2 T lr H`` — exactly the SGLD noise.
"""

from __future__ import annotations

import torch
from torch import Tensor


def langevin_noise_variance(denom: Tensor, temperature: float, lr: float) -> Tensor:
    """Per-coordinate parameter-noise variance (in the eigenbasis) for the pSGLD step.

    Args:
        denom: the per-coordinate denominator the deterministic update divides by, i.e.
            ``V^α + damping + eps``. The applied preconditioner is ``H_coord = 1/denom``.
        temperature: SGLD temperature ``T`` (= ``1/(2 N_e^eff)`` in the biological reading).
        lr: optimizer step size (the Euler–Maruyama discretization step).

    Returns:
        Tensor of the same shape as ``denom``: variance ``2 * T * lr / denom`` per coordinate.
    """
    return 2.0 * temperature * lr * denom.reciprocal()


def langevin_noise_std(denom: Tensor, temperature: float, lr: float) -> Tensor:
    """Per-coordinate noise standard deviation = sqrt of :func:`langevin_noise_variance`."""
    return langevin_noise_variance(denom, temperature, lr).sqrt()


def parameter_space_covariance(qL: Tensor | None, qR: Tensor | None,
                               noise_var_rot: Tensor) -> Tensor:
    """Analytic parameter-space covariance of the injected noise, for verification.

    For a 2D parameter the rotated-basis noise has independent per-entry variance
    ``noise_var_rot[i, j]``. Projecting back is ``N_param = qL @ N_rot @ qRᵀ``; the
    covariance of ``vec(N_param)`` is ``(qR ⊗ qL) diag(vec(noise_var_rot)) (qR ⊗ qL)ᵀ``.

    This is used only by tests to confirm that the in-basis injection + project-back yields
    the intended Kronecker-structured covariance (≈ ``2 T lr H`` when ``H`` is the applied
    preconditioner). ``qL``/``qR`` may be ``None`` (identity on that axis).
    """
    m, n = noise_var_rot.shape
    eye_m = torch.eye(m, dtype=noise_var_rot.dtype, device=noise_var_rot.device)
    eye_n = torch.eye(n, dtype=noise_var_rot.dtype, device=noise_var_rot.device)
    QL = eye_m if qL is None else qL
    QR = eye_n if qR is None else qR
    kron = torch.kron(QR, QL)  # (qR ⊗ qL), acts on vec (column-major-ish via our convention)
    cov_rot = torch.diag(noise_var_rot.t().reshape(-1))  # vec stacks columns
    return kron @ cov_rot @ kron.t()

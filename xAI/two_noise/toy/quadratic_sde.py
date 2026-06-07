"""§2.2 / §2.3 — the toy anisotropic quadratic, where the two-noise SDE is exact.

Setup (CONVENTIONS §4). Loss ``L(θ) = ½ θᵀ A θ`` with PSD curvature ``A`` (controlled condition
number). Under the Fisher identity we take the gradient-noise covariance ``C = A``. An optimizer
applying preconditioner power ``α`` (``H = C^{-α}``) gives, near the optimum, the linear SDE

    dθ = -C^{1-α}(θ-θ*) dt + (1/√S) C^{-α} B dW,   BBᵀ = C,

which in A's eigenbasis decouples per coordinate ``i`` (curvature ``a_i``) into an OU process

    dθ_i = -a_i^{1-α} θ_i dt + (1/√S) a_i^{1/2-α} dW_i,

with **stationary variance ``V_stat(a_i) = a_i^{-α} / (2S)``** (framework §7) — so the log-log
slope of ``V_stat`` vs curvature is ``-α``. The optimal exponent ``α*`` has no closed form from
the *stationary* loss alone (it pushes to a boundary); the noise dependence appears in the
**finite-budget** loss, where fast isotropic relaxation at ``α→1`` trades off against
over-amplified flat-direction noise. That trade-off is closed-form for OU and is what
:func:`analytic_optimal_alpha` evaluates.

This module is numpy-based for the exact/analytic parts (fast, transparent) and provides a
torch + SOAPFullPower simulator to connect the *actual optimizer* to the SDE (§2.3).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------- the landscape
@dataclass
class Quadratic:
    eigvals: np.ndarray   # curvature eigenvalues a_i (descending), shape (d,)
    eigvecs: np.ndarray   # eigenvectors (columns), shape (d, d). A = V diag(eigvals) Vᵀ
    A: np.ndarray         # the d×d curvature matrix

    @property
    def dim(self) -> int:
        return self.eigvals.shape[0]


def make_quadratic(dim: int = 20, cond_number: float = 1e3, seed: int = 0,
                   rotate: bool = True, center: bool = True) -> Quadratic:
    """Anisotropic PSD quadratic with log-spaced eigenvalues spanning ``cond_number``.

    ``center=True`` (default) places the eigenvalues *geometrically around 1*, i.e. over
    ``[cond^{-1/2}, cond^{1/2}]``, so the spectrum contains genuinely **flat** directions
    (curvature < 1) as well as sharp ones. This matters for the α* prediction: the noise term
    ``a^{-α}/(2S)`` over-amplifies low-curvature directions only when ``a < 1``. ``center=False``
    spans ``[1, cond_number]`` (all curvatures ≥ 1), where the stationary loss trivially favors
    α=1 and the noise/transient trade-off does not produce an interior α*.

    ``rotate=False`` keeps A diagonal (eigenbasis = standard basis), the exact decoupled regime;
    ``rotate=True`` applies a random orthogonal rotation (general, still commuting C=A).
    """
    rng = np.random.default_rng(seed)
    half = np.log10(cond_number) / 2.0
    if center:
        eigvals = np.logspace(half, -half, dim).copy()        # descending, geo-mean 1
    else:
        eigvals = np.logspace(np.log10(cond_number), 0, dim).copy()  # descending, [1, cond]
    if rotate:
        Q, _ = np.linalg.qr(rng.standard_normal((dim, dim)))
    else:
        Q = np.eye(dim)
    A = (Q * eigvals) @ Q.T
    A = 0.5 * (A + A.T)
    return Quadratic(eigvals=eigvals, eigvecs=Q, A=A)


# --------------------------------------------------------------------------- analytic OU
def analytic_stationary_variance(eigvals: np.ndarray, alpha: float, S: float) -> np.ndarray:
    """Continuous-time per-eigendirection stationary variance ``a^{-α} / (2S)`` (framework §7)."""
    return eigvals ** (-alpha) / (2.0 * S)


def _discrete_ou_moments(eigvals, alpha, S, lr, n_steps, theta0_sq):
    """Exact mean²/variance of the discrete OU step
        θ_{t+1} = (1 - lr a^{1-α}) θ_t + sqrt(lr/S) a^{1/2-α} ξ_t
    after ``n_steps``, per eigendirection. Returns (mean_sq[T], var[T])."""
    k = eigvals ** (1.0 - alpha)                # drift rate
    rho = 1.0 - lr * k                          # per-step contraction
    q = (lr / S) * eigvals ** (1.0 - 2.0 * alpha)  # variance injected per step
    # stationary discrete variance q / (1 - rho^2); guard rho^2 -> 1
    denom = 1.0 - rho ** 2
    v_stat_disc = np.where(np.abs(denom) > 1e-300, q / np.where(denom == 0, 1.0, denom), np.inf)
    rho2T = rho ** (2 * n_steps)
    var_T = v_stat_disc * (1.0 - rho2T)
    mean_sq_T = theta0_sq * rho2T
    return mean_sq_T, var_T


def analytic_expected_loss(quad: Quadratic, alpha: float, S: float, lr: float,
                           n_steps: int, theta0_norm: float = 1.0,
                           test_curvature: str = "isotropic") -> float:
    """Closed-form expected loss at fixed budget for the discrete OU.

    ``E[L(T)] = ½ Σ w_i (E[θ_i]² + Var[θ_i])`` with per-direction weight ``w_i``:

    - ``test_curvature="isotropic"`` (default): ``w_i = 1`` — expected *test/parameter* error
      ``½ E||θ-θ*||²``. Flat training directions (small ``a``) get over-amplified noise
      ``a^{-α}/(2S)`` that α→1 makes worse, so this objective has the interior α* the framework
      predicts (the generalization picture; §2.1/§2.2 measure val loss, not train loss).
    - ``test_curvature="train"``: ``w_i = a_i`` — the *training* loss ``½θᵀAθ``, which on a
      geomean-1 spectrum is trivially minimized at α=1 (no interior optimum); kept for contrast.

    Start ``θ0`` isotropic with squared norm ``theta0_norm²`` spread over coordinates.
    """
    a = quad.eigvals
    w = np.ones_like(a) if test_curvature == "isotropic" else a
    theta0_sq = np.full_like(a, theta0_norm ** 2 / a.shape[0])
    mean_sq_T, var_T = _discrete_ou_moments(a, alpha, S, lr, n_steps, theta0_sq)
    return 0.5 * float(np.sum(w * (mean_sq_T + var_T)))


def analytic_optimal_alpha(quad: Quadratic, S, n_steps: int = 2000,
                           alpha_grid: np.ndarray = None, theta0_norm: float = 1.0,
                           test_curvature: str = "isotropic", n_lr: int = 24):
    """Locate ``α*`` minimizing finite-budget expected (test) loss, per noise level ``S``.

    For each (α, S) the learning rate is optimized over a stability-bounded grid
    (``lr < 1.9 / max_i a_i^{1-α}``): α=1 admits larger stable steps (Newton-like), but at high
    noise those steps inject more flat-direction noise — this is the speed/noise trade-off that
    produces a noise-dependent α*. Returns (alpha_star, loss_grid) where ``loss_grid`` is the
    best-lr loss of shape (len(S), len(alpha_grid)).
    """
    if alpha_grid is None:
        alpha_grid = np.linspace(0.0, 1.0, 41)
    S_arr = np.atleast_1d(np.asarray(S, dtype=float))
    loss_grid = np.empty((S_arr.size, alpha_grid.size))
    a = quad.eigvals
    for j, al in enumerate(alpha_grid):
        lr_max = 1.9 / float((a ** (1.0 - al)).max())
        lr_grid = np.linspace(lr_max / n_lr, lr_max, n_lr)
        for i, s in enumerate(S_arr):
            best = np.inf
            for lr in lr_grid:
                L = analytic_expected_loss(quad, al, s, lr, n_steps, theta0_norm, test_curvature)
                if L < best:
                    best = L
            loss_grid[i, j] = best
    star = alpha_grid[np.argmin(loss_grid, axis=1)]
    if np.isscalar(S) or np.ndim(S) == 0:
        return float(star[0]), loss_grid[0]
    return star, loss_grid


# --------------------------------------------------------------- exact diagonal SDE simulator
def simulate_exact_sde(quad: Quadratic, alpha: float, S: float, lr: float,
                       n_steps: int, burn_in: int, seed: int = 0, n_chains: int = 256):
    """Euler–Maruyama on the decoupled OU (in A's eigenbasis), many parallel chains.

    Returns the empirical stationary variance per eigendirection (averaged over chains and
    post-burn-in steps). This is the 'exact regime' reference for the analytic formula.
    """
    rng = np.random.default_rng(seed)
    a = quad.eigvals
    d = a.shape[0]
    k = a ** (1.0 - alpha)
    noise_std = np.sqrt(lr / S) * a ** (0.5 - alpha)
    theta = rng.standard_normal((n_chains, d))  # start dispersed
    sq_acc = np.zeros(d)
    mean_acc = np.zeros(d)
    count = 0
    for t in range(n_steps):
        theta = theta * (1.0 - lr * k) + noise_std * rng.standard_normal((n_chains, d))
        if t >= burn_in:
            sq_acc += np.sum(theta ** 2, axis=0)
            mean_acc += np.sum(theta, axis=0)
            count += n_chains
    mean = mean_acc / count
    var = sq_acc / count - mean ** 2
    return var


def fit_loglog_slope(curvature: np.ndarray, vstat: np.ndarray):
    """Least-squares slope of log(vstat) on log(curvature). Returns (slope, intercept)."""
    x = np.log(curvature)
    y = np.log(vstat)
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(coef[0]), float(coef[1])

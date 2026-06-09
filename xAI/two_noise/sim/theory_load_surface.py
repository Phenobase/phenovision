"""
theory_load_surface.py
======================
Analytic objects of the (z̄, G, M) SDE theory (sde_validation_plan.md Part 0).
Pure numpy so it imports in either conda env (two_noise / two_noise_jax) and is
trivially unit-testable. This is the SINGLE SOURCE OF TRUTH for:
  * V1's prediction surface  ℒ_eff(M)  and its gradient,
  * V6's coupled-SDE drift   (Ĝ, Σ_∞, the M-flow),
  * the natural-gradient form used by the design-(ii) metric-control test.

Symbols (d×d SPD, d=2 suffices; CONVENTIONS.md):
  A   curvature of log-fitness (stabilizing selection); larger eig = steeper.
  M   mutational input covariance (the slow evolving variable).
  G   additive genetic covariance; conditional equilibrium Ĝ(M) = A⁻¹ # M.
  N*  effective gradient-sampling size;  v/n ~ 1/N_e demographic coefficient.

vech convention (UPPER triangle, row-major):  for d=2  vech(M) = [m00, m01, m11].
unvech rebuilds the symmetric matrix (m01 placed at both [0,1] and [1,0]).

THE FACTOR-OF-2 (documented to avoid the classic bug): a vech gradient g_vech has
ONE entry per off-diagonal pair, but the symmetric matrix derivative ∇_M L carries
that derivative on BOTH [i,j] and [j,i]. So when converting g_vech → ∇_M (matrix),
the OFF-DIAGONAL entries are HALVED:  (∇_M L)[i,j] = (∇_M L)[j,i] = g_vech_ij / 2,
while diagonal entries are copied as-is. `grad_matrix_from_vech` does exactly this,
and `natural_gradient_M` relies on it.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# SPD helpers
# --------------------------------------------------------------------------- #

def _spd_sqrt(X):
    w, V = np.linalg.eigh(0.5 * (X + X.T))
    w = np.clip(w, 1e-300, None)
    return (V * np.sqrt(w)) @ V.T


def _spd_inv_sqrt(X):
    w, V = np.linalg.eigh(0.5 * (X + X.T))
    w = np.clip(w, 1e-300, None)
    return (V * (1.0 / np.sqrt(w))) @ V.T


def matrix_geometric_mean(X, Y):
    """Geometric mean X # Y = X^{1/2} (X^{-1/2} Y X^{-1/2})^{1/2} X^{1/2} of two SPD matrices.
    Symmetric in (X, Y). To form Ĝ = A⁻¹ # M, pass X = inv(A)."""
    Xh = _spd_sqrt(X)
    Xih = _spd_inv_sqrt(X)
    mid = _spd_sqrt(Xih @ Y @ Xih)
    return Xh @ mid @ Xh


def ghat(A, M):
    """Conditional equilibrium of G given M: Ĝ = A⁻¹ # M (the MSB solution of GAG=M)."""
    return matrix_geometric_mean(np.linalg.inv(A), M)


def kappa(N_star):
    """Load prefactor κ = 1/2 + 1/(4 N*)  (sde_validation_plan.md §0.7)."""
    return 0.5 + 1.0 / (4.0 * N_star)


def sigma_infinity(A, G, v_over_n, N_star):
    """Stationary covariance of the mean deviation x = z̄ - θ:
        Σ_∞ = (v/2n) A⁻¹ + (1/2N*) G        (C = A near the optimum)."""
    return 0.5 * v_over_n * np.linalg.inv(A) + (1.0 / (2.0 * N_star)) * G


# --------------------------------------------------------------------------- #
# vech / unvech (upper triangle, row-major)
# --------------------------------------------------------------------------- #

def _triu_index(d):
    return [(i, j) for i in range(d) for j in range(i, d)]


def vech(M):
    d = M.shape[0]
    return np.array([M[i, j] for (i, j) in _triu_index(d)], dtype=float)


def unvech(v, d=None):
    v = np.asarray(v, dtype=float)
    if d is None:
        # solve d(d+1)/2 = len(v)
        d = int((np.sqrt(8 * len(v) + 1) - 1) / 2 + 0.5)
    M = np.zeros((d, d))
    for k, (i, j) in enumerate(_triu_index(d)):
        M[i, j] = v[k]
        M[j, i] = v[k]
    return M


# --------------------------------------------------------------------------- #
# Effective load ℒ_eff(M) and the three regimes (§0.7, §0.8)
# --------------------------------------------------------------------------- #
# ℒ_eff(M) = κ·tr(A·Ĝ(M)) - ℬ(M).  We compute the cost κ·tr(A·Ĝ) directly and the
# three regimes via explicit terms that REPRODUCE THE FIXED POINTS (the metric-
# independent predictions V1/V3 must match):
#   static       ℬ = 0                          -> M shrinks to a floor (canalization)
#   tracking     ℬ = -(1/4) tr(Ω Ĝ⁻¹)  (lag load enters ℒ_eff as +cost) -> m_i ∝ ω_i
#   exploration  ℬ = (λ/2) logdet M             -> m_i ∝ 1/a_i  (full inverse-curvature)
# (The tracking "benefit" is the NEGATIVE of the lag load; documented sign quirk.)

def _cost(M, A, N_star):
    return kappa(N_star) * np.trace(A @ ghat(A, M))


def L_eff(M, A, N_star, regime="static", Omega=None, lam=1.0):
    """Effective fitness-load surface the slow M-flow descends. regime ∈
    {static, tracking, exploration}. tracking needs Omega; exploration needs lam."""
    val = _cost(M, A, N_star)
    if regime == "static":
        return val
    if regime == "tracking":
        if Omega is None:
            raise ValueError("tracking regime needs Omega (optimum displacement cov)")
        # lag load (a COST): (1/4) tr(Ω Ĝ⁻¹); aligned = (1/4) Σ ω_i/g_i
        return val + 0.25 * np.trace(Omega @ np.linalg.inv(ghat(A, M)))
    if regime == "exploration":
        sign, logdet = np.linalg.slogdet(M)
        return val - 0.5 * lam * logdet
    raise ValueError(f"unknown regime {regime!r}")


def grad_L_eff_vech(M, A, N_star, regime="static", mode="fd", Omega=None, lam=1.0,
                    h=1e-5):
    """Gradient of ℒ_eff w.r.t. vech(M).
    mode='aligned': closed-form for DIAGONAL (commuting) A, M (off-diag entries 0);
    mode='fd':      central finite differences in vech coords (general, non-commuting),
                    symmetrizing the off-diagonal perturbation."""
    d = M.shape[0]
    if mode == "aligned":
        a = np.diag(A).astype(float)
        m = np.diag(M).astype(float)
        k = kappa(N_star)
        # cost term ∂/∂m_i [κ √(a_i m_i)] = (κ/2)√(a_i/m_i)
        dcost = 0.5 * k * np.sqrt(a / m)
        if regime == "static":
            dm = dcost
        elif regime == "tracking":
            om = np.diag(Omega).astype(float)
            # lag load (1/4)Σ ω_i √(a_i/m_i): ∂/∂m_i = -(1/8) ω_i √a_i m_i^{-3/2}
            dm = dcost - 0.125 * om * np.sqrt(a) * m ** (-1.5)
        elif regime == "exploration":
            dm = dcost - 0.5 * lam / m
        else:
            raise ValueError(f"unknown regime {regime!r}")
        # assemble into vech (diagonal entries only; off-diagonals 0 at a diagonal M)
        g = np.zeros(len(_triu_index(d)))
        for k_idx, (i, j) in enumerate(_triu_index(d)):
            if i == j:
                g[k_idx] = dm[i]
        return g
    elif mode == "fd":
        v0 = vech(M)
        g = np.zeros_like(v0)
        for k_idx in range(len(v0)):
            vp = v0.copy(); vp[k_idx] += h
            vm = v0.copy(); vm[k_idx] -= h
            g[k_idx] = (L_eff(unvech(vp, d), A, N_star, regime, Omega, lam)
                        - L_eff(unvech(vm, d), A, N_star, regime, Omega, lam)) / (2 * h)
        return g
    raise ValueError(f"unknown mode {mode!r}")


def grad_matrix_from_vech(g_vech, d=None):
    """Convert a vech gradient to the symmetric matrix gradient ∇_M L.
    OFF-DIAGONAL entries are HALVED (the factor-of-2; see module docstring)."""
    if d is None:
        d = int((np.sqrt(8 * len(g_vech) + 1) - 1) / 2 + 0.5)
    G = np.zeros((d, d))
    for k, (i, j) in enumerate(_triu_index(d)):
        if i == j:
            G[i, i] = g_vech[k]
        else:
            G[i, j] = G[j, i] = 0.5 * g_vech[k]
    return G


def natural_gradient_M(M, A, N_star, regime="static", Omega=None, lam=1.0, h=1e-5):
    """Natural-gradient flow of M on the Gaussian manifold N(0,M):
        Ṁ = -2 M (∇_M ℒ_eff) M
    This is what the design-(ii) metric-control IBM (𝒢_M ∝ Fisher) should match.
    Returns the symmetric matrix Ṁ direction."""
    g_vech = grad_L_eff_vech(M, A, N_star, regime, mode="fd", Omega=Omega, lam=lam, h=h)
    gradM = grad_matrix_from_vech(g_vech, M.shape[0])
    return -2.0 * M @ gradM @ M


def fisher_metric_vech(M):
    """Fisher metric of N(0,M) in vech coords: g = (1/2) D^T (M⁻¹ ⊗ M⁻¹) D, where D is
    the duplication matrix (vech -> vec). natural grad = -g⁻¹ grad_vech equals
    vech(-2 M ∇_M L M). Provided for the reduction test."""
    d = M.shape[0]
    Minv = np.linalg.inv(M)
    idx = _triu_index(d)
    p = len(idx)
    # duplication matrix D: vec(M) (d*d) = D @ vech(M) (p)
    D = np.zeros((d * d, p))
    for col, (i, j) in enumerate(idx):
        D[i * d + j, col] = 1.0
        if i != j:
            D[j * d + i, col] = 1.0
    kron = np.kron(Minv, Minv)
    return 0.5 * D.T @ kron @ D

"""
drift_estimator.py — V0 re-established (substrate-agnostic ensemble-drift harness).

Re-creates the lost `drift_estimator_prototype.py` as a reusable module. The drift SOURCE is
injected as a callable `increment(M, rng) -> M_next`, so the SAME harness drives:
  (a) the Gaussian-closure surrogate now (where the true drift -𝒢_M∇ℒ_eff is known by
      construction), validating the ESTIMATOR (not the theory), and
  (b) the evolvable-M IBM later (V3/V4) — a drop-in swap of `increment`.

What V0 established (and these tests re-establish):
  * the ensemble drift estimator recovers the drift DIRECTION to cos>0.999 at R≈200;
  * the metric correction is necessary (bare -∇ℒ_eff gives ~0.70; -𝒢_M∇ℒ_eff gives ~0.999);
  * the eigenvalue-scaling / eigenvector-rotation split is sound;
  * the Helmholtz curl detector reads ~0 for a pure-gradient field and jumps for rotation.

Design lessons baked into defaults (V0 'magnitude trap'): keep eta_M·tau·|b| small (so the
population does not travel far during the window) and tau above the fast correlation time.
"""
from __future__ import annotations

import numpy as np

from sim.theory_load_surface import (
    grad_L_eff_vech, vech, unvech, natural_gradient_M,
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def cos_direction(u, v):
    u = np.ravel(np.asarray(u, dtype=float)); v = np.ravel(np.asarray(v, dtype=float))
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-300 or nv < 1e-300:
        return 0.0
    return float(u @ v / (nu * nv))


def _project_spd(M, floor=1e-9):
    w, V = np.linalg.eigh(0.5 * (M + M.T))
    w = np.clip(w, floor, None)
    return (V * w) @ V.T


# --------------------------------------------------------------------------- #
# the ensemble drift estimator (V3/V4 core)
# --------------------------------------------------------------------------- #

def ensemble_drift(increment, M0, R, tau, seed=0, eta_M=1.0):
    """Clone M0 across R replicates, advance each tau slow-steps, estimate the conditional
    drift  Ê[ΔM|M] = (1/R) Σ_r (M_r(τ) - M0) / (τ·eta_M).

    Returns dict(matrix, vech, reps): the drift as a symmetric matrix, its vech, and the
    per-replicate (M_r(τ)-M0)/(τ·eta_M) endpoints (for CIs and the scaling/rotation split)."""
    d = M0.shape[0]
    ss = np.random.SeedSequence(seed)
    rngs = [np.random.default_rng(s) for s in ss.spawn(R)]
    deltas = np.zeros((R, d, d))
    for i, rng in enumerate(rngs):
        M = M0.copy()
        for _ in range(tau):
            M = increment(M, rng)
        deltas[i] = (M - M0) / (tau * eta_M)
    bhat = deltas.mean(0)
    bhat = 0.5 * (bhat + bhat.T)
    return dict(matrix=bhat, vech=vech(bhat), reps=deltas)


def estimate_G_M(modifier_breeding_values):
    """Metric 𝒢_M = sample covariance of modifier breeding values (in their modifier
    coordinate, e.g. vech(M) or log-eigenvalue space). Shape (p, p)."""
    bv = np.asarray(modifier_breeding_values, dtype=float)
    return np.cov(bv, rowvar=False)


def split_scaling_rotation(M0, bhat_matrix):
    """Decompose a drift matrix in M0's eigenbasis into the eigenvalue-SCALING part (diagonal,
    the canalization knob) and the eigenvector-ROTATION part (off-diagonal, the alignment knob).
    Returns the two components (back in M-space) and their Frobenius fractions of the total."""
    w, V = np.linalg.eigh(0.5 * (M0 + M0.T))
    B = V.T @ bhat_matrix @ V
    diag = np.diag(np.diag(B))
    off = B - diag
    tot = np.linalg.norm(B) + 1e-300
    return dict(
        scaling=V @ diag @ V.T, rotation=V @ off @ V.T,
        scaling_frac=float(np.linalg.norm(diag) / tot),
        rotation_frac=float(np.linalg.norm(off) / tot),
    )


# --------------------------------------------------------------------------- #
# the Gaussian-closure surrogate (self-validates V0 without an IBM)
# --------------------------------------------------------------------------- #

def gaussian_surrogate_increment(A, N_star, regime, G_M, eta_M=1.0, dt=1.0,
                                 noise=0.0, curl_strength=0.0, **bargs):
    """Build an increment(M, rng) whose TRUE drift is exactly -𝒢_M ∇_vech ℒ_eff(M) (in vech
    coords), with optional isotropic noise and an optional injected NON-CONSERVATIVE (curl)
    term for testing the curl detector. 𝒢_M is the (p×p) metric in vech coordinates.

    curl_strength>0 adds a rotational field orthogonal to the gradient (a skew-symmetric
    generator applied to vech), which the Helmholtz detector should flag."""
    p = G_M.shape[0]
    d = A.shape[0]
    # a fixed skew-symmetric operator on vech space for the (optional) injected curl
    K = np.zeros((p, p))
    if p >= 2:
        K[0, 1] = 1.0; K[1, 0] = -1.0
    if p >= 3:
        K[1, 2] = 1.0; K[2, 1] = -1.0

    def increment(M, rng):
        g = grad_L_eff_vech(M, A, N_star, regime, mode="fd", **bargs)
        drift = -G_M @ g
        if curl_strength != 0.0:
            drift = drift + curl_strength * (K @ g)
        v = vech(M) + eta_M * dt * drift
        if noise > 0.0:
            v = v + np.sqrt(noise * dt) * rng.standard_normal(p)
        return _project_spd(unvech(v, d))

    return increment


def true_drift_vech(M, A, N_star, regime, G_M, **bargs):
    """The surrogate's analytic true drift in vech coords: -𝒢_M ∇_vech ℒ_eff(M)."""
    g = grad_L_eff_vech(M, A, N_star, regime, mode="fd", **bargs)
    return -G_M @ g


# --------------------------------------------------------------------------- #
# Helmholtz curl detector (V4)
# --------------------------------------------------------------------------- #

def helmholtz_curl_fraction_2d(xs, ys, U, V):
    """Helmholtz curl fraction of a 2-D vector field (U,V) sampled on a regular grid xs×ys.
    Fits a scalar potential φ by least squares so ∇φ ≈ (U,V); returns
        ||(U,V) - ∇φ|| / ||(U,V)||   (the rotational residual fraction; a presence/absence
    detector that saturates — read it as a detector, per V0).

    For the M-drift field, premultiply the raw drift by 𝒢_M⁻¹ BEFORE passing it here (the
    metric-corrected field is the one predicted to be a gradient flow)."""
    nx, ny = len(xs), len(ys)
    dx = xs[1] - xs[0]; dy = ys[1] - ys[0]
    # unknowns: phi at each grid node (flatten C-order: idx = i*ny + j)
    rows, cols, vals, rhs = [], [], [], []
    eq = 0
    for i in range(nx):
        for j in range(ny):
            # central/forward difference for d phi/dx = U
            if i < nx - 1:
                rows += [eq, eq]; cols += [(i + 1) * ny + j, i * ny + j]
                vals += [1.0 / dx, -1.0 / dx]; rhs.append(U[i, j]); eq += 1
            if j < ny - 1:
                rows += [eq, eq]; cols += [i * ny + (j + 1), i * ny + j]
                vals += [1.0 / dy, -1.0 / dy]; rhs.append(V[i, j]); eq += 1
    Amat = np.zeros((eq, nx * ny))
    Amat[rows, cols] = vals
    rhs = np.array(rhs)
    phi, *_ = np.linalg.lstsq(Amat, rhs, rcond=None)
    # curl fraction = the part of the field NOT explained by any potential, measured on the
    # SAME finite-difference stencil used to fit phi (consistent, so a true gradient -> ~0).
    resid = np.linalg.norm(Amat @ phi - rhs)
    tot = np.linalg.norm(rhs) + 1e-300
    return float(resid / tot)

"""GATE 0 — analytic objects of the (z̄,G,M) SDE theory (sim/theory_load_surface.py).
Pure numpy; exact/near-exact tolerances. These identities de-risk V1/V3/V6 downstream."""
import numpy as np
import pytest

from sim.theory_load_surface import (
    ghat, sigma_infinity, kappa, L_eff, grad_L_eff_vech, natural_gradient_M,
    vech, unvech, fisher_metric_vech, grad_matrix_from_vech,
)


def _rand_spd(d, seed, cond=5.0):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    eigs = np.linspace(1.0, cond, d)
    return (Q * eigs) @ Q.T


def test_ghat_msb_identity_commuting_and_noncommuting():
    # Ĝ A Ĝ = M (mutation-selection balance), for aligned and misaligned A, M.
    A_diag = np.diag([1.0, 4.0])
    M_diag = np.diag([0.2, 0.05])
    G = ghat(A_diag, M_diag)
    assert np.allclose(G @ A_diag @ G, M_diag, atol=1e-10)
    # non-commuting
    A = _rand_spd(2, 1, cond=6.0)
    M = _rand_spd(2, 2, cond=4.0) * 0.1
    G = ghat(A, M)
    assert np.allclose(G @ A @ G, M, atol=1e-9)


def test_ghat_aligned_closed_form():
    a = np.array([1.0, 4.0, 9.0]); m = np.array([0.3, 0.1, 0.05])
    G = ghat(np.diag(a), np.diag(m))
    assert np.allclose(np.diag(G), np.sqrt(m / a), atol=1e-10)
    # off-diagonals ~ 0 for aligned inputs
    assert np.allclose(G - np.diag(np.diag(G)), 0.0, atol=1e-10)


def test_sigma_infinity_solves_lyapunov():
    # (GA) Σ + Σ (AG) = (v/n) G + GAG/N*   (C = A); Σ_∞ = (v/2n)A⁻¹ + (1/2N*)G
    A = _rand_spd(2, 3, cond=7.0)
    G = _rand_spd(2, 4, cond=3.0) * 0.5
    v_over_n, N_star = 0.02, 50.0
    S = sigma_infinity(A, G, v_over_n, N_star)
    GA = G @ A
    lhs = GA @ S + S @ GA.T
    rhs = v_over_n * G + (G @ A @ G) / N_star
    assert np.allclose(lhs, rhs, atol=1e-10)


def test_aligned_grad_matches_fd():
    A = np.diag([1.0, 5.0]); M = np.diag([0.25, 0.04]); N_star = 40.0
    Omega = np.diag([0.3, 0.1])
    for regime, kw in [("static", {}), ("tracking", {"Omega": Omega}),
                       ("exploration", {"lam": 0.5})]:
        ga = grad_L_eff_vech(M, A, N_star, regime, mode="aligned", **kw)
        gf = grad_L_eff_vech(M, A, N_star, regime, mode="fd", **kw)
        assert np.allclose(ga, gf, atol=1e-5), f"{regime}: {ga} vs {gf}"


def test_regime_fixed_points():
    A = np.diag([1.0, 4.0]); N_star = 30.0; k = kappa(N_star)
    # exploration: m_i* = (lam/kappa)^2 / a_i  -> grad ~ 0, and m_i a_i = const
    lam = 0.7
    a = np.diag(A)
    m_star = (lam / k) ** 2 / a
    g = grad_L_eff_vech(np.diag(m_star), A, N_star, "exploration", mode="fd", lam=lam)
    assert np.allclose(g, 0.0, atol=1e-5)
    assert np.allclose(m_star * a, (m_star * a)[0])  # m_i ∝ 1/a_i
    # tracking: m_i* = 0.25 ω_i / kappa  -> grad ~ 0, and m_i ∝ ω_i
    Omega = np.diag([0.6, 0.15])
    m_trk = 0.25 * np.diag(Omega) / k
    g = grad_L_eff_vech(np.diag(m_trk), A, N_star, "tracking", mode="fd", Omega=Omega)
    assert np.allclose(g, 0.0, atol=1e-5)
    assert np.allclose(m_trk / np.diag(Omega), (m_trk / np.diag(Omega))[0])
    # static: gradient strictly positive everywhere (M shrinks toward floor)
    g = grad_L_eff_vech(np.diag([0.2, 0.05]), A, N_star, "static", mode="aligned")
    assert np.all(g[[0, 2]] > 0)  # diagonal entries (vech idx 0 and 2 for d=2)


def test_natural_gradient_descends_and_aligned_reduction():
    A = np.diag([1.0, 4.0]); M = np.diag([0.5, 0.5]); N_star = 30.0; lam = 0.7
    Mdot = natural_gradient_M(M, A, N_star, "exploration", lam=lam)
    # (a) descends L_eff
    L0 = L_eff(M, A, N_star, "exploration", lam=lam)
    eps = 1e-4
    L1 = L_eff(M + eps * Mdot, A, N_star, "exploration", lam=lam)
    assert L1 < L0
    # (b) aligned reduction: vech(Ṁ) == -inv(Fisher) grad_vech for diagonal M
    g = grad_L_eff_vech(M, A, N_star, "exploration", mode="fd", lam=lam)
    F = fisher_metric_vech(M)
    nat_vech = -np.linalg.solve(F, g)
    assert np.allclose(vech(Mdot), nat_vech, atol=1e-4)


def test_vech_roundtrip_and_factor_of_two():
    M = np.array([[1.0, 0.3], [0.3, 2.0]])
    assert np.allclose(unvech(vech(M)), M)
    # off-diagonal of the matrix gradient is half the vech gradient entry
    g_vech = np.array([1.0, 2.0, 3.0])  # [g00, g01, g11]
    Gm = grad_matrix_from_vech(g_vech, 2)
    assert Gm[0, 0] == 1.0 and Gm[1, 1] == 3.0
    assert Gm[0, 1] == 1.0 and Gm[1, 0] == 1.0  # 2.0 / 2

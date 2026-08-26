"""§2.2 toy quadratic tests: the stationary-variance law V_stat ∝ a^{-α}, and the
noise-dependent optimal exponent α*."""
import numpy as np
import pytest

from toy.quadratic_sde import (make_quadratic, simulate_exact_sde, fit_loglog_slope,
                               analytic_stationary_variance, analytic_optimal_alpha)


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
def test_exact_sde_matches_discrete_ou(alpha):
    """Empirical stationary variance from the exact diagonal SDE matches the closed-form
    discrete-OU stationary variance per eigendirection."""
    quad = make_quadratic(dim=16, cond_number=1e3, seed=0, rotate=False, center=True)
    S = 50.0
    lr = 0.5 / float((quad.eigvals ** (1 - alpha)).max())  # per-alpha: good mixing + stable
    v_emp = simulate_exact_sde(quad, alpha, S, lr=lr, n_steps=60000, burn_in=15000,
                               seed=1, n_chains=512)
    k = quad.eigvals ** (1 - alpha)
    rho = 1 - lr * k
    q = (lr / S) * quad.eigvals ** (1 - 2 * alpha)
    v_disc = q / (1 - rho ** 2)
    rel = np.abs(v_emp - v_disc) / v_disc
    assert rel.max() < 0.06, f"alpha={alpha}: max rel err {rel.max():.3f}"


@pytest.mark.parametrize("alpha", [0.5, 1.0])
def test_vstat_loglog_slope_is_minus_alpha(alpha):
    """The headline law: log-log slope of V_stat vs curvature is -alpha."""
    quad = make_quadratic(dim=24, cond_number=1e4, seed=2, rotate=False, center=True)
    S = 100.0
    lr = 0.4 / float((quad.eigvals ** (1 - alpha)).max())
    v_emp = simulate_exact_sde(quad, alpha, S, lr=lr, n_steps=80000, burn_in=20000,
                               seed=3, n_chains=512)
    slope, _ = fit_loglog_slope(quad.eigvals, v_emp)
    assert abs(slope - (-alpha)) < 0.06, f"alpha={alpha}: slope {slope:.3f} != {-alpha}"


def test_analytic_stationary_variance_formula():
    """analytic_stationary_variance returns a^{-alpha}/(2S) exactly."""
    a = np.array([0.5, 1.0, 4.0])
    for alpha in (0.0, 0.5, 1.0):
        v = analytic_stationary_variance(a, alpha, S=10.0)
        assert np.allclose(v, a ** (-alpha) / 20.0)


def test_alpha_star_increases_with_batch():
    """α* (isotropic-test objective) is monotone non-decreasing in batch size S; the training
    loss objective instead pins α*≈1 (contrast)."""
    quad = make_quadratic(dim=20, cond_number=1e3, seed=0, rotate=False, center=True)
    S_grid = np.array([2.0, 10.0, 50.0, 250.0, 1250.0, 6000.0])
    astar, _ = analytic_optimal_alpha(quad, S_grid, n_steps=1500, test_curvature="isotropic")
    assert np.all(np.diff(astar) >= -1e-9), f"alpha* not monotone in S: {astar}"
    assert astar[-1] > astar[0] + 1e-6, f"alpha* did not increase: {astar[0]}->{astar[-1]}"
    astar_tr, _ = analytic_optimal_alpha(quad, S_grid, n_steps=1500, test_curvature="train")
    assert np.all(astar_tr > 0.9), f"train-loss alpha* should be ~1, got {astar_tr}"

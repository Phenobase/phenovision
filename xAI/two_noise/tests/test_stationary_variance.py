"""§2.3 — the stationary-variance / FDT test with the ACTUAL SOAPFullPower optimizer.

Confirms that running SOAP at precond_power=α on a separable quadratic with Fisher gradient
noise (C=A) yields stationary fluctuation variance scaling as curvature^{-α} — i.e. the
power-reshaped Gibbs law the framework predicts (slope −α; −0.5 whitening, −1 full inverse).
"""
import numpy as np
import pytest

from toy.quadratic_sde import simulate_soap_stationary, fit_loglog_slope


@pytest.mark.parametrize("alpha", [0.5, 0.75, 1.0])
def test_soap_stationary_variance_slope(alpha):
    p = np.logspace(-0.5, 0.5, 6)
    q = np.logspace(-0.5, 0.5, 6)
    kappa, vstat = simulate_soap_stationary(p, q, alpha=alpha, S=200.0, lr=0.02,
                                            n_steps=40000, burn_in=10000, seed=0,
                                            damping=1e-3)
    slope, _ = fit_loglog_slope(kappa, vstat)
    r = np.corrcoef(np.log(kappa), np.log(vstat))[0, 1]
    assert abs(r) > 0.98, f"alpha={alpha}: weak log-log fit R={r:.3f}"
    assert abs(slope - (-alpha)) < 0.08, f"alpha={alpha}: slope {slope:.3f} != {-alpha}"


def test_stationary_slope_monotone_in_alpha():
    """The realized exponent tracks the set α: steeper (more negative) slope for larger α."""
    p = np.logspace(-0.5, 0.5, 6)
    q = np.logspace(-0.5, 0.5, 6)
    slopes = []
    for alpha in (0.5, 1.0):
        kappa, vstat = simulate_soap_stationary(p, q, alpha=alpha, S=200.0, lr=0.02,
                                                n_steps=40000, burn_in=10000, seed=1,
                                                damping=1e-3)
        s, _ = fit_loglog_slope(kappa, vstat)
        slopes.append(s)
    assert slopes[1] < slopes[0], f"slope not steeper at alpha=1: {slopes}"

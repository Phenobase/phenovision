"""Exploration fixed-point eigenvalue scaling (sim/exploration_scaling.py). Wiring + the
qualitative contrast: a fluctuating optimum with Ω∝A⁻¹ drives the evolved G toward the
inverse-curvature scaling (slope of log g† vs log a more negative), whereas isotropic Ω does not.
NOTE: Ω∝A⁻¹ is a STRUCTURED-environment assumption (see directional_alignment.py for the natural,
no-assumption transient mechanism). Small/fast scale; the clean exponent is the production run."""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from sim.exploration_scaling import exponent_sweep, _fluctuating_theta
import jax, numpy as np


def test_fluctuating_theta_covariance_structure():
    A = np.diag([1.0, 9.0])
    th = np.asarray(_fluctuating_theta(jax.random.PRNGKey(0), 4000, A, env_sigma=1.0,
                                       omega_mode="Ainv"))
    cov = np.cov(th, rowvar=False)
    # Ω∝A⁻¹: more optimum-fluctuation variance in the FLAT (small-a) direction
    assert cov[0, 0] > cov[1, 1]


def test_Ainv_drives_more_negative_G_scaling_than_isotropic():
    common = dict(N=350, L=8, n_generations=800, n_replicates=24, seed=0)
    r_ainv = exponent_sweep(A_ratios=(2.0, 6.0), env_sigma=2.5, omega_mode="Ainv",
                            evolve=True, **common)
    r_iso = exponent_sweep(A_ratios=(2.0, 6.0), env_sigma=2.5, omega_mode="isotropic",
                           evolve=True, **common)
    assert np.isfinite(r_ainv["slope_g"]) and np.isfinite(r_iso["slope_g"])
    # Ω∝A⁻¹ pushes the standing-variance scaling toward inverse-curvature (more negative slope)
    assert r_ainv["slope_g"] < r_iso["slope_g"]

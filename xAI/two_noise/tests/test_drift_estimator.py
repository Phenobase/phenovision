"""V0 — the ensemble-drift estimator + metric correction + scaling/rotation split + curl
detector, validated against the Gaussian-closure surrogate (where the true drift is known).
A low cosine here later (against a real IBM) would indict the THEORY, not the estimator."""
import numpy as np

from sim.theory_load_surface import vech, unvech
from sim.drift_estimator import (
    ensemble_drift, estimate_G_M, split_scaling_rotation, cos_direction,
    gaussian_surrogate_increment, true_drift_vech, helmholtz_curl_fraction_2d,
)


def _setup(seed=0):
    A = np.diag([1.0, 3.0])
    N_star = 40.0
    # a NON-identity metric in vech space so the metric correction is necessary
    rng = np.random.default_rng(seed)
    L = np.array([[1.0, 0.0, 0.0], [0.4, 0.8, 0.0], [0.2, 0.3, 0.7]])
    G_M = L @ L.T * 0.02
    M0 = np.diag([0.4, 0.15])
    return A, N_star, G_M, M0


def test_recovery_direction_high_cosine():
    A, N_star, G_M, M0 = _setup()
    inc = gaussian_surrogate_increment(A, N_star, "exploration", G_M, eta_M=1.0,
                                       dt=0.02, noise=0.0, lam=0.5)
    bhat = ensemble_drift(inc, M0, R=200, tau=30, seed=1, eta_M=1.0)
    truth = true_drift_vech(M0, A, N_star, "exploration", G_M, lam=0.5)
    assert cos_direction(bhat["vech"], truth) > 0.999


def test_metric_correction_is_necessary():
    A, N_star, G_M, M0 = _setup()
    inc = gaussian_surrogate_increment(A, N_star, "exploration", G_M, eta_M=1.0,
                                       dt=0.02, noise=0.0, lam=0.5)
    bhat = ensemble_drift(inc, M0, R=300, tau=30, seed=2, eta_M=1.0)
    from sim.theory_load_surface import grad_L_eff_vech
    bare = -grad_L_eff_vech(M0, A, N_star, "exploration", mode="fd", lam=0.5)
    corrected = -G_M @ grad_L_eff_vech(M0, A, N_star, "exploration", mode="fd", lam=0.5)
    cos_bare = cos_direction(bhat["vech"], bare)
    cos_corr = cos_direction(bhat["vech"], corrected)
    assert cos_corr > 0.99
    assert cos_bare < cos_corr - 0.05      # bare gradient is a worse match (metric bends drift)


def test_estimate_G_M_recovers_injected_metric():
    A, N_star, G_M, M0 = _setup()
    # draw N modifier breeding values from N(0, G_M); sample cov should recover G_M
    rng = np.random.default_rng(3)
    bvs = rng.multivariate_normal(np.zeros(3), G_M, size=400)
    Ghat = estimate_G_M(bvs)
    assert cos_direction(Ghat, G_M) > 0.99


def test_scaling_rotation_split_isolates_components():
    M0 = np.diag([0.5, 0.2])
    # pure scaling drift (diagonal in M0 eigenbasis)
    pure_scale = np.diag([0.1, -0.05])
    s = split_scaling_rotation(M0, pure_scale)
    assert s["scaling_frac"] > 0.99 and s["rotation_frac"] < 0.05
    # pure rotation drift (off-diagonal in M0 eigenbasis)
    pure_rot = np.array([[0.0, 0.07], [0.07, 0.0]])
    r = split_scaling_rotation(M0, pure_rot)
    assert r["rotation_frac"] > 0.99 and r["scaling_frac"] < 0.05


def test_curl_detector_zero_for_gradient_jumps_for_rotation():
    A, N_star, G_M, _ = _setup()
    Ginv = np.linalg.inv(G_M)
    # sweep the two eigenvalues of a diagonal M on a grid; field = metric-corrected drift
    xs = np.linspace(0.15, 0.6, 7)
    ys = np.linspace(0.08, 0.4, 7)

    def fields(curl_strength):
        U = np.zeros((len(xs), len(ys))); V = np.zeros((len(xs), len(ys)))
        for i, mx in enumerate(xs):
            for j, my in enumerate(ys):
                M = np.diag([mx, my])
                # raw drift incl. optional injected curl, then metric-correct (G_M^{-1} v)
                from sim.theory_load_surface import grad_L_eff_vech
                g = grad_L_eff_vech(M, A, N_star, "exploration", mode="fd", lam=0.5)
                drift = -G_M @ g
                if curl_strength:
                    K = np.array([[0, 1.0, 0], [-1.0, 0, 0], [0, 0, 0]])
                    drift = drift + curl_strength * (K @ g)
                u = Ginv @ drift            # metric-corrected field (should be a gradient)
                U[i, j] = u[0]; V[i, j] = u[2]   # the two diagonal-vech components (m00, m11)
        return U, V

    cf0 = helmholtz_curl_fraction_2d(xs, ys, *fields(0.0))
    cf1 = helmholtz_curl_fraction_2d(xs, ys, *fields(0.5))
    assert cf0 < 0.15                      # pure gradient -> small curl residual
    assert cf1 > cf0 + 0.1                 # injected rotation -> detector jumps

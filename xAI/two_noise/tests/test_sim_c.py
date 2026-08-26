"""Sim C (§3.3): CMA-ES covariance converges to A^{-1}; natural-gradient == realized response."""
import numpy as np

from sim.sim_c_naturalgrad import cma_covariance_vs_hessian, natural_gradient_vs_response


def test_cma_covariance_converges_to_Ainv():
    rng = np.random.default_rng(0)
    Q, _ = np.linalg.qr(rng.standard_normal((4, 4)))
    A = (Q * np.array([8.0, 3.0, 1.0, 0.4])) @ Q.T
    A = 0.5 * (A + A.T)
    res = cma_covariance_vs_hessian(A, n_gen=300, seed=0)
    # C aligns with A^{-1}: matrix cosine close to 1 and eigenvalue slope close to -1
    assert res["final_cosine"] > 0.9, f"C-Ainv cosine {res['final_cosine']:.3f}"
    assert abs(res["final_slope"] - (-1.0)) < 0.25, f"loglog slope {res['final_slope']:.3f}"


def test_natural_gradient_matches_response():
    # If realized response equals G@beta exactly, cosine == 1
    rng = np.random.default_rng(1)
    G = rng.standard_normal((3, 3)); G = G @ G.T + np.eye(3)
    beta = rng.standard_normal(3)
    dz = G @ beta
    out = natural_gradient_vs_response(G, beta, dz)
    assert out["cosine_ng_response"] > 0.999

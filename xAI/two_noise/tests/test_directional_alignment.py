"""Directional-selection alignment (sim/directional_alignment.py): the natural-gradient benefit
measured DURING the chase toward a distant optimum (the optimizer analog; no contrived environment).
In the low-noise regime the response heads straighter to the optimum (Newton) as α→1 and follows
the raw gradient as α→0."""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

from sim.directional_alignment import run_alpha, _M_for_alpha


def test_M_for_alpha_gives_target_G_scaling():
    # M ∝ A^{1-2α}: α=1 -> M∝A⁻¹, α=0 -> M∝A, α=0.5 -> M∝const
    A = np.diag([1.0, 4.0])
    m1 = _M_for_alpha(A, 1.0, 0.08)      # ∝ A⁻¹: larger in the flat (small-a) direction
    assert m1[0] > m1[1]
    m0 = _M_for_alpha(A, 0.0, 0.08)      # ∝ A: larger in the steep direction
    assert m0[1] > m0[0]
    mh = _M_for_alpha(A, 0.5, 0.08)      # ∝ const (isotropic)
    assert np.allclose(mh[0], mh[1], rtol=1e-6)


def test_newton_alignment_increases_with_alpha():
    A = np.diag([1.0, 6.0]); theta = np.array([4.0, 4.0])    # moderate (low-noise) anisotropy
    cfg = dict(N=600, L=12, n_generations=400, n_replicates=24, seed=0)
    r0 = run_alpha(A, 0.0, theta, **cfg)
    r1 = run_alpha(A, 1.0, theta, **cfg)
    # α=1 (G∝A⁻¹) heads straighter to the optimum (Newton) than α=0 (G∝I, gradient-following)
    assert r1["cos_newton"] > r0["cos_newton"]
    # and α=0 is more gradient-aligned than α=1 (the crossover)
    assert r0["cos_grad"] > r1["cos_grad"]

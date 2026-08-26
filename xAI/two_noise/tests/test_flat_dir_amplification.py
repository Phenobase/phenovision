"""O1 — the a^{1-2 alpha} update-noise law and the alpha* (B) boundary.
Mirrors test_toy_quadratic.py tolerances (~0.06-0.08 for stochastic slopes)."""
import numpy as np

from toy.flat_dir_amplification import (
    flat_dir_amplification, stationary_noise_slope, alpha_star_of_batch,
    realized_exponent_sweep,
)


def test_stationary_noise_slope_is_one_minus_two_alpha():
    # the headline law (C=A): update-noise variance ~ a^{1-2 alpha} per eigendirection
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        slope = stationary_noise_slope(alpha, B=256, T=6000, seed=0, true_curvature=True)
        assert abs(slope - (1 - 2 * alpha)) < 0.08, (alpha, slope)
    # the empirical-C optimizer deviates at high alpha but the slope still DECREASES with alpha
    emp = [stationary_noise_slope(a, B=256, T=6000, seed=0, true_curvature=False)
           for a in [0.0, 0.5, 1.0]]
    assert emp[0] > emp[1] > emp[2]


def test_flat_direction_amplifies_with_alpha_and_low_batch():
    # flat-direction wander grows with alpha (>1/2) and is worse at small batch
    f_lo_a = flat_dir_amplification(0.5, B=16, T=4000, seed=0)[0]
    f_hi_a = flat_dir_amplification(1.0, B=16, T=4000, seed=0)[0]
    assert f_hi_a > f_lo_a                              # alpha=1 wanders more than alpha=1/2
    f_b16 = flat_dir_amplification(1.0, B=16, T=4000, seed=0)[0]
    f_b256 = flat_dir_amplification(1.0, B=256, T=4000, seed=0)[0]
    assert f_b16 > f_b256                                # worse at small batch


def test_shrink_stabilizes_alpha_one():
    f_noshrink = flat_dir_amplification(1.0, B=16, T=4000, shrink=0.0, seed=0)[0]
    f_shrink = flat_dir_amplification(1.0, B=16, T=4000, shrink=0.5, seed=0)[0]
    assert f_shrink < f_noshrink                         # shrinkage reduces flat-dir wander


def test_alpha_star_increases_with_batch():
    # alpha*(B) delegated to toy.quadratic_sde.analytic_optimal_alpha (proper finite-budget
    # loss). The noise-dependent law: alpha* non-decreasing in batch.
    rows = alpha_star_of_batch(batches=(8, 64, 512, 4096), n_steps=1200, seed=0)
    astar = [r["alpha_star"] for r in rows]
    assert all(astar[i] <= astar[i + 1] + 1e-9 for i in range(len(astar) - 1)), astar


def test_realized_exponent_decreases_with_shrink():
    rex = realized_exponent_sweep(rhos=(0.0, 0.3, 0.6, 0.9))
    e = [r["realized_exponent"] for r in rex]
    assert abs(e[0] - 1.0) < 0.1                          # shrink=0 -> exponent ~1
    assert all(e[i + 1] <= e[i] + 1e-6 for i in range(len(e) - 1))

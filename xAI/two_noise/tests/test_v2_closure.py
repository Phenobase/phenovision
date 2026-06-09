"""V2 — fast equilibria + Gaussian closure (sim/v2_closure.py). Pilot scale. Validates that the
MSB closure Ĝ=A⁻¹#M is right in SHAPE and off only by a CONSTANT scale factor (resolving the V1
finding), that Σ_∞ ∝ A⁻¹, and that the Gaussian closure holds (small skew/kurtosis)."""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

from sim.v2_closure import run_fixed_M


def _runs():
    A = [1.0, 3.0]
    cfg = dict(N=400, L=10, mu=0.02, n_generations=350, burn_in=200, n_replicates=10, seed=0)
    M_list = [(0.08, 0.03), (0.1, 0.06)]
    rows = []
    for rot in (0.0, 0.6):                       # commuting + non-commuting
        for m in M_list:
            rows.append(run_fixed_M(A, m, rotate_A=rot, **cfg))
    return rows


def test_closure_shape_and_constant_scale():
    rows = _runs()
    # SHAPE: measured G aligns with Ĝ=A⁻¹#M (Frobenius cosine high), commuting + non-commuting
    assert all(r["shape_cos"] > 0.95 for r in rows), [r["shape_cos"] for r in rows]
    # SCALE: the discrepancy is a CONSTANT factor (the calibratable Gaussian-approx coefficient)
    sf = np.array([r["scale_factor"] for r in rows])
    assert np.std(sf) / np.mean(sf) < 0.2, sf      # near-constant across M / anisotropy / rotation
    assert np.mean(sf) > 1.5                        # IBM variance exceeds the bare MSB closure


def test_sigma_infinity_shape_is_A_inverse():
    rows = _runs()
    assert all(r["sigma_shape_cos"] > 0.9 for r in rows), [r["sigma_shape_cos"] for r in rows]


def test_gaussian_closure_holds():
    rows = _runs()
    # small skew and excess kurtosis along principal axes => Gaussian (not house-of-cards)
    assert all(r["skew_max"] < 0.4 for r in rows), [r["skew_max"] for r in rows]
    assert all(r["kurt_max"] < 1.0 for r in rows), [r["kurt_max"] for r in rows]

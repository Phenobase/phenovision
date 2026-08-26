"""V1 — fixed-M fitness scan diagnostic (sim/v1_fitness_scan.py). Pilot-scale; per the sim_b
'scale caveat', we assert the ROBUST parts (scan well-formed, G diagonal under diagonal A, the
load functional version-B aligns with U_IBM) and RECORD the closure-version-A discrepancy (the
~2.6x Gaussian-vs-house-of-cards scale factor V2 quantifies)."""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

from sim.v1_fitness_scan import scan_diagonal_M, gradient_cosines


def test_scan_wellformed_G_diagonal_and_loadfunctional_aligns():
    A = np.diag([1.0, 3.0])
    m1 = np.linspace(0.03, 0.11, 3)
    m2 = np.linspace(0.03, 0.11, 3)
    rows = scan_diagonal_M(A, m1, m2, N=300, L=10, mu=0.02, n_generations=200,
                           burn_in=120, n_replicates=6, Lm=4, seed=0)
    assert len(rows) == 9
    # all U_IBM finite and negative (mean log fitness below the peak)
    assert all(np.isfinite(r["U_IBM"]) and r["U_IBM"] < 0 for r in rows)
    # measured G is diagonal under diagonal A + per-trait-independent injection
    corr = [abs(r["G12"]) / (r["G11"] * r["G22"]) ** 0.5 for r in rows]
    assert max(corr) < 0.15, max(corr)
    # the LOAD FUNCTIONAL (version B, measured G) tracks U_IBM in shape
    cos = gradient_cosines(rows, m1, m2)
    assert cos["cos_U_vs_negLeff_B"] > 0.55, cos
    # the closure (version A, analytic Ghat) is right in SHAPE: G/Ghat ratio roughly constant
    ratios = [r["G11"] / r["Ghat11"] for r in rows] + [r["G22"] / r["Ghat22"] for r in rows]
    assert np.std(ratios) / np.mean(ratios) < 0.6     # near-scalar rescaling (V2 nails the factor)
    assert np.mean(ratios) > 1.2                       # IBM variance exceeds the bare MSB closure

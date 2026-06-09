"""V5 — adiabaticity dialing (sim/v5_adiabatic.py). Pilot scale. The drift stays robustly aligned
with the natural gradient across the ε sweep; the clean cos-improves-as-ε→0 (and curl ∝ ε) is the
production-reps result (at pilot reps small ε => smaller drift => noisier estimate)."""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

from sim.v5_adiabatic import adiabaticity_sweep, epsilon


def test_drift_robustly_aligned_across_epsilon():
    A = np.diag([1.0, 4.0]); M0 = np.diag([0.06, 0.06])
    # THE adiabaticity prediction: the reduced-SDE (gradient-flow) approximation IMPROVES as ε→0.
    # On the clean canalization regime, the drift aligns with the static natural gradient better at
    # small ε (small modifier mutation) and degrades at large ε (modifier mutation dominates the
    # drift). grid is mut_var_mod DESCENDING => ε descending => cos should rise.
    rows = adiabaticity_sweep(M0, A, regime="canalization", mut_var_mod_grid=(0.04, 0.01),
                              N=350, L=8, Lm=8, burn_in=200, tau=50, n_replicates=64,
                              env_sigma=0.0, seed=0)
    assert rows[-1]["cos_drift_vs_theory"] > rows[0]["cos_drift_vs_theory"]  # improves as ε→0
    assert rows[-1]["cos_drift_vs_theory"] > 0.5  # aligned in the adiabatic (small-ε) limit
    # epsilon is monotone in mut_var_mod and positive
    assert rows[0]["epsilon"] > rows[1]["epsilon"] > 0


def test_epsilon_formula():
    assert epsilon(0.02, Lm=8, L=10) > epsilon(0.01, Lm=8, L=10) > 0

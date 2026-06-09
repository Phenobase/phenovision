"""V4 — gradient flow vs curl (sim/v4_curl.py). Pilot scale. Core leading-order prediction: the
metric-corrected M-drift field is curl-free in the ALIGNED (diagonal A => A,G commute) regime."""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

from sim.v4_curl import curl_fraction


def test_aligned_curl_is_small():
    A = np.diag([1.0, 4.0])
    cf, U, V = curl_fraction(A, regime="static", grid=(0.04, 0.07, 0.10),
                             N=300, L=8, burn_in=150, tau=40, n_replicates=40, seed=0)
    # the metric-corrected drift is (to leading order) -∇ℒ_eff => a gradient => small curl residual
    assert 0.0 <= cf < 0.35, cf
    assert np.all(np.isfinite(U)) and np.all(np.isfinite(V))

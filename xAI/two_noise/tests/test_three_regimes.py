"""The three M-evolution regimes via emergent dynamics (sim/three_regimes.py). Pilot scale.
Canalization shrinks M; exploration (static peak + the emergent random-disaster bet-hedging
challenge -- NO imposed term) parks maintained variance in the flat direction, lifting the
evolved-M ratio toward the A⁻¹ target."""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

from sim.three_regimes import run_three_regimes


def test_regimes_canalization_and_emergent_exploration():
    rows = run_three_regimes(N=500, L=10, burn_in=300, tau=70, n_replicates=96,
                             chal_strength=1.0, chal_sigma=1.5, track_sigma=0.15, seed=0)
    by = {r["regime"]: r for r in rows}
    # canalization: total mutational variance shrinks under a static peak, no challenge
    assert by["canalization"]["trM_end"] < by["canalization"]["trM0"]
    # emergent exploration: the challenge lifts the FLAT-direction drift vs canalization,
    # and the evolved-M ratio moves toward the A⁻¹ target (a_steep/a_flat = 4)
    assert by["exploration"]["dM_flat"] > by["canalization"]["dM_flat"]
    assert by["exploration"]["ratio_flat_over_steep"] > by["canalization"]["ratio_flat_over_steep"]
    # exploration ratio is in the right ballpark of the A⁻¹ target (not collapsed, not runaway)
    assert 2.5 < by["exploration"]["ratio_flat_over_steep"] < 8.0

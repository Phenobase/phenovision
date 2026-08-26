"""V6 — full trajectory match (sim/v6_trajectory.py). Pilot scale. Validates the sliced-Wasserstein
machinery and that the reduced SDE captures the DIRECTION of the IBM's M-evolution (canalization
under a static peak). The full distributional match is the production run (the pilot also surfaces
that the deterministic natural-gradient flow overshoots without a mutational-floor term)."""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

from sim.v6_trajectory import sliced_wasserstein, euler_maruyama_M, compare


def test_sliced_wasserstein_metric():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((200, 2))
    assert sliced_wasserstein(a, a) < 1e-9            # zero for identical clouds
    assert sliced_wasserstein(a, a + 3.0) > 1.0       # grows with separation


def test_em_integrator_runs_spd():
    A = np.diag([1.0, 4.0]); M0 = np.diag([0.06, 0.06])
    traj = euler_maruyama_M(M0, A, "static", rate=0.5, n_steps=50, dt=1.0, diffusion=1e-4, seed=0)
    assert traj.shape == (51, 2, 2)
    assert all(np.linalg.eigvalsh(M).min() > 0 for M in traj)   # SPD preserved


def test_sde_captures_canalization_direction():
    A = np.diag([1.0, 4.0]); M0 = np.diag([0.06, 0.06])
    r = compare(M0, A, regime="static", N=300, L=8, n_generations=150, n_replicates=32, seed=0)
    # both the IBM and the reduced SDE shrink M under a static peak (canalization direction)
    assert r["trM_ibm_end"] < r["trM0"]
    assert r["trM_sde_end"] < r["trM0"]
    assert r["sliced_wasserstein"] >= 0 and np.isfinite(r["sliced_wasserstein"])

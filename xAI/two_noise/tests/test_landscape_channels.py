"""The two benefit channels (notes_landscape_fluctuation_benefit.md): the evolved M tracks an
ANISOTROPIC random-disaster covariance Ω (Channel 1, M∝Ω) but hedges to A⁻¹ for a DIRECTIONLESS
(isotropic) one (Channel 2). Discriminated by Ω's directional structure, not amplitude."""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

from sim.v3_drift import measure_drift


def test_anisotropic_omega_tracked_isotropic_hedges_to_Ainv():
    A = np.diag([1.0, 4.0]); M0 = np.diag([0.06, 0.06])   # A⁻¹ wants flat(0)>steep(1): ratio>1
    cfg = dict(regime="canalization", N=500, L=10, burn_in=300, tau=80,
               n_replicates=96, challenge_strength=1.0, challenge_sigma=2.0, seed=0)
    # Channel 1: strong challenge with Ω favoring the STEEP direction -> M follows Ω (steep, ratio<1)
    r_aniso = measure_drift(M0, A, challenge_aniso=[0.4, 2.0], **cfg)
    # Channel 2: same strength, ISOTROPIC Ω -> no direction to track -> M ∝ A⁻¹ (flat, ratio>1)
    r_iso = measure_drift(M0, A, challenge_aniso=[1.0, 1.0], **cfg)
    ra = r_aniso["M_end"][0, 0] / max(r_aniso["M_end"][1, 1], 1e-9)
    ri = r_iso["M_end"][0, 0] / max(r_iso["M_end"][1, 1], 1e-9)
    assert ra < 1.0, ra          # anisotropic-steep Ω -> M leans steep (tracks Ω, Channel 1)
    assert ri > 1.5, ri          # isotropic Ω -> M leans flat (A⁻¹, Channel 2)
    assert ri > ra               # the two channels are clearly separated

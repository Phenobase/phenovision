"""§3.2 (clean): incoherent errors-in-variables noise compresses G anisotropy (Force 2),
without lag-load, with flat G-A alignment in the clean regime."""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from sim.sim_b_compression import run_incoherent_sweep


def test_g_anisotropy_compresses_with_gradient_noise():
    # small/fast config; coarse noise grid
    rows = run_incoherent_sweep(env_amp_grid=(0.0, 0.8, 2.0), A_eigratio=6.0, N=400, L=15,
                                Ne=400, n_generations=300, burn_in=180, n_replicates=16, seed=0)
    g = [r["G_anisotropy"] for r in rows]
    # compression: high-noise anisotropy clearly below zero-noise anisotropy, toward isotropy (1.0)
    assert g[-1] < g[0] - 0.15, f"G anisotropy did not compress: {g}"
    assert g[-1] >= 0.95, f"G anisotropy went below isotropy (unphysical): {g}"
    # alignment stays small (no eigenvalue-order inversion) in this regime
    assert all(r["G_A_angle"] < 10.0 for r in rows[:-1]), \
        f"alignment not flat: {[r['G_A_angle'] for r in rows]}"

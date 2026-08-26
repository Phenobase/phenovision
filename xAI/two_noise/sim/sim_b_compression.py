"""§3.2 (clean) — the gradient-noise COMPRESSION signature via incoherent errors-in-variables.

The spatial-correlation N* knob (sim_b_phase.py) confounds compression (Force 2, attenuation) with
lag-load (Force 3, enhancement), because lowering N* via correlation makes the per-generation
optimum perturbation coherent across individuals (a per-generation shift). This module isolates the
compression mechanism with an INCOHERENT perturbation: independent per-individual optimum noise
δ_i ~ N(0, env_amp² I) (an IDENTITY kernel in run_sim_perturbed), which has no coherent shift.

Biological interpretation (errors-in-variables): perturbing each individual's optimum independently
is identical to independent ENVIRONMENTAL/DEVELOPMENTAL variance in the expressed phenotype
(P = G + E) — selection acts on a noisy read of the breeding value, i.e. regression dilution /
low heritability. Inflating this per-individual variance inflates the gradient covariance C, which
is equivalent to lowering the effective N* in the gradient-noise term C/N*. So env_amp is a
gradient-noise (inverse-N*) knob; predict anisotropy compression as it rises.

Measure G (additive-genetic covariance) anisotropy — it responds on the standing-variance timescale,
unlike M which needs the (much slower) architectural M-evolution timescale to develop anisotropy.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from sim.engine import make_config
from sim.sim_b_phase import run_sim_perturbed, m_anisotropy, m_a_alignment_angle


def run_incoherent_sweep(env_amp_grid=(0.0, 0.4, 0.8, 1.5, 2.5), A_eigratio=6.0, N=600, L=20,
                         Ne=600, mu=5e-2, mut_var=0.05, sigma_eps=0.3, n_generations=500,
                         burn_in=300, n_replicates=24, seed=0):
    """Sweep the incoherent gradient-noise magnitude; return a tidy list of dicts with G/M
    anisotropy and G–A alignment angle per noise level. Ne large (low drift) to isolate the
    gradient-noise (compression) effect from demographic (Langevin) inflation."""
    s = float(np.sqrt(A_eigratio))
    A_np = np.array([[s, 0.0], [0.0, 1.0 / s]])
    ident = jnp.eye(N, dtype=jnp.float32)  # identity kernel => independent per-individual noise
    cfg = make_config(N=N, L=L, n_traits=2, mu=mu, mut_var=mut_var, r=0.5,
                      sigma_eps=sigma_eps, A=jnp.asarray(A_np, dtype=jnp.float32),
                      theta=jnp.zeros(2), Ne=Ne)
    rows = []
    for amp in env_amp_grid:
        traj = run_sim_perturbed(jax.random.PRNGKey(seed), cfg, env_L_sqrt=ident,
                                 env_amp=float(amp), n_generations=n_generations,
                                 n_replicates=n_replicates)
        G = np.array(traj["G"])[:, burn_in:].mean(axis=1).mean(axis=0)
        M = np.array(traj["M"])[:, burn_in:].mean(axis=1).mean(axis=0)
        rows.append({"env_amp": float(amp), "G_anisotropy": float(m_anisotropy(G)),
                     "M_anisotropy": float(m_anisotropy(M)),
                     "G_A_angle": float(m_a_alignment_angle(G, A_np)),
                     "A_eigratio": float(A_eigratio), "Ne": int(Ne)})
    return rows


if __name__ == "__main__":
    import csv
    from pathlib import Path
    rows = run_incoherent_sweep()
    out = Path("runs/sim_b_compression/results.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"[sim_b_compression] wrote {out}")
    for r in rows:
        print(f"  env_amp={r['env_amp']:.2f}  G_aniso={r['G_anisotropy']:.3f}  "
              f"M_aniso={r['M_anisotropy']:.3f}  G-A angle={r['G_A_angle']:.2f}")

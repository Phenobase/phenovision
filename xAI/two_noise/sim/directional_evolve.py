"""Evolving M under sustained DIRECTIONAL selection -> the noise-dependent optimal exponent α*(N).

This is the SLOW-timescale (M-evolution) test, and the natural ('optimum keeps moving away')
mechanism: a relocating optimum keeps the population perpetually ADAPTING (transient descent,
never at MSB). Second-order selection then favors the M-structure that adapts FASTEST. The
fixed-M analysis (directional_alignment.py) showed:
  * α=1 (G∝A⁻¹) = Newton = straight to the optimum = fastest adaptation, BUT
  * α=1 destabilizes in flat directions via noise amplification (the a^{1-2α} law), worse at
    smaller N (more sampling/drift noise).
So evolving M should converge to a NOISE-DEPENDENT optimal exponent α*: toward 1 (full inverse /
Newton) when N is large (low noise), backing off toward 1/2 (whitening) when N is small. That is
the evolutionary mirror of the optimizer's α*(batch) law (O1) — the cross-substrate prediction.

Mechanism (no contrived environment): the optimum jumps to a fresh RANDOM distant point every
`relocate_every` generations (isotropic relocation challenges all directions equally, so the
curvature-aware A⁻¹ structure is what wins on average). M evolves freely (eig_diag); we read the
evolved exponent α from the standing G eigenvalues vs A across the two trait axes:
    α = log(g_lo / g_hi) / log(a_hi / a_lo)   (G ∝ A^{-α}).
Sweep N -> α*(N). CPU/JAX. Writes runs/directional_evolve/results.csv.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp

from sim.engine import make_config
from sim.evolvable_m import (make_hyper, m_input_scale, init_evo_population, continue_evo_sim)


def _relocating_theta(key, n_generations, n, radius, relocate_every):
    """Optimum jumps to a fresh random point at distance `radius` every relocate_every gens
    (isotropic directions -> sustained directional selection challenging all directions)."""
    n_blocks = (n_generations + relocate_every - 1) // relocate_every
    dirs = jax.random.normal(key, (n_blocks, n))
    dirs = dirs / (jnp.linalg.norm(dirs, axis=1, keepdims=True) + 1e-30)
    targets = radius * dirs                                   # (n_blocks, n)
    seq = jnp.repeat(targets, relocate_every, axis=0)[:n_generations]
    return seq


def evolved_exponent(A, *, N=800, L=14, mu=0.03, Lm=8, mu_mod=0.3, mut_var_mod=0.04,
                     n_generations=4000, burn_frac=0.5, n_replicates=48, radius=3.0,
                     relocate_every=60, M0_scale=0.06, seed=0):
    """Evolve M under a relocating optimum; return the evolved standing-G eigenvalues and the
    fitted exponent α (G ∝ A^{-α}) from the two trait axes (time-avg post-burn-in)."""
    n = A.shape[0]
    config = make_config(N=N, L=L, n_traits=n, mu=mu, A=jnp.asarray(A, jnp.float32), Ne=N)
    hyper = make_hyper(design="eig_diag", n_traits=n, Lm=Lm, mu_mod=mu_mod, mut_var_mod=mut_var_mod)
    scale = m_input_scale(config, hyper)
    target_b = jnp.log(jnp.full((n,), M0_scale / scale, dtype=jnp.float32))   # isotropic M0
    k0, k1 = jax.random.split(jax.random.PRNGKey(seed))
    rep_keys = jax.random.split(k0, n_replicates)
    states0 = jax.vmap(lambda k: init_evo_population(k, config, hyper, "eig_diag", target_b))(rep_keys)
    theta_seq = _relocating_theta(k1, n_generations, n, radius, relocate_every)
    traj, _ = continue_evo_sim(states0, config, hyper, "eig_diag", theta_seq)
    burn = int(burn_frac * n_generations)
    G = np.asarray(traj["G"][:, burn:].mean(axis=1)).mean(axis=0)              # (n,n)
    M = np.asarray(traj["M"][:, burn:].mean(axis=1)).mean(axis=0)
    a = np.diag(A); g = np.clip(np.diag(G), 1e-9, None)
    # G ∝ A^{-α}  =>  α = log(g_lo/g_hi) / log(a_hi/a_lo)  (lo/hi by curvature)
    order = np.argsort(a)
    a_s, g_s = a[order], g[order]
    alpha = float(np.log(g_s[0] / g_s[-1]) / np.log(a_s[-1] / a_s[0]))
    return dict(alpha=alpha, g_lo=float(g_s[0]), g_hi=float(g_s[-1]),
                m_lo=float(np.diag(M)[order][0]), m_hi=float(np.diag(M)[order][-1]),
                trG=float(np.trace(G)))


def sweep_N(A, N_grid=(200, 500, 1500, 5000), **kw):
    rows = []
    for N in N_grid:
        r = evolved_exponent(A, N=N, **kw)
        r["N"] = N
        rows.append(r)
        print(f"[dir-evolve] N={N:5d}  evolved α(G∝A^-α)={r['alpha']:+.2f}  "
              f"g=({r['g_lo']:.4f},{r['g_hi']:.4f})  trG={r['trG']:.4f}")
    return rows


def _write_csv(path, rows):
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("wrote", path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--L", type=int, default=14)
    p.add_argument("--generations", type=int, default=4000)
    p.add_argument("--replicates", type=int, default=48)
    p.add_argument("--anisotropy", type=float, default=6.0, help="A = diag(1, anisotropy)")
    p.add_argument("--radius", type=float, default=3.0)
    p.add_argument("--relocate-every", type=int, default=60)
    p.add_argument("--N-grid", type=int, nargs="+", default=[200, 500, 1500, 5000])
    p.add_argument("--out-dir", default=str(ROOT / "runs" / "directional_evolve"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    A = np.diag([1.0, args.anisotropy])
    print(f"=== evolving M under a relocating optimum; A=diag(1,{args.anisotropy}); sweep N ===")
    rows = sweep_N(A, N_grid=tuple(args.N_grid), L=args.L, n_generations=args.generations,
                   n_replicates=args.replicates, radius=args.radius,
                   relocate_every=args.relocate_every, seed=args.seed)
    _write_csv(Path(args.out_dir) / "results.csv", rows)
    al = [r["alpha"] for r in rows]; Ns = [r["N"] for r in rows]
    print(f"\n[dir-evolve] α*(N): {[f'{n}:{a:+.2f}' for n, a in zip(Ns, al)]}")
    print("Prediction: α* increases toward 1 (Newton/full-inverse) as N grows (less noise) --")
    print("the evolutionary mirror of the optimizer's α*(batch) law.")


if __name__ == "__main__":
    main()

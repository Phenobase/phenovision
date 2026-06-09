"""Exploration regime: the inverse-curvature fixed point M ∝ A⁻¹ (the evolutionary analog of the
optimizer's full inverse, alpha=1). The headline cross-substrate eigenvalue-scaling test (E2).

THE MECHANISM (derived carefully). Under a FLUCTUATING optimum θ_t (fast, uncorrelated -> the
population mean cannot track it; only within-generation variance buffers fitness), the long-run
growth rate is ⟨log W̄⟩ = -½ Σ log(1+g_i a_i) - ½ Σ ω_i a_i/(1+g_i a_i), where ω_i is the optimum's
fluctuation variance in eigendirection i. The evolved standing variance g_i (hence the mutational
input m_i that sustains it) depends on the SHAPE of the fluctuation Ω:

  * Ω ISOTROPIC (ω_i = σ²):       g_i = σ² - 1/a_i   -> MORE variance in STEEP directions
                                  (the tracking/lag regime; M aligns WITH A).
  * Ω ∝ A⁻¹ (ω_i = σ²/a_i):       1 + g_i a_i = σ²  -> g_i = (σ²-1)/a_i ∝ 1/a_i
                                  -> the EXPLORATION fixed point M ∝ A⁻¹ (variance accumulates in
                                  FLAT directions, where it is cheap; the alpha=1 analog).

So the proper exploration mechanism is a fluctuating optimum whose variability is LARGER in the
directions selection constrains LEAST (Ω ∝ A⁻¹). This module evolves M to its fixed point under
each Ω structure, across an A-anisotropy sweep, and fits the log-log slope of the evolved m_i (and
the resulting standing g_i) against the curvature a_i:

    slope of log m_i† vs log a_i  ->  -1 for Ω∝A⁻¹ (M∝A⁻¹),  ~0/positive for isotropic Ω.

Compared against the fixed-isotropic-M control (no M-evolution), whose standing variance gives the
Sim-A exponent (~ -1/2 in the Gaussian/high-mu regime). CPU/JAX. Writes
runs/exploration_scaling/results.csv.
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
from sim.evolvable_m import (make_hyper, m_input_scale, init_evo_population,
                             continue_evo_sim)


def _fluctuating_theta(key, n_generations, A, env_sigma, omega_mode):
    """iid (uncorrelated) optimum θ_t with covariance Ω: isotropic σ²I or σ²·A⁻¹.
    A⁻¹ structure (more fluctuation where selection is flatter) drives the M∝A⁻¹ regime."""
    n = A.shape[0]
    z = jax.random.normal(key, (n_generations, n))
    if omega_mode == "isotropic":
        Lom = jnp.eye(n)
    elif omega_mode == "Ainv":
        w, V = jnp.linalg.eigh(A)
        Lom = V @ jnp.diag(1.0 / jnp.sqrt(w)) @ V.T          # A^{-1/2}
    else:
        raise ValueError(omega_mode)
    return env_sigma * z @ Lom.T                              # (n_generations, n), Cov = σ²Ω


def run_to_fixed_point(A, env_sigma, omega_mode, *, evolve=True, N=700, L=12, mu=0.02,
                       Lm=8, mu_mod=0.3, mut_var_mod=0.04, n_generations=2500, burn_frac=0.5,
                       n_replicates=48, M0_scale=0.06, seed=0):
    """Evolve M under fluctuating selection to its fixed point; return time-averaged evolved
    M† and standing G† (post-burn-in, replicate-averaged)."""
    n = A.shape[0]
    config = make_config(N=N, L=L, n_traits=n, mu=mu, A=jnp.asarray(A, jnp.float32), Ne=N)
    hyper = make_hyper(design="eig_diag", n_traits=n, Lm=Lm,
                       mu_mod=(mu_mod if evolve else 0.0), mut_var_mod=mut_var_mod)
    scale = m_input_scale(config, hyper)
    target_b = jnp.log(jnp.full((n,), M0_scale / scale, dtype=jnp.float32))   # isotropic M0
    k0, k1 = jax.random.split(jax.random.PRNGKey(seed))
    rep_keys = jax.random.split(k0, n_replicates)
    states0 = jax.vmap(lambda k: init_evo_population(k, config, hyper, "eig_diag", target_b))(rep_keys)
    theta_seq = _fluctuating_theta(k1, n_generations, jnp.asarray(A, jnp.float32),
                                   env_sigma, omega_mode)
    traj, _ = continue_evo_sim(states0, config, hyper, "eig_diag", theta_seq)
    burn = int(burn_frac * n_generations)
    M_star = np.asarray(traj["M"][:, burn:].mean(axis=1)).mean(axis=0)         # (n,n)
    G_star = np.asarray(traj["G"][:, burn:].mean(axis=1)).mean(axis=0)
    return M_star, G_star


def exponent_sweep(A_ratios=(2.0, 4.0, 8.0), env_sigma=2.0, omega_mode="Ainv", evolve=True,
                   geom_mean_a=2.0, **kw):
    """Across A-anisotropy ratios, pool (a_i, m_i†, g_i†) and fit the log-log slopes
    (m vs a) and (g vs a). Predict slope_m ≈ -1 for omega_mode='Ainv' (M∝A⁻¹)."""
    a_all, m_all, g_all = [], [], []
    rows = []
    for ratio in A_ratios:
        a_hi = geom_mean_a * np.sqrt(ratio); a_lo = geom_mean_a / np.sqrt(ratio)
        A = np.diag([a_lo, a_hi])
        M_star, G_star = run_to_fixed_point(A, env_sigma, omega_mode, evolve=evolve, **kw)
        a = np.diag(A); m = np.diag(M_star); g = np.diag(G_star)
        a_all += list(a); m_all += list(m); g_all += list(g)
        rows.append(dict(ratio=ratio, a_lo=a_lo, a_hi=a_hi,
                         m_lo=m[0], m_hi=m[1], g_lo=g[0], g_hi=g[1]))
        print(f"[expl] ratio={ratio:.1f} omega={omega_mode} evolve={evolve}  "
              f"a=({a_lo:.2f},{a_hi:.2f}) m†=({m[0]:.4f},{m[1]:.4f}) g†=({g[0]:.4f},{g[1]:.4f})")
    a_all = np.array(a_all); m_all = np.array(m_all); g_all = np.array(g_all)
    ok = (m_all > 1e-9) & (g_all > 1e-9)
    slope_m = np.polyfit(np.log(a_all[ok]), np.log(m_all[ok]), 1)[0]
    slope_g = np.polyfit(np.log(a_all[ok]), np.log(g_all[ok]), 1)[0]
    return dict(slope_m=float(slope_m), slope_g=float(slope_g), rows=rows,
                a=a_all.tolist(), m=m_all.tolist(), g=g_all.tolist())


def _write_csv(path, res, tag):
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["tag", "slope_m", "slope_g", "ratio", "a_lo", "a_hi",
                        "m_lo", "m_hi", "g_lo", "g_hi"])
        for r in res["rows"]:
            w.writerow([tag, res["slope_m"], res["slope_g"], r["ratio"], r["a_lo"], r["a_hi"],
                        r["m_lo"], r["m_hi"], r["g_lo"], r["g_hi"]])
    print("appended", tag, "->", path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--N", type=int, default=900)
    p.add_argument("--L", type=int, default=14)
    p.add_argument("--generations", type=int, default=3000)
    p.add_argument("--replicates", type=int, default=64)
    p.add_argument("--env-sigma", type=float, default=2.0)
    p.add_argument("--ratios", type=float, nargs="+", default=[2.0, 4.0, 8.0])
    p.add_argument("--out-dir", default=str(ROOT / "runs" / "exploration_scaling"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    out = Path(args.out_dir) / "results.csv"
    if out.exists():
        out.unlink()
    common = dict(N=args.N, L=args.L, n_generations=args.generations,
                  n_replicates=args.replicates, seed=args.seed)
    print("\n=== exploration fixed-point eigenvalue scaling (M vs A) ===")
    for tag, omega, evolve in [("Ainv_evolve", "Ainv", True),
                               ("isotropic_evolve", "isotropic", True),
                               ("Ainv_fixedM", "Ainv", False)]:
        res = exponent_sweep(A_ratios=tuple(args.ratios), env_sigma=args.env_sigma,
                             omega_mode=omega, evolve=evolve, **common)
        print(f"  >>> {tag}: slope(log m† vs log a) = {res['slope_m']:+.2f}  "
              f"slope(log g† vs log a) = {res['slope_g']:+.2f}")
        _write_csv(out, res, tag)
    print("\nPrediction: Ainv_evolve slope_m ≈ -1 (M∝A⁻¹, the alpha=1 analog); isotropic_evolve "
          "slope_m >= 0 (steep-favoring); fixedM standing-G slope ~ -1/2 (Sim-A).")


if __name__ == "__main__":
    main()

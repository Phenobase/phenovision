"""V1 — the effective potential, directly (fixed-M fitness scan). The cheapest decisive test.

On a grid of FIXED M (frozen modifiers in the evolvable-M IBM), run each population to
stationarity and measure the long-run fitness U_IBM(M) = ⟨ln W̄⟩ (time-average of mean log
fitness). The theory says ⟨ln W̄⟩ = const - ℒ_eff(M), so U_IBM should track -ℒ_eff(M) in shape
and ∇U_IBM should align with -∇ℒ_eff(M).

A-vs-B closure diagnostic (sde_validation_plan.md V1):
  version A : ℒ_eff using the ANALYTIC equilibrium Ĝ = A⁻¹#M  (tests closure + load jointly).
  version B : the load κ·tr(A·Ĝ) using the IBM's MEASURED equilibrium G at each M
              (tests only the load functional, given the correct fast state).
If B matches U_IBM but A does not, the error is in the MSB closure Ĝ=A⁻¹#M, not the
second-order-selection load functional.

Normalization: the IBM's realized per-generation input M = M_INPUT_SCALE·S_ind is placed by
encoding S_ind = M / M_INPUT_SCALE in the (frozen) modifiers; the SAME M goes into ℒ_eff.

CPU/local pilot (JAX). Writes runs/v1_fitness_scan/results.csv.
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
from sim.evolvable_m import make_hyper, m_input_scale, run_evo_sim
from sim.theory_load_surface import ghat, kappa, L_eff


def _cos(u, v):
    u = np.ravel(u); v = np.ravel(v)
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    return float(u @ v / (nu * nv)) if nu > 0 and nv > 0 else 0.0


def scan_diagonal_M(A, m1_grid, m2_grid, *, N=500, L=10, mu=0.02, n_generations=400,
                    burn_in=250, n_replicates=8, Lm=4, Ne=None, seed=0, N_star=1e6):
    """Scan a diagonal M = diag(m1, m2) over a grid; return tidy rows with U_IBM, measured G,
    and the analytic ℒ_eff (versions A and B). A must be diagonal (aligned/commuting case)."""
    config = make_config(N=N, L=L, n_traits=2, mu=mu, A=jnp.asarray(A, jnp.float32),
                         Ne=Ne if Ne is not None else N)
    hyper = make_hyper(design="eig_diag", n_traits=2, Lm=Lm, mu_mod=0.0)  # frozen M
    scale = m_input_scale(config, hyper)
    A_np = np.asarray(A, dtype=float)
    k = jax.random.PRNGKey(seed)
    rows = []
    for i, m1 in enumerate(m1_grid):
        for j, m2 in enumerate(m2_grid):
            M = np.diag([m1, m2])                       # theory M (per-generation input cov)
            S0 = np.diag(M) / scale                     # per-event eigenvalues
            target_b = jnp.log(jnp.asarray(S0, jnp.float32))   # eig_diag: b_mod = log eigs
            k, kk = jax.random.split(k)
            traj = run_evo_sim(kk, config, hyper, "eig_diag", n_generations, n_replicates,
                               target_b_mod=target_b)
            ln_w = np.asarray(traj["ln_mean_fitness"])  # (rep, gen)
            U = float(ln_w[:, burn_in:].mean())         # U_IBM = ⟨ln W̄⟩
            G_meas = np.asarray(traj["G"][:, burn_in:].mean(axis=1)).mean(axis=0)  # (2,2)
            kap = kappa(N_star)
            L_A = float(L_eff(M, A_np, N_star, "static"))                 # version A (analytic Ĝ)
            L_B = float(kap * np.trace(A_np @ G_meas))                    # version B (measured G)
            rows.append(dict(i=i, j=j, m1=m1, m2=m2, U_IBM=U,
                             G11=G_meas[0, 0], G22=G_meas[1, 1], G12=G_meas[0, 1],
                             L_eff_A=L_A, L_eff_B=L_B,
                             Ghat11=ghat(A_np, M)[0, 0], Ghat22=ghat(A_np, M)[1, 1]))
    return rows


def gradient_cosines(rows, m1_grid, m2_grid):
    """Finite-difference ∇U_IBM over the grid vs -∇ℒ_eff (versions A and B); return cosines."""
    nx, ny = len(m1_grid), len(m2_grid)
    U = np.full((nx, ny), np.nan); LA = np.full((nx, ny), np.nan); LB = np.full((nx, ny), np.nan)
    for r in rows:
        U[r["i"], r["j"]] = r["U_IBM"]; LA[r["i"], r["j"]] = r["L_eff_A"]; LB[r["i"], r["j"]] = r["L_eff_B"]
    gUx, gUy = np.gradient(U, m1_grid, m2_grid)
    gAx, gAy = np.gradient(-LA, m1_grid, m2_grid)
    gBx, gBy = np.gradient(-LB, m1_grid, m2_grid)
    gU = np.stack([gUx, gUy], -1).reshape(-1, 2)
    gA = np.stack([gAx, gAy], -1).reshape(-1, 2)
    gB = np.stack([gBx, gBy], -1).reshape(-1, 2)
    return dict(cos_gradU_vs_negGradLeff_A=_cos(gU, gA),
                cos_gradU_vs_negGradLeff_B=_cos(gB, gU),
                cos_U_vs_negLeff_A=_cos(U - U.mean(), -(LA - LA.mean())),
                cos_U_vs_negLeff_B=_cos(U - U.mean(), -(LB - LB.mean())))


def _write_csv(path, rows):
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("wrote", path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--N", type=int, default=600)
    p.add_argument("--L", type=int, default=12)
    p.add_argument("--generations", type=int, default=500)
    p.add_argument("--burn-in", type=int, default=300)
    p.add_argument("--replicates", type=int, default=12)
    p.add_argument("--grid", type=int, default=5, help="grid points per M eigenvalue")
    p.add_argument("--out-dir", default=str(ROOT / "runs" / "v1_fitness_scan"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    A = np.diag([1.0, 3.0])                              # aligned (commuting) case
    m1_grid = np.linspace(0.02, 0.12, args.grid)
    m2_grid = np.linspace(0.02, 0.12, args.grid)
    rows = scan_diagonal_M(A, m1_grid, m2_grid, N=args.N, L=args.L,
                           n_generations=args.generations, burn_in=args.burn_in,
                           n_replicates=args.replicates, seed=args.seed)
    _write_csv(Path(args.out_dir) / "results.csv", rows)
    cos = gradient_cosines(rows, m1_grid, m2_grid)
    print("[v1] gradient/shape cosines:")
    for k, v in cos.items():
        print(f"   {k} = {v:.3f}")


if __name__ == "__main__":
    main()

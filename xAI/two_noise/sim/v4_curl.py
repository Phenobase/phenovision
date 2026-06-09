"""V4 — gradient flow versus curl (the distinctive structural test of §0.9).

Assemble the measured M-drift field v(M) on a grid, metric-correct it, and Helmholtz-decompose:
the theory predicts the metric-corrected field is curl-free TO LEADING ORDER (it is -∇ℒ_eff),
with a rotational residual that is higher-order — it grows with A–G misalignment and with noise
(1/Ne, 1/N*) and VANISHES when A and G are co-diagonalized (aligned).

Coordinate / metric correction (design eig_diag, diagonal M): the natural-gradient flow gives
dM_ii = -2 m_i² ∂ℒ/∂m_i, so the Euclidean gradient component is u_i = -dM_ii / (2 m_i²). The field
(u_1, u_2) over the (m1, m2) grid should be a GRADIENT (curl-free) — small
drift_estimator.helmholtz_curl_fraction_2d.

This is the leading-order ALIGNED test (diagonal A ⇒ A,G commute ⇒ curl→0). The misalignment- and
noise-scaling of the curl (the higher-order term) is the production extension (rotated A + eig_rot
design, sweeping Ne/N*). Each grid point is a full V3 drift measurement, so the grid is coarse.
Writes runs/v4_curl/results.csv.
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

from sim.v3_drift import measure_drift
from sim.drift_estimator import helmholtz_curl_fraction_2d


def drift_field(A, m1_grid, m2_grid, *, regime="static", N=500, L=10, burn_in=300, tau=60,
                n_replicates=64, mut_var_mod=0.02, env_sigma=0.0, seed=0):
    """Metric-corrected drift field u_i = -dM_ii/(2 m_i²) on the (m1,m2) grid (eig_diag)."""
    nx, ny = len(m1_grid), len(m2_grid)
    U = np.zeros((nx, ny)); V = np.zeros((nx, ny))
    for i, m1 in enumerate(m1_grid):
        for j, m2 in enumerate(m2_grid):
            M0 = np.diag([m1, m2])
            r = measure_drift(M0, np.asarray(A, float), design="eig_diag", regime=regime,
                              N=N, L=L, burn_in=burn_in, tau=tau, n_replicates=n_replicates,
                              mut_var_mod=mut_var_mod, env_sigma=env_sigma, seed=seed + i * ny + j)
            dM = r["dM"]
            U[i, j] = -dM[0, 0] / (2 * m1 ** 2)
            V[i, j] = -dM[1, 1] / (2 * m2 ** 2)
    return U, V


def curl_fraction(A, *, regime="static", grid=(0.04, 0.07, 0.10), **kw):
    g = np.array(grid)
    U, V = drift_field(A, g, g, regime=regime, **kw)
    return helmholtz_curl_fraction_2d(g, g, U, V), U, V


def _write_csv(path, rows):
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("wrote", path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--N", type=int, default=700)
    p.add_argument("--L", type=int, default=12)
    p.add_argument("--burn-in", type=int, default=450)
    p.add_argument("--tau", type=int, default=80)
    p.add_argument("--replicates", type=int, default=128)
    p.add_argument("--grid", type=float, nargs="+", default=[0.03, 0.05, 0.07, 0.09])
    p.add_argument("--out-dir", default=str(ROOT / "runs" / "v4_curl"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    A = np.diag([1.0, 4.0])
    cf, U, V = curl_fraction(A, regime="static", grid=tuple(args.grid), N=args.N, L=args.L,
                             burn_in=args.burn_in, tau=args.tau, n_replicates=args.replicates,
                             seed=args.seed)
    print(f"[v4] aligned (diagonal A) curl fraction = {cf:.3f}  (prediction: small => gradient flow)")
    _write_csv(Path(args.out_dir) / "results.csv",
               [dict(case="aligned", curl_fraction=cf, grid=";".join(f"{g:g}" for g in args.grid))])


if __name__ == "__main__":
    main()

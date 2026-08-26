"""V5 — adiabaticity dialing. Vary the timescale-separation parameter ε (the modifier genetic
variance relative to the focal traits) and check that the reduced-SDE drift approximation improves
as ε→0: the measured M-drift aligns better with the natural-gradient theory (and, with V4's
machinery, the curl residual shrinks ∝ ε).

ε ≈ (Lm · mut_var_mod) / (L · mut_var): we dial it via mut_var_mod (modifier mutational variance)
at fixed Lm. Small ε ⇒ M evolves slowly ⇒ the fast (z̄,G) subsystem stays at its conditional
equilibrium ⇒ the adiabatic reduction (gradient flow on ℒ_eff) is accurate.

Reuses sim.v3_drift.measure_drift at each ε. CPU/local pilot; production burst sweep via the CLI.
Writes runs/v5_adiabatic/results.csv.
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


def epsilon(mut_var_mod, Lm, L, mu_focal_var=0.05 * 2):
    """Heuristic timescale-separation ratio ε = modifier genetic input / focal genetic input."""
    return (Lm * mut_var_mod) / (L * mu_focal_var)


def adiabaticity_sweep(M0, A, *, regime="exploration", mut_var_mod_grid=(0.08, 0.04, 0.02, 0.01),
                       N=600, L=12, Lm=8, burn_in=400, tau=80, n_replicates=128,
                       env_sigma=0.6, seed=0):
    rows = []
    for mvm in mut_var_mod_grid:
        r = measure_drift(M0, A, design="eig_diag", regime=regime, N=N, L=L, Lm=Lm,
                          burn_in=burn_in, tau=tau, n_replicates=n_replicates,
                          mut_var_mod=mvm, env_sigma=env_sigma, seed=seed)
        eps = epsilon(mvm, Lm, L)
        rows.append(dict(mut_var_mod=mvm, epsilon=eps,
                         cos_drift_vs_theory=r["cos_drift_vs_theory"],
                         scaling_frac=r["scaling_frac"], trM0=r["trM0"], trM_end=r["trM_end"]))
        print(f"[v5] mut_var_mod={mvm:.3f} eps={eps:.3f} cos={r['cos_drift_vs_theory']:.3f} "
              f"trM {r['trM0']:.4f}->{r['trM_end']:.4f}")
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
    p.add_argument("--N", type=int, default=900)
    p.add_argument("--L", type=int, default=14)
    p.add_argument("--Lm", type=int, default=8)
    p.add_argument("--burn-in", type=int, default=600)
    p.add_argument("--tau", type=int, default=100)
    p.add_argument("--replicates", type=int, default=160)
    p.add_argument("--regime", default="exploration")
    p.add_argument("--env-sigma", type=float, default=0.6)
    p.add_argument("--out-dir", default=str(ROOT / "runs" / "v5_adiabatic"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    A = np.diag([1.0, 4.0]); M0 = np.diag([0.06, 0.06])
    rows = adiabaticity_sweep(M0, A, regime=args.regime,
                              N=args.N, L=args.L, Lm=args.Lm, burn_in=args.burn_in,
                              tau=args.tau, n_replicates=args.replicates,
                              env_sigma=args.env_sigma, seed=args.seed)
    _write_csv(Path(args.out_dir) / "results.csv", rows)
    cos = [r["cos_drift_vs_theory"] for r in rows]
    eps = [r["epsilon"] for r in rows]
    print(f"\n[v5] cos vs eps (eps decreasing): {[f'{e:.2f}:{c:.2f}' for e, c in zip(eps, cos)]}")
    print("[v5] prediction: cos improves as eps -> 0 (better adiabatic reduction)")


if __name__ == "__main__":
    main()

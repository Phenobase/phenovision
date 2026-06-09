"""The three M-evolution regimes (+ the intermediate), measured as drift from an isotropic M0 by
the validated V3 short-τ method (fixed points run away under modifier mutational drift). Each
regime is distinguished by the SHAPE the drift heads toward, read as the evolved-M eigenvalue ratio
m_flat/m_steep against the regime's predicted target:

  canalization (static peak, no premium)      : M shrinks (both directions), shape transiently A⁻¹.
  exploration (static + diversity premium)     : M ∝ A⁻¹   -> ratio -> a_steep/a_flat (the α=1 analog).
  tracking (moving optimum, lag-load)          : M ∝ Ω     -> ratio -> ω_flat/ω_steep (movement axes).
  intermediate (premium + moving optimum)      : a COMPROMISE between A⁻¹ and Ω (the four-forces mix).

Exploration imposes the balanced net load ½c·tr(AS) - ½λ·logdet S (the IBM's emergent mutation-load
cost is too weak to balance the bet-hedging benefit, so the cost is imposed); its minimizer is
S ∝ A⁻¹. Canalization and tracking use the emergent dynamics only. CPU/JAX; burst-friendly.
Writes runs/three_regimes/results.csv.
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


def run_three_regimes(A=np.diag([1.0, 4.0]), M0=np.diag([0.06, 0.06]), *, N=800, L=12,
                      burn_in=500, tau=90, n_replicates=160, mu_mod=0.25, mut_var_mod=0.02,
                      chal_strength=1.0, chal_sigma=1.5, track_sigma=0.15, seed=0):
    a = np.diag(A)
    flat, steep = int(np.argmin(a)), int(np.argmax(a))
    Ainv_ratio = a[steep] / a[flat]          # exploration target: m_flat/m_steep -> a_steep/a_flat
    common = dict(N=N, L=L, burn_in=burn_in, tau=tau, n_replicates=n_replicates,
                  mu_mod=mu_mod, mut_var_mod=mut_var_mod, seed=seed)
    # EXPLORATION uses the GENUINE emergent bet-hedging (shared random-disaster challenge each
    # generation; the portfolio effect makes diversity protective), NOT an imposed load term.
    chal = dict(challenge_strength=chal_strength, challenge_sigma=chal_sigma)
    specs = [
        ("canalization", dict(regime="canalization")),
        ("exploration",  dict(regime="canalization", **chal)),                 # static peak + disaster
        ("tracking",     dict(regime="tracking", env_sigma=track_sigma)),
        ("intermediate", dict(regime="tracking", env_sigma=track_sigma, **chal)),
    ]
    rows = []
    for name, kw in specs:
        r = measure_drift(np.asarray(M0, float), np.asarray(A, float), **kw, **common)
        Me = np.diag(r["M_end"]); dM = np.diag(r["dM"])
        ratio = float(Me[flat] / max(Me[steep], 1e-9))
        rows.append(dict(regime=name, m_flat_end=float(Me[flat]), m_steep_end=float(Me[steep]),
                         ratio_flat_over_steep=ratio, Ainv_target=Ainv_ratio,
                         dM_flat=float(dM[flat]), dM_steep=float(dM[steep]),
                         trM0=r["trM0"], trM_end=r["trM_end"]))
        print(f"[3reg] {name:13s}: dM=({dM[flat]:+.4f} flat,{dM[steep]:+.4f} steep)  "
              f"M_end ratio(flat/steep)={ratio:.2f} (A⁻¹ target {Ainv_ratio:.1f})  "
              f"trM {r['trM0']:.3f}->{r['trM_end']:.3f}")
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
    p.add_argument("--N", type=int, default=1000)
    p.add_argument("--L", type=int, default=14)
    p.add_argument("--burn-in", type=int, default=600)
    p.add_argument("--tau", type=int, default=110)
    p.add_argument("--replicates", type=int, default=256)
    p.add_argument("--anisotropy", type=float, default=4.0)
    p.add_argument("--chal-strength", type=float, default=1.0)
    p.add_argument("--chal-sigma", type=float, default=1.5)
    p.add_argument("--track-sigma", type=float, default=0.15)
    p.add_argument("--out-dir", default=str(ROOT / "runs" / "three_regimes"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    A = np.diag([1.0, args.anisotropy]); M0 = np.diag([0.06, 0.06])
    print(f"=== three M-evolution regimes (emergent bet-hedging); A=diag(1,{args.anisotropy}) ===")
    rows = run_three_regimes(A, M0, N=args.N, L=args.L, burn_in=args.burn_in, tau=args.tau,
                             n_replicates=args.replicates, chal_strength=args.chal_strength,
                             chal_sigma=args.chal_sigma, track_sigma=args.track_sigma,
                             seed=args.seed)
    _write_csv(Path(args.out_dir) / "results.csv", rows)
    print("\nExpect: canalization shrinks; exploration ratio -> A⁻¹ target; tracking grows along the\n"
          "moving axis; intermediate sits between A⁻¹ and the movement axis (the four-forces mix).")


if __name__ == "__main__":
    main()

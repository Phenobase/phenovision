"""T1/T2 — the anisotropy sign-flip (notes_landscape_disaster_benefit.md §3.1/§3.3/§3.4).

The headline falsification. Under an ISOTROPIC disaster Ω (challenge_aniso=[1,1]) the evolved
mutational covariance M is predicted to flip which way it leans on the curvature A=diag(1,4):

  * MILD / Gaussian / large-N  -> the ARITHMETIC regime (§3.1): standing variance is maintained
    by mutation-selection balance m_i ∝ a_i (σ² large) or, when mild, the flat direction falls to
    the floor and only the steep direction keeps variance -> M ∝ A⁺  -> ratio m_flat/m_steep < 1,
    slope(log m vs log a) -> +1 (or steeper under flooring).
  * SEVERE / heavy-tailed / small-N -> the CATASTROPHE regime (§3.3): rare huge disasters dominate
    the geometric mean; survival demands no direction be left uncovered (a log-det/bet-hedge
    premium) -> M ∝ A⁻¹  -> ratio m_flat/m_steep > 1, slope -> -1.

So the SAME codepath, swept over severity (challenge_sigma) × tail (gaussian vs student_t) × N,
should make the ratio m_flat/m_steep CROSS 1. T2 is the internal falsification: the existing IBM
(soft Gaussian fecundity, σ≈1.5, N≈800) sits in the catastrophe regime (ratio≈4 ~ A⁻¹); softening
σ and raising N must flip it below 1.

Measurement reuses the validated V3 short-τ drift from an isotropic M0 (sim.v3_drift.measure_drift);
the drift DIRECTION from isotropic M0 is exactly the sign test. CPU/JAX, burst-friendly.
Writes runs/anisotropy_flip/results.csv.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.v3_drift import measure_drift


def _slope_logm_vs_loga(a, m):
    """Slope of log m vs log a (note §7.2). ratio<1 (A⁺) -> slope>0; ratio>1 (A⁻¹) -> slope<0."""
    a = np.asarray(a, float); m = np.maximum(np.asarray(m, float), 1e-12)
    return float(np.polyfit(np.log(a), np.log(m), 1)[0])


def run_flip(A=np.diag([1.0, 4.0]), M0=np.diag([0.06, 0.06]), *,
             sigmas=(0.5, 1.0, 1.5, 2.5), tails=("gaussian", "student_t"), Ns=(400, 1500),
             df=3.0, mode="fecundity", radius=0.0, strength=1.0,
             L=12, burn_in=400, tau=80, n_replicates=160, mu_mod=0.25, mut_var_mod=0.02, seed=0):
    a = np.diag(A)
    flat, steep = int(np.argmin(a)), int(np.argmax(a))
    Ainv_ratio = a[steep] / a[flat]                       # the A⁻¹ ratio (catastrophe target)
    rows = []
    for tail in tails:
        for N in Ns:
            for sigma in sigmas:
                r = measure_drift(
                    np.asarray(M0, float), np.asarray(A, float), regime="canalization",
                    N=N, L=L, burn_in=burn_in, tau=tau, n_replicates=n_replicates,
                    mu_mod=mu_mod, mut_var_mod=mut_var_mod, seed=seed,
                    challenge_strength=strength, challenge_sigma=sigma,
                    challenge_aniso=[1.0, 1.0],            # ISOTROPIC Ω — the whole point of T1
                    challenge_tail=tail, challenge_df=df,
                    challenge_mode=mode, challenge_radius=radius)
                Me = np.diag(r["M_end"])
                ratio = float(Me[flat] / max(Me[steep], 1e-9))
                slope = _slope_logm_vs_loga(a, Me)
                regime = "catastrophe(A^-1)" if ratio > 1.0 else "arithmetic(A^+)"
                rows.append(dict(tail=tail, df=df, mode=mode, N=N, sigma=sigma, strength=strength,
                                 m_flat=float(Me[flat]), m_steep=float(Me[steep]),
                                 ratio_flat_over_steep=ratio, slope_logm_vs_loga=slope,
                                 Ainv_ratio=Ainv_ratio, leans=regime,
                                 trM0=r["trM0"], trM_end=r["trM_end"],
                                 n_replicates=n_replicates, seed=seed))
                print(f"[flip] tail={tail:9s} N={N:5d} σ={sigma:4.2f} | "
                      f"M_end=({Me[flat]:.4f} flat,{Me[steep]:.4f} steep) "
                      f"ratio={ratio:5.2f} slope={slope:+5.2f}  -> {regime}")
    return rows


def _report_crossings(rows):
    """For each (tail, N) series report where ratio crosses 1 as σ increases."""
    print("\n=== ratio m_flat/m_steep crossings (arithmetic A^+ <-> catastrophe A^-1) ===")
    keyset = sorted({(r["tail"], r["N"]) for r in rows})
    for tail, N in keyset:
        series = sorted((r for r in rows if r["tail"] == tail and r["N"] == N),
                        key=lambda r: r["sigma"])
        sgn = [(r["sigma"], r["ratio_flat_over_steep"]) for r in series]
        crossed = any(s0[1] < 1.0 <= s1[1] or s1[1] < 1.0 <= s0[1]
                      for s0, s1 in zip(sgn, sgn[1:]))
        lo, hi = sgn[0][1], sgn[-1][1]
        print(f"  tail={tail:9s} N={N:5d}: ratio σ={sgn[0][0]:.2f}->{sgn[-1][0]:.2f} : "
              f"{lo:.2f} -> {hi:.2f}  {'CROSSES 1' if crossed else '(no crossing in range)'}")


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("wrote", path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sigmas", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.5])
    p.add_argument("--tails", nargs="+", default=["gaussian", "student_t"])
    p.add_argument("--Ns", type=int, nargs="+", default=[400, 1500])
    p.add_argument("--df", type=float, default=3.0, help="Student-t dof (heavier = smaller)")
    p.add_argument("--mode", default="fecundity", help="fecundity | viability")
    p.add_argument("--radius", type=float, default=0.0, help="viability survival radius (0=>sigma)")
    p.add_argument("--strength", type=float, default=1.0)
    p.add_argument("--anisotropy", type=float, default=4.0)
    p.add_argument("--L", type=int, default=12)
    p.add_argument("--burn-in", type=int, default=400)
    p.add_argument("--tau", type=int, default=80)
    p.add_argument("--replicates", type=int, default=160)
    p.add_argument("--out-dir", default=str(ROOT / "runs" / "anisotropy_flip"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    A = np.diag([1.0, args.anisotropy]); M0 = np.diag([0.06, 0.06])
    print(f"=== T1/T2 anisotropy flip; isotropic Ω; A=diag(1,{args.anisotropy}); "
          f"A⁻¹ ratio target={args.anisotropy:.0f} ===")
    rows = run_flip(A, M0, sigmas=tuple(args.sigmas), tails=tuple(args.tails), Ns=tuple(args.Ns),
                    df=args.df, mode=args.mode, radius=args.radius, strength=args.strength,
                    L=args.L, burn_in=args.burn_in, tau=args.tau, n_replicates=args.replicates,
                    seed=args.seed)
    _report_crossings(rows)
    _write_csv(Path(args.out_dir) / "results.csv", rows)
    print("\nPredict (notes §3.4): mild σ / gaussian / large N -> ratio<1 (A⁺, slope→+1);\n"
          "severe σ / student_t / small N -> ratio>1 (A⁻¹, slope→-1). The crossing is the headline.")


if __name__ == "__main__":
    main()

"""Flip diagnostic — is A⁺ ever a fixed point, or does the IBM always pull to A⁻¹?

The anisotropy_flip sweep (sim/anisotropy_flip.py) found the evolved-M SHAPE stays A⁻¹-leaning
(ratio m_flat/m_steep > 1) across the whole mild→severe range — the predicted flip to A⁺ (§3.1)
never appeared. But that sweep measures SHORT-τ DRIFT FROM ISOTROPIC M0, which at mild σ is
dominated by the canalization transient (M shrinks, faster in steep dirs since the variance load
∝ a_i, giving a transient A⁻¹ shape) — so it may not isolate the arithmetic equilibrium.

This script removes that confound with a STABILITY PROBE: start M0 AT each candidate shape
(A⁺: m∝a, isotropic, A⁻¹: m∝1/a) and ask which way the drift moves the ratio, at MILD σ / large N
and across SELECTION STRENGTH (the other "mildness" knob, held at 1.0 in the sweep).

  * If A⁺ is the arithmetic fixed point at mild/weak selection: starting from M0=A⁺ (ratio<1) the
    drift is ≈0 or toward A⁺ (ratio stays <1) — A⁺ is (weakly) stable.
  * If the catastrophe/bet-hedge pull always wins: M0=A⁺ drifts BACK toward A⁻¹ (ratio rises >1).

Reports, per (M0 shape, strength): ratio_start -> ratio_end and the drift sign. CPU/JAX.
Writes runs/flip_diagnostic/results.csv.
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

SHAPES = {"aplus": 0.25, "iso": 1.0, "ainv": 4.0}   # ratio m_flat/m_steep at M0


def _m0_from_ratio(ratio, trace=0.12):
    """M0=diag(m_flat,m_steep) with m_flat/m_steep=ratio and fixed trace (A=diag(1,4))."""
    m_steep = trace / (1.0 + ratio)
    m_flat = ratio * m_steep
    return np.diag([m_flat, m_steep])


def run_diag(A=np.diag([1.0, 4.0]), *, sigma=0.4, Ns=(2000,), strengths=(0.1, 0.3, 1.0),
             shapes=("aplus", "iso", "ainv"), tail="gaussian", L=14, burn_in=600, tau=120,
             n_replicates=160, seed=0):
    a = np.diag(A); flat, steep = int(np.argmin(a)), int(np.argmax(a))
    rows = []
    for N in Ns:
        for shape in shapes:
            M0 = _m0_from_ratio(SHAPES[shape])
            r0 = M0[flat, flat] / M0[steep, steep]
            for s in strengths:
                r = measure_drift(M0, np.asarray(A, float), regime="canalization",
                                  N=N, L=L, burn_in=burn_in, tau=tau, n_replicates=n_replicates,
                                  challenge_strength=s, challenge_sigma=sigma,
                                  challenge_aniso=[1.0, 1.0], challenge_tail=tail, seed=seed)
                Me = np.diag(r["M_end"]); r1 = float(Me[flat] / max(Me[steep], 1e-9))
                toward = "A^-1" if r1 > r0 * 1.05 else ("A^+" if r1 < r0 * 0.95 else "~stable")
                rows.append(dict(shape=shape, ratio_start=round(r0, 3), N=N, sigma=sigma,
                                 strength=s, tail=tail, m_flat=float(Me[flat]),
                                 m_steep=float(Me[steep]), ratio_end=round(r1, 3),
                                 drift_toward=toward, trM_start=float(np.trace(M0)),
                                 trM_end=r["trM_end"]))
                print(f"[diag] M0={shape:5s}(r0={r0:.2f}) N={N} s={s:<4} sigma={sigma} | "
                      f"ratio {r0:.2f} -> {r1:.2f}  drift toward {toward}")
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sigma", type=float, default=0.4)
    p.add_argument("--Ns", type=int, nargs="+", default=[2000])
    p.add_argument("--strengths", type=float, nargs="+", default=[0.1, 0.3, 1.0])
    p.add_argument("--shapes", nargs="+", default=["aplus", "iso", "ainv"])
    p.add_argument("--tail", default="gaussian")
    p.add_argument("--L", type=int, default=14)
    p.add_argument("--burn-in", type=int, default=600)
    p.add_argument("--tau", type=int, default=120)
    p.add_argument("--replicates", type=int, default=160)
    p.add_argument("--out-dir", default=str(ROOT / "runs" / "flip_diagnostic"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    print(f"=== flip stability diagnostic: does A⁺ hold at mild/weak selection? "
          f"(A=diag(1,4), sigma={args.sigma}) ===")
    rows = run_diag(sigma=args.sigma, Ns=tuple(args.Ns), strengths=tuple(args.strengths),
                    shapes=tuple(args.shapes), tail=args.tail, L=args.L, burn_in=args.burn_in,
                    tau=args.tau, n_replicates=args.replicates, seed=args.seed)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("wrote", out / "results.csv")
    print("\nKEY: if M0=aplus drifts toward A^-1 even at weak selection -> the catastrophe pull\n"
          "always wins (no arithmetic regime in this IBM); if it holds (~stable / toward A^+) ->\n"
          "the sweep's no-flip was the canalization-transient confound, not a falsification.")


if __name__ == "__main__":
    main()

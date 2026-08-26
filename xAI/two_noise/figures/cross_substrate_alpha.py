"""T6 — the cross-substrate α*(noise) overlay (the unifying deliverable).

Puts ONE prediction in three substrates on a common "effective sample size" (less-noise →) axis,
y = realized preconditioner/mutational exponent α (G ∝ A^{−α}):

  * SDE (analytic)      : α*(S) from toy/quadratic_sde — the shared theory curve.
  * Optimizer (ML)      : StableEvolutionSOAP's REALIZED mean_exponent vs effective batch
                          (runs/alpha_vs_batch/*_stable_evo.csv) — the selection-driven α*(batch).
  * Biology (IBM)       : the evolved-M exponent α_evolved = −slope(log m vs log a) vs an effective
                          sample size mapped from disaster severity/N (runs/anisotropy_flip) — the
                          T1/T2 flip: mild/large-N → low α (A⁺), severe/heavy-tail/small-N → α→1 (A⁻¹).

Reuses figures/export_csv.py to (re)assemble the tidy panel CSVs, then draws a single matplotlib
overlay to figures/cross_substrate_alpha.png. Degrades gracefully: panels whose source runs do not
exist yet are skipped, so the analytic curve (and any completed panel) renders before the expensive
sweeps finish. Re-run after the SLURM jobs land to fill in the optimizer + biology arms.

Run:  python3 -m figures.cross_substrate_alpha
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from figures import export_csv

OUT = ROOT / "figures"
PANELS = ROOT / "runs" / "figures"


def _bio_eff_sample(df: pd.DataFrame) -> pd.Series:
    """Heuristic, MONOTONE map from disaster severity/N to a shared effective-sample axis: more
    individuals and milder disasters => more effective coverage => less noise. eff ≈ N / σ²
    (documented heuristic, matching export_csv.sim_panel's env_amp→N* style)."""
    return df["N"] / (df["sigma"] ** 2).clip(lower=1e-6)


def build_figure(outpath: Path = OUT / "cross_substrate_alpha.png"):
    export_csv.build()                                   # (re)emit the panel CSVs
    fig, ax = plt.subplots(figsize=(7.2, 5.0))

    # --- SDE analytic curve (always available) ---
    sde_f = PANELS / "sde_curve.csv"
    if sde_f.exists():
        sde = pd.read_csv(sde_f)
        ax.plot(sde["eff_sample_size"], sde["alpha_star"], "-", color="black", lw=2,
                label="SDE α*(S) (analytic)", zorder=3)

    # --- Optimizer arm: StableEvolutionSOAP realized exponent vs batch ---
    se_f = PANELS / "stable_evo_panel.csv"
    if se_f.exists():
        se = pd.read_csv(se_f)
        ax.plot(se["batch_size"], se["mean_exponent"], "o-", color="C0", ms=7,
                label="optimizer: stable_evo realized α (vs batch)", zorder=4)
    else:
        print("[note] stable_evo panel absent -> optimizer arm skipped")

    # --- Biology arm: evolved-M exponent vs effective sample (severity/N), by tail ---
    flip_f = PANELS / "anisotropy_flip_panel.csv"
    if flip_f.exists():
        flip = pd.read_csv(flip_f)
        flip = flip.copy(); flip["eff_sample"] = _bio_eff_sample(flip)
        for tail, sub in flip.groupby("tail"):
            sub = sub.sort_values("eff_sample")
            ax.plot(sub["eff_sample"], sub["alpha_evolved"], "s--", ms=6, alpha=0.85,
                    label=f"biology IBM: evolved α ({tail})", zorder=2)
        ax.axhline(1.0, color="grey", ls=":", lw=1)      # A⁻¹ (bet-hedge) line
        ax.axhline(0.0, color="grey", ls=":", lw=1)      # A⁺ / SGD line
    else:
        print("[note] anisotropy_flip panel absent -> biology arm skipped")

    ax.set_xscale("log")
    ax.set_xlabel("effective sample size  (← more noise   |   less noise →)")
    ax.set_ylabel("realized exponent α   (G ∝ A^(−α);  ½=whiten, 1=inverse/A⁻¹)")
    ax.set_title("Cross-substrate α*(noise): SDE / optimizer / evolution on one axis")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    print("wrote", outpath)
    return outpath


if __name__ == "__main__":
    build_figure()

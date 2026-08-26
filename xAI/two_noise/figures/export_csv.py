"""§4/§5 — assemble the shared-prediction figure's tidy CSVs (Python emits, R draws).

The spine figure puts ONE prediction in two substrates on a common "effective sample size" axis:
 - LEFT  (ML):  alpha* vs effective batch S        (from ml_experiments/alpha_vs_batch)
 - RIGHT (bio): M-anisotropy vs N*                 (from sim/sim_b_phase)
 - OVERLAY:     the analytic SDE alpha* curve       (from toy/quadratic_sde, available now)

This module writes three CSVs into runs/figures/ that xAI/R/two_noise_figures.R consumes. It
degrades gracefully: panels whose source runs don't exist yet are skipped with a note, so the
analytic curve (and any completed panel) can be drawn before the expensive runs finish.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RUNS = ROOT / "runs"
OUT = RUNS / "figures"


def analytic_sde_curve(cond_number=1e3, dim=24,
                       S_grid=np.logspace(0.3, 4, 24)) -> pd.DataFrame:
    """The SDE prediction alpha*(S) overlaid on both panels (the shared theory curve)."""
    from toy.quadratic_sde import make_quadratic, analytic_optimal_alpha
    quad = make_quadratic(dim=dim, cond_number=cond_number, seed=0, rotate=False, center=True)
    astar, _ = analytic_optimal_alpha(quad, S_grid, n_steps=1500, test_curvature="isotropic")
    return pd.DataFrame({"eff_sample_size": S_grid, "alpha_star": np.atleast_1d(astar),
                         "source": "SDE (analytic)"})


def optimizer_panel() -> pd.DataFrame | None:
    """ML optimizer panel: the FULL-INVERSE PENALTY vs batch size.

    The argmin alpha* is flat at 0.5 across batches (whitening robustly best at this scale), so it
    hides the law. The signal is in the GAP: val_loss(alpha=1) - val_loss(alpha=0.5) per batch,
    which shrinks monotonically as batch grows (full inverse becomes more favorable as gradient
    noise falls) — the noise-dependent-alpha* prediction in the right direction. Returns one row
    per batch with the gap (and the per-alpha val_losses for reference)."""
    d = RUNS / "alpha_vs_batch"
    if not d.exists():
        return None
    # only the fixed-alpha SOAP grid feeds the alpha-pivot; the stable_evo overlay (single alpha,
    # realized mean_exponent) is a SEPARATE panel (stable_evo_panel) so it can't pollute the pivot.
    frames = [pd.read_csv(f) for f in d.glob("*.csv") if not f.name.endswith("_stable_evo.csv")]
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    if "optimizer" in df.columns:
        df = df[df["optimizer"] == "soap"]
    piv = df.pivot_table(index="batch_size", columns="alpha", values="val_loss", aggfunc="first")
    if 1.0 not in piv.columns or 0.5 not in piv.columns:
        return df  # fall back to raw if the needed alphas are absent
    out = pd.DataFrame({
        "batch_size": piv.index,
        "full_inverse_penalty": (piv[1.0] - piv[0.5]).values,  # val_loss(a=1) - val_loss(a=0.5)
        "val_loss_a05": piv[0.5].values,
        "val_loss_a10": piv[1.0].values,
    }).reset_index(drop=True)
    return out


def sim_panel(N_for_neff: int = 600) -> pd.DataFrame | None:
    """Biology panel: G-anisotropy vs effective N*. Prefer the clean incoherent compression sweep
    (sim_b_compression: errors-in-variables noise, no lag-load); fall back to the spatial Sim B.

    The incoherent sweep is parameterized by env_amp (gradient-noise magnitude). We map it to an
    effective N* via the C-inflation equivalence (environmental variance inflates the gradient
    covariance C, equivalent to lowering N* in the C/N* term): N*_eff = N / (1 + env_amp^2) — a
    heuristic, monotone mapping so the x-axis is shared with the optimizer panel (larger eff sample
    size = less noise). Anisotropy compresses (falls) as N*_eff falls."""
    f = RUNS / "sim_b_compression" / "results.csv"
    if f.exists():
        df = pd.read_csv(f)
        df = df.copy()
        df["eff_N_star"] = N_for_neff / (1.0 + df["env_amp"] ** 2)
        return df
    f2 = RUNS / "sim_b" / "results.csv"
    return pd.read_csv(f2) if f2.exists() else None


def alpha1_panel() -> pd.DataFrame | None:
    """The at-scale import panel: full-inverse (alpha=1) ViT stability by condition."""
    f = RUNS / "alpha1_stability" / "results.csv"
    return pd.read_csv(f) if f.exists() else None


def stable_evo_panel() -> pd.DataFrame | None:
    """StableEvolutionSOAP overlay: the REALIZED operative exponent (mean_exponent) vs batch size
    — the selection-driven optimizer's own alpha*(batch), which should rise with batch (the
    cross-substrate prediction, realized by the optimizer for free)."""
    d = RUNS / "alpha_vs_batch"
    if not d.exists():
        return None
    frames = [pd.read_csv(f) for f in d.glob("*_stable_evo.csv")]
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    keep = [c for c in ("model", "dataset", "batch_size", "mean_exponent", "val_loss") if c in df.columns]
    return df[keep].sort_values("batch_size").reset_index(drop=True)


def anisotropy_flip_panel() -> pd.DataFrame | None:
    """Biology severity panel (T1/T2): the evolved-M exponent vs disaster severity / N / tail.
    Maps the measured ratio to the shared y-axis 'realized alpha' via M∝A^{−α}  =>
    slope(log m vs log a) = −α, so alpha_evolved = −slope_logm_vs_loga. ratio>1 (A⁻¹) -> alpha→1
    (catastrophe/hedge); ratio<1 (A⁺) -> alpha<0 (arithmetic/track)."""
    f = RUNS / "anisotropy_flip" / "results.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f).copy()
    df["alpha_evolved"] = -df["slope_logm_vs_loga"]
    return df


def build(outdir: Path = OUT):
    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    sde = analytic_sde_curve()
    sde.to_csv(outdir / "sde_curve.csv", index=False); written.append("sde_curve.csv")
    opt = optimizer_panel()
    if opt is not None:
        opt.to_csv(outdir / "optimizer_panel.csv", index=False); written.append("optimizer_panel.csv")
    else:
        print("[note] no alpha_vs_batch runs yet -> optimizer panel skipped")
    sim = sim_panel()
    if sim is not None:
        sim.to_csv(outdir / "sim_panel.csv", index=False); written.append("sim_panel.csv")
    else:
        print("[note] no sim_b results yet -> sim panel skipped")
    a1 = alpha1_panel()
    if a1 is not None:
        a1.to_csv(outdir / "alpha1_panel.csv", index=False); written.append("alpha1_panel.csv")
    else:
        print("[note] no alpha1_stability results yet -> alpha1 panel skipped")
    se = stable_evo_panel()
    if se is not None:
        se.to_csv(outdir / "stable_evo_panel.csv", index=False); written.append("stable_evo_panel.csv")
    else:
        print("[note] no stable_evo run yet -> stable_evo panel skipped")
    flip = anisotropy_flip_panel()
    if flip is not None:
        flip.to_csv(outdir / "anisotropy_flip_panel.csv", index=False); written.append("anisotropy_flip_panel.csv")
    else:
        print("[note] no anisotropy_flip results yet -> flip panel skipped")
    print("wrote:", ", ".join(written), "to", outdir)
    return written


if __name__ == "__main__":
    build()

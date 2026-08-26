"""Sim A — the G–A eigenvalue exponent and the per-locus regime test (analysis_plan.md §3.1).

This is the Strand-2 LEAD result. It drives the JAB-style IBM engine (`sim/engine.py`) to
mutation-selection-drift balance under ANISOTROPIC stabilizing selection, then measures the
log-log slope of G's eigenvalues against A's eigenvalues. The negative of that slope is the
G–A exponent  α  in  G ∝ A^(-α)  (CONVENTIONS.md §3).

============================================================================================
 THE SCIENTIFIC CLAIM (two_noise_framework_summary.md §7, analysis_plan.md §3.1)
============================================================================================
  G ∝ A^(-α),  with α = (γ+1)/2  when  M ∝ A^(-γ)  under GAUSSIAN closure.

  Predictions discriminated here:
    * Zhang–Hill regime  (LOW per-locus mutation rate μ): α ≈ 1  (G ∝ A^(-1)) EVEN with
      isotropic M (M_isotropic arm), via rare-allele cancellation — residence time ∝ 1/s ∝ V_s
      cancels input rate ∝ μ, so per-locus contribution ∝ V_s ∝ a^(-1) for ANY M.
    * Gaussian regime   (HIGH μ) with isotropic M: α ≈ 1/2  (G ∝ A^(-1/2)) — the
      Bulmer–Lande–Week square-root baseline (γ=0 ⇒ α=1/2).
    * M_free arm (σ_eps>0, M can evolve toward A^(-1)):  α → 1 in BOTH regimes.

============================================================================================
 EXPONENT-ESTIMATION METHOD  (the scientific core — be rigorous)
============================================================================================
  n_traits = 2 gives only TWO eigenvalues per run, so a single run cannot fit a power law.
  We therefore SWEEP the A-anisotropy ratio (a1/a2 ∈ {1.5, 3, 6, 12, ...}) at fixed geometric
  mean (det(A)=1 ⇒ a1·a2=1), giving a wide spread of A eigenvalues across runs. We then POOL
  every (log a_eig, log g_eig) point — both axes, all runs at a given (arm, μ) — and fit ONE
  ordinary-least-squares line. The fitted slope is  -α  (so α = -slope).

  Why this is the right estimator:
    * Because the additive map and the isotropic mutational input do not couple traits and A
      is diagonal, G stays (to within sampling noise) DIAGONAL and shares A's eigenbasis
      (verified: off-diagonal correlation ~3e-4). So the per-axis eigenvalue pairing
      (a_axis_k -> g_axis_k) is unambiguous and we do not need a Procrustes/CPC matching.
    * Pooling across the anisotropy sweep turns the 2-point-per-run problem into a regression
      over a wide curvature range, which is exactly what makes the slope identifiable and
      robust (the exponent is more robust than the magnitude — see the engine's ~factor-2 MSB
      caveat). We pair eigenvalues by SORTED order within each run (descending), which equals
      the axis pairing for diagonal A/G but is also robust to small eigenvector rotation.

  We report the slope with a bootstrap-over-replicates CI so the regime discrimination has an
  uncertainty attached.

Run (CPU dev):
    JAX_PLATFORMS=cpu mamba run -n two_noise_jax python -m sim.sim_a_exponent
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # dev default; SLURM script overrides for GPU

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from sim.engine import make_config, run_sim


# ============================================================================================
# Building an anisotropic A with a controlled eigenvalue ratio and fixed geometric mean
# ============================================================================================

def anisotropic_A(eigratio: float, geom_mean: float = 1.0) -> np.ndarray:
    """Diagonal 2x2 curvature A = diag(a1, a2) with a1/a2 = eigratio and sqrt(a1*a2) = geom_mean.

    Holding the geometric mean fixed keeps the OVERALL selection strength comparable across the
    anisotropy sweep, so only the eigenvalue SPREAD (the thing the exponent is about) changes.
    a1 = geom_mean * sqrt(eigratio),  a2 = geom_mean / sqrt(eigratio)  ⇒  a1/a2 = eigratio,
    a1*a2 = geom_mean^2.
    """
    s = float(np.sqrt(eigratio))
    a1 = geom_mean * s
    a2 = geom_mean / s
    return np.array([[a1, 0.0], [0.0, a2]], dtype=np.float64)


# ============================================================================================
# One condition: run the engine to MSB balance and read off G & A eigenvalues
# ============================================================================================

def run_condition(
    key,
    A_eigratio: float,
    mu: float,
    arm: str,
    *,
    N: int = 1000,
    L: int = 15,
    mut_var: float = 0.05,
    r: float = 0.5,
    Ne: int | None = None,
    geom_mean: float = 1.0,
    sigma_eps_free: float = 0.3,
    n_generations: int = 400,
    burn_in: int = 250,
    n_replicates: int = 16,
) -> dict:
    """Run ONE (arm, μ, A_eigratio) condition to mutation-selection-drift balance.

    Builds an ANISOTROPIC diagonal A with the given eigenvalue ratio (geometric mean fixed,
    default 1), runs the engine, averages G over replicates and over the post-burn-in
    generations, and returns G, A, and the per-axis (sorted-descending) eigenvalues.

    arm:
      * "M_isotropic": σ_eps = 0  ⇒ M fixed isotropic (the BLW / Zhang–Hill regime test).
      * "M_free":      σ_eps > 0  ⇒ multilinear epistasis lets the realized M deviate from
                       isotropy (modifier-locus / M-evolution arm). Predicts α → 1.

    Returns dict with:
      arm, mu, A_eigratio, A (2x2), G (2x2, replicate+time averaged),
      a_eigs (descending), g_eigs (descending, paired by sort order to a_eigs),
      G_reps (n_replicates x 2x2, per-replicate time-averaged G — used for bootstrap CIs).
    """
    if arm == "M_isotropic":
        sigma_eps = 0.0
    elif arm == "M_free":
        sigma_eps = sigma_eps_free
    else:
        raise ValueError(f"arm must be 'M_isotropic' or 'M_free', got {arm!r}")

    A_np = anisotropic_A(A_eigratio, geom_mean=geom_mean)
    cfg = make_config(
        N=N, L=L, n_traits=2, mu=mu, mut_var=mut_var, r=r,
        sigma_eps=sigma_eps, A=jnp.asarray(A_np, dtype=jnp.float32),
        theta=jnp.zeros(2), Ne=Ne,
    )

    traj = run_sim(key, cfg, n_generations=n_generations, n_replicates=n_replicates)
    G = np.array(traj["G"])                       # (rep, gen, 2, 2)

    # per-replicate, time-averaged G over the post-burn-in window (MSB plateau)
    G_reps = G[:, burn_in:, :, :].mean(axis=1)    # (rep, 2, 2)
    G_mean = G_reps.mean(axis=0)                   # (2, 2) pooled estimate

    # A eigenvalues (descending) — A is diagonal so these are just the sorted diagonal.
    a_eigs = np.sort(np.linalg.eigvalsh(A_np))[::-1]
    # Pairing rule: the SHARP A axis (large a) carries the SMALL realized G, so we must
    # associate the LARGEST a with the SMALLEST g. We therefore pair a DESCENDING with g
    # ASCENDING (axis-wise for diagonal A/G; robust to small eigenvector rotation). This gives
    # the (large a, small g) points whose log-log slope is negative, with slope = -α.
    g_eigs_paired = np.sort(np.linalg.eigvalsh(G_mean))  # ascending

    return {
        "arm": arm,
        "mu": mu,
        "A_eigratio": A_eigratio,
        "A": A_np,
        "G": G_mean,
        "a_eigs": a_eigs,                 # descending: [a_large, a_small]
        "g_eigs": g_eigs_paired,          # ascending:  [g_small, g_large]  (paired to a_eigs)
        "G_reps": G_reps,                 # (rep, 2, 2)
    }


# ============================================================================================
# Fitting the G–A exponent across the anisotropy sweep
# ============================================================================================

def fit_GA_exponent(G_eigs: Sequence[float], A_eigs: Sequence[float]) -> float:
    """Fit α in  G ∝ A^(-α)  by OLS of log(G_eig) on log(A_eig); returns α = -slope.

    G_eigs, A_eigs are PAIRED, POOLED arrays of eigenvalues across all runs/axes in the sweep
    (each A eigenvalue with the G eigenvalue realized along the SAME axis). The OLS slope of
    log G on log A estimates -α; we return α (positive in [~0.5, ~1] for the predicted
    regimes). Requires ≥2 distinct A eigenvalues.
    """
    logA = np.log(np.asarray(A_eigs, dtype=np.float64))
    logG = np.log(np.asarray(G_eigs, dtype=np.float64))
    mask = np.isfinite(logA) & np.isfinite(logG)
    logA, logG = logA[mask], logG[mask]
    if logA.size < 2 or np.allclose(logA, logA[0]):
        return float("nan")
    slope = np.polyfit(logA, logG, 1)[0]
    return float(-slope)


def _pool_eig_points(conditions: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Pool (a_eig, g_eig) pairs across a list of run_condition results (same arm & μ)."""
    A_pts, G_pts = [], []
    for c in conditions:
        # a_eigs descending, g_eigs ascending -> paired axis-wise (large a <-> small g).
        A_pts.extend(list(c["a_eigs"]))
        G_pts.extend(list(c["g_eigs"]))
    return np.asarray(A_pts), np.asarray(G_pts)


def _bootstrap_exponent_ci(conditions: list[dict], n_boot: int = 500,
                           seed: int = 0) -> tuple[float, float, float]:
    """Bootstrap the fitted α over REPLICATES (resample replicates within each condition).

    Returns (alpha_point, alpha_lo, alpha_hi) for a 95% percentile interval. Resampling
    replicates (not eigen-points) propagates the dominant source of noise — finite-N drift
    across replicates — into the slope CI.
    """
    rng = np.random.default_rng(seed)

    # point estimate from the full pooled fit
    A_all, G_all = _pool_eig_points(conditions)
    alpha_point = fit_GA_exponent(G_all, A_all)

    boots = []
    for _ in range(n_boot):
        A_pts, G_pts = [], []
        for c in conditions:
            reps = c["G_reps"]                      # (rep, 2, 2)
            nrep = reps.shape[0]
            idx = rng.integers(0, nrep, size=nrep)  # resample replicates with replacement
            G_mean = reps[idx].mean(axis=0)
            a_eigs = np.sort(np.linalg.eigvalsh(c["A"]))[::-1]  # descending
            g_eigs = np.sort(np.linalg.eigvalsh(G_mean))        # ascending (paired)
            A_pts.extend(list(a_eigs))
            G_pts.extend(list(g_eigs))
        a = fit_GA_exponent(G_pts, A_pts)
        if np.isfinite(a):
            boots.append(a)
    if not boots:
        return alpha_point, float("nan"), float("nan")
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return alpha_point, float(lo), float(hi)


# ============================================================================================
# Full sweep: arms × μ × A_eigratio  ->  tidy DataFrame + fitted-slope summary
# ============================================================================================

def run_sim_a(
    *,
    arms: Sequence[str] = ("M_isotropic", "M_free"),
    mus: Sequence[float] = (3e-4, 5e-2),          # (Zhang–Hill low μ, Gaussian high μ)
    A_eigratios: Sequence[float] = (1.5, 3.0, 6.0, 12.0),
    seed: int = 0,
    n_boot: int = 500,
    write_csv: bool = True,
    out_dir: str | Path | None = None,
    **condition_kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full Sim A sweep and return (tidy results, slope summary).

    tidy results: one row per (arm, μ, A_eigratio, axis) with columns
        arm, mu, A_eigratio, axis, a_eig, g_eig, g_eig_sd  (sd across replicates).
    slope summary: one row per (arm, μ) with columns
        arm, mu, alpha, alpha_lo, alpha_hi, n_points, predicted_alpha, regime.

    Writes the tidy results to runs/sim_a/results.csv (created) and the summary to
    runs/sim_a/slopes.csv unless write_csv=False. Figures are drawn in R later — emit CSV only.
    """
    key = jax.random.PRNGKey(seed)
    tidy_rows = []
    summary_rows = []

    for arm in arms:
        for mu in mus:
            conditions = []
            for ratio in A_eigratios:
                key, sub = jax.random.split(key)
                c = run_condition(sub, A_eigratio=ratio, mu=mu, arm=arm, **condition_kwargs)
                conditions.append(c)

                # tidy rows: pair a (descending) with g (ascending) axis-wise.
                a_eigs = c["a_eigs"]                          # descending
                g_eigs = c["g_eigs"]                          # ascending (paired)
                # per-replicate eigenvalues for an SD on g (ascending order to stay paired)
                g_rep_eigs = np.sort(
                    np.linalg.eigvalsh(c["G_reps"]), axis=1
                )                                            # (rep, 2) ascending
                g_sd = g_rep_eigs.std(axis=0)                 # (2,)
                for axis in range(2):
                    tidy_rows.append({
                        "arm": arm,
                        "mu": mu,
                        "A_eigratio": ratio,
                        "axis": axis,                        # 0 = sharp-A/small-G, 1 = shallow
                        "a_eig": float(a_eigs[axis]),
                        "g_eig": float(g_eigs[axis]),
                        "g_eig_sd": float(g_sd[axis]),
                    })

            alpha, lo, hi = _bootstrap_exponent_ci(conditions, n_boot=n_boot, seed=seed)
            A_all, _ = _pool_eig_points(conditions)
            # predicted α & regime label
            predicted, regime = _predict_alpha(arm, mu, mus)
            summary_rows.append({
                "arm": arm,
                "mu": mu,
                "alpha": alpha,
                "alpha_lo": lo,
                "alpha_hi": hi,
                "n_points": int(A_all.size),
                "predicted_alpha": predicted,
                "regime": regime,
            })

    tidy = pd.DataFrame(tidy_rows)
    summary = pd.DataFrame(summary_rows)

    if write_csv:
        out = Path(out_dir) if out_dir is not None else _default_out_dir()
        out.mkdir(parents=True, exist_ok=True)
        tidy.to_csv(out / "results.csv", index=False)
        summary.to_csv(out / "slopes.csv", index=False)

    return tidy, summary


def _predict_alpha(arm: str, mu: float, mus: Sequence[float]) -> tuple[float, str]:
    """Predicted α and a regime label for a condition (framework §7 / plan §3.1)."""
    mu_sorted = sorted(set(mus))
    is_low = mu <= mu_sorted[0]
    if arm == "M_free":
        return 1.0, "M_free (-> alpha~1 both regimes)"
    # M_isotropic
    if is_low:
        return 1.0, "Zhang-Hill (low mu, isotropic M -> alpha~1)"
    return 0.5, "Gaussian/BLW (high mu, isotropic M -> alpha~1/2)"


def _default_out_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "runs" / "sim_a"


# ============================================================================================
# CLI: run a representative sweep and print the slope summary
# ============================================================================================

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Sim A — G–A exponent sweep")
    p.add_argument("--N", type=int, default=1000)
    p.add_argument("--L", type=int, default=15)
    p.add_argument("--generations", type=int, default=400)
    p.add_argument("--burn-in", type=int, default=250)
    p.add_argument("--replicates", type=int, default=16)
    p.add_argument("--n-boot", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--low-mu", type=float, default=3e-4)
    p.add_argument("--high-mu", type=float, default=5e-2)
    args = p.parse_args()

    tidy, summary = run_sim_a(
        mus=(args.low_mu, args.high_mu),
        seed=args.seed,
        n_boot=args.n_boot,
        N=args.N, L=args.L,
        n_generations=args.generations,
        burn_in=args.burn_in,
        n_replicates=args.replicates,
    )
    pd.set_option("display.width", 140)
    print("\n=== Sim A: fitted G–A exponent alpha (G ∝ A^-alpha) ===")
    print(summary.to_string(index=False))
    print(f"\nTidy results: {len(tidy)} rows -> {_default_out_dir() / 'results.csv'}")

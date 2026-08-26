"""Sim B — the N* phase diagram and two-signature discrimination (analysis_plan.md §3.2).

This is build-order step §3.2 / framework extensions #1,#2,#3 (two_noise_framework_summary.md
§11). It adds the SECOND noise source — STRUCTURED within-generation GRADIENT noise — to the
JAB-style IBM engine (`sim/engine.py`) and maps how it reshapes the mutational architecture M
across the (N*, Ne) plane.

============================================================================================
 THE MECHANISM (analysis_plan.md §3.2, framework §1, §2, §5.5, §6 Force 2)
============================================================================================
  Each individual i sees a PERTURBED optimum  θ_i = θ + δ_i  within a generation: it lives in
  a slightly different environmental realization and so "sees" a slightly different fitness
  peak. The perturbations δ_i are NOT independent — they carry a SPATIAL CORRELATION set by a
  length ℓ on a 1-D lattice of individuals (limited dispersal / philopatry: neighbours share
  microenvironment). The population fitness gradient is averaged across individuals, and that
  average has residual covariance C/N* by the CLT, with the EFFECTIVE sample size given by the
  Kish correction (framework §"Parameters, limits, and FDT"):

        N* ≈ N / ( 1 + (N-1) ρ̄ ),   ρ̄ = mean pairwise correlation of the δ_i.

  Independent δ (ℓ→0)  ⇒ ρ̄→0 ⇒ N*≈N (gradient noise averages away, classical Lande limit).
  Strongly correlated δ (ℓ→∞) ⇒ ρ̄→1 ⇒ N*→1 (the colony/philopatry regime: one shared
  environment, the population gradient is a sample of size ~1).

  We use an EXPONENTIAL spatial kernel  K_ij = exp(-|i-j|/ℓ)  (CONVENTIONS-compatible; the
  plan permits 1-D or 2-D lattice — 1-D is used, it is the cleanest knob). δ for each trait is
  a zero-mean Gaussian field with covariance  env_var · K, drawn via a stable spectral square
  root of K (banded-Toeplitz; cheap). ρ̄ = mean off-diagonal of K, so ρ̄ and hence N* depend
  ONLY on ℓ (and N), independent of env_var — env_var sets the gradient-noise MAGNITUDE, ℓ
  sets its effective sample size. We sweep N* (via ℓ) and Ne INDEPENDENTLY.

============================================================================================
 THE TWO-SIGNATURE PREDICTION (the science — analysis_plan.md §3.2, framework §5.5 end, §6)
============================================================================================
  (i)  M EIGENVALUE ANISOTROPY (max/min) vs N*  → COMPRESSION toward isotropy as N* falls.
       Gradient noise is an errors-in-variables / regression-DILUTION attenuation bias: a
       noisy per-individual optimum blurs the selection signal the population responds to, so
       the realized M anisotropy (which under epistasis tracks A's anisotropy, Force 1) is
       ATTENUATED. Smaller N* ⇒ more shared noise ⇒ stronger attenuation ⇒ M → isotropic.
  (ii) M EIGENVECTOR ALIGNMENT ANGLE to A's axes vs N* → UNCHANGED / FLAT, NO eigenvalue-order
       inversion. (The explicit correction to the retracted inversion claim, framework
       "executive summary": "anisotropy compresses while eigenvector alignment holds".)
  (iii) ACROSS-REPLICATE VARIANCE of M's structure vs Ne → INFLATION with smaller Ne. This is
       the DEMOGRAPHIC-noise (Langevin) signature: variance inflation around an UNBIASED mean,
       structurally distinct from gradient noise's eigenvalue BIAS. The two fingerprints
       separate in the (Ne, N*) plane.

============================================================================================
 WHY THIS DOES NOT EDIT engine.py
============================================================================================
  The engine's `step_generation` applies one SHARED optimum `config.theta` via `fitness()`.
  Per-individual θ_i is a Sim-B-specific mechanism, so we COMPOSE engine primitives
  (genotype_to_phenotype, _segregate_gametes, _mutate_gametes, init_population, measure) into
  a perturbed-optimum generation step HERE, leaving engine.py untouched (per task constraint).
  The only change versus the engine's step is the fitness line:
        W_i = exp(-½ (z_i - θ_i)ᵀ A (z_i - θ_i))     [per-individual θ_i]
  everything else (finite-Ne pool, fitness-weighted sampling, segregation+recombination,
  mutation) is identical to engine.step_generation. M is measured with the engine's `measure`.

  M-anisotropy requires M to be background-dependent, so Sim B runs the EPISTASIS arm
  (sigma_eps > 0): with sigma_eps = 0 the engine's M is fixed isotropic (mut_var/n)·I and
  cannot compress. sigma_eps > 0 lets selection shape the realized M toward A^(-1) (Force 1),
  giving an anisotropy that gradient noise can then attenuate (Force 2). See engine.py header
  "THE G vs M DISTINCTION" and `measure`'s M estimator.

============================================================================================
 RNG DISCIPLINE (CONVENTIONS.md §5): explicit jax.random.PRNGKey split lattice, never global.
============================================================================================

Run (CPU dev — Sim B is CPU-only; GPU is forbidden on the burst QOS used for the full run):
    JAX_PLATFORMS=cpu mamba run -n two_noise_jax python -m sim.sim_b_phase            # pilot
    JAX_PLATFORMS=cpu mamba run -n two_noise_jax python -m sim.sim_b_phase --cell 7   # one cell
"""
from __future__ import annotations

import json
import os
from functools import partial
from pathlib import Path
from typing import Sequence

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # dev default; the SLURM script also sets it.

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import lax

from sim.engine import (
    Config,
    State,
    genotype_to_phenotype,
    init_population,
    make_config,
    measure,
    _mutate_gametes,
    _segregate_gametes,
)


# ============================================================================================
# Spatial correlation kernel and the Kish N* operationalization
# ============================================================================================

def exponential_kernel_1d(N: int, ell: float) -> np.ndarray:
    """Exponential spatial correlation kernel on a 1-D lattice of N individuals.

        K_ij = exp(-|i-j| / ell),     K_ii = 1.

    ell = 0  ⇒  K = I (independent environments). Larger ell ⇒ longer-range correlation.
    This is the (banded-Toeplitz) correlation structure of the per-individual environmental
    perturbations δ_i; ρ̄ (and hence N*) is a deterministic function of K alone.
    """
    pos = np.arange(N)
    D = np.abs(pos[:, None] - pos[None, :]).astype(np.float64)
    if ell <= 0:
        return np.eye(N)
    return np.exp(-D / float(ell))


def rho_bar_of_kernel(K: np.ndarray) -> float:
    """Mean pairwise (off-diagonal) correlation ρ̄ of a correlation matrix K (unit diagonal)."""
    N = K.shape[0]
    if N < 2:
        return 0.0
    off_sum = K.sum() - np.trace(K)
    return float(off_sum / (N * (N - 1)))


def kish_n_star(N: int, rho_bar: float) -> float:
    """Kish effective sample size  N* = N / (1 + (N-1) ρ̄)  (framework §Parameters/FDT).

    ρ̄ = 0 ⇒ N* = N (independent); ρ̄ = 1 ⇒ N* = 1 (one shared environment).
    """
    return float(N / (1.0 + (N - 1) * rho_bar))


def n_star_of_ell(N: int, ell: float) -> tuple[float, float]:
    """(N*, ρ̄) realized by an exponential 1-D kernel of length ell on N individuals."""
    K = exponential_kernel_1d(N, ell)
    rb = rho_bar_of_kernel(K)
    return kish_n_star(N, rb), rb


def solve_ell_for_n_star(N: int, n_star_target: float,
                         ell_hi: float = 1e7, tol: float = 1e-3) -> float:
    """Find the correlation length ell giving a target N* (bisection; N* is monotone ↓ in ell).

    N*(ell) decreases monotonically from N (ell→0) to 1 (ell→∞), so a simple bisection on ell
    inverts it. We clamp the target into the achievable open interval (1, N). n_star_target ≥ N
    returns ell = 0 (independent); n_star_target ≤ 1 returns a very large ell (N*→1).
    """
    if n_star_target >= N:
        return 0.0
    if n_star_target <= 1.0 + tol:
        return ell_hi
    lo, hi = 0.0, ell_hi
    # N*(lo)=N >= target ; N*(hi)≈1 <= target. Bisect on ell.
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        ns, _ = n_star_of_ell(N, mid)
        if ns > n_star_target:      # too little correlation ⇒ increase ell
            lo = mid
        else:                       # too much ⇒ decrease ell
            hi = mid
        if abs(ns - n_star_target) < tol:
            break
    return 0.5 * (lo + hi)


def _kernel_sqrt(K: np.ndarray) -> np.ndarray:
    """Symmetric PSD square root L (L Lᵀ = K) via eigendecomposition (K small: N≲few thousand).

    Used to draw a correlated Gaussian field  δ = L z,  z ~ N(0, I)  with Cov(δ) = K. Eigh is
    numerically safe for the near-singular all-ones limit (ell→∞) where a Cholesky can fail.
    """
    w, V = np.linalg.eigh(K)
    w = np.clip(w, 0.0, None)
    return (V * np.sqrt(w)) @ V.T


# ============================================================================================
# Perturbed-optimum generation step (COMPOSES engine primitives; does not edit engine.py)
# ============================================================================================

def _perturbed_fitness(z, theta_i, A):
    """Gaussian stabilizing selection with a PER-INDIVIDUAL optimum θ_i (the Sim B term).

        W_i = exp( -½ (z_i - θ_i)ᵀ A (z_i - θ_i) ).

    z : (N, n) ; theta_i : (N, n) [per-individual] ; A : (n, n)  ->  W : (N,).
    This is the ONLY line that differs from engine.fitness (which uses a shared θ). Everything
    downstream (pool, sampling, segregation, mutation) matches engine.step_generation exactly.
    """
    d = z - theta_i                                   # (N, n)
    quad = jnp.einsum("ni,ij,nj->n", d, A, d)         # (N,)
    return jnp.exp(-0.5 * quad)


def step_generation_perturbed(key, state: State, config: Config,
                              env_L_sqrt: jnp.ndarray, env_amp: float) -> State:
    """One generation with spatially-correlated per-individual optimum perturbations.

    Mirrors engine.step_generation step-for-step, with the single change that fitness uses a
    per-individual optimum θ_i = θ + δ_i. δ is drawn fresh each generation as a spatially
    correlated field: for each trait, δ[:, t] = env_amp · (env_L_sqrt @ z_t), z_t ~ N(0, I), so
    Cov(δ[:, t]) = env_amp² · K (independent across traits — the environmental perturbation is
    isotropic in trait space; spatial correlation is across INDIVIDUALS, not traits).

    env_L_sqrt : (N, N) PSD square root of the spatial kernel K (so L Lᵀ = K).
    env_amp    : SD scale of the perturbation (the gradient-noise MAGNITUDE; ℓ sets its N*).

    jit-compatible (no data-dependent Python control flow).
    """
    N, L, n = config.shape_key
    Ne = config.Ne

    (key, k_env, k_pool, k_p1, k_p2,
     k_seg1, k_seg2, k_mut1, k_mut2) = jax.random.split(key, 9)

    z = genotype_to_phenotype(state, config)                       # (N, n)

    # --- spatially-correlated per-individual optimum perturbation δ (the Sim B term) ---------
    zz = jax.random.normal(k_env, (N, n))                          # iid standard normal
    delta = env_amp * (env_L_sqrt @ zz)                            # (N, n), Cov over inds = amp²·K
    theta_i = config.theta[None, :] + delta                        # (N, n) per-individual optimum

    W = _perturbed_fitness(z, theta_i, config.A)                   # (N,)

    # --- finite-Ne demographic step (identical to engine.step_generation) -------------------
    pool_idx = jax.random.choice(k_pool, N, shape=(Ne,), replace=False)          # (Ne,)
    pool_y = state.y[pool_idx]                                                   # (Ne, L, 2, n)
    pool_W = W[pool_idx]                                                         # (Ne,)
    wsum = jnp.sum(pool_W)
    p = jnp.where(wsum > 0, pool_W / wsum, jnp.ones_like(pool_W) / Ne)           # (Ne,)

    # --- fitness-weighted parent sampling (identical) ---------------------------------------
    par1 = jax.random.choice(k_p1, Ne, shape=(N,), replace=True, p=p)            # (N,)
    par2 = jax.random.choice(k_p2, Ne, shape=(N,), replace=True, p=p)            # (N,)
    par1_y = pool_y[par1]
    par2_y = pool_y[par2]

    # --- segregation w/ recombination + mutation (identical) --------------------------------
    g1 = _segregate_gametes(k_seg1, par1_y, config.r)
    g2 = _segregate_gametes(k_seg2, par2_y, config.r)
    g1 = _mutate_gametes(k_mut1, g1, config.mu, config.mut_var)
    g2 = _mutate_gametes(k_mut2, g2, config.mu, config.mut_var)

    new_y = jnp.stack([g1, g2], axis=2)                                          # (N, L, 2, n)
    return State(y=new_y, eps=state.eps, key=key)


def _run_one_replicate_perturbed(key, config, env_L_sqrt, env_amp, n_generations):
    """Run one replicate under perturbed selection; record `measure` (G, M, ...) each gen."""
    state0 = init_population(key, config)

    def body(state, _):
        new_state = step_generation_perturbed(state.key, state, config, env_L_sqrt, env_amp)
        m = measure(new_state, config)
        return new_state, m

    final_state, traj = lax.scan(body, state0, xs=None, length=n_generations)
    return final_state, traj


def run_sim_perturbed(key, config, env_L_sqrt, env_amp, n_generations, n_replicates):
    """vmap a per-replicate perturbed-selection scan over n_replicates independent replicates.

    Returns a dict of stacked per-generation measurements (leaves (n_replicates, n_gen, ...)),
    exactly like engine.run_sim but with the perturbed (per-individual θ) selection step.
    """
    rep_keys = jax.random.split(key, n_replicates)
    env_L_sqrt = jnp.asarray(env_L_sqrt)
    run_fn = partial(_run_one_replicate_perturbed, config=config, env_L_sqrt=env_L_sqrt,
                     env_amp=env_amp, n_generations=n_generations)
    _, traj = jax.vmap(run_fn)(rep_keys)
    return traj


# ============================================================================================
# M anisotropy / alignment estimators (be rigorous — CONVENTIONS.md §6: sort descending)
# ============================================================================================

def m_anisotropy(M: np.ndarray) -> float:
    """Eigenvalue anisotropy ratio max/min of a (2,2) PSD matrix M (≥ 1; 1 = isotropic)."""
    ev = np.linalg.eigvalsh(M)                         # ascending, real (M symmetric PSD)
    ev = np.clip(ev, 1e-30, None)
    return float(ev[-1] / ev[0])


def m_a_alignment_angle(M: np.ndarray, A: np.ndarray) -> float:
    """EIGENVALUE-ORDER-INVARIANT eigenframe-rotation angle (degrees, [0, 45]) between M and A.

    Prediction (ii) is about whether M's EIGENVECTORS stay locked to A's principal AXES across
    N* — NOT about which axis carries the larger M eigenvalue. So the estimator must NOT depend
    on eigenvalue ORDERING: when M is near-isotropic (exactly the compressed, low-N* regime),
    sampling noise flips which eigenvalue is larger, which would make a "leading-eigvec-to-
    leading-axis" angle jump between 0° and 90° spuriously. That ordering flip is the very
    eigenvalue-order-inversion the framework RETRACTED as an artifact — we must not reintroduce
    it in the estimator.

    Instead we measure the ROTATION of M's eigenframe relative to A's eigenframe, folded into
    [0, 45°] (the symmetry cell for a 2-frame: a 90° swap is a relabel, not a rotation). We
    rotate M into A's eigenbasis and read the off-diagonal rotation angle:

        M_in_A = Vaᵀ M Va ;  φ = ½ atan2( 2·M_in_A[0,1] , M_in_A[0,0] - M_in_A[1,1] ),

    then fold φ to [0, 45°]. φ ≈ 0 ⇔ M is DIAGONAL in A's basis (eigenvectors aligned with A's
    axes, EITHER ordering); φ → 45° ⇔ M's axes are maximally rotated off A's. Flatness of φ vs
    N* is prediction (ii); it stays well-defined as M → isotropic (where the angle is simply
    undefined-but-small because the off-diagonal → 0). Eigh sorts ascending (CONVENTIONS §6
    sorts descending; the rotation measure is independent of that choice by construction).
    """
    _, VA = np.linalg.eigh(A)                          # A eigenvectors (columns)
    M_in_A = VA.T @ np.asarray(M, dtype=np.float64) @ VA
    off = M_in_A[0, 1]
    diff = M_in_A[0, 0] - M_in_A[1, 1]
    phi = 0.5 * np.arctan2(2.0 * off, diff)            # eigenframe rotation, radians
    deg = abs(np.degrees(phi)) % 90.0                  # fold the 90° relabel symmetry
    if deg > 45.0:
        deg = 90.0 - deg                               # fold into [0, 45]
    return float(deg)


# ============================================================================================
# A single phase-diagram cell
# ============================================================================================

def run_cell(
    key,
    n_star_target: float,
    Ne: int,
    *,
    A_eigratio: float = 6.0,
    geom_mean: float = 1.0,
    N: int = 600,
    L: int = 15,
    mut_var: float = 0.05,
    mu: float = 5e-2,
    r: float = 0.5,
    sigma_eps: float = 0.3,
    env_amp: float = 0.6,
    n_generations: int = 400,
    burn_in: int = 200,
    n_replicates: int = 16,
) -> dict:
    """Run ONE (N*, Ne) phase-diagram cell to M-equilibrium under perturbed selection.

    Builds an anisotropic diagonal A (eigenvalue ratio A_eigratio, fixed geometric mean), finds
    the correlation length ℓ realizing the target N* via the Kish inversion, draws the spatial
    kernel square root once, runs `run_sim_perturbed`, and reduces the post-burn-in M:

      * per-replicate, time-averaged M over the post-burn-in window  → M_reps (rep, 2, 2)
      * pooled mean M = mean over replicates                          → M_mean (2, 2)
      * M anisotropy = max/min eigenvalue of M_mean
      * M–A alignment angle = angle(M_mean leading eigvec, A leading eigvec)
      * per-replicate anisotropy/angle → across-replicate variance (the Ne signature)

    Returns a dict with the realized ℓ, ρ̄, N*, the M matrices, the scalar M-summaries, and the
    per-replicate arrays (for across-replicate variance and bootstrap CIs).
    """
    # anisotropic A (sharp axis = index 0 after descending sort); fixed geometric mean.
    s = float(np.sqrt(A_eigratio))
    A_np = np.array([[geom_mean * s, 0.0], [0.0, geom_mean / s]], dtype=np.float64)

    # Kish inversion: ℓ realizing the target N* (and the realized ρ̄, N*).
    ell = solve_ell_for_n_star(N, n_star_target)
    K = exponential_kernel_1d(N, ell)
    rho_bar = rho_bar_of_kernel(K)
    n_star = kish_n_star(N, rho_bar)
    env_L_sqrt = _kernel_sqrt(K)

    cfg = make_config(
        N=N, L=L, n_traits=2, mu=mu, mut_var=mut_var, r=r,
        sigma_eps=sigma_eps, A=jnp.asarray(A_np, dtype=jnp.float32),
        theta=jnp.zeros(2), Ne=Ne,
    )

    traj = run_sim_perturbed(
        key, cfg, env_L_sqrt=jnp.asarray(env_L_sqrt, dtype=jnp.float32),
        env_amp=float(env_amp), n_generations=n_generations, n_replicates=n_replicates,
    )
    M = np.array(traj["M"])                            # (rep, gen, 2, 2)
    M_reps = M[:, burn_in:, :, :].mean(axis=1)        # (rep, 2, 2) time-averaged plateau
    M_mean = M_reps.mean(axis=0)                       # (2, 2) pooled

    # per-replicate scalar summaries (for across-replicate variance, the Ne/Langevin signature)
    aniso_reps = np.array([m_anisotropy(Mr) for Mr in M_reps])
    angle_reps = np.array([m_a_alignment_angle(Mr, A_np) for Mr in M_reps])

    return {
        "n_star_target": float(n_star_target),
        "Ne": int(Ne),
        "A_eigratio": float(A_eigratio),
        "ell": float(ell),
        "rho_bar": float(rho_bar),
        "n_star": float(n_star),
        "A": A_np,
        "M_mean": M_mean,
        "M_reps": M_reps,                             # (rep, 2, 2)
        "m_anisotropy": m_anisotropy(M_mean),
        "m_a_angle": m_a_alignment_angle(M_mean, A_np),
        "aniso_reps": aniso_reps,                     # (rep,)
        "angle_reps": angle_reps,                     # (rep,)
        # across-replicate variance of M's structure = the demographic (Langevin) signature.
        "aniso_rep_var": float(np.var(aniso_reps, ddof=1)) if aniso_reps.size > 1 else 0.0,
        "angle_rep_var": float(np.var(angle_reps, ddof=1)) if angle_reps.size > 1 else 0.0,
        "n_replicates": int(n_replicates),
    }


# ============================================================================================
# The (N*, Ne) phase-diagram grid + per-cell checkpointing (idempotent / preemption-safe)
# ============================================================================================

def _default_out_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "runs" / "sim_b"


def enumerate_cells(n_star_grid: Sequence[float], Ne_grid: Sequence[int]) -> list[dict]:
    """Flatten the (N*, Ne) grid into an ordered list of cells with a stable integer index.

    The integer index is the SLURM array task id (the array job runs one cell per task). The
    order is row-major over (N*, Ne); it is stable as long as the grids are unchanged, which is
    what makes the per-cell .done checkpointing reproducible across requeues.
    """
    cells = []
    idx = 0
    for ns in n_star_grid:
        for ne in Ne_grid:
            cells.append({"cell": idx, "n_star_target": float(ns), "Ne": int(ne)})
            idx += 1
    return cells


def _cell_paths(out_dir: Path, cell: int) -> tuple[Path, Path]:
    """(per-cell results csv, per-cell .done marker) for a given cell index."""
    cdir = out_dir / "cells"
    return cdir / f"cell_{cell:04d}.csv", cdir / f"cell_{cell:04d}.done"


def _cell_to_rows(cell_idx: int, res: dict) -> list[dict]:
    """One tidy row PER REPLICATE for a finished cell (matches the aggregated schema)."""
    rows = []
    for rep in range(res["n_replicates"]):
        Mr = res["M_reps"][rep]
        rows.append({
            "cell": int(cell_idx),
            "n_star_target": res["n_star_target"],
            "Ne": res["Ne"],
            "A_eigratio": res["A_eigratio"],
            "ell": res["ell"],
            "rho_bar": res["rho_bar"],
            "n_star": res["n_star"],
            "replicate": rep,
            "m_anisotropy": float(res["aniso_reps"][rep]),
            "m_a_angle": float(res["angle_reps"][rep]),
            "M00": float(Mr[0, 0]), "M01": float(Mr[0, 1]),
            "M10": float(Mr[1, 0]), "M11": float(Mr[1, 1]),
        })
    return rows


def run_one_cell_checkpointed(
    cell_idx: int,
    n_star_grid: Sequence[float],
    Ne_grid: Sequence[int],
    *,
    seed: int = 0,
    out_dir: str | Path | None = None,
    force: bool = False,
    **cell_kwargs,
) -> Path:
    """Run a SINGLE phase-diagram cell by index, with per-cell checkpointing (idempotent).

    If the cell's .done marker exists and force=False, the cell is SKIPPED (returns the existing
    per-cell csv path) — this is what makes a requeued/preempted SLURM array job resume cleanly
    on the preemptible burst QOS. Each cell gets an INDEPENDENT key folded from (seed, cell_idx)
    so cells are reproducible and mutually independent regardless of run order.

    Writes runs/sim_b/cells/cell_XXXX.csv (one row per replicate) and a cell_XXXX.done marker.
    """
    out = Path(out_dir) if out_dir is not None else _default_out_dir()
    (out / "cells").mkdir(parents=True, exist_ok=True)

    cells = enumerate_cells(n_star_grid, Ne_grid)
    if cell_idx < 0 or cell_idx >= len(cells):
        raise IndexError(f"cell {cell_idx} out of range 0..{len(cells)-1}")
    spec = cells[cell_idx]

    csv_path, done_path = _cell_paths(out, cell_idx)
    if done_path.exists() and not force:
        return csv_path  # idempotent skip — already finished (preemption-safe)

    # independent per-cell key folded from (seed, cell_idx) — order-independent, reproducible.
    key = jax.random.fold_in(jax.random.PRNGKey(seed), cell_idx)
    res = run_cell(key, n_star_target=spec["n_star_target"], Ne=spec["Ne"], **cell_kwargs)

    pd.DataFrame(_cell_to_rows(cell_idx, res)).to_csv(csv_path, index=False)
    done_path.write_text("ok\n")  # atomic-enough marker written AFTER the csv exists
    return csv_path


def aggregate_cells(
    n_star_grid: Sequence[float],
    Ne_grid: Sequence[int],
    *,
    out_dir: str | Path | None = None,
    write_csv: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
    """Concatenate all finished per-cell csvs into runs/sim_b/results.csv + aggregated surfaces.

    Returns (tidy, surfaces, missing_cells):
      tidy     : one row per (cell, replicate) — the per-replicate results (M-anisotropy, angle,
                 M entries, realized N*/ρ̄). Written to results.csv.
      surfaces : one row per cell — replicate MEAN and MEDIAN anisotropy & angle PLUS the
                 across-replicate VARIANCE of anisotropy/angle (the Ne/Langevin signature).
                 Written to surfaces.csv. This is the (N*, Ne) phase-diagram surface table.
                 NOTE: the per-cell anisotropy distribution is RIGHT-SKEWED at low N* (the
                 correlated environmental perturbation occasionally produces a large coherent
                 optimum shift that transiently inflates a replicate's M anisotropy). The
                 MEDIAN is therefore the robust location estimator for the compression signature
                 (i) — the mean is pulled up by the tail at low N*. Use m_anisotropy_median for
                 the compression-vs-N* slice; the across-replicate VARIANCE (Ne signature, iii)
                 captures exactly that tail growth at small Ne.
      missing_cells : indices whose .done marker is absent (so the caller knows the grid is
                 incomplete — e.g. preempted array tasks still pending).
    """
    out = Path(out_dir) if out_dir is not None else _default_out_dir()
    cells = enumerate_cells(n_star_grid, Ne_grid)

    frames, missing = [], []
    for spec in cells:
        csv_path, done_path = _cell_paths(out, spec["cell"])
        if done_path.exists() and csv_path.exists():
            frames.append(pd.read_csv(csv_path))
        else:
            missing.append(spec["cell"])

    if not frames:
        return pd.DataFrame(), pd.DataFrame(), missing

    tidy = pd.concat(frames, ignore_index=True).sort_values(["cell", "replicate"])

    # aggregated surfaces: per-cell replicate means + across-replicate variances.
    g = tidy.groupby(["cell", "n_star_target", "Ne", "A_eigratio", "ell", "rho_bar", "n_star"])
    surfaces = g.agg(
        m_anisotropy_mean=("m_anisotropy", "mean"),
        m_anisotropy_median=("m_anisotropy", "median"),   # robust location (skewed at low N*)
        m_anisotropy_var=("m_anisotropy", lambda x: float(np.var(x, ddof=1)) if len(x) > 1 else 0.0),
        m_a_angle_mean=("m_a_angle", "mean"),
        m_a_angle_median=("m_a_angle", "median"),
        m_a_angle_var=("m_a_angle", lambda x: float(np.var(x, ddof=1)) if len(x) > 1 else 0.0),
        n_rep=("replicate", "count"),
    ).reset_index()

    if write_csv:
        out.mkdir(parents=True, exist_ok=True)
        tidy.to_csv(out / "results.csv", index=False)
        surfaces.to_csv(out / "surfaces.csv", index=False)

    return tidy, surfaces, missing


def write_status(
    n_star_grid: Sequence[float],
    Ne_grid: Sequence[int],
    *,
    out_dir: str | Path | None = None,
) -> dict:
    """Write a STATUS manifest (runs/sim_b/STATUS) listing done/missing cells (preemption view).

    A requeued array can read this (or the per-cell .done markers directly) to see what remains.
    """
    out = Path(out_dir) if out_dir is not None else _default_out_dir()
    out.mkdir(parents=True, exist_ok=True)
    cells = enumerate_cells(n_star_grid, Ne_grid)
    done = [c["cell"] for c in cells if _cell_paths(out, c["cell"])[1].exists()]
    missing = [c["cell"] for c in cells if c["cell"] not in done]
    status = {
        "n_cells": len(cells),
        "n_done": len(done),
        "n_missing": len(missing),
        "done": done,
        "missing": missing,
        "n_star_grid": [float(x) for x in n_star_grid],
        "Ne_grid": [int(x) for x in Ne_grid],
        "cells": cells,
    }
    (out / "STATUS").write_text(json.dumps(status, indent=2))
    return status


def run_phase_diagram(
    n_star_grid: Sequence[float],
    Ne_grid: Sequence[int],
    *,
    seed: int = 0,
    out_dir: str | Path | None = None,
    force: bool = False,
    **cell_kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the FULL (N*, Ne) phase diagram cell-by-cell (checkpointed), then aggregate.

    Iterates every cell through `run_one_cell_checkpointed` (so already-.done cells are skipped),
    writes the STATUS manifest, and returns (tidy results, aggregated surfaces). Equivalent to
    launching the SLURM array and then aggregating — used for the pilot / local full runs.

    tidy → runs/sim_b/results.csv (one row per (N*, Ne, replicate)); surfaces → surfaces.csv.
    """
    for spec in enumerate_cells(n_star_grid, Ne_grid):
        run_one_cell_checkpointed(
            spec["cell"], n_star_grid, Ne_grid,
            seed=seed, out_dir=out_dir, force=force, **cell_kwargs,
        )
    write_status(n_star_grid, Ne_grid, out_dir=out_dir)
    tidy, surfaces, _missing = aggregate_cells(n_star_grid, Ne_grid, out_dir=out_dir)
    return tidy, surfaces


# ============================================================================================
# Default publication grid (overridable on the CLI)
# ============================================================================================

# N* swept via ℓ across ~2 orders of magnitude (independent → philopatric colony regime);
# Ne swept independently across the drift range. The compression-with-N* signature (i) is
# cleanest with Ne < N (some demographic sampling present but held fixed while N* varies — see
# the build report / Sim B finding): with Ne = N the high-N* baseline anisotropy is suppressed
# and a large correlated perturbation behaves like lag-load instead of attenuation. So the
# default census N (=800 in the CLI) is set ABOVE the largest Ne (600) so EVERY cell has Ne < N.
DEFAULT_N_STAR_GRID = (800.0, 100.0, 30.0, 10.0, 3.0)
DEFAULT_NE_GRID = (600, 200, 60, 20)


# ============================================================================================
# CLI: run one cell (SLURM array) or the whole pilot grid
# ============================================================================================

def _parse_grid(s: str, cast):
    return tuple(cast(x) for x in s.split(",") if x.strip())


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Sim B — N* phase diagram (§3.2)")
    p.add_argument("--cell", type=int, default=None,
                   help="SLURM array task id: run ONLY this cell (checkpointed). "
                        "Omit to run the whole grid locally and aggregate.")
    p.add_argument("--n-star-grid", type=str, default=None,
                   help="comma-separated N* targets (default: publication grid)")
    p.add_argument("--Ne-grid", type=str, default=None,
                   help="comma-separated Ne values (default: publication grid)")
    p.add_argument("--N", type=int, default=800)   # census N > max(Ne) so every cell has Ne < N
    p.add_argument("--L", type=int, default=15)
    p.add_argument("--A-eigratio", type=float, default=6.0)
    p.add_argument("--sigma-eps", type=float, default=0.3)
    p.add_argument("--env-amp", type=float, default=0.6)
    p.add_argument("--mu", type=float, default=5e-2)
    p.add_argument("--generations", type=int, default=400)
    p.add_argument("--burn-in", type=int, default=200)
    p.add_argument("--replicates", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--force", action="store_true", help="ignore .done markers and recompute")
    p.add_argument("--aggregate-only", action="store_true",
                   help="skip computation; just aggregate finished per-cell csvs")
    args = p.parse_args()

    n_star_grid = (_parse_grid(args.n_star_grid, float)
                   if args.n_star_grid else DEFAULT_N_STAR_GRID)
    Ne_grid = (_parse_grid(args.Ne_grid, int)
               if args.Ne_grid else DEFAULT_NE_GRID)

    cell_kwargs = dict(
        A_eigratio=args.A_eigratio, N=args.N, L=args.L, mu=args.mu,
        sigma_eps=args.sigma_eps, env_amp=args.env_amp,
        n_generations=args.generations, burn_in=args.burn_in,
        n_replicates=args.replicates,
    )

    out_dir = args.out_dir

    if args.aggregate_only:
        tidy, surfaces, missing = aggregate_cells(n_star_grid, Ne_grid, out_dir=out_dir)
        write_status(n_star_grid, Ne_grid, out_dir=out_dir)
        print(f"Aggregated {len(tidy)} rows; {len(missing)} cells still missing: {missing}")
    elif args.cell is not None:
        path = run_one_cell_checkpointed(
            args.cell, n_star_grid, Ne_grid, seed=args.seed, out_dir=out_dir,
            force=args.force, **cell_kwargs,
        )
        # refresh STATUS opportunistically (cheap; reflects this cell's completion)
        write_status(n_star_grid, Ne_grid, out_dir=out_dir)
        print(f"cell {args.cell} done -> {path}")
    else:
        tidy, surfaces = run_phase_diagram(
            n_star_grid, Ne_grid, seed=args.seed, out_dir=out_dir,
            force=args.force, **cell_kwargs,
        )
        pd.set_option("display.width", 160)
        print("\n=== Sim B: (N*, Ne) phase-diagram surfaces ===")
        print(surfaces.to_string(index=False))
        out = Path(out_dir) if out_dir is not None else _default_out_dir()
        print(f"\nTidy: {len(tidy)} rows -> {out / 'results.csv'}")
        print(f"Surfaces -> {out / 'surfaces.csv'}")

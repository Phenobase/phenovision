"""§3.0 gate tests for the Strand-2 JAB-style IBM engine (sim/engine.py).

The five definition-of-done checks from analysis_plan.md §3.0:
  1. Determinism: same PRNGKey -> identical trajectory (exact).
  2. vmap over replicates == Python loop over single replicates (exact, small case).
  3. JAB 2003 baseline: σ_eps=0, isotropic A, static peak -> mutation-selection-drift balance
     with G ~ isotropic and magnitude near the BLW/Gaussian-MSB expectation (within ~factor-2).
  4. Mean fitness monotone non-decreasing (up to drift) starting OFF the peak (σ_eps=0).
  5. Anisotropy sanity: anisotropic A, σ_eps=0 (fixed isotropic M) -> finite G with the
     expected shape and some eigenvalue structure (previews Sim A).

Plus unit checks on the scientific core: the multilinear epistasis map reduces to additive at
σ_eps=0, and the M estimator equals the fixed isotropic mutational covariance at σ_eps=0.

Run on CPU:
    mamba run -n two_noise_jax python -m pytest tests/test_engine.py -q
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # tests are CPU; no GPU on login node

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sim.engine import (
    make_config, init_population, genotype_to_phenotype, breeding_values, fitness,
    step_generation, measure, run_sim, _run_one_replicate,
)


# --------------------------------------------------------------------------------------------
# Unit checks on the scientific core
# --------------------------------------------------------------------------------------------

def test_multilinear_reduces_to_additive_when_sigma_eps_zero():
    """σ_eps=0 ⇒ multilinear phenotype == pure additive (Σ loci & alleles) EXACTLY."""
    cfg = make_config(N=64, L=12, n_traits=2, sigma_eps=0.0)
    key = jax.random.PRNGKey(7)
    st = init_population(key, cfg)
    # give the population some nonzero allelic effects so the test is non-trivial
    st = st._replace(y=jax.random.normal(jax.random.PRNGKey(8), st.y.shape))
    z = genotype_to_phenotype(st, cfg)
    z_add = jnp.sum(st.y, axis=(1, 2))            # additive map = Σ loci & alleles
    np.testing.assert_allclose(np.array(z), np.array(z_add), rtol=1e-5, atol=1e-5)
    # eps tensor is identically zero
    assert float(jnp.max(jnp.abs(st.eps))) == 0.0


def test_multilinear_differs_from_additive_when_sigma_eps_positive():
    """σ_eps>0 ⇒ multilinear map departs from additive (epistasis is actually active)."""
    cfg = make_config(N=64, L=12, n_traits=2, sigma_eps=0.5)
    st = init_population(jax.random.PRNGKey(1), cfg)
    st = st._replace(y=0.3 * jax.random.normal(jax.random.PRNGKey(2), st.y.shape))
    z = genotype_to_phenotype(st, cfg)
    z_add = jnp.sum(st.y, axis=(1, 2))
    assert float(jnp.max(jnp.abs(z - z_add))) > 1e-3
    assert float(jnp.max(jnp.abs(st.eps))) > 0.0
    # ε(i,i) must be exactly zero (a locus has no epistatic effect on itself)
    diag = jnp.einsum("iit->it", st.eps)
    assert float(jnp.max(jnp.abs(diag))) == 0.0


def test_M_estimator_is_fixed_isotropic_at_sigma_eps_zero():
    """σ_eps=0 ⇒ M = (mut_var / n) · I, background-independent (the fixed-isotropic-M arm)."""
    mut_var, n = 0.05, 2
    cfg = make_config(N=128, L=10, n_traits=n, mut_var=mut_var, sigma_eps=0.0)
    st = init_population(jax.random.PRNGKey(3), cfg)
    # nonzero background must NOT change M when σ_eps=0
    st = st._replace(y=jax.random.normal(jax.random.PRNGKey(4), st.y.shape))
    m = measure(st, cfg)["M"]
    expected = (mut_var / n) * jnp.eye(n)
    np.testing.assert_allclose(np.array(m), np.array(expected), rtol=1e-6, atol=1e-6)


def test_fitness_peaks_at_optimum():
    """W(θ)=1 and W decreases away from θ; matches the explicit Gaussian formula."""
    A = jnp.array([[2.0, 0.5], [0.5, 1.0]])
    theta = jnp.array([1.0, -1.0])
    z = jnp.array([[1.0, -1.0], [1.5, -0.5], [3.0, 2.0]])
    W = fitness(z, theta, A)
    assert np.isclose(float(W[0]), 1.0)            # at the optimum
    assert W[1] > W[2]                              # closer to peak ⇒ higher fitness
    # cross-check against explicit quadratic
    d = z - theta
    expected = np.exp(-0.5 * np.einsum("ni,ij,nj->n", d, np.array(A), d))
    np.testing.assert_allclose(np.array(W), expected, rtol=1e-6)


# --------------------------------------------------------------------------------------------
# §3.0 gate test 1: determinism
# --------------------------------------------------------------------------------------------

def test_determinism_same_key_identical_trajectory():
    """Same PRNGKey ⇒ bit-identical trajectory (RNG discipline: no global state)."""
    cfg = make_config(N=300, L=12, n_traits=2, sigma_eps=0.0)
    key = jax.random.PRNGKey(123)
    t1 = run_sim(key, cfg, n_generations=40, n_replicates=4)
    t2 = run_sim(key, cfg, n_generations=40, n_replicates=4)
    for k in t1:
        np.testing.assert_array_equal(np.array(t1[k]), np.array(t2[k]))
    # a different key gives a different trajectory (sanity: RNG is actually used)
    t3 = run_sim(jax.random.PRNGKey(124), cfg, n_generations=40, n_replicates=4)
    assert not np.allclose(np.array(t1["G"]), np.array(t3["G"]))


# --------------------------------------------------------------------------------------------
# §3.0 gate test 2: vmap over replicates == python loop
# --------------------------------------------------------------------------------------------

def test_vmap_equals_python_loop_over_replicates():
    """vmap-over-replicates must equal a manual loop using the same per-replicate split keys."""
    cfg = make_config(N=200, L=10, n_traits=2, sigma_eps=0.0)
    key = jax.random.PRNGKey(99)
    n_rep, n_gen = 5, 30

    traj_vmap = run_sim(key, cfg, n_generations=n_gen, n_replicates=n_rep)

    # reproduce run_sim's split lattice exactly, then loop
    rep_keys = jax.random.split(key, n_rep)
    loop = {k: [] for k in traj_vmap}
    for i in range(n_rep):
        _, tr = _run_one_replicate(rep_keys[i], cfg, n_gen)
        for k in tr:
            loop[k].append(np.array(tr[k]))
    for k in traj_vmap:
        stacked = np.stack(loop[k], axis=0)
        np.testing.assert_allclose(np.array(traj_vmap[k]), stacked, rtol=1e-5, atol=1e-6)


# --------------------------------------------------------------------------------------------
# §3.0 gate test 3: JAB 2003 baseline — isotropic MSB balance
# --------------------------------------------------------------------------------------------

def test_jab_baseline_isotropic_msb_balance():
    """σ_eps=0, isotropic A, static peak: reach mutation-selection-drift balance with G
    approximately ISOTROPIC and magnitude near the BLW/Gaussian-MSB expectation.

    Analytic target (Lande 1980 / Bulmer–Lande–Week Gaussian closure, per-trait commuting
    case): equilibrium per-trait genetic variance  g ≈ sqrt(2 · V_m · V_s),  where
        V_m = per-generation mutational input variance per trait = 2·L·μ·mut_var
              (diploid: 2 allele copies × L loci × μ rate × increment variance),
        V_s = 1/a  (a = isotropic curvature; selection "variance" / width).

    TOLERANCE: within a factor of 2 of the analytic value, and anisotropy ratio < 1.5.
    Justification: the analytic value is a deterministic-Gaussian-closure, infinite-N
    approximation; with finite N=1000, finite L, and drift the realized variance is
    house-of-cards-suppressed below the Gaussian value. The spec explicitly allows
    order-of-magnitude / within-factor-2. Empirically (calibration runs) we land ~7% low and
    nearly perfectly isotropic, well inside these bounds. Burn-in 250 gens reaches the plateau.
    """
    N, L, n = 1000, 15, 2
    mu, mut_var, a = 1e-3, 0.05, 1.0
    cfg = make_config(N=N, L=L, n_traits=n, mu=mu, mut_var=mut_var, r=0.5,
                      sigma_eps=0.0, A=jnp.eye(n) * a, theta=jnp.zeros(n))
    traj = run_sim(jax.random.PRNGKey(0), cfg, n_generations=500, n_replicates=8)

    G = np.array(traj["G"])                          # (rep, gen, n, n)
    G_late = G[:, 250:, :, :].mean(axis=(0, 1))      # average over reps & post-burn-in gens

    V_m = 2 * L * mu * mut_var
    V_s = 1.0 / a
    g_analytic = np.sqrt(2 * V_m * V_s)

    g_diag = np.array([G_late[0, 0], G_late[1, 1]])
    ratio_to_analytic = g_diag / g_analytic
    assert np.all(ratio_to_analytic > 0.5) and np.all(ratio_to_analytic < 2.0), (
        f"per-trait G {g_diag} vs analytic {g_analytic:.4f} "
        f"(ratios {ratio_to_analytic}) outside factor-2"
    )

    # G approximately isotropic: eigenvalue ratio near 1, off-diagonal small
    evals = np.linalg.eigvalsh(G_late)
    aniso = evals.max() / evals.min()
    assert aniso < 1.5, f"isotropic-A G not isotropic: eigenvalue ratio {aniso:.3f}"
    offdiag = abs(G_late[0, 1])
    assert offdiag < 0.3 * g_diag.mean(), f"unexpected G off-diagonal {offdiag:.4f}"


# --------------------------------------------------------------------------------------------
# §3.0 gate test 4: off-peak fitness climbs (population moves toward θ)
# --------------------------------------------------------------------------------------------

def test_mean_fitness_increases_off_peak():
    """Starting monomorphic at z=0 with θ≠0, under pure stabilizing selection the population
    mean moves toward θ and mean fitness is monotone non-decreasing up to drift noise."""
    N, L, n = 1000, 15, 2
    cfg = make_config(N=N, L=L, n_traits=n, mu=1e-3, mut_var=0.05, r=0.5, sigma_eps=0.0,
                      A=jnp.eye(n) * 0.5, theta=jnp.array([2.0, -1.5]))
    traj = run_sim(jax.random.PRNGKey(2), cfg, n_generations=200, n_replicates=8)

    mf = np.array(traj["mean_fitness"]).mean(axis=0)     # (gen,) averaged over reps
    # end fitness clearly above start (climbed toward the peak)
    assert mf[-1] > mf[0] + 0.3, f"fitness did not climb: {mf[0]:.3f} -> {mf[-1]:.3f}"

    # monotone non-decreasing up to drift noise: smooth (moving average) and require no
    # appreciable sustained decrease.
    w = 5
    ma = np.convolve(mf, np.ones(w) / w, mode="valid")
    dec = np.diff(ma)
    assert dec.min() > -0.02, f"mean fitness decreased beyond drift tol: min Δ={dec.min():.4f}"

    # the trait mean actually approaches θ
    tm = np.array(traj["trait_mean"]).mean(axis=0)       # (gen, n)
    final_dist = np.linalg.norm(tm[-1] - np.array([2.0, -1.5]))
    init_dist = np.linalg.norm(tm[0] - np.array([2.0, -1.5]))
    assert final_dist < 0.25 * init_dist, (
        f"trait mean did not approach θ: dist {init_dist:.3f} -> {final_dist:.3f}"
    )


# --------------------------------------------------------------------------------------------
# §3.0 gate test 5: anisotropy sanity (previews Sim A)
# --------------------------------------------------------------------------------------------

def test_anisotropic_A_produces_structured_finite_G():
    """Anisotropic A, σ_eps=0 (fixed isotropic M): after equilibration G is finite, correct
    shape, and shows eigenvalue structure with MORE variance in the WEAKLY-selected direction
    (smaller curvature ⇒ larger G). This previews the Sim A G–A exponent test."""
    N, L, n = 1000, 15, 2
    A = jnp.array([[4.0, 0.0], [0.0, 0.5]])              # sharp on trait 0, shallow on trait 1
    cfg = make_config(N=N, L=L, n_traits=n, mu=1e-3, mut_var=0.05, r=0.5,
                      sigma_eps=0.0, A=A, theta=jnp.zeros(n))
    traj = run_sim(jax.random.PRNGKey(1), cfg, n_generations=400, n_replicates=8)

    G = np.array(traj["G"])
    G_late = G[:, 250:, :, :].mean(axis=(0, 1))

    assert G_late.shape == (n, n)
    assert np.all(np.isfinite(G_late))
    evals = np.linalg.eigvalsh(G_late)
    assert np.all(evals > 0), f"G not PSD: eigenvalues {evals}"
    # clear anisotropy
    assert evals.max() / evals.min() > 1.5, f"expected structured G, ratio {evals.max()/evals.min():.2f}"
    # weakly-selected trait (a=0.5) carries MORE variance than sharply-selected (a=4)
    assert G_late[1, 1] > G_late[0, 0], (
        f"weak-selection trait should have larger G: "
        f"G00(a=4)={G_late[0,0]:.4f}, G11(a=0.5)={G_late[1,1]:.4f}"
    )


# --------------------------------------------------------------------------------------------
# extra: epistasis run executes and yields finite, sensible G/M (Sim A free-M arm smoke test)
# --------------------------------------------------------------------------------------------

def test_epistasis_run_executes_and_is_finite():
    """σ_eps>0 (free-M arm) runs end-to-end and produces finite G and a (now possibly
    anisotropic / background-dependent) M of the right shape."""
    N, L, n = 600, 12, 2
    cfg = make_config(N=N, L=L, n_traits=n, mu=2e-3, mut_var=0.05, r=0.5,
                      sigma_eps=0.3, A=jnp.array([[3.0, 0.0], [0.0, 0.7]]))
    traj = run_sim(jax.random.PRNGKey(5), cfg, n_generations=150, n_replicates=4)
    G = np.array(traj["G"])[:, -1].mean(axis=0)
    M = np.array(traj["M"])[:, -1].mean(axis=0)
    assert G.shape == (n, n) and M.shape == (n, n)
    assert np.all(np.isfinite(G)) and np.all(np.isfinite(M))
    assert np.all(np.linalg.eigvalsh(G) > 0)
    assert np.all(np.diag(M) > 0)


# --------------------------------------------------------------------------------------------
# extra: finite-Ne injects more across-replicate spread (demographic drift smoke test)
# --------------------------------------------------------------------------------------------

def test_finite_Ne_increases_replicate_spread():
    """Smaller Ne (more demographic drift) ⇒ larger across-replicate variance in G trace.
    A Langevin/drift signature smoke test; not a precise law, just the right direction."""
    N, L, n = 800, 12, 2
    common = dict(N=N, L=L, n_traits=n, mu=1e-3, mut_var=0.05, r=0.5, sigma_eps=0.0,
                  A=jnp.eye(n))
    cfg_big = make_config(Ne=N, **common)
    cfg_small = make_config(Ne=40, **common)
    nrep, ngen = 12, 250
    tb = run_sim(jax.random.PRNGKey(11), cfg_big, n_generations=ngen, n_replicates=nrep)
    ts = run_sim(jax.random.PRNGKey(12), cfg_small, n_generations=ngen, n_replicates=nrep)

    def trace_late(tr):
        G = np.array(tr["G"])[:, 150:]               # (rep, gen, n, n)
        return (G[:, :, 0, 0] + G[:, :, 1, 1]).mean(axis=1)   # (rep,) per-replicate mean trace

    var_big = trace_late(tb).var()
    var_small = trace_late(ts).var()
    assert var_small > var_big, (
        f"smaller Ne should inflate replicate variance: Ne=N var={var_big:.2e}, "
        f"Ne=40 var={var_small:.2e}"
    )

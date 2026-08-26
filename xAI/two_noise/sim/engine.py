"""Strand 2 — JAB-style individual-based quantitative-genetics simulation ENGINE (JAX).

This is build-order step §3.0 of the two_noise project: the foundation for Sims A/B/C
(see notes/analysis_plan.md §3.0–§3.3 and notes/two_noise_framework_summary.md §6, §11).

A reimplementation of the Jones–Arnold–Bürger (JAB 2003/2007/2014) individual-based,
diploid, multivariate quantitative-genetics model, with the multilinear epistasis map of
Hansen & Wagner (2001) so that the mutational architecture M can (in later sims) evolve.

============================================================================================
 A vs ω CONVENTION  (binding — see CONVENTIONS.md §1)
============================================================================================
  * `A`  = selection-surface CURVATURE: the matrix in Gaussian stabilizing selection
               W(z) = exp( -1/2 (z - θ)ᵀ A (z - θ) ).
           `A` is PSD. LARGER eigenvalue  ⇒  SHARPER selection  ⇒  stronger constraint.
  * `ω`  = selection WIDTH matrix.   A = ω⁻¹   (so larger width ⇒ weaker selection).
  * RULE: use `A` (curvature) everywhere internally. If a width `omega` enters, invert it
          to `A` AT THE BOUNDARY with an explicit comment. `omega` must never reach the core.
  This A/ω flip is a documented cross-paper source of confusion; keep it explicit.

  Related (CONVENTIONS.md §2): `A` (curvature) and `C` (per-sample gradient/noise covariance)
  are SEPARATE objects. The Fisher identity C = A is an approximation asserted only in tests.
  This engine only uses A; C-machinery (gradient noise / N*) is added in Sim B (§3.2).

============================================================================================
 RNG DISCIPLINE  (binding — see CONVENTIONS.md §5)
============================================================================================
  Explicit jax.random.PRNGKey split lattice; NEVER global RNG state. Each replicate gets an
  independent split key, which is what makes vmap-over-replicates correct and what makes a
  fixed top-level key reproduce a trajectory exactly.

============================================================================================
 LAYOUT
============================================================================================
  vmap(over REPLICATES) [ lax.scan(over GENERATIONS) [ vectorized ops over INDIVIDUALS ] ]
  Many replicates run in parallel (this is what Sim B's many-cell phase diagram needs).

  State arrays for a SINGLE replicate are unbatched; `run_sim` vmaps the per-replicate scan,
  adding a leading replicate axis. `Config` carries scalars/shared matrices (A, θ) that are
  identical across replicates and is held constant (closed over) inside the scan.

============================================================================================
 THE G vs M DISTINCTION  (the scientific core — see two_noise_framework_summary.md, m_vs_g)
============================================================================================
  * G = additive-genetic covariance of BREEDING VALUES across the CURRENT individuals.
        Breeding value of an individual = its ADDITIVE genotypic value = the sum over loci &
        alleles of allelic effects (the phenotype under the σ_eps=0 additive map; epistasis
        and environment are NOT part of the breeding value by the standard QG definition).
        G is a POPULATION STATISTIC of the standing genetic variation.
  * M = covariance of the MUTATIONAL EFFECTS ENTERING the population this generation, given
        the current genetic background. A NEW mutation of size α at locus i changes the
        phenotype by ΔX (the multilinear map below); M = Cov(ΔX) over the distribution of
        new mutations (which locus, which background). M is a property of the MUTATION
        PROCESS, not of the standing variation. For σ_eps = 0 this is the FIXED additive
        mutational covariance (background-independent) — the fixed-isotropic-M arm.
  These are different matrices with different roles; do not conflate them.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax


# ============================================================================================
# Config and State (pytrees)
# ============================================================================================

class Config(NamedTuple):
    """Run configuration. Scalars + shared matrices (identical across replicates).

    These are registered as a pytree (NamedTuple), so static/traced handling is automatic.
    Arrays (A, theta) are traced; the integer shape fields N, L, n_traits are treated as
    Python ints and used as static shapes (so jit recompiles if they change — intended).
    """
    N: int                # census population size (individuals per generation)
    L: int                # number of diploid loci
    n_traits: int         # number of quantitative traits (start n=2)
    mu: float             # per-locus, per-allele, per-generation mutation rate
    mut_var: float        # variance of a mutational increment (per trait, isotropic input)
    r: float              # recombination rate between adjacent loci (per-locus, free=0.5)
    sigma_eps: float      # SD of epistatic coefficients ε(i,j) ~ N(0, σ_eps²); 0 ⇒ additive
    A: jnp.ndarray        # selection-surface CURVATURE (n_traits, n_traits), PSD  [see header]
    theta: jnp.ndarray    # optimum phenotype (n_traits,)
    Ne: int               # effective number of reproducing parents (Ne <= N ⇒ drift)

    @property
    def shape_key(self):
        return (int(self.N), int(self.L), int(self.n_traits))


class State(NamedTuple):
    """Per-replicate population state (UNBATCHED; run_sim vmaps to add replicate axis).

    y        : per-locus diploid allelic effects, shape (N, L, 2, n_traits).
               y[ind, locus, allele, trait] is the additive effect that allele copy
               contributes to that trait.
    eps      : epistatic coefficient tensor, FIXED per run, shape (L, L, n_traits).
               eps[i, j, t] = ε(i,j) acting on trait t (ε(i,i) := 0). Drawn once at init.
    key      : the replicate's PRNGKey (threaded through the scan; never global state).
    """
    y: jnp.ndarray
    eps: jnp.ndarray
    key: jnp.ndarray


# ============================================================================================
# Initialization
# ============================================================================================

def init_population(key, config: Config) -> State:
    """Initialize one replicate's population.

    Allelic effects start at 0 (monomorphic, all individuals at phenotype 0); variation is
    then generated by mutation. (Starting monomorphic at 0 with θ possibly ≠ 0 lets us test
    the population MOVING TOWARD the peak — definition-of-done #4.)

    Epistatic coefficients ε(i,j) ~ N(0, σ_eps²) are drawn ONCE here and held fixed for the
    whole run (they are part of the genetic architecture, not a per-generation random).
    Diagonal ε(i,i) is forced to 0 (a locus has no epistatic effect on itself).
    """
    N, L, n = config.shape_key
    k_eps, k_state = jax.random.split(key)

    y = jnp.zeros((N, L, 2, n))

    eps = config.sigma_eps * jax.random.normal(k_eps, (L, L, n))
    # zero the diagonal ε(i,i) for every trait
    diag_mask = jnp.eye(L)[:, :, None]            # (L, L, 1)
    eps = eps * (1.0 - diag_mask)

    return State(y=y, eps=eps, key=k_state)


# ============================================================================================
# Genotype -> phenotype  (multilinear epistasis map; Hansen & Wagner 2001)
# ============================================================================================

def _additive_value(y):
    """Pure additive genotypic value = breeding value: sum over loci & alleles of effects.

    y : (..., L, 2, n)  ->  (..., n)
    This is exactly the phenotype map when σ_eps = 0, and it is the definition of an
    individual's BREEDING VALUE used for G (see header).
    """
    return jnp.sum(y, axis=(-3, -2))


def _multilinear_phenotype_one(y_ind, eps):
    """Multilinear-epistasis phenotype for ONE individual (Hansen & Wagner 2001).

    A mutation of size α at locus i contributes  ΔX = α·(1 + Σ_{j≠i} ε(i,j)·y(j)),
    where y(j) is the *background* (reference) genotypic state at locus j. We realize this at
    the level of standing allelic effects: each allele copy y[i, a] at locus i contributes

        contrib(i, a, t) = y[i, a, t] · ( 1 + Σ_{j≠i} ε(i, j, t) · ybg(j, t) )

    and the phenotype is the sum of all contributions over (i, a):

        z[t] = Σ_i Σ_a  y[i, a, t] · ( 1 + Σ_{j≠i} ε(i, j, t) · ybg(j, t) ).

    Background ybg(j, t): the locus-j genotypic state summed over its two allele copies,
    ybg(j, t) = y[j, 0, t] + y[j, 1, t]. ε(i,i) = 0 already, so the j≠i restriction is
    automatic and we sum over all j.

    σ_eps = 0  ⇒  eps ≡ 0  ⇒  factor (1 + 0) = 1  ⇒  z = Σ_i Σ_a y = additive (exact).

    y_ind : (L, 2, n) ;  eps : (L, L, n)  ->  z : (n,)
    """
    n = y_ind.shape[-1]
    ybg = jnp.sum(y_ind, axis=1)                       # (L, n)   background per locus per trait
    # epistatic modulation for each focal locus i, each trait t:
    #   modfac[i, t] = 1 + Σ_j ε(i, j, t) · ybg(j, t)
    # eps : (L_i, L_j, n) ; ybg : (L_j, n)  -> einsum over j
    mod = jnp.einsum("ijt,jt->it", eps, ybg)          # (L, n)
    modfac = 1.0 + mod                                 # (L, n)
    # contribution of each locus = (sum over its 2 alleles) * modfac
    contrib = ybg * modfac                             # (L, n)
    return jnp.sum(contrib, axis=0)                    # (n,)


def genotype_to_phenotype(state: State, config: Config) -> jnp.ndarray:
    """Phenotypes z for all individuals, shape (N, n_traits).

    Uses the multilinear epistasis map. With σ_eps = 0 (eps ≡ 0) this reduces EXACTLY to the
    pure additive map z = Σ_loci Σ_alleles y  (the fixed-isotropic-M arm). See header.
    """
    return jax.vmap(_multilinear_phenotype_one, in_axes=(0, None))(state.y, state.eps)


def breeding_values(state: State) -> jnp.ndarray:
    """Additive breeding values of all individuals, shape (N, n_traits).

    Breeding value = additive genotypic value = Σ over loci & alleles of allelic effects.
    This is the quantity whose across-individual covariance is G (NOT the epistatic phenotype).
    """
    return _additive_value(state.y)


# ============================================================================================
# Fitness (Gaussian stabilizing selection)
# ============================================================================================

def fitness(z, theta, A) -> jnp.ndarray:
    """Gaussian stabilizing selection.

        W(z) = exp( -1/2 (z - θ)ᵀ A (z - θ) ),   A = curvature (see header).

    z : (N, n) ; theta : (n,) ; A : (n, n)  ->  W : (N,).
    """
    d = z - theta                                     # (N, n)
    quad = jnp.einsum("ni,ij,nj->n", d, A, d)         # (N,)
    return jnp.exp(-0.5 * quad)


# ============================================================================================
# One generation
# ============================================================================================

def _segregate_gametes(key, parent_y, r):
    """Mendelian segregation WITH recombination to produce one gamete per parent.

    parent_y : (M, L, 2, n)  — M parents, each diploid at L loci.
    Returns  : (M, L, n)     — one haploid gamete per parent.

    Mechanism (linear chromosome, per-locus recombination rate r between ADJACENT loci):
      * Each parent has two homologs (allele axis = 2). A gamete is built by walking the loci
        and, at each locus, copying the currently-selected homolog; between adjacent loci a
        crossover (switch of homolog) happens with probability r. r = 0.5 ⇒ free recombination
        (every locus independent); r = 0 ⇒ full linkage (whole homolog inherited).
      * The starting homolog at locus 0 is chosen 50/50.
    Implemented as a cumulative parity of Bernoulli(r) switches → a 0/1 strand index per locus.
    """
    M, L, _, n = parent_y.shape
    k_start, k_sw = jax.random.split(key)
    start = jax.random.bernoulli(k_start, 0.5, (M, 1)).astype(jnp.int32)        # (M,1)
    # switches between locus l-1 and l, for l = 1..L-1 ; locus 0 has no preceding switch
    switches = jax.random.bernoulli(k_sw, r, (M, L)).astype(jnp.int32)
    switches = switches.at[:, 0].set(0)                                          # no switch before locus 0
    strand = (start + jnp.cumsum(switches, axis=1)) % 2                          # (M, L) in {0,1}
    # gather the chosen homolog at each locus
    gamete = jnp.take_along_axis(parent_y, strand[:, :, None, None], axis=2)     # (M, L, 1, n)
    return gamete[:, :, 0, :]                                                    # (M, L, n)


def _mutate_gametes(key, gametes, mu, mut_var):
    """Per-locus mutation with Gaussian increments.

    gametes : (M, L, n).  Each locus, each trait, mutates independently with probability `mu`;
    a mutation ADDS a Gaussian increment N(0, mut_var) to that locus's allelic effect (a
    "continuum-of-alleles" model). The mutational input is ISOTROPIC across traits (variance
    mut_var per trait, no cross-trait covariance) — this is the fixed-isotropic-M input.
    Returns mutated gametes of the same shape.
    """
    M, L, n = gametes.shape
    k_hit, k_inc = jax.random.split(key)
    hit = jax.random.bernoulli(k_hit, mu, (M, L, n)).astype(gametes.dtype)       # (M, L, n)
    inc = jnp.sqrt(mut_var) * jax.random.normal(k_inc, (M, L, n))
    return gametes + hit * inc


def step_generation(key, state: State, config: Config) -> State:
    """Advance one generation. jit-compatible (no Python-level data-dependent control flow).

    Steps:
      1. phenotypes z (multilinear map) and fitness W.
      2. Finite-Ne demographic step: choose Ne reproducing parents (uniform without-ish
         replacement via random choice) — only these contribute gametes. Ne < N injects
         demographic drift (a finite breeding pool). Ne = N ⇒ all individuals eligible.
      3. Fitness-weighted parent sampling: for each of N offspring draw 2 parents from the
         reproducing pool with probability ∝ W (selection acts through differential
         reproduction).
      4. Mendelian segregation with recombination (one gamete per chosen parent).
      5. Combine the two gametes into a diploid offspring; per-locus mutation.

    Returns the next-generation State (key advanced).
    """
    N, L, n = config.shape_key
    Ne = config.Ne

    key, k_pool, k_p1, k_p2, k_seg1, k_seg2, k_mut1, k_mut2 = jax.random.split(key, 8)

    z = genotype_to_phenotype(state, config)                       # (N, n)
    W = fitness(z, config.theta, config.A)                         # (N,)

    # --- finite-Ne demographic step: subsample the reproducing pool -------------------------
    # Choose Ne individuals (the breeding pool) uniformly at random from the N adults. With
    # Ne < N this is the demographic-drift injection: only a finite set reproduces, so the
    # parental gene pool is a finite sample (Wright-style). Ne = N ⇒ everyone is eligible.
    pool_idx = jax.random.choice(k_pool, N, shape=(Ne,), replace=False)          # (Ne,)
    pool_y = state.y[pool_idx]                                                   # (Ne, L, 2, n)
    pool_W = W[pool_idx]                                                         # (Ne,)
    # selection weights within the pool (fitness-proportional). Guard all-zero fitness.
    wsum = jnp.sum(pool_W)
    p = jnp.where(wsum > 0, pool_W / wsum, jnp.ones_like(pool_W) / Ne)           # (Ne,)

    # --- fitness-weighted parent sampling: 2 parents per offspring -------------------------
    par1 = jax.random.choice(k_p1, Ne, shape=(N,), replace=True, p=p)            # (N,)
    par2 = jax.random.choice(k_p2, Ne, shape=(N,), replace=True, p=p)            # (N,)
    par1_y = pool_y[par1]                                                        # (N, L, 2, n)
    par2_y = pool_y[par2]

    # --- segregation w/ recombination: one gamete from each parent -------------------------
    g1 = _segregate_gametes(k_seg1, par1_y, config.r)                            # (N, L, n)
    g2 = _segregate_gametes(k_seg2, par2_y, config.r)                            # (N, L, n)

    # --- mutation -------------------------------------------------------------------------
    g1 = _mutate_gametes(k_mut1, g1, config.mu, config.mut_var)
    g2 = _mutate_gametes(k_mut2, g2, config.mu, config.mut_var)

    # --- form diploid offspring: stack the two gametes on the allele axis ------------------
    new_y = jnp.stack([g1, g2], axis=2)                                          # (N, L, 2, n)

    return State(y=new_y, eps=state.eps, key=key)


# ============================================================================================
# Measurement (G, M, means, mean fitness)
# ============================================================================================

def _cov(x):
    """Population covariance of rows of x : (m, n) -> (n, n)  (divisor m, not m-1)."""
    xc = x - jnp.mean(x, axis=0, keepdims=True)
    return (xc.T @ xc) / x.shape[0]


def _mutational_effect_vectors(state: State, config: Config):
    """ΔX vectors for a NEW single mutation, over a sample of (locus, individual, trait-axis).

    For M we need the covariance of the mutational effect ΔX that a new mutation introduces to
    the PHENOTYPE given the current background. A unit-increment mutation at locus i, on trait
    axis t, in individual `ind`, perturbs that locus's allelic effect; under the multilinear
    map its phenotypic effect on trait t' is

        ∂z[t'] / ∂y[i, ·, t]  =  δ_{t,t'} · ( 1 + Σ_{j≠i} ε(i, j, t) · ybg(j, t) )  =: δ_{t,t'}·s(i,t)

    (epistasis modulates only the trait the locus codes for, by construction of the map; the
    cross-trait covariance of M arises only if the mutational INPUT is cross-trait-correlated,
    which here it is not — input is isotropic). So a mutation of random size α ~ N(0, mut_var)
    on axis t at locus i in individual ind produces the phenotypic vector

        ΔX = α · s(i, t) · e_t ,

    and M = Cov(ΔX) over the mutational input distribution (uniform locus, uniform individual,
    each of the n trait-axes equally likely, α ~ N(0, mut_var)). We return the per-axis scaled
    basis vectors  v = s(i,t)·e_t  so that  M = mut_var · E[ v vᵀ ].

    σ_eps = 0  ⇒  s(i,t) ≡ 1  ⇒  v = e_t  ⇒  M = mut_var · (1/n)·... ; see `measure` for the
    exact normalization. This is the FIXED isotropic additive mutational covariance.

    Returns S_mod : (N, L, n) where S_mod[ind, i, t] = s(i, t) for that individual's background.
    """
    # background per individual per locus per trait: sum over the 2 allele copies
    ybg = jnp.sum(state.y, axis=2)                                # (N, L, n)
    # s(i, t) for each individual = 1 + Σ_j ε(i, j, t) ybg(j, t)
    mod = jnp.einsum("ijt,njt->nit", state.eps, ybg)             # (N, L, n)
    return 1.0 + mod                                             # (N, L, n)


def measure(state: State, config: Config) -> dict:
    """Per-generation measurements.

    Returns a dict (leaves are jax arrays):
      G          : (n, n) additive-genetic covariance of BREEDING VALUES across individuals.
      M          : (n, n) covariance of mutational effects ENTERING the population given the
                   current background (see header & `_mutational_effect_vectors`).
      trait_mean : (n,)   mean PHENOTYPE (multilinear map) across individuals.
      mean_fitness : scalar mean Gaussian fitness.

    --- G estimator ---
      G = Cov(breeding_values) with population divisor N. Breeding value = additive genotypic
      value (Σ loci & alleles). This is the standing additive-genetic (co)variance — a
      property of the population, distinct from M.

    --- M estimator ---
      A new mutation of size α~N(0,mut_var) on trait-axis t at locus i in individual `ind`
      gives phenotypic effect ΔX = α·s(i,t)·e_t. Averaging over the mutational input
      distribution (i uniform over L, ind uniform over N, t uniform over n trait-axes,
      α independent with var mut_var):

          M = mut_var · (1/n) · diag_t( E_{i,ind}[ s(i,t)² ] ),

      i.e. M is diagonal with M[t,t] = mut_var · (1/n) · mean over (ind,i) of s(i,t)². The
      (1/n) reflects that any single new mutation hits ONE trait-axis (probability 1/n each).
      σ_eps = 0 ⇒ s ≡ 1 ⇒ M = (mut_var / n) · I  — FIXED, isotropic, background-independent.

      (Note: M is the per-mutation effect covariance. The total per-generation mutational
      INPUT to genetic variance scales this by the expected number of mutations 2·L·μ·n; we
      report the per-mutation effect covariance, which is the architecture-level object the
      M-evolution program tracks. The scaling is a constant for fixed L, μ, n and does not
      affect M's SHAPE/anisotropy, which is what Sims A/B measure.)
    """
    n = config.n_traits

    bv = breeding_values(state)                                  # (N, n)
    G = _cov(bv)                                                 # (n, n)

    z = genotype_to_phenotype(state, config)                    # (N, n)
    trait_mean = jnp.mean(z, axis=0)                            # (n,)

    W = fitness(z, config.theta, config.A)
    mean_fitness = jnp.mean(W)

    s = _mutational_effect_vectors(state, config)              # (N, L, n) = s(i,t)
    mean_s2 = jnp.mean(s ** 2, axis=(0, 1))                     # (n,)  E_{ind,i}[ s(i,t)² ]
    M = jnp.diag(config.mut_var * mean_s2 / n)                  # (n, n) diagonal

    return {
        "G": G,
        "M": M,
        "trait_mean": trait_mean,
        "mean_fitness": mean_fitness,
    }


# ============================================================================================
# Full simulation: scan over generations, vmap over replicates
# ============================================================================================

def _run_one_replicate(key, config: Config, n_generations: int):
    """Run a single replicate: scan over generations, recording `measure` each generation.

    Returns (final_state, measurements) where measurements is a dict of arrays each with a
    leading generation axis of length n_generations.
    """
    state0 = init_population(key, config)

    def body(state, _):
        new_state = step_generation(state.key, state, config)
        m = measure(new_state, config)
        return new_state, m

    final_state, traj = lax.scan(body, state0, xs=None, length=n_generations)
    return final_state, traj


def run_sim(key, config: Config, n_generations: int, n_replicates: int):
    """Top-level driver: vmap a per-replicate scan over `n_replicates` independent replicates.

    Layout: vmap(over replicates)[ scan(over generations)[ vectorized over individuals ] ].
    Each replicate gets an INDEPENDENT split key (RNG discipline) — this is what makes the
    vmap correct and the whole run reproducible from `key`.

    Returns a dict of stacked per-generation measurements, each leaf shaped
    (n_replicates, n_generations, ...). G and M are (n_replicates, n_generations, n, n);
    trait_mean (n_replicates, n_generations, n); mean_fitness (n_replicates, n_generations).
    """
    rep_keys = jax.random.split(key, n_replicates)

    run_fn = partial(_run_one_replicate, config=config, n_generations=n_generations)
    _, traj = jax.vmap(run_fn)(rep_keys)
    return traj


# ============================================================================================
# Convenience: a small helper to build a Config with sensible defaults / validated A
# ============================================================================================

def make_config(*, N, L, n_traits=2, mu=1e-3, mut_var=0.05, r=0.5, sigma_eps=0.0,
                A=None, omega=None, theta=None, Ne=None) -> Config:
    """Build a Config. Provide selection as curvature `A` OR width `omega` (A = ω⁻¹).

    Exactly one of `A`, `omega` may be given (boundary inversion is done HERE with a comment,
    per the A/ω convention — `omega` never propagates inward). If neither is given, A = I.
    """
    if A is not None and omega is not None:
        raise ValueError("Give selection as either A (curvature) or omega (width), not both.")
    if omega is not None:
        # BOUNDARY INVERSION: width -> curvature. omega must not propagate past this point.
        A = jnp.linalg.inv(jnp.asarray(omega, dtype=jnp.float32))
    if A is None:
        A = jnp.eye(n_traits)
    A = jnp.asarray(A, dtype=jnp.float32)
    if theta is None:
        theta = jnp.zeros((n_traits,))
    theta = jnp.asarray(theta, dtype=jnp.float32)
    if Ne is None:
        Ne = N
    return Config(N=int(N), L=int(L), n_traits=int(n_traits), mu=float(mu),
                  mut_var=float(mut_var), r=float(r), sigma_eps=float(sigma_eps),
                  A=A, theta=theta, Ne=int(Ne))

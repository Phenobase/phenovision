"""
evolvable_m.py — the EVOLVABLE-M individual-based model (the keystone of the SDE arm).

The engine (sim/engine.py) has a FIXED, isotropic mutational input: `_mutate_gametes` adds
independent N(0, mut_var) increments per (locus, trait). The (z̄,G,M) theory needs M to be a
heritable, anisotropic, ROTATABLE matrix that EVOLVES — which the engine cannot express. This
module adds that by COMPOSING engine primitives (engine.py is NOT edited):

  * each individual carries MODIFIER loci `m` (N, Lm, 2, P) whose additive breeding value
    b_mod (N, P) encodes that individual's per-EVENT mutational covariance S_ind (n×n SPD)
    via one of two metric-control designs;
  * focal-trait mutation is replaced by a per-individual CORRELATED injection
    (`mutate_gametes_anisotropic`): when a locus mutates it adds v ~ N(0, S_ind) (correlated
    across traits), so the realized per-generation mutational input to the breeding value is
        M_input = M_INPUT_SCALE * S_ind,   M_INPUT_SCALE = 2 * L * mu
    (2 gametes × L loci × per-locus mutation prob mu). THIS NORMALIZATION IS SHARED with the
    theory module: a target theory-M = M0 is placed by setting S_ind = M0 / M_INPUT_SCALE.

Metric-control designs (sde_validation_plan.md, "metric confound"):
  * 'eig_diag'  (MVV)         : P=n, b_mod = log-eigenvalues, S = diag(exp(b_mod)). Modifiers
                                act log-additively on eigenvalues -> 𝒢_M ∝ Fisher (design ii),
                                aligned (no rotation). The minimal viable version.
  * 'eig_rot'   (design ii)   : P = n + n(n-1)/2, adds rotation knob(s); for d=2 P=3
                                (ℓ1, ℓ2, ψ), S = R(ψ) diag(exp ℓ) R(ψ)ᵀ -> 𝒢_M ∝ Fisher.
  * 'chol'      (design i)    : P = n(n+1)/2, b_mod = Cholesky entries (diag via softplus),
                                S = L Lᵀ, SPD by construction, arbitrary 𝒢_M (measured).

Modifier loci segregate (engine._segregate_gametes) and mutate (engine._mutate_gametes, isotropic
in the modifier coordinate) like focal loci; the modifier mutational variance sets ε (the slow
timescale, V5) and 𝒢_M. RNG: explicit jax PRNGKey splits (engine discipline).
"""
from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from sim import engine
from sim.engine import (Config, State, _cov, _mutate_gametes, _segregate_gametes,
                        breeding_values, fitness, genotype_to_phenotype)


# ============================================================================================
# Evolvable-M state and hyperparameters
# ============================================================================================

class EvoState(NamedTuple):
    """Per-replicate state. `y` (focal) and `eps` mirror engine.State; `m` adds modifier loci.

    y   : (N, L, 2, n)  focal-trait diploid allelic effects (engine layout).
    m   : (N, Lm, 2, P) modifier diploid allelic effects; b_mod = Σ_{loci,alleles} m -> (N, P).
    eps : (L, L, n)     fixed multilinear-epistasis tensor (0 in the additive arm).
    key : PRNGKey.
    """
    y: jnp.ndarray
    m: jnp.ndarray
    eps: jnp.ndarray
    key: jnp.ndarray


class EvoHyper(NamedTuple):
    """Modifier-architecture hyperparameters (static ints / traced floats)."""
    Lm: int            # number of modifier loci
    P: int             # number of modifier parameters (= dim of b_mod; design-dependent)
    mu_mod: float      # per-locus modifier mutation rate (0 => frozen M; the V1/V2 fixed-M case)
    mut_var_mod: float # variance of a modifier mutational increment (sets ε and 𝒢_M scale)
    diversity_lambda: float = 0.0  # (legacy) imposed bet-hedging benefit ½λ logdet S; runs away
                                   # without a balancing cost. Prefer the EMERGENT challenge below.
    mut_load_coef: float = 0.0     # (legacy) imposed anisotropic mutation-load cost ½c tr(A S).
    # --- GENUINE bet-hedging: a shared per-generation random selective challenge (the "portfolio"
    #     / random-disaster mechanism). Each generation a random optimum θ_chal ~ N(0, σ²I) imposes
    #     extra HARD selection exp(-½ s |z-θ_chal|²); diverse families always have some offspring
    #     near θ_chal, clustered families occasionally get wiped out. Because lineage success is
    #     multiplicative across generations, avoiding wipeout earns the geometric-mean (logdet)
    #     premium -> the bet-hedging benefit EMERGES; the static A selection supplies the cost.
    challenge_strength: float = 0.0  # s: strength of the random per-generation challenge (0=off)
    challenge_sigma: float = 0.0     # σ: overall std of the random challenge optimum θ_chal
    challenge_aniso: jnp.ndarray = None  # (n,) per-trait relative std of θ_chal; Cov(θ_chal)=
                                         # diag((σ·aniso)²)=Ω_chal. None => isotropic (ones). Used
                                         # to test Channel 1 (M∝Ω) vs Channel 2 (M∝A⁻¹).
    # --- T1/T2 (notes_landscape_disaster_benefit.md §3.4): the SEVERITY/TAIL axis that selects the
    #     arithmetic regime (M∝A⁺, §3.1) vs the catastrophe regime (M∝A⁻¹, §3.3). Defaults reproduce
    #     the original soft-Gaussian-fecundity challenge exactly.
    challenge_tail: str = "gaussian"   # "gaussian" | "student_t": θ_chal magnitude tail. Heavy tail
                                       # (student_t, low df) => rare huge disasters dominate the
                                       # geometric mean => the log-det/catastrophe regime.
    challenge_df: float = 4.0          # Student-t degrees of freedom (only used when tail=student_t;
                                       # smaller = heavier; df→∞ recovers gaussian).
    challenge_mode: str = "fecundity"  # "fecundity" (soft exp-weight) | "viability" (hard survival
                                       # floor: outside challenge_radius the disaster kills, the
                                       # §3.3 unrecoverable-bad-generation mechanism).
    challenge_radius: float = 0.0      # viability survival radius (0 => default to challenge_sigma).


def n_modifier_params(design: str, n: int) -> int:
    if design == "eig_diag":
        return n
    if design == "eig_rot":
        return n + (n * (n - 1)) // 2
    if design == "chol":
        return (n * (n + 1)) // 2
    raise ValueError(f"unknown design {design!r}")


# ============================================================================================
# modifier breeding value -> per-event mutational covariance S_ind (and its Cholesky)
# ============================================================================================

def _softplus(x):
    return jax.nn.softplus(x) + 1e-6


def s_chol_from_bmod(b_mod, design: str, n: int):
    """Map ONE individual's modifier breeding value b_mod (P,) to the Cholesky factor L of its
    per-event mutational covariance S = L Lᵀ (n, n). vmappable."""
    if design == "eig_diag":
        # b_mod = log-eigenvalues; S = diag(exp(b_mod)); L = diag(exp(b_mod/2))
        return jnp.diag(jnp.exp(0.5 * b_mod))
    if design == "eig_rot":
        # b_mod = (log-eigenvalues..., rotation angle(s)); d=2: (l1, l2, psi)
        ell = b_mod[:n]
        if n == 2:
            psi = b_mod[2]
            c, s = jnp.cos(psi), jnp.sin(psi)
            R = jnp.array([[c, -s], [s, c]])
            S = R @ jnp.diag(jnp.exp(ell)) @ R.T
            return jnp.linalg.cholesky(S)
        raise NotImplementedError("eig_rot implemented for n=2")
    if design == "chol":
        # b_mod fills the lower triangle; diagonal via softplus for positivity
        L = jnp.zeros((n, n))
        idx = 0
        for i in range(n):
            for j in range(i + 1):
                val = b_mod[idx]
                val = _softplus(val) if i == j else val
                L = L.at[i, j].set(val)
                idx += 1
        return L
    raise ValueError(f"unknown design {design!r}")


def s_chol_all(m, design: str, n: int):
    """Per-individual Cholesky factors of S from the modifier genotypes m (N, Lm, 2, P)."""
    b_mod = jnp.sum(m, axis=(1, 2))                       # (N, P) additive modifier breeding value
    return jax.vmap(lambda b: s_chol_from_bmod(b, design, n))(b_mod), b_mod


def m_input_scale(config: Config, hyper: EvoHyper) -> float:
    """M_INPUT_SCALE = 2 L mu: total per-generation input cov = scale * S (shared with theory)."""
    return 2.0 * config.L * config.mu


# ============================================================================================
# anisotropic correlated mutation (replaces engine._mutate_gametes for the focal loci)
# ============================================================================================

def mutate_gametes_anisotropic(key, gametes, s_chol, mu):
    """Per-locus CORRELATED mutation. gametes (N, L, n); s_chol (N, n, n) per-individual Cholesky
    of the per-event covariance S_ind. Each locus mutates with prob `mu` (one event hitting all
    traits jointly); the increment is v = S_chol · z, z ~ N(0, I_n) (correlated across traits).
    Returns mutated gametes (N, L, n)."""
    N, L, n = gametes.shape
    k_hit, k_z = jax.random.split(key)
    hit = jax.random.bernoulli(k_hit, mu, (N, L, 1)).astype(gametes.dtype)     # (N, L, 1)
    z = jax.random.normal(k_z, (N, L, n))                                      # (N, L, n)
    inc = jnp.einsum("nij,nlj->nli", s_chol, z)                                # correlated increments
    return gametes + hit * inc


# ============================================================================================
# initialization
# ============================================================================================

def init_evo_population(key, config: Config, hyper: EvoHyper, design: str,
                        target_b_mod=None) -> EvoState:
    """Initialize one replicate. Focal alleles start at 0 (monomorphic). Modifier alleles are set
    so the additive modifier breeding value b_mod equals `target_b_mod` (P,) for every individual
    (split evenly across the 2*Lm modifier allele copies); default target = 0 (-> S = I)."""
    N, L, n = config.shape_key
    k_eps, k_rest = jax.random.split(key)

    y = jnp.zeros((N, L, 2, n))
    eps = config.sigma_eps * jax.random.normal(k_eps, (L, L, n))
    eps = eps * (1.0 - jnp.eye(L)[:, :, None])

    if target_b_mod is None:
        target_b_mod = jnp.zeros((hyper.P,))
    target_b_mod = jnp.asarray(target_b_mod, dtype=jnp.float32)
    per_allele = target_b_mod / (2.0 * hyper.Lm)                  # (P,)
    m = jnp.broadcast_to(per_allele, (N, hyper.Lm, 2, hyper.P)).astype(jnp.float32)

    return EvoState(y=y, m=m, eps=eps, key=k_rest)


# ============================================================================================
# one generation
# ============================================================================================

def step_generation_evolvable(key, state: EvoState, config: Config, hyper: EvoHyper,
                              design: str, theta=None) -> EvoState:
    """Advance one generation (jit/scan-compatible). Mirrors engine.step_generation for the
    finite-Ne pool, fitness-weighted parent sampling, segregation and recombination; the
    DIFFERENCES are: (a) modifier loci are inherited+mutated alongside focal loci, and (b) focal
    mutation is the per-individual CORRELATED injection whose covariance is the offspring's own
    modifier-encoded S_ind. `theta` overrides config.theta for this generation (V3 regimes:
    static / drifting / fluctuating optimum)."""
    N, L, n = config.shape_key
    Ne = config.Ne
    th = config.theta if theta is None else theta
    key, k_pool, k_p1, k_p2, k_seg1, k_seg2, k_segm1, k_segm2, k_mut1, k_mut2, k_mm1, k_mm2, k_chal = \
        jax.random.split(key, 13)

    # phenotype + fitness from the focal genotype (epistasis via engine; additive if sigma_eps=0)
    eng_state = State(y=state.y, eps=state.eps, key=key)
    z = genotype_to_phenotype(eng_state, config)                  # (N, n)
    W = fitness(z, th, config.A)                                  # (N,) static stabilizing selection

    # GENUINE bet-hedging (the portfolio / random-disaster mechanism): a SHARED per-generation
    # random challenge optimum θ_chal ~ N(0, σ²I) imposes extra hard selection. Diverse families
    # always have offspring near the random θ_chal; clustered families risk a total wipeout. The
    # benefit of diversity (and hence M -> A⁻¹, balanced against the static-A cost) EMERGES from
    # the multiplicative-across-generations structure -- no imposed cost/benefit term.
    if hyper.challenge_strength > 0.0:
        aniso = jnp.ones((n,)) if hyper.challenge_aniso is None else hyper.challenge_aniso
        # draw the disaster optimum θ_chal. gaussian: Ω_chal=diag((σ·aniso)²). student_t: a scale
        # mixture (isotropic direction, heavy-tailed magnitude) -> rare huge displacements (note §3.3).
        if hyper.challenge_tail == "student_t":
            k_chal, k_scale = jax.random.split(k_chal)
            zc = jax.random.normal(k_chal, (n,))
            chi = jax.random.chisquare(k_scale, hyper.challenge_df)         # shared scale mixer
            s_mix = jnp.sqrt(hyper.challenge_df / jnp.maximum(chi, 1e-6))
            theta_chal = hyper.challenge_sigma * aniso * zc * s_mix
        else:  # gaussian (default; reproduces the original challenge exactly)
            theta_chal = hyper.challenge_sigma * aniso * jax.random.normal(k_chal, (n,))
        d2 = jnp.sum((z - theta_chal[None, :]) ** 2, axis=1)               # |z-θ_chal|² per individual
        if hyper.challenge_mode == "viability":
            # hard survival floor: outside the radius the disaster kills (W->~0); in finite N a
            # single near-total-wipeout generation is unrecoverable (the §3.3 catastrophe mechanism).
            radius2 = (hyper.challenge_radius if hyper.challenge_radius > 0.0
                       else hyper.challenge_sigma) ** 2
            W = W * jnp.where(d2 <= radius2, 1.0, 1e-6)
        else:  # fecundity: soft Gaussian selection (the arithmetic-regime-friendly default)
            W = W * jnp.exp(-0.5 * hyper.challenge_strength * d2)

    # EXPLORATION regime: the net second-order load  ½c·tr(A S) - ½λ·logdet S  applied as a
    # per-individual selection differential. The benefit ½λ logdet S (bet-hedging / non-collapse)
    # rewards diversity direction-blindly; the cost ½c tr(A S) penalizes variance ∝ curvature.
    # Their balance is minimized at S ∝ A⁻¹ (s_i = λ/(c a_i)) -- the inverse-curvature fixed point.
    # The cost is imposed because the IBM's EMERGENT mutation load is too weak to balance the
    # benefit (it runs away). λ=0 => off (canalization / tracking use the emergent dynamics only).
    if hyper.diversity_lambda > 0.0:
        s_chol_par, _ = s_chol_all(state.m, design, n)            # (N, n, n)
        S = jnp.einsum("nij,nkj->nik", s_chol_par, s_chol_par)    # S = L Lᵀ per individual
        logdet_S = 2.0 * jnp.sum(jnp.log(jnp.diagonal(s_chol_par, axis1=1, axis2=2) + 1e-12), axis=1)
        trAS = jnp.einsum("ij,nji->n", config.A, S)               # tr(A S) per individual
        W = W * jnp.exp(0.5 * hyper.diversity_lambda * logdet_S - 0.5 * hyper.mut_load_coef * trAS)

    # finite-Ne breeding pool (demographic drift if Ne < N)
    pool_idx = jax.random.choice(k_pool, N, shape=(Ne,), replace=False)
    pool_y = state.y[pool_idx]                                    # (Ne, L, 2, n)
    pool_m = state.m[pool_idx]                                    # (Ne, Lm, 2, P)
    pool_W = W[pool_idx]
    wsum = jnp.sum(pool_W)
    p = jnp.where(wsum > 0, pool_W / wsum, jnp.ones_like(pool_W) / Ne)

    # fitness-weighted parents (SAME indices select focal AND modifier genotypes)
    par1 = jax.random.choice(k_p1, Ne, shape=(N,), replace=True, p=p)
    par2 = jax.random.choice(k_p2, Ne, shape=(N,), replace=True, p=p)

    # --- modifier inheritance FIRST: segregate + mutate -> offspring modifier genotype ---------
    mg1 = _segregate_gametes(k_segm1, pool_m[par1], config.r)     # (N, Lm, P)
    mg2 = _segregate_gametes(k_segm2, pool_m[par2], config.r)
    mg1 = _mutate_gametes(k_mm1, mg1, hyper.mu_mod, hyper.mut_var_mod)
    mg2 = _mutate_gametes(k_mm2, mg2, hyper.mu_mod, hyper.mut_var_mod)
    m_new = jnp.stack([mg1, mg2], axis=2)                         # (N, Lm, 2, P)

    # offspring per-event Cholesky from ITS OWN modifier breeding value
    s_chol, _ = s_chol_all(m_new, design, n)                      # (N, n, n)

    # --- focal inheritance: segregate, then CORRELATED anisotropic mutation --------------------
    g1 = _segregate_gametes(k_seg1, pool_y[par1], config.r)       # (N, L, n)
    g2 = _segregate_gametes(k_seg2, pool_y[par2], config.r)
    g1 = mutate_gametes_anisotropic(k_mut1, g1, s_chol, config.mu)
    g2 = mutate_gametes_anisotropic(k_mut2, g2, s_chol, config.mu)
    y_new = jnp.stack([g1, g2], axis=2)                           # (N, L, 2, n)

    return EvoState(y=y_new, m=m_new, eps=state.eps, key=key)


# ============================================================================================
# measurement
# ============================================================================================

def measure_evolvable(state: EvoState, config: Config, hyper: EvoHyper, design: str) -> dict:
    """Per-generation measurements for the evolvable-M IBM.

      G        : (n,n) additive-genetic covariance of breeding values (engine definition).
      M        : (n,n) realized per-generation mutational INPUT covariance
                 = M_INPUT_SCALE * mean_individuals(S_ind)  (shares the theory normalization).
      G_M      : (P,P) covariance of modifier breeding values (the metric on M-space, design i).
      b_mod_mean : (P,) mean modifier breeding value.
      trait_mean, mean_fitness, ln_mean_fitness : as in engine.measure (+ log for U_IBM=⟨ln W̄⟩).
    """
    n = config.n_traits
    G = _cov(breeding_values(state))

    s_chol, b_mod = s_chol_all(state.m, design, n)               # (N,n,n), (N,P)
    S_ind = jnp.einsum("nij,nkj->nik", s_chol, s_chol)           # S = L Lᵀ per individual
    M = m_input_scale(config, hyper) * jnp.mean(S_ind, axis=0)   # (n,n) realized input cov

    G_M = _cov(b_mod)                                            # (P,P) modifier metric
    b_mod_mean = jnp.mean(b_mod, axis=0)

    eng_state = State(y=state.y, eps=state.eps, key=state.key)
    z = genotype_to_phenotype(eng_state, config)
    trait_mean = jnp.mean(z, axis=0)
    W = fitness(z, config.theta, config.A)
    mean_fitness = jnp.mean(W)

    return {
        "G": G, "M": M, "G_M": G_M, "b_mod_mean": b_mod_mean,
        "trait_mean": trait_mean, "mean_fitness": mean_fitness,
        "ln_mean_fitness": jnp.log(mean_fitness + 1e-30),
    }


# ============================================================================================
# full simulation (scan over generations, vmap over replicates)
# ============================================================================================

def _run_one_replicate(key, config, hyper, design, n_generations, target_b_mod):
    state0 = init_evo_population(key, config, hyper, design, target_b_mod)

    def body(state, _):
        new_state = step_generation_evolvable(state.key, state, config, hyper, design)
        return new_state, measure_evolvable(new_state, config, hyper, design)

    final_state, traj = lax.scan(body, state0, xs=None, length=n_generations)
    return final_state, traj


def run_evo_sim(key, config: Config, hyper: EvoHyper, design: str, n_generations: int,
                n_replicates: int, target_b_mod=None, return_final=False):
    """Vmap a per-replicate scan over independent replicates (engine layout/RNG discipline).
    Returns a dict of per-generation measurements (n_replicates, n_generations, ...); if
    return_final, also returns the final EvoState (for the ensemble-drift estimator)."""
    rep_keys = jax.random.split(key, n_replicates)
    run_fn = partial(_run_one_replicate, config=config, hyper=hyper, design=design,
                     n_generations=n_generations, target_b_mod=target_b_mod)
    final, traj = jax.vmap(run_fn)(rep_keys)
    return (traj, final) if return_final else traj


def _run_one_from_state(state, config, hyper, design, theta_seq):
    """Continue ONE replicate from a given EvoState for len(theta_seq) generations, applying a
    per-generation optimum theta_seq (G, n). Returns (final_state, per-gen measurements)."""
    def body(s, theta_t):
        ns = step_generation_evolvable(s.key, s, config, hyper, design, theta=theta_t)
        return ns, measure_evolvable(ns, config, hyper, design)
    return lax.scan(body, state, theta_seq)


def continue_evo_sim(states_batched, config: Config, hyper: EvoHyper, design: str, theta_seq):
    """Continue a BATCH of replicate states (leading replicate axis, e.g. from run_evo_sim with
    return_final) for theta_seq.shape[0] generations under a shared per-generation optimum
    schedule theta_seq (n_generations, n). Returns (traj, final) — traj leaves have a leading
    (replicate, generation, ...) shape. Used by V3: frozen-burn-in equilibration (run_evo_sim,
    mu_mod=0) -> release modifiers (this, mu_mod>0) and measure the slow M-drift."""
    run_fn = partial(_run_one_from_state, config=config, hyper=hyper, design=design,
                     theta_seq=theta_seq)
    final, traj = jax.vmap(run_fn)(states_batched)
    return traj, final


def make_hyper(*, design="eig_diag", n_traits=2, Lm=6, mu_mod=0.0, mut_var_mod=0.01,
               diversity_lambda=0.0, mut_load_coef=0.0,
               challenge_strength=0.0, challenge_sigma=0.0, challenge_aniso=None,
               challenge_tail="gaussian", challenge_df=4.0,
               challenge_mode="fecundity", challenge_radius=0.0) -> EvoHyper:
    """Build an EvoHyper, computing P from the design and n_traits."""
    aniso = None if challenge_aniso is None else jnp.asarray(challenge_aniso, jnp.float32)
    return EvoHyper(Lm=int(Lm), P=int(n_modifier_params(design, n_traits)),
                    mu_mod=float(mu_mod), mut_var_mod=float(mut_var_mod),
                    diversity_lambda=float(diversity_lambda), mut_load_coef=float(mut_load_coef),
                    challenge_strength=float(challenge_strength),
                    challenge_sigma=float(challenge_sigma), challenge_aniso=aniso,
                    challenge_tail=str(challenge_tail), challenge_df=float(challenge_df),
                    challenge_mode=str(challenge_mode), challenge_radius=float(challenge_radius))

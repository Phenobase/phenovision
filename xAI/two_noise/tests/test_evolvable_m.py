"""Keystone gates for the evolvable-M IBM (sim/evolvable_m.py). Mirrors test_engine.py style;
JAX CPU; small/fast configs. THE critical gate is realized-M == encoded-M (the injection loop)."""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

from sim.engine import make_config
from sim.evolvable_m import (
    make_hyper, n_modifier_params, s_chol_from_bmod, mutate_gametes_anisotropic,
    m_input_scale, run_evo_sim, measure_evolvable, init_evo_population,
    step_generation_evolvable,
)


def test_design_param_counts():
    assert n_modifier_params("eig_diag", 2) == 2
    assert n_modifier_params("eig_rot", 2) == 3
    assert n_modifier_params("chol", 2) == 3


def test_anisotropic_injection_covariance():
    # THE injection mechanism: increments v = S_chol z have empirical Cov ≈ S = L Lᵀ.
    n = 2
    L = jnp.array([[1.0, 0.0], [0.5, 0.8]])
    S = L @ L.T
    N, Lloci = 6000, 2
    s_chol = jnp.broadcast_to(L, (N, n, n))
    gametes = jnp.zeros((N, Lloci, n))
    out = mutate_gametes_anisotropic(jax.random.PRNGKey(0), gametes, s_chol, mu=1.0)
    inc = np.asarray(out - gametes).reshape(-1, n)     # (N*Lloci, n) increments
    emp = np.cov(inc, rowvar=False)
    assert np.allclose(emp, np.asarray(S), atol=0.05), (emp, np.asarray(S))


def test_realized_M_equals_encoded():
    # frozen modifiers (mu_mod=0) at a target encoding S0 -> measured M == M_INPUT_SCALE * S0.
    config = make_config(N=400, L=10, n_traits=2, mu=0.01, A=jnp.eye(2))
    hyper = make_hyper(design="eig_diag", n_traits=2, Lm=4, mu_mod=0.0)
    S0 = np.array([0.04, 0.01])                         # target per-event eigenvalues (diagonal S)
    target_b = jnp.log(jnp.array(S0))                   # eig_diag: b_mod = log eigenvalues
    traj = run_evo_sim(jax.random.PRNGKey(1), config, hyper, "eig_diag",
                       n_generations=5, n_replicates=2, target_b_mod=target_b)
    M = np.asarray(traj["M"][:, -1])                    # (n_rep, n, n)
    expected = m_input_scale(config, hyper) * np.diag(S0)
    assert np.allclose(M.mean(0), expected, atol=1e-6), (M.mean(0), expected)


def test_determinism():
    config = make_config(N=300, L=8, n_traits=2, mu=0.01, A=jnp.eye(2))
    hyper = make_hyper(design="eig_diag", n_traits=2, Lm=4, mu_mod=1e-2, mut_var_mod=0.01)
    a = run_evo_sim(jax.random.PRNGKey(7), config, hyper, "eig_diag", 6, 3)
    b = run_evo_sim(jax.random.PRNGKey(7), config, hyper, "eig_diag", 6, 3)
    assert np.allclose(np.asarray(a["M"]), np.asarray(b["M"]))
    assert np.allclose(np.asarray(a["G"]), np.asarray(b["G"]))


def test_spd_always_chol_design():
    # chol design must keep every per-individual S SPD across a run with active modifier mutation
    config = make_config(N=300, L=8, n_traits=2, mu=0.02, A=jnp.eye(2))
    hyper = make_hyper(design="chol", n_traits=2, Lm=5, mu_mod=0.05, mut_var_mod=0.05)
    traj, final = run_evo_sim(jax.random.PRNGKey(3), config, hyper, "chol", 30, 2,
                              return_final=True)
    # reconstruct S per individual from the final state's modifiers and check SPD
    from sim.evolvable_m import s_chol_all
    for rep in range(2):
        m_rep = final.m[rep]
        s_chol, _ = s_chol_all(m_rep, "chol", 2)
        S = jnp.einsum("nij,nkj->nik", s_chol, s_chol)
        eigs = np.linalg.eigvalsh(np.asarray(S))
        assert (eigs > 0).all(), eigs.min()
    # realized M is finite and SPD-ish
    M = np.asarray(traj["M"][:, -1]).mean(0)
    assert np.all(np.isfinite(M)) and np.linalg.eigvalsh(M).min() > 0


def test_heritability_M_moves_and_GM_positive():
    # with active modifier mutation, modifier variance accumulates (𝒢_M grows) and M evolves
    config = make_config(N=400, L=10, n_traits=2, mu=0.02, A=jnp.diag(jnp.array([1.0, 4.0])))
    hyper = make_hyper(design="eig_diag", n_traits=2, Lm=6, mu_mod=0.1, mut_var_mod=0.05)
    traj = run_evo_sim(jax.random.PRNGKey(5), config, hyper, "eig_diag", 120, 4)
    GM = np.asarray(traj["G_M"])                        # (rep, gen, P, P)
    # modifier genetic variance accumulates from 0
    assert np.trace(GM[:, -1].mean(0)) > np.trace(GM[:, 2].mean(0))
    # M's eigenvalues change over the run (it evolves)
    M0 = np.asarray(traj["M"][:, 5]).mean(0)
    MT = np.asarray(traj["M"][:, -1]).mean(0)
    assert not np.allclose(np.linalg.eigvalsh(M0), np.linalg.eigvalsh(MT), atol=1e-4)


def test_eig_design_metric_is_isotropic():
    # design (ii) eig_diag with isotropic modifier mutation -> 𝒢_M isotropic in log-eig space
    # (the sufficient condition that makes 𝒢_M ∝ Fisher). Off-diagonal small vs diagonal.
    config = make_config(N=500, L=8, n_traits=2, mu=0.02, A=jnp.eye(2))
    hyper = make_hyper(design="eig_diag", n_traits=2, Lm=8, mu_mod=0.1, mut_var_mod=0.05)
    traj = run_evo_sim(jax.random.PRNGKey(9), config, hyper, "eig_diag", 100, 4)
    GM = np.asarray(traj["G_M"][:, -1]).mean(0)         # (2,2) in log-eigenvalue space
    offdiag = abs(GM[0, 1]); diag = 0.5 * (GM[0, 0] + GM[1, 1])
    assert offdiag < 0.4 * diag, (GM,)                  # approximately isotropic

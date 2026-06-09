"""V3 — the M-drift direction (the core second-order-selection mechanism test).

Protocol (sde_validation_plan.md V3 + V0 ensemble estimator):
  1. encode a fixed M0 in FROZEN modifiers and burn in so the fast variables (z̄, G) reach
     their conditional equilibrium at M0 (run_evo_sim, mu_mod=0);
  2. RELEASE the modifiers (mu_mod>0) and run τ generations under the regime's optimum schedule
     (static / drifting / fluctuating); the mean ΔM over R replicates estimates the conditional
     drift Ê[ΔM|M0];
  3. compare the drift DIRECTION to the theory:
       design (ii) eig_*  -> the natural gradient  -2 M ∇ℒ_eff M  (𝒢_M ∝ Fisher), so the
                             direction is invariant to 𝒢_M's magnitude;
       design (i) chol    -> -𝒢_M ∇ℒ_eff with the MEASURED 𝒢_M.
  4. DECOMPOSE the drift into eigenvalue-SCALING (canalization) and eigenvector-ROTATION
     (alignment) parts (drift_estimator.split_scaling_rotation).

The drift DIRECTION is invariant to V2's constant closure factor (~2.8x): a constant c on Ĝ
scales ℒ_eff and its gradient by c, leaving the direction unchanged.

Regimes / predicted fixed points (the drift's zeros):
  static       -> canalization: M shrinks (∇ℒ_eff>0), fastest in steep directions.
  exploration  -> M ∝ A⁻¹ (fluctuating optimum makes variation valuable; the α=1 analog).
  tracking     -> M ∝ Ω (optimum drifts with covariance Ω).

CPU burst-friendly (JAX). Writes runs/v3_drift/results.csv.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp

from sim.engine import make_config
from sim.evolvable_m import make_hyper, m_input_scale, run_evo_sim, continue_evo_sim
from sim.theory_load_surface import natural_gradient_M, grad_L_eff_vech, vech, unvech
from sim.drift_estimator import split_scaling_rotation, cos_direction


def _theta_seq(regime, tau, n, env_sigma, key):
    # static, canalization, AND exploration all use a STATIC peak (theta=0); exploration differs
    # only by the diversity premium (Force 1 + non-collapse), NOT a fluctuating optimum. Only the
    # TRACKING regime moves the optimum (Force 3, lag-load).
    if regime in ("static", "canalization", "exploration"):
        return jnp.zeros((tau, n))
    if regime == "tracking":
        # ISOTROPIC random-walk (drifting) optimum -> lag-load (Force 3) -> M grows toward the
        # movement covariance Ω∝I (isotropic), contrasting with exploration's A⁻¹ and canalization.
        return jnp.cumsum(env_sigma * jax.random.normal(key, (tau, n)), axis=0)
    raise ValueError(regime)


def measure_drift(M0, A, *, design="eig_diag", regime="static", N=600, L=12, mu=0.02,
                  burn_in=400, tau=60, n_replicates=64, Lm=8, mu_mod=0.25, mut_var_mod=0.02,
                  N_star=1e6, env_sigma=0.0, seed=0, lam=1.0, Omega=None,
                  diversity_lambda=0.0, mut_load_coef=0.0,
                  challenge_strength=0.0, challenge_sigma=0.0):
    n = 2
    A = np.asarray(A, float)
    config = make_config(N=N, L=L, n_traits=n, mu=mu, A=jnp.asarray(A, jnp.float32), Ne=N)
    scale = m_input_scale(config, make_hyper(design=design, n_traits=n, Lm=Lm))
    M0 = np.asarray(M0, float)
    # encode M0 in frozen modifiers (eig designs: b_mod = log eigenvalues of S0 = M0/scale)
    S0 = M0 / scale
    if design == "eig_diag":
        target_b = jnp.log(jnp.asarray(np.diag(S0), jnp.float32))
    elif design == "eig_rot":
        w, _ = np.linalg.eigh(S0)
        target_b = jnp.asarray(np.concatenate([np.log(w), [0.0]]), jnp.float32)
    else:  # chol
        Lc = np.linalg.cholesky(S0)
        # invert softplus on the diagonal so the encoded factor reproduces S0
        import math
        diag = np.array([math.log(math.expm1(max(Lc[i, i], 1e-6))) for i in range(n)])
        target_b = jnp.asarray(np.array([diag[0], Lc[1, 0], diag[1]]), jnp.float32)

    hyper_frozen = make_hyper(design=design, n_traits=n, Lm=Lm, mu_mod=0.0)
    hyper_active = make_hyper(design=design, n_traits=n, Lm=Lm, mu_mod=mu_mod,
                              mut_var_mod=mut_var_mod, diversity_lambda=diversity_lambda,
                              mut_load_coef=mut_load_coef, challenge_strength=challenge_strength,
                              challenge_sigma=challenge_sigma)
    k0, k1, k2 = jax.random.split(jax.random.PRNGKey(seed), 3)
    # phase 1: frozen burn-in -> fast variables equilibrated at M0
    _, final = run_evo_sim(k1, config, hyper_frozen, design, burn_in, n_replicates,
                           target_b_mod=target_b, return_final=True)
    # phase 2: release modifiers, apply regime optimum schedule, measure M trajectory
    theta_seq = _theta_seq(regime, tau, n, env_sigma, k2)
    traj2, _ = continue_evo_sim(final, config, hyper_active, design, theta_seq)
    M_traj = np.asarray(traj2["M"])                      # (rep, tau, n, n)
    dM = (M_traj[:, -1] - M0).mean(axis=0)               # mean drift matrix Ê[ΔM]
    GM = np.asarray(traj2["G_M"][:, -1]).mean(axis=0)

    # theory drift direction
    bargs = {}
    if regime == "exploration":
        bargs = dict(lam=lam)
    elif regime == "tracking":
        bargs = dict(Omega=np.asarray(Omega) if Omega is not None else np.eye(n))
    th_regime = {"canalization": "static"}.get(regime, regime)
    theory = np.asarray(natural_gradient_M(M0, A, N_star, th_regime, **bargs))  # design (ii)

    split = split_scaling_rotation(M0, dM)
    theory_split = split_scaling_rotation(M0, theory)
    return dict(
        regime=regime, design=design, M0=M0, dM=dM, GM=GM, theory=theory,
        cos_drift_vs_theory=cos_direction(vech(dM), vech(theory)),
        scaling_frac=split["scaling_frac"], rotation_frac=split["rotation_frac"],
        cos_scaling=cos_direction(vech(split["scaling"]), vech(theory_split["scaling"])),
        trM0=float(np.trace(M0)), trM_end=float(np.trace(M_traj[:, -1].mean(0))),
        M_end=M_traj[:, -1].mean(0),
    )


def _write_csv(path, rows):
    import csv
    flat = []
    for r in rows:
        flat.append(dict(regime=r["regime"], design=r["design"],
                         cos_drift_vs_theory=r["cos_drift_vs_theory"],
                         scaling_frac=r["scaling_frac"], rotation_frac=r["rotation_frac"],
                         trM0=r["trM0"], trM_end=r["trM_end"],
                         M0_11=r["M0"][0, 0], M0_22=r["M0"][1, 1],
                         Mend_11=r["M_end"][0, 0], Mend_22=r["M_end"][1, 1]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0].keys()))
        w.writeheader(); w.writerows(flat)
    print("wrote", path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--N", type=int, default=800)
    p.add_argument("--L", type=int, default=12)
    p.add_argument("--burn-in", type=int, default=500)
    p.add_argument("--tau", type=int, default=80)
    p.add_argument("--replicates", type=int, default=96)
    p.add_argument("--out-dir", default=str(ROOT / "runs" / "v3_drift"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    A = np.diag([1.0, 4.0])
    M0 = np.diag([0.06, 0.06])
    rows = []
    for regime, env in [("static", 0.0), ("exploration", 0.6)]:
        r = measure_drift(M0, A, design="eig_diag", regime=regime, N=args.N, L=args.L,
                          burn_in=args.burn_in, tau=args.tau, n_replicates=args.replicates,
                          env_sigma=env, seed=args.seed)
        rows.append(r)
        print(f"[v3] regime={regime:12s} cos(drift,theory)={r['cos_drift_vs_theory']:.3f} "
              f"scaling_frac={r['scaling_frac']:.2f} trM {r['trM0']:.4f}->{r['trM_end']:.4f} "
              f"M_end_diag=({r['M_end'][0,0]:.4f},{r['M_end'][1,1]:.4f})")
    _write_csv(Path(args.out_dir) / "results.csv", rows)


if __name__ == "__main__":
    main()

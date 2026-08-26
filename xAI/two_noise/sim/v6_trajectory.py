"""V6 — full trajectory / distribution match (the strongest, most expensive test; done last).

Integrate the slow M-flow SDE  dM = rate · (-2 M ∇ℒ_eff M) dt + diffusion  (the natural-gradient
flow, design ii) by Euler–Maruyama and compare M(t) to the evolvable-M IBM ensemble:
  * the deterministic mean M(t) trajectory (eigenvalue paths), and
  * a sliced-Wasserstein distance on the M-eigenvalue distributions across the ensemble.

The single `rate` constant absorbs V2's constant closure factor (~2.8) and the modifier-genetics
timescale (units are free as long as consistent); it is fit so the SDE matches the IBM's overall
pace, then the SHAPE of the trajectory and the distributional spread are the test.

CPU/local (numpy SDE + JAX IBM). Writes runs/v6_trajectory/results.csv.
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
from sim.evolvable_m import make_hyper, m_input_scale, run_evo_sim
from sim.theory_load_surface import natural_gradient_M


def sliced_wasserstein(A, B, n_proj=64, seed=0):
    """Sliced-Wasserstein-1 distance between two point clouds A (na,d), B (nb,d)."""
    A = np.atleast_2d(A); B = np.atleast_2d(B)
    d = A.shape[1]
    rng = np.random.default_rng(seed)
    dirs = rng.standard_normal((n_proj, d))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-30
    tot = 0.0
    for u in dirs:
        pa = np.sort(A @ u); pb = np.sort(B @ u)
        m = min(len(pa), len(pb))
        qa = np.quantile(pa, np.linspace(0, 1, m)); qb = np.quantile(pb, np.linspace(0, 1, m))
        tot += np.mean(np.abs(qa - qb))
    return float(tot / n_proj)


def euler_maruyama_M(M0, A, regime, rate, n_steps, dt, diffusion=0.0, seed=0, **bargs):
    """Integrate dM = rate·natural_grad(M) dt + sqrt(diffusion·dt)·sym-noise; project to SPD.
    Returns the M trajectory (n_steps+1, d, d) for one path."""
    from sim.drift_estimator import _project_spd
    rng = np.random.default_rng(seed)
    d = M0.shape[0]
    M = M0.copy(); traj = [M.copy()]
    for _ in range(n_steps):
        drift = rate * np.asarray(natural_gradient_M(M, A, 1e6, regime, **bargs))
        M = M + dt * drift
        if diffusion > 0:
            z = rng.standard_normal((d, d)); z = 0.5 * (z + z.T)
            M = M + np.sqrt(diffusion * dt) * z
        M = _project_spd(M)
        traj.append(M.copy())
    return np.array(traj)


def ibm_trajectory(M0, A, regime, *, N=600, L=12, mu=0.02, Lm=8, mu_mod=0.25, mut_var_mod=0.02,
                   n_generations=300, n_replicates=64, env_sigma=0.0, seed=0):
    """IBM M(t) ensemble: run_evo_sim with ACTIVE modifiers from M0. Returns M traj (rep, gen, d, d)
    and the per-generation optimum schedule applied (static here; regimes handled in v3 via theta)."""
    config = make_config(N=N, L=L, n_traits=2, mu=mu, A=jnp.asarray(A, jnp.float32), Ne=N)
    hyper = make_hyper(design="eig_diag", n_traits=2, Lm=Lm, mu_mod=mu_mod, mut_var_mod=mut_var_mod)
    scale = m_input_scale(config, hyper)
    target_b = jnp.log(jnp.asarray(np.diag(M0) / scale, jnp.float32))
    traj = run_evo_sim(jax.random.PRNGKey(seed), config, hyper, "eig_diag", n_generations,
                       n_replicates, target_b_mod=target_b)
    return np.asarray(traj["M"])


def compare(M0, A, regime="static", n_generations=300, **kw):
    M_ibm = ibm_trajectory(M0, A, regime, n_generations=n_generations, **kw)   # (rep, gen, d, d)
    mean_ibm = M_ibm.mean(axis=0)                                              # (gen, d, d)
    # fit the SDE rate so its end-point trace change matches the IBM's (timescale calibration)
    d_tr_ibm = np.trace(mean_ibm[-1]) - np.trace(M0)
    probe = euler_maruyama_M(M0, A, regime, rate=1.0, n_steps=n_generations, dt=1.0)
    d_tr_sde = np.trace(probe[-1]) - np.trace(M0)
    rate = float(d_tr_ibm / d_tr_sde) if abs(d_tr_sde) > 1e-9 else 1.0
    M_sde = euler_maruyama_M(M0, A, regime, rate=rate, n_steps=n_generations, dt=1.0)
    # trajectory shape agreement: correlation of the two M-eigenvalue paths over time
    ev_ibm = np.linalg.eigvalsh(mean_ibm)                                      # (gen, d) ascending
    ev_sde = np.linalg.eigvalsh(M_sde[1:])                                     # (gen, d)
    corr = np.corrcoef(ev_ibm.ravel(), ev_sde.ravel())[0, 1]
    # distributional: SW on final-time M-eigenvalues, IBM ensemble vs an SDE+diffusion ensemble
    final_ibm_ev = np.linalg.eigvalsh(M_ibm[:, -1])                            # (rep, d)
    diff = float(np.mean(np.var(final_ibm_ev, axis=0)))                        # match ensemble spread
    sde_ens = np.array([euler_maruyama_M(M0, A, regime, rate=rate, n_steps=n_generations, dt=1.0,
                                         diffusion=diff / n_generations, seed=s)[-1]
                        for s in range(final_ibm_ev.shape[0])])
    final_sde_ev = np.linalg.eigvalsh(sde_ens)
    sw = sliced_wasserstein(final_ibm_ev, final_sde_ev)
    return dict(rate=rate, traj_corr=float(corr), sliced_wasserstein=sw,
                trM0=float(np.trace(M0)), trM_ibm_end=float(np.trace(mean_ibm[-1])),
                trM_sde_end=float(np.trace(M_sde[-1])))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--N", type=int, default=800)
    p.add_argument("--L", type=int, default=12)
    p.add_argument("--generations", type=int, default=400)
    p.add_argument("--replicates", type=int, default=96)
    p.add_argument("--out-dir", default=str(ROOT / "runs" / "v6_trajectory"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    A = np.diag([1.0, 4.0]); M0 = np.diag([0.06, 0.06])
    rows = []
    for regime in ("static",):
        r = compare(M0, A, regime=regime, N=args.N, L=args.L,
                    n_generations=args.generations, n_replicates=args.replicates, seed=args.seed)
        r["regime"] = regime; rows.append(r)
        print(f"[v6] regime={regime} traj_corr={r['traj_corr']:.3f} SW={r['sliced_wasserstein']:.4f} "
              f"trM IBM {r['trM0']:.4f}->{r['trM_ibm_end']:.4f} SDE->{r['trM_sde_end']:.4f}")
    import csv
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("wrote", out / "results.csv")


if __name__ == "__main__":
    main()

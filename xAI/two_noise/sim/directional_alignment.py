"""Directional-selection alignment: the natural-gradient benefit measured DURING the chase
(transient), not at mutation-selection-balance. This is the faithful evolutionary analog of an
optimizer DESCENDING toward a distant/moving target (it is never at stationarity).

The breeder's equation gives the response directly:
    Δz̄ = G β,    β = -A(z̄ - θ),    so the Newton / straight-to-optimum direction is
    A⁻¹β = (θ - z̄).
With the standing variance G = Ĝ(M) = A⁻¹#M, choosing M ∝ A^{1-2α} gives G ∝ A^{-α} (the SAME
α knob as the optimizer's precond_power), and the realized response is
    Δz̄ ∝ A^{-α} · A (θ-z̄) = A^{1-α}(θ-z̄):
  * α=1 (M∝A⁻¹, G∝A⁻¹):   Δz̄ ∝ (θ-z̄)        -> STRAIGHT to the optimum (Newton step).
  * α=1/2 (M∝I, G∝A^{-1/2}): Δz̄ ∝ A^{1/2}(θ-z̄) -> partly curved (whitening).
  * α=0 (M∝A, G∝I):         Δz̄ ∝ A(θ-z̄)       -> the raw gradient (curves to steep dirs).

We set a DISTANT optimum, let the standing G reach its fast conditional equilibrium, and measure,
during the approach: (i) cos(Δz̄, θ-z̄) — how straight the population heads to the optimum (the
natural-gradient / Newton alignment), and (ii) the adaptation speed (fitness gain / lag). No
contrived environment: the inverse-curvature benefit (α=1 = straight + fastest) is intrinsic to
directional selection, and second-order selection that rewards faster adaptation is what would
drive evolving M toward A⁻¹ (the evolve-M meta-loop's biological counterpart).

CPU/JAX. Writes runs/directional_alignment/results.csv.
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


def _M_for_alpha(A, alpha, m_scale):
    """M ∝ A^{1-2α} (normalized to geometric-mean eigenvalue m_scale) so that G = Ĝ(M) ∝ A^{-α}."""
    a = np.diag(A)
    m = a ** (1.0 - 2.0 * alpha)
    m = m / np.exp(np.mean(np.log(m))) * m_scale       # set geometric mean to m_scale
    return m


def run_alpha(A, alpha, theta, *, N=800, L=14, mu=0.03, Lm=4, n_generations=400,
              n_replicates=24, m_scale=0.08, seed=0):
    """Frozen M ∝ A^{1-2α}; distant optimum theta; measure the response alignment with the
    straight-to-optimum (Newton) direction during the approach, plus adaptation speed."""
    n = A.shape[0]
    config = make_config(N=N, L=L, n_traits=n, mu=mu, A=jnp.asarray(A, jnp.float32),
                         theta=jnp.asarray(theta, jnp.float32), Ne=N)
    hyper = make_hyper(design="eig_diag", n_traits=n, Lm=Lm, mu_mod=0.0)   # frozen M
    scale = m_input_scale(config, hyper)
    m = _M_for_alpha(A, alpha, m_scale)
    target_b = jnp.log(jnp.asarray(m / scale, jnp.float32))
    traj = run_evo_sim(jax.random.PRNGKey(seed), config, hyper, "eig_diag",
                       n_generations, n_replicates, target_b_mod=target_b)
    zbar = np.asarray(traj["trait_mean"])              # (rep, gen, n)
    th = np.asarray(theta)
    A_np = np.asarray(A, float)
    # per-generation response and directions, averaged over replicates and the approach window
    cos_newton, cos_grad, dist = [], [], []
    for rep in range(zbar.shape[0]):
        z = zbar[rep]                                  # (gen, n)
        dz = np.diff(z, axis=0)                         # (gen-1, n) response Δz̄
        to_opt = th[None, :] - z[:-1]                   # (gen-1, n) straight-to-optimum
        grad = (A_np @ to_opt.T).T                      # β-direction = A(θ-z̄)
        for t in range(dz.shape[0]):
            if np.linalg.norm(to_opt[t]) < 0.2 * np.linalg.norm(th):
                continue                                # stop counting once essentially arrived
            d = dz[t]
            if np.linalg.norm(d) < 1e-8:
                continue
            cos_newton.append(d @ to_opt[t] / (np.linalg.norm(d) * np.linalg.norm(to_opt[t])))
            cos_grad.append(d @ grad[t] / (np.linalg.norm(d) * np.linalg.norm(grad[t])))
        dist.append(np.linalg.norm(z - th[None, :], axis=1))
    dist = np.array(dist).mean(axis=0)                  # mean distance-to-optimum over time
    # generations to halve the initial distance (adaptation speed)
    d0 = dist[0]
    half = np.argmax(dist < 0.5 * d0)
    t_half = int(half) if dist.min() < 0.5 * d0 else n_generations
    return dict(alpha=alpha, cos_newton=float(np.mean(cos_newton)),
                cos_grad=float(np.mean(cos_grad)), t_half=t_half,
                final_dist=float(dist[-1]), init_dist=float(d0))


def _write_csv(path, rows):
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("wrote", path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--N", type=int, default=900)
    p.add_argument("--L", type=int, default=14)
    p.add_argument("--generations", type=int, default=500)
    p.add_argument("--replicates", type=int, default=32)
    p.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    p.add_argument("--out-dir", default=str(ROOT / "runs" / "directional_alignment"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    A = np.diag([1.0, 6.0])                             # anisotropic (cond 6)
    theta = np.array([4.0, 4.0])                        # distant optimum
    rows = []
    for alpha in args.alphas:
        r = run_alpha(A, alpha, theta, N=args.N, L=args.L, n_generations=args.generations,
                      n_replicates=args.replicates, seed=args.seed)
        rows.append(r)
        print(f"[dir] alpha={alpha:.2f} (M∝A^{1-2*alpha:+.0f}, G∝A^{-alpha:+.1f})  "
              f"cos(Δz̄, straight-to-opt)={r['cos_newton']:.3f}  cos(Δz̄, gradient)={r['cos_grad']:.3f}  "
              f"gens-to-halve={r['t_half']}")
    _write_csv(Path(args.out_dir) / "results.csv", rows)
    print("\nPrediction: cos(Δz̄, straight-to-optimum) rises toward 1 as α→1 (G∝A⁻¹ = Newton =\n"
          "straight to the optimum & fastest), and is lower at α=1/2 (whitening) and α=0 (gradient).")


if __name__ == "__main__":
    main()

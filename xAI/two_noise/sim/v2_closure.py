"""V2 — the fast equilibria and the Gaussian closure (fixed-M).

Validates the substitutions V1's "version A" relies on, and CHARACTERIZES the ~2.6x closure
discrepancy V1 found. At fixed M (frozen modifiers) measure:

  * the realized equilibrium G  vs  the analytic Ĝ = A⁻¹#M  (the MSB closure). Report:
      - SHAPE agreement: Frobenius cosine <G,Ĝ>/(||G|| ||Ĝ||) and the principal angle between
        their leading eigenvectors (does G align with Ĝ even if the SCALE is off?);
      - SCALE factor: (det G / det Ĝ)^(1/n). The V1 finding predicts this is ~constant across M
        (=> a single calibratable coefficient: the theory holds up to a constant, which the doc
        permits) rather than varying with M / A-anisotropy (=> genuine closure breakdown).
  * the stationary mean-displacement covariance Σ_∞ = Cov_t(z̄)  vs  the closed form
      Σ_∞ = (v/2n)A⁻¹ + (1/2N*)G  (C=A). The basic engine has N*→∞, so Σ_∞ ≈ (v/2n)A⁻¹: we
      check the SHAPE (∝ A⁻¹) and fit the demographic coefficient v/2n = tr(Σ_∞ A)/n.
  * the GAUSSIAN CLOSURE directly: phenotype skew and excess kurtosis along principal axes
      (≈0 if the population is Gaussian; large => the house-of-cards regime).

Run for COMMUTING (diagonal A) and NON-COMMUTING (rotated A, diagonal M) cases — the closed-form
Σ_∞ and Ĝ are claimed to hold in general. CPU/local (JAX). Writes runs/v2_closure/results.csv.
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

from sim.engine import make_config, State, genotype_to_phenotype
from sim.evolvable_m import make_hyper, m_input_scale, run_evo_sim
from sim.theory_load_surface import ghat


def _frob_cos(X, Y):
    x = np.ravel(X); y = np.ravel(Y)
    return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-30))


def _principal_angle(X, Y):
    """Angle (deg) between the leading eigenvectors of two symmetric matrices."""
    _, Vx = np.linalg.eigh(X); _, Vy = np.linalg.eigh(Y)
    c = abs(float(Vx[:, -1] @ Vy[:, -1]))
    return float(np.degrees(np.arccos(min(1.0, c))))


def _scale_factor(G, Ghat):
    n = G.shape[0]
    dG = np.linalg.det(G); dH = np.linalg.det(Ghat)
    return float((dG / dH) ** (1.0 / n)) if dH > 0 else float("nan")


def _phenotype_moments(final_state, config):
    """Skew and excess kurtosis of the phenotype along each principal axis, at the final gen,
    averaged over replicates (final_state has a leading replicate axis)."""
    skews, kurts = [], []
    nrep = final_state.y.shape[0]
    for rep in range(nrep):
        st = State(y=final_state.y[rep], eps=final_state.eps[rep], key=final_state.key[rep])
        z = np.asarray(genotype_to_phenotype(st, config))     # (N, n)
        zc = z - z.mean(0)
        C = np.cov(zc, rowvar=False)
        _, V = np.linalg.eigh(C)
        proj = zc @ V                                          # principal-axis coordinates
        sd = proj.std(0) + 1e-12
        skews.append((((proj / sd) ** 3).mean(0)))
        kurts.append((((proj / sd) ** 4).mean(0) - 3.0))
    return np.mean(skews, axis=0), np.mean(kurts, axis=0)


def run_fixed_M(A, M_diag_eigs, *, rotate_A=0.0, N=600, L=12, mu=0.02, n_generations=600,
                burn_in=350, n_replicates=12, Lm=4, Ne=None, seed=0):
    """One fixed-M run. M is placed as diag(M_diag_eigs) in TRAIT coordinates; A may be rotated
    by `rotate_A` radians off the trait axes (non-commuting case)."""
    n = 2
    A0 = np.diag(np.asarray(A, float))
    if rotate_A:
        c, s = np.cos(rotate_A), np.sin(rotate_A)
        R = np.array([[c, -s], [s, c]])
        A0 = R @ A0 @ R.T
    config = make_config(N=N, L=L, n_traits=n, mu=mu, A=jnp.asarray(A0, jnp.float32),
                         Ne=Ne if Ne is not None else N)
    hyper = make_hyper(design="eig_diag", n_traits=n, Lm=Lm, mu_mod=0.0)  # frozen M
    scale = m_input_scale(config, hyper)
    M = np.diag(np.asarray(M_diag_eigs, float))               # theory M (per-gen input cov)
    S0 = np.diag(M) / scale
    target_b = jnp.log(jnp.asarray(S0, jnp.float32))
    traj, final = run_evo_sim(jax.random.PRNGKey(seed), config, hyper, "eig_diag",
                              n_generations, n_replicates, target_b_mod=target_b,
                              return_final=True)
    G = np.asarray(traj["G"][:, burn_in:].mean(axis=1)).mean(axis=0)    # (n,n) equilibrium G
    zbar = np.asarray(traj["trait_mean"][:, burn_in:])                  # (rep, gen', n)
    zbar = zbar.reshape(-1, n)
    Sigma_inf = np.cov(zbar, rowvar=False)                             # Cov_t(z-bar)
    Gh = np.asarray(ghat(A0, M))
    skew, kurt = _phenotype_moments(final, config)
    # demographic coefficient fit: Sigma_inf ≈ (v/2n) A^-1  =>  v/2n = tr(Sigma_inf A)/n
    v_over_2n = float(np.trace(Sigma_inf @ A0) / n)
    return dict(
        rotate_A=rotate_A, m1=float(M[0, 0]), m2=float(M[1, 1]),
        G=G, Ghat=Gh, Sigma_inf=Sigma_inf, A=A0,
        shape_cos=_frob_cos(G, Gh), principal_angle_deg=_principal_angle(G, Gh),
        scale_factor=_scale_factor(G, Gh),
        sigma_shape_cos=_frob_cos(Sigma_inf, np.linalg.inv(A0)), v_over_2n=v_over_2n,
        skew_max=float(np.max(np.abs(skew))), kurt_max=float(np.max(np.abs(kurt))),
    )


def closure_scan(**kw):
    """Commuting (rotate_A=0) and non-commuting (rotate_A=0.6 rad) cases over a few M."""
    A = [1.0, 3.0]
    M_list = [(0.04, 0.04), (0.08, 0.03), (0.03, 0.09), (0.10, 0.06)]
    rows = []
    for rot in (0.0, 0.6):
        for m in M_list:
            r = run_fixed_M(A, m, rotate_A=rot, **kw)
            rows.append(r)
            print(f"[v2] rot={rot:.1f} M=diag{m}  shape_cos={r['shape_cos']:.3f} "
                  f"angle={r['principal_angle_deg']:.1f}deg scale={r['scale_factor']:.2f} "
                  f"Sigma~A^-1 cos={r['sigma_shape_cos']:.3f} skew={r['skew_max']:.2f} "
                  f"kurt={r['kurt_max']:.2f}")
    return rows


def _write_csv(path, rows):
    import csv
    flat = []
    for r in rows:
        flat.append(dict(rotate_A=r["rotate_A"], m1=r["m1"], m2=r["m2"],
                         shape_cos=r["shape_cos"], principal_angle_deg=r["principal_angle_deg"],
                         scale_factor=r["scale_factor"], sigma_shape_cos=r["sigma_shape_cos"],
                         v_over_2n=r["v_over_2n"], skew_max=r["skew_max"], kurt_max=r["kurt_max"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0].keys()))
        w.writeheader(); w.writerows(flat)
    print("wrote", path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--N", type=int, default=600)
    p.add_argument("--L", type=int, default=12)
    p.add_argument("--generations", type=int, default=700)
    p.add_argument("--burn-in", type=int, default=400)
    p.add_argument("--replicates", type=int, default=16)
    p.add_argument("--out-dir", default=str(ROOT / "runs" / "v2_closure"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    rows = closure_scan(N=args.N, L=args.L, n_generations=args.generations,
                        burn_in=args.burn_in, n_replicates=args.replicates, seed=args.seed)
    _write_csv(Path(args.out_dir) / "results.csv", rows)
    sf = [r["scale_factor"] for r in rows]
    print(f"\n[v2] closure SCALE factor: mean={np.mean(sf):.2f} std={np.std(sf):.2f} "
          f"(constant => single calibratable coefficient)")
    print(f"[v2] shape cos mean={np.mean([r['shape_cos'] for r in rows]):.3f} "
          f"(G aligns with Ĝ); Σ∞~A⁻¹ cos mean={np.mean([r['sigma_shape_cos'] for r in rows]):.3f}")


if __name__ == "__main__":
    main()

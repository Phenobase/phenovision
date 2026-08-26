"""O1 — the alpha=1/2 stability boundary, from first principles (CPU, local).

Closes the Phase-1 loop: we found empirically that full-power (alpha=1) preconditioning
diverges on ViT; Part III of riccati_optimizer_and_plan.md explains it as update-noise
variance ~ a_i^{1-2 alpha} per curvature eigendirection. alpha<1/2 suppresses flat-direction
noise, alpha=1/2 is the boundary, alpha>1/2 amplifies it (worst at small batch), alpha=1
amplifies as a_i^{-1}. Shrinkage toward isotropy reduces the effective exponent and cures it.

This module reproduces that law two ways:
  (1) `flat_dir_amplification` — the doc's diagonal-quadratic diagnostic (numpy), the cheapest
      reproduction: preconditioned SGD with A-shaped gradient noise; returns flat/steep wander.
  (2) `stationary_noise_slope` — fits the per-eigendirection stationary variance vs curvature;
      the slope is the predicted 1-2 alpha (reuses toy.quadratic_sde.fit_loglog_slope).
  (3) `riccati_flat_dir` — runs the REAL optim.riccati_precond.RiccatiPrecond (inverse mode,
      sweeping shrink) on the same separable quadratic, mapping shrink -> effective exponent
      (this also feeds O3).

CLI writes runs/flat_dir/o1_amplification.csv (alpha x batch wander + fitted slope) and
runs/flat_dir/o3_realized_exponent.csv (shrink -> realized exponent), consumed by figures/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toy.quadratic_sde import fit_loglog_slope  # noqa: E402

RUNS = ROOT / "runs" / "flat_dir"


# --------------------------------------------------------------------------- #
# (1) the doc's diagonal diagnostic
# --------------------------------------------------------------------------- #

def flat_dir_amplification(alpha, B, d=24, T=4000, eta=0.05, beta=0.95,
                           delta=1e-8, shrink=0.0, sigma2=1.0, seed=0,
                           burn_in=1000, true_curvature=False):
    """Preconditioned SGD on a diagonal quadratic with A-shaped gradient noise.

    Returns (flat_wander, steep_wander, a, vpos, vupd):
      flat/steep_wander : |x| in the flattest/steepest direction at the end,
      a                 : curvature eigenvalues,
      vpos              : stationary POSITION variance per direction (~ a^{-alpha}),
      vupd              : stationary UPDATE-NOISE variance per direction (~ a^{1-2 alpha}).
    The headline alpha=1/2 law is in vupd: alpha>1/2 amplifies flat directions, worse at
    low B; shrinkage toward isotropy reduces it (cures the full-inverse instability).
    """
    r = np.random.default_rng(seed)
    a = np.logspace(-3, 1, d)
    x = np.ones(d)
    C = np.full(d, 1e-3)
    pos_acc = np.zeros(d)
    upd_noise_acc = np.zeros(d)     # E[(eta C^{-a} xi)^2], xi = gradient noise (mean 0)
    nacc = 0
    for t in range(T):
        noise = r.standard_normal(d) * np.sqrt(a * sigma2 / B)
        g = a * x + noise
        C = beta * C + (1 - beta) * g * g
        if true_curvature:
            # use the TRUE curvature A as the preconditioner basis (C = A, the score
            # identity) -> isolates the theoretical update-noise law var ~ a^{1-2 alpha}.
            Cs = (1 - shrink) * a + shrink * a.mean()
        else:
            Cs = (1 - shrink) * C + shrink * C.mean()
        precond = (Cs + delta) ** (-alpha)
        x = x - eta * precond * g
        if t >= burn_in:
            pos_acc += x * x
            upd_noise = eta * precond * noise
            upd_noise_acc += upd_noise * upd_noise
            nacc += 1
    vpos = pos_acc / max(nacc, 1)
    vupd = upd_noise_acc / max(nacc, 1)
    return float(abs(x[0])), float(abs(x[-1])), a, vpos, vupd


def stationary_noise_slope(alpha, B, true_curvature=True, **kw):
    """Log-log slope of the UPDATE-NOISE variance vs curvature; predicted = 1 - 2 alpha.
    Defaults to true_curvature=True (C=A) to isolate the theoretical law; with the empirical
    EMA preconditioner the slope deviates at high alpha (C != A away from the optimum)."""
    _, _, a, _, vupd = flat_dir_amplification(alpha, B, true_curvature=true_curvature, **kw)
    slope, _ = fit_loglog_slope(a, vupd)
    return float(slope)


def sweep_alpha_batch(alphas=(0.0, 0.25, 0.5, 0.75, 1.0),
                      batches=(16, 64, 256, 1024), **kw):
    """Tidy rows: {alpha, batch, flat_wander, steep_wander, noise_slope, predicted_slope}.
    noise_slope is the fitted update-noise slope; predicted_slope = 1 - 2 alpha."""
    rows = []
    for B in batches:
        for al in alphas:
            flat, steep, a, _, vupd = flat_dir_amplification(al, B, **kw)
            slope, _ = fit_loglog_slope(a, vupd)
            rows.append(dict(alpha=al, batch=B, flat_wander=flat, steep_wander=steep,
                             noise_slope=slope, predicted_slope=1 - 2 * al))
    return rows


def alpha_star_of_batch(batches=(16, 64, 256, 1024), dim=24, cond_number=1e3,
                        n_steps=1500, seed=0, **_ignored):
    """alpha*(B): the noise-dependent optimal exponent, increasing with batch.

    Delegates to the established finite-budget-loss machinery in toy.quadratic_sde
    (analytic_optimal_alpha), which balances convergence speed (favors high alpha) against
    noise amplification (favors low alpha) on an isotropic-test objective -- the same curve
    the shared figure already overlays as sde_curve.csv. The flat-direction diagnostic above
    cannot locate alpha* because it ignores convergence speed (pure noise always prefers
    alpha=0); this is why O1's alpha*(B) is sourced from the proper loss surface."""
    from toy.quadratic_sde import make_quadratic, analytic_optimal_alpha
    quad = make_quadratic(dim=dim, cond_number=cond_number, seed=seed,
                          rotate=False, center=True)
    S = np.asarray(batches, dtype=float)
    astar, _ = analytic_optimal_alpha(quad, S, n_steps=n_steps, test_curvature="isotropic")
    astar = np.atleast_1d(astar)
    return [dict(batch=int(b), alpha_star=float(a)) for b, a in zip(batches, astar)]


# --------------------------------------------------------------------------- #
# (3) the REAL optimizer: shrink -> realized exponent (also O3)
# --------------------------------------------------------------------------- #

def riccati_flat_dir(shrink, d=16, steps=60, seed=0):
    """Run RiccatiPrecond (inverse mode) on a diagonal SPD curvature and read the
    realized preconditioner exponent = -slope(log eig(G) vs log eig(C)). Returns the
    realized exponent; shrink=0 -> ~1, increasing shrink reduces it toward 0."""
    import torch
    from optim.riccati_precond import _ns_inverse
    eigs = np.logspace(-1, 0, d)               # max eig 1.0 (NS converges from I)
    C = torch.diag(torch.tensor(eigs, dtype=torch.float64))
    eye = torch.eye(d, dtype=torch.float64)
    G = _ns_inverse(C, eye, steps=steps, eta_p=0.5, rho=float(shrink), safeguard=8.0)
    log_c = np.log(eigs)
    log_g = np.log(np.diag(G.numpy()))
    slope = np.polyfit(log_c, log_g, 1)[0]
    return float(-slope)


def realized_exponent_sweep(rhos=(0.0, 0.2, 0.4, 0.6, 0.8, 0.95), **kw):
    return [dict(shrink=r, realized_exponent=riccati_flat_dir(r, **kw)) for r in rhos]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _write_csv(path, rows):
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default=str(RUNS))
    p.add_argument("--T", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    out = Path(args.out_dir)

    amp = sweep_alpha_batch(T=args.T, seed=args.seed)
    _write_csv(out / "o1_amplification.csv", amp)
    astar = alpha_star_of_batch(T=args.T, seed=args.seed)
    _write_csv(out / "o1_alpha_star.csv", astar)
    rex = realized_exponent_sweep()
    _write_csv(out / "o3_realized_exponent.csv", rex)

    # quick console summary
    print("\nalpha* by batch:", {r["batch"]: r["alpha_star"] for r in astar})
    print("realized exponent by shrink:",
          {r["shrink"]: round(r["realized_exponent"], 3) for r in rex})


if __name__ == "__main__":
    main()

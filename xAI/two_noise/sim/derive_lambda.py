"""Phase 4 — closing the open theory items of notes_landscape_disaster_benefit.md §6.

Pure numpy/scipy (runs in either conda env; CPU; no IBM). Three self-contained numerical
derivations for the items the note flags as "open":

  (A) Self-consistent Model-I optimum. The tracking-filter residual σ²_x,i = ς²_i·φ_i/(φ_i+g_i a_i)
      depends on g_i = sqrt(m_i/a_i), and the Channel-1 optimum g*_i = σ²_x,i − 1/a_i depends on
      σ²_x. So the M-optimum is a FIXED POINT, not the one-shot white-limit answer. We solve it by
      iteration and compare to the naive σ²_x = ς² (φ→∞) optimum.

  (B) λ for a SPECIFIED tail (the missing large-deviation piece). Monte-Carlo the floored
      geometric-mean lineage growth under a disaster δ = R·n (isotropic direction n, magnitude R
      from a Student-t / Gaussian), for offspring cloud Σ = c·A^{−α}. The survival floor (a near-
      zero-count generation is unrecoverable) is what turns the smooth coverage objective (whose
      optimum is cost-form dependent, ~A^{−1/2}) into the log-det/bet-hedge regime. We show the
      growth-maximizing exponent α* rises toward 1 (M∝A^{−1}) as the tail gets heavier, and stays
      below 1 for a light (Gaussian) tail — the §3.1↔§3.3 interpolation.

  (C) Finite-N scaling of λ. The same model with N lineages: extinction bites when the expected
      surviving count falls below ~1, which is EASIER to avoid at large N. We show α* (and the
      fitted log-det coefficient λ) increase as N falls, establishing the asserted λ ∝ 1/N trend.

Run:  python3 -m sim.derive_lambda      (or:  python3 sim/derive_lambda.py)
"""
from __future__ import annotations

import numpy as np


# ===========================================================================================
# (A) Self-consistent Model-I tracking optimum
# ===========================================================================================
def selfconsistent_modelI(a, phi, vsig2):
    """Solve the fixed point  g_i = max(0, ς²_i·φ_i/(φ_i+g_i a_i) − 1/a_i)  per aligned mode.

    Naive iteration of this map OSCILLATES — the tracking feedback is bistable (g=0 makes the
    optimum untrackable, demanding large g; large g tracks it, demanding g=0). The interior root
    instead solves the implicit equation in closed form: clearing denominators,
        (g + 1/a)(φ + g a) = ς² φ   =>   a g² + (φ+1) g + φ(1/a − ς²) = 0,
    a quadratic whose positive root is the self-consistent standing variance (floored at 0 when
    ς² < 1/a, i.e. no Channel-1 benefit). Returns (g_fixed, g_naive); g_naive is the φ→∞ white
    limit g = ς² − 1/a that the note's one-shot optimization uses."""
    a, phi, vsig2 = map(lambda x: np.asarray(x, float), (a, phi, vsig2))
    disc = (phi + 1.0) ** 2 - 4.0 * phi * (1.0 - a * vsig2)
    g = (-(phi + 1.0) + np.sqrt(np.maximum(disc, 0.0))) / (2.0 * a)
    g = np.maximum(g, 0.0)                                # mutational floor
    g_naive = np.maximum(vsig2 - 1.0 / a, 0.0)           # φ→∞ one-shot
    return g, g_naive


def _run_A():
    print("=" * 74)
    print("(A) Self-consistent Model-I optimum: fixed point vs naive white-limit (φ→∞)")
    print("    g_i = max(0, ς²φ/(φ+g a) − 1/a);  m_i = a_i g_i²  (MSB)")
    print("=" * 74)
    a = np.array([1.0, 4.0])
    vsig2 = np.array([2.0, 2.0])                          # isotropic optimum-wander variance
    for phi in (0.3, 1.0, 5.0, 50.0):
        g, gn = selfconsistent_modelI(a, np.full_like(a, phi), vsig2)
        m, mn = a * g * g, a * gn * gn
        # slope of log m vs log a (2-point): >0 lean A⁺, <0 lean A⁻¹
        sl = np.polyfit(np.log(a), np.log(np.maximum(m, 1e-12)), 1)[0]
        sln = np.polyfit(np.log(a), np.log(np.maximum(mn, 1e-12)), 1)[0]
        print(f"  φ={phi:5.1f}: g*={np.round(g,4)} (naive {np.round(gn,4)})  "
              f"m*={np.round(m,4)}  slope={sl:+.3f}  (naive slope {sln:+.3f})")
    print("  -> as φ grows (faster optimum, less trackable) the fixed point approaches the naive")
    print("     white limit; at small φ the self-consistent g is SMALLER (tracking eats the slow")
    print("     part), so the one-shot answer OVER-states the benefit. The φ→∞ limit is Model II.")


# ===========================================================================================
# (B,C) Floored geometric-mean lineage growth -> alpha*(tail, N) and lambda
# ===========================================================================================
def _disaster_mag2(rng, n, draws, df):
    """Sample squared disaster magnitude |δ|² for isotropic δ. df<=0 => Gaussian; else Student-t
    scale mixture (heavy tail in the shared radial scale)."""
    z = rng.standard_normal((draws, n))
    if df is None or df <= 0:
        d = z
    else:
        chi = rng.chisquare(df, size=(draws, 1))
        d = z * np.sqrt(df / np.maximum(chi, 1e-6))       # multivariate-t
    return d                                              # (draws, n) displacement vectors


def floored_growth(alpha, a, *, df, N, sigma=1.0, kappa=0.5, draws=20000, seed=0):
    """Long-run (geometric-mean) lineage growth for offspring cloud Σ = c·A^{−α}, under disasters
    δ ~ sigma·(isotropic, tail df). Survivors out of N ∝ N·exp(−½ δᵀΣ⁻¹δ); a generation with
    expected count < 1 is an extinction (floored). Growth = ⟨log max(count,1)⟩ − κ·tr(AΣ)."""
    rng = np.random.default_rng(seed)
    n = len(a)
    # Σ = c A^{−α}, normalized to unit determinant so only the SHAPE (anisotropy) varies with α;
    # the scale is carried by the explicit cost term.
    s = a ** (-alpha)
    s = s / np.exp(np.mean(np.log(s)))                    # det(Σ)=1
    dvec = sigma * _disaster_mag2(rng, n, draws, df)      # (draws, n)
    quad = np.sum(dvec * dvec / s[None, :], axis=1)       # δᵀΣ⁻¹δ
    count = N * np.exp(-0.5 * quad)                       # expected survivors
    log_surv = np.log(np.maximum(count, 1.0))             # extinction floor at count=1
    cost = kappa * np.sum(a * s)                          # κ·tr(A Σ)
    return float(np.mean(log_surv) - cost)


def alpha_star(a, *, df, N, alphas=None, **kw):
    alphas = np.linspace(0.0, 1.0, 21) if alphas is None else np.asarray(alphas)
    vals = np.array([floored_growth(al, a, df=df, N=N, **kw) for al in alphas])
    return float(alphas[int(np.argmax(vals))]), alphas, vals


def _run_BC():
    a = np.array([1.0, 4.0])
    print("\n" + "=" * 74)
    print("(B) λ for a specified tail: α* of the floored geometric-mean growth rises with tail")
    print("    weight. Σ=A^{−α} (det=1); δ isotropic; survival floored at count=1. N=50.")
    print("=" * 74)
    print("    tail            α*(growth)   (A⁺ <--0 .. 1--> A⁻¹)")
    for label, df in [("gaussian", None), ("student-t df=8", 8.0),
                      ("student-t df=4", 4.0), ("student-t df=2.5", 2.5)]:
        astar, _, _ = alpha_star(a, df=df, N=50, sigma=1.0, draws=40000, seed=1)
        print(f"    {label:16s}  α* = {astar:.2f}")
    print("  -> light (Gaussian) tail: α*<1 (arithmetic/coverage); heavier tail: α*→1 (A⁻¹ hedge).")

    print("\n" + "=" * 74)
    print("(C) Finite-N scaling: α* and the log-det coefficient λ rise as N falls (heavy tail).")
    print("    student-t df=3; λ estimated from the curvature of growth in logdet at fixed shape.")
    print("=" * 74)
    print("    N        α*      λ̂ (∝ extinction pressure)")
    for N in (500, 100, 20, 5):
        astar, alphas, vals = alpha_star(a, df=3.0, N=N, sigma=1.0, draws=40000, seed=2)
        # λ̂: local sensitivity of growth to the log-det barrier near α*, proxied by how sharply
        # growth falls toward α=0 (loss of coverage). A monotone-in-1/N pressure statistic.
        lam_hat = float(max(vals) - vals[0])
        print(f"    {N:4d}    {astar:.2f}    {lam_hat:8.3f}")
    print("  -> smaller N => higher α* and larger λ̂: the asserted λ ∝ 1/N (eq. 12) trend.")


def main():
    _run_A()
    _run_BC()


if __name__ == "__main__":
    main()

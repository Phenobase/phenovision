"""Sim C (§3.3): the natural-gradient identity and the CMA-ES bridge — three mechanisms
(biological M-evolution, the breeder's-equation response, rank-mu CMA-ES) reaching the same
C ∝ A^{-1} attractor.

This module implements the two mechanisms that don't need the JAX engine:
 - rank-mu CMA-ES (pycma) on a quadratic: covariance C -> A^{-1} (Akimoto et al. 2012).
 - the natural-gradient vs realized-response check, given G and the selection gradient beta
   (callers pass G/beta measured from the engine; here we provide the comparison math).
The biological M-evolution -> A^{-1} panel reuses Sim A's G-vs-A result (runs/sim_a).

Runs in either env (depends only on numpy + cma, both pure-python-importable).
"""

from __future__ import annotations

import numpy as np


def cma_covariance_vs_hessian(A: np.ndarray, n_gen: int = 400, sigma0: float = 0.5,
                              popsize: int = None, seed: int = 0):
    """Run CMA-ES minimizing L(x)=½ xᵀAx and track how its covariance C aligns with A^{-1}.

    Returns dict with per-generation: the matrix-cosine between (normalized) C and A^{-1}, and the
    log-log slope of eig(C) vs eig(A) (predicted -> -1, i.e. C ∝ A^{-1}). Akimoto et al. (2012):
    rank-mu CMA-ES is natural-gradient ascent on the Gaussian manifold; C -> A^{-1} on quadratics.
    """
    import cma
    d = A.shape[0]
    Ainv = np.linalg.inv(A)
    Ainv_n = Ainv / np.linalg.norm(Ainv)
    es = cma.CMAEvolutionStrategy(d * [1.0], sigma0, {
        'verbose': -9, 'seed': seed + 1,
        **({'popsize': popsize} if popsize else {}),
        'maxiter': n_gen,
    })
    cosines, slopes, gens = [], [], []
    g = 0
    while not es.stop() and g < n_gen:
        sols = es.ask()
        es.tell(sols, [0.5 * float(x @ A @ x) for x in sols])
        C = es.C * (es.sigma ** 2)  # full search covariance
        Cn = C / np.linalg.norm(C)
        cosines.append(float((Cn * Ainv_n).sum()))  # matrix cosine in [-1,1]
        # eigenvalue scaling: eig(C) vs eig(A) -> slope ~ -1. C ∝ A^{-1} means the LARGEST C
        # eigenvalue shares an eigenvector with the SMALLEST A eigenvalue, so pair A-descending
        # with C-ASCENDING (commuting case).
        ca = np.sort(np.linalg.eigvalsh(C))          # ascending
        aa = np.sort(np.linalg.eigvalsh(A))[::-1]    # descending
        coef = np.polyfit(np.log(aa), np.log(ca), 1)
        slopes.append(float(coef[0]))
        gens.append(g)
        g += 1
    return {"gen": np.array(gens), "cosine_C_Ainv": np.array(cosines),
            "loglog_slope": np.array(slopes), "final_cosine": cosines[-1],
            "final_slope": slopes[-1]}


def natural_gradient_vs_response(G: np.ndarray, beta: np.ndarray, realized_dz: np.ndarray):
    """Compare the natural-gradient direction to the realized breeder's response.

    The deterministic breeder's equation IS the (Riemannian) natural-gradient step:
    Δz̄ = G·β, with G the metric. Given measured G, selection gradient β, and the realized mean
    shift Δz̄, return the cosine between G·β and Δz̄ (should -> 1 as the sim follows the response
    map) and the natural-gradient direction. As G -> A^{-1}, G·β is the Newton/natural-gradient
    direction on the fitness landscape.
    """
    ng = G @ beta
    cos = float(ng @ realized_dz / (np.linalg.norm(ng) * np.linalg.norm(realized_dz) + 1e-30))
    return {"cosine_ng_response": cos, "natural_gradient": ng}

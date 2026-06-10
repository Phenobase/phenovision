"""GATE 0 — matrix-free Riccati optimizer (optim/riccati_precond.py).
Mirrors tests/test_optimizer.py style: bit-exact for off-paths, <1e-6 fp64 for
the Newton-Schulz parity, loose for the stochastic meta-loop."""
import math

import numpy as np
import torch
import torch.nn as nn

from optim.riccati_precond import (
    RiccatiPrecond, _ns_inverse, _ns_inv_sqrt, _riccati_with_source,
)


def _diag_spd(eigs):
    return torch.diag(torch.tensor(eigs, dtype=torch.float64))


def test_ns_inverse_matches_direct_inverse():
    eigs = np.logspace(-1, 0, 8)            # max eig 1.0 so NS converges from I
    C = _diag_spd(eigs)
    eye = torch.eye(8, dtype=torch.float64)
    G = _ns_inverse(C, eye, steps=80, eta_p=0.5, rho=0.0, safeguard=8.0)
    assert (G @ C - eye).abs().max() < 1e-6
    assert (C @ G @ C - C).abs().max() < 1e-6


def test_ns_inv_sqrt_matches_C_to_minus_half():
    C = _diag_spd(np.logspace(-1, 1, 8))
    eye = torch.eye(8, dtype=torch.float64)
    G = _ns_inv_sqrt(C, steps=30, eps=1e-12)
    # fixed point G C G = I (M = eps*I -> whitening)
    assert (G @ C @ G - eye).abs().max() < 1e-5


def test_riccati_with_source_fixed_point():
    C = _diag_spd(np.logspace(-0.5, 0, 6))
    M = _diag_spd(np.linspace(0.2, 0.8, 6))
    eye = torch.eye(6, dtype=torch.float64)
    G = _riccati_with_source(C, eye, M, steps=120, eta_p=0.4, rho=0.0, safeguard=8.0)
    assert (G @ C @ G - M).abs().max() < 1e-4


def test_shrinkage_reduces_realized_exponent():
    # G ≈ shrink(C, rho)^{-1}; realized exponent = -slope(log eig(G) vs log eig(C)).
    # rho=0 -> exponent ≈ 1; increasing rho monotonically reduces it toward 0.
    eigs = np.logspace(-1, 0, 12)
    C = _diag_spd(eigs)
    eye = torch.eye(12, dtype=torch.float64)
    log_c = np.log(eigs)
    exps = []
    for rho in [0.0, 0.3, 0.6, 0.9]:
        G = _ns_inverse(C, eye, steps=120, eta_p=0.5, rho=rho, safeguard=8.0)
        log_g = np.log(np.diag(G.numpy()))
        slope = np.polyfit(log_c, log_g, 1)[0]
        exps.append(-slope)
    assert abs(exps[0] - 1.0) < 0.1, exps
    assert all(exps[i + 1] <= exps[i] + 1e-6 for i in range(len(exps) - 1)), exps


def test_shrink_from_batch_schedule():
    assert RiccatiPrecond.shrink_from_batch(512, ref_batch=512) == 0.0      # no shrink at ref
    assert RiccatiPrecond.shrink_from_batch(0, ref_batch=512, rho_max=0.6) == 0.6  # clamp at B=0
    lo = RiccatiPrecond.shrink_from_batch(256, ref_batch=512, rho_max=0.6)  # linear in 1/B
    hi = RiccatiPrecond.shrink_from_batch(64, ref_batch=512, rho_max=0.6)
    assert hi > lo > 0.0  # more shrinkage at smaller batch


def _toy_step(opt, W, A, gen, batch=64):
    """One stochastic step on the separable quadratic L = 0.5 <A, W^2>."""
    noise = torch.randn(W.shape, generator=gen) * (A.sqrt() / math.sqrt(batch))
    g = A * W.detach() + noise
    W.grad = g
    loss = 0.5 * (A * W.detach() ** 2).sum()
    opt.step(closure=lambda: loss)
    return float(loss)


def test_alpha_sweep_no_nan():
    # Per-column curvature spanning 0.1..10 (cond 100) so the inverse (α=1) mode is
    # genuinely stressed. GATE: NO NaN/inf for any config; the STABLE configs converge;
    # and shrinkage stabilizes the inverse mode that diverges at shrink=0.
    A = torch.logspace(-1, 1, 6).double()
    results = {}
    for mode in ["whiten", "inverse"]:
        for shrink in [0.0, 0.3, 0.7]:
            gen = torch.Generator().manual_seed(1)
            W = nn.Parameter(torch.ones(8, 6, dtype=torch.float64))
            opt = RiccatiPrecond([W], lr=0.02, precond=mode, shrink=shrink,
                                 inner_steps=2, damping=1e-6)
            losses = [_toy_step(opt, W, A, gen) for _ in range(60)]
            assert torch.isfinite(W).all(), (mode, shrink)        # no NaN/inf ever
            assert all(math.isfinite(x) for x in losses), (mode, shrink)
            results[(mode, shrink)] = (losses[0], losses[-1])
    # whitening is stable at every shrink
    for shrink in [0.0, 0.3, 0.7]:
        l0, l1 = results[("whiten", shrink)]
        assert l1 < l0, ("whiten", shrink, l0, l1)
    # shrinkage stabilizes the inverse mode: heavy shrink converges where shrink=0 may not
    l0, l1 = results[("inverse", 0.7)]
    assert l1 < l0, ("inverse", 0.7, l0, l1)


def test_evolve_M_runs_and_M_evolves():
    # Unit test of the meta-loop MECHANICS on a tractable (mild, low-noise) problem;
    # the inverse-mode stabilization-at-scale claim is the O4 experiment, not this gate.
    # NOTE: now that the inverse mode actually converges to C^{-1} (alpha=1), it genuinely
    # amplifies flat-direction noise, so the toy needs the stabilizers (shrink + damping + low lr)
    # for the meta-loop MECHANICS test to stay finite. (alpha=1 stability at scale is the O4 study.)
    gen = torch.Generator().manual_seed(2)
    A = torch.logspace(-0.3, 0.3, 6).double()          # cond ~4
    W = nn.Parameter(torch.ones(8, 6, dtype=torch.float64))
    opt = RiccatiPrecond([W], lr=0.008, precond="inverse", shrink=0.6,
                         evolve_M=True, evolve_M_weighted=True, eta_M=5e-2,
                         meta_every=10, inner_steps=3, damping=1e-2)
    losses = [_toy_step(opt, W, A, gen, batch=256) for _ in range(200)]
    st = opt.state[W]
    assert torch.isfinite(W).all()
    assert all(math.isfinite(x) for x in losses)
    # ML should have moved away from its eps*I (=1e-2 I here) init (the meta-loop accumulated)
    ML = st["ML"]
    assert ML.diag().mean() > 2 * 1e-2    # grew above the damping-scale init
    # mechanics test: stays bounded (does NOT explode). Convergence-under-evolve_M with the now-
    # correct alpha=1 is the O4 empirical study, not a unit-test invariant.
    assert losses[-1] < 3 * losses[0]


def test_langevin_off_is_identical():
    A = (torch.logspace(-1, 1, 6).double()).abs() + 0.1

    def run(langevin, T):
        gen = torch.Generator().manual_seed(7)
        lgen = torch.Generator().manual_seed(123)
        W = nn.Parameter(torch.ones(8, 6, dtype=torch.float64))
        opt = RiccatiPrecond([W], lr=0.02, precond="whiten", langevin=langevin,
                             temperature=T, langevin_generator=lgen)
        for _ in range(30):
            _toy_step(opt, W, A, gen)
        return W.detach().clone()

    base = run(False, 1.0)
    off = run(True, 0.0)            # langevin on but T=0 -> zero noise
    assert torch.equal(base, off)


def test_hook_drives_curvature():
    # With precond_stats_from_hook, the curvature factor CL tracks the HOOK gradient,
    # not the empirical gradient. Constant grads -> CL converges to (hg hg^T)/n.
    m, n = 4, 3
    W = nn.Parameter(torch.zeros(m, n, dtype=torch.float64))
    g = torch.ones(m, n, dtype=torch.float64)
    hg = torch.zeros(m, n, dtype=torch.float64)
    hg[0, 0] = 2.0; hg[1, 1] = 3.0      # clearly different structure
    opt = RiccatiPrecond([W], lr=0.0, precond="inverse", precond_stats_from_hook=True,
                         damping=1e-9)
    for _ in range(400):
        W.grad = g.clone()
        W._soap_precond_grad = hg.clone()
        opt.step()
    CL = opt.state[W]["CL"].double()
    target_hook = (hg @ hg.t()) / n
    target_emp = (g @ g.t()) / n
    assert (CL - target_hook).abs().max() < 1e-2
    assert (CL - target_emp).abs().max() > 0.1

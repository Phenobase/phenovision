"""Tests for StableEvolutionSOAP (the GENERATE-don't-INVERT, selection-driven optimizer).

Mirrors tests/test_optimizer.py patterns. Covers: alpha-sweep no-NaN + descent; the realized
operative exponent surfaced via mean_exponent() stays in [alpha_min, alpha_max]; selection_off
forces the ceiling; the multiplicative generative recursion stays bounded under curvature noise
where the direct inverse has a heavy tail (the core stability claim, reference [4a]); and the
max_update_norm trust region genuinely clamps the step.
"""
import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from optim.stable_evolution_optimizer import StableEvolutionSOAP


# ------------------------------------------------------------ no-NaN + descent (DoD)
@pytest.mark.parametrize("selection_off,alpha_max", [(False, 0.9), (True, 0.5),
                                                     (True, 0.75), (True, 1.0)])
def test_no_nan_and_decreases(selection_off, alpha_max):
    """End-to-end on a 2-layer MLP: finite params/loss and loss decreases, for selection on
    and for forced fixed exponents up to alpha=1 (where the generative path must NOT blow up)."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(16, 32), nn.Tanh(), nn.Linear(32, 4))
    opt = StableEvolutionSOAP(model.parameters(), lr=1e-3, alpha_max=alpha_max,
                              selection_off=selection_off, damping=1e-2,
                              precondition_frequency=10, max_update_norm=1.0)
    g = torch.Generator().manual_seed(1)
    x = torch.randn(64, 16, generator=g)
    y = torch.randn(64, 4, generator=g)
    loss_fn = nn.MSELoss()
    losses = []
    for _ in range(60):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert all(math.isfinite(v) for v in losses)
    assert losses[-1] < losses[0] + 1e-6


# ------------------------------------------------------------ realized exponent readout
def test_mean_exponent_in_range():
    """mean_exponent() returns the size-weighted realized per-coordinate exponent, in
    [alpha_min, alpha_max], after at least one preconditioned step."""
    torch.manual_seed(0)
    W = nn.Linear(12, 8, bias=False)
    opt = StableEvolutionSOAP(W.parameters(), lr=1e-3, alpha_min=0.5, alpha_max=0.9,
                              precondition_frequency=5, damping=1e-2)
    assert math.isnan(opt.mean_exponent())                 # nothing accumulated yet
    g = torch.Generator().manual_seed(2)
    X = torch.randn(128, 12, generator=g)
    Y = X @ torch.randn(8, 12, generator=g).t() + 0.5 * torch.randn(128, 8, generator=g)
    for _ in range(40):
        opt.zero_grad(); ((W(X) - Y) ** 2).mean().backward(); opt.step()
    a = opt.mean_exponent()
    assert math.isfinite(a)
    assert 0.5 - 1e-6 <= a <= 0.9 + 1e-6, a


def test_exponent_stats_distribution():
    """exponent_stats() returns a coherent distribution summary: mean/std/max in [alpha_min,
    alpha_max], fracs in [0,1], not NaN after training, and consistent with mean_exponent()."""
    torch.manual_seed(0)
    n, m, B = 16, 8, 256
    X = torch.randn(B, n) * torch.logspace(-1.0, 1.0, n)
    Y = X @ torch.randn(m, n).t() + 0.5 * torch.randn(B, m)
    W = nn.Linear(n, m, bias=False)
    opt = StableEvolutionSOAP(W.parameters(), lr=2e-3, alpha_min=0.5, alpha_max=0.9,
                              precondition_frequency=5, damping=1e-2)
    assert all(math.isnan(v) for v in opt.exponent_stats().values())   # nothing accumulated yet
    for _ in range(40):
        opt.zero_grad(); ((W(X) - Y) ** 2).mean().backward(); opt.step()
    es = opt.exponent_stats()
    assert 0.5 - 1e-6 <= es["mean"] <= 0.9 + 1e-6
    assert 0.5 - 1e-6 <= es["max"] <= 0.9 + 1e-6
    assert es["max"] >= es["mean"] - 1e-6 and es["std"] >= 0.0
    assert 0.0 <= es["frac_high"] <= 1.0 and 0.0 <= es["frac_floor"] <= 1.0
    assert abs(opt.mean_exponent() - es["mean"]) < 1e-6


def test_selection_off_forces_ceiling():
    """selection_off=True pins every coordinate at alpha_max (the recapitulation framing)."""
    torch.manual_seed(0)
    W = nn.Linear(10, 6, bias=False)
    opt = StableEvolutionSOAP(W.parameters(), lr=1e-3, alpha_max=1.0, selection_off=True,
                              damping=1e-2, precondition_frequency=5)
    g = torch.Generator().manual_seed(3)
    X = torch.randn(64, 10, generator=g); Y = X @ torch.randn(6, 10, generator=g).t()
    for _ in range(20):
        opt.zero_grad(); ((W(X) - Y) ** 2).mean().backward(); opt.step()
    assert abs(opt.mean_exponent() - 1.0) < 1e-6


# ------------------------------------------------ GENERATE bounded; INVERT heavy-tailed (ref [4a])
def test_generative_recursion_bounded_under_curvature_noise():
    """The multiplicative geometric recursion the optimizer uses (P <- P*(P_target/P)**kappa,
    P_target = v_damp**(-alpha)) stays bounded under log-normal curvature noise, where the
    direct inverse 1/c on a flat direction has a heavy upper tail. Diagonal-in-eigenbasis port
    of stable_evolution_reference.py [4a]."""
    a = np.array([1.0, 4.0, 0.25]); kappa = 0.4
    rng = np.random.default_rng(1)
    g = np.ones(3); gmax = 0.0
    for _ in range(6000):
        c = a * np.exp(0.5 * rng.standard_normal(3))       # 50% log-normal curvature noise
        g = np.clip(g * np.power(1.0 / (g * g * c + 1e-12), kappa), 1e-8, 1e8)
        gmax = max(gmax, g.max())
    assert gmax < 10.0, gmax                                # generative excursion is bounded
    assert np.allclose(g, np.sqrt(1.0 / a), rtol=0.5)      # tracks the whitening target
    rng2 = np.random.default_rng(2)
    inv = 1.0 / (0.25 * np.exp(0.5 * rng2.standard_normal(50000)))
    assert np.percentile(inv, 99.9) > 3 * np.median(inv)   # direct inverse has the heavy tail


# ---------------------------------------------------------------- trust region clamps the step
def test_max_update_norm_clamps():
    """A tight max_update_norm bounds the per-step parameter movement (||Δp|| <= lr*clip),
    and a tight clip moves the params less than a loose clip."""
    def run(clip):
        torch.manual_seed(0)
        W = nn.Linear(16, 8, bias=False)
        w0 = W.weight.detach().clone()
        # weight_decay=0 so the ONLY parameter movement is the (clamped) update itself.
        opt = StableEvolutionSOAP(W.parameters(), lr=0.1, max_update_norm=clip,
                                  weight_decay=0.0, damping=1e-2, precondition_frequency=5)
        g = torch.Generator().manual_seed(4)
        X = torch.randn(64, 16, generator=g) * torch.logspace(-1, 1, 16)
        Y = X @ torch.randn(8, 16, generator=g).t()
        max_delta = 0.0
        prev = w0.clone()
        for _ in range(20):
            opt.zero_grad(); ((W(X) - Y) ** 2).mean().backward(); opt.step()
            max_delta = max(max_delta, (W.weight.detach() - prev).norm().item())
            prev = W.weight.detach().clone()
        return (W.weight.detach() - w0).norm().item(), max_delta

    moved_tight, dmax_tight = run(0.1)
    moved_loose, _ = run(5.0)
    assert dmax_tight <= 0.1 * 0.1 + 1e-5                   # ||Δp|| <= lr * clip
    assert moved_tight < moved_loose                       # tighter clip => less total movement


# ------------------------------------------------ demographic-noise (pSGLD) injection
def _demo_run(demo, T):
    torch.manual_seed(0)
    W = nn.Linear(12, 8, bias=False)
    opt = StableEvolutionSOAP(W.parameters(), lr=1e-2, demographic_noise=demo,
                              demographic_temperature=T,
                              demographic_generator=torch.Generator().manual_seed(7),
                              precondition_frequency=5, weight_decay=0.0)
    g = torch.Generator().manual_seed(5)
    X = torch.randn(64, 12, generator=g); Y = X @ torch.randn(8, 12, generator=g).t()
    for _ in range(40):
        opt.zero_grad(); ((W(X) - Y) ** 2).mean().backward(); opt.step()
    return W.weight.detach().clone()


def test_demographic_noise_off_is_identical():
    """demographic_noise=False (default) and T=0-with-flag-on are both bit-identical to no noise."""
    base = _demo_run(False, 0.0)
    flag_T0 = _demo_run(True, 0.0)                          # flag on but T=0 -> guard keeps it off
    assert torch.equal(base, flag_T0)


def test_demographic_noise_on_perturbs_and_is_finite():
    """T>0 injects noise: trajectory changes but stays finite (bounded pSGLD step)."""
    base = _demo_run(False, 0.0)
    noisy = _demo_run(True, 1e-2)
    assert torch.isfinite(noisy).all()
    assert (base - noisy).abs().max() > 1e-4               # the injected noise genuinely moves params

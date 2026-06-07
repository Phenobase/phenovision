"""§2.0 optimizer tests: parity, Adam-fallback, bias correction, demographic-noise off,
and the alpha-sweep no-NaN definition-of-done."""
import copy

import pytest
import torch
import torch.nn as nn

from optim.soap_full_power import SOAPFullPower
from optim._reference.soap import SOAP
from optim.parity_test import run_parity


# ----------------------------------------------------------------------------- parity
def test_parity_vs_official_soap():
    """§2.0(a): vanilla_soap_config reproduces official SOAP (Kronecker path)."""
    worst, _ = run_parity(seed=0, n_steps=8, dtype=torch.float32, verbose=False)
    assert worst < 1e-5, f"parity max|Δparam|={worst:.3e} exceeds 1e-5"


def test_adam_fallback_bit_exact():
    """With Kronecker disabled (max_precond_dim=1) the inner loop matches SOAP exactly."""
    torch.manual_seed(0)
    shapes = [(16, 32), (32, 8)]
    g = torch.Generator().manual_seed(0)
    p_ref = [nn.Parameter(torch.randn(s, generator=g) * 0.1) for s in shapes]
    p_our = [nn.Parameter(p.detach().clone()) for p in p_ref]
    common = dict(lr=3e-3, betas=(0.95, 0.95), eps=1e-8,
                  precondition_frequency=10000, correct_bias=True)
    opt_ref = SOAP(p_ref, weight_decay=0.0, merge_dims=False, precondition_1d=False,
                   normalize_grads=False, max_precond_dim=1, **common)
    ok = dict(common); ok.update(SOAPFullPower.vanilla_soap_config()); ok["max_precond_dim"] = 1
    opt_our = SOAPFullPower(p_our, **ok)
    gg = torch.Generator().manual_seed(7)
    for _ in range(6):
        grads = [torch.randn(s, generator=gg) for s in shapes]
        for params, opt in ((p_ref, opt_ref), (p_our, opt_our)):
            for p, gr in zip(params, grads):
                p.grad = gr.clone()
            opt.step()
        d = max((a - b).abs().max().item() for a, b in zip(p_ref, p_our))
    assert d == 0.0, f"Adam-fallback path not bit-exact: {d:.3e}"


# ----------------------------------------------------------------- bias correction (§2.0b)
@pytest.mark.parametrize("alpha", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_bias_correction_constant_gradient_constant_update(alpha):
    """Generalized bias correction: with a CONSTANT gradient, v_hat=g^2 and m_hat=g exactly
    at every step, so the per-step update is constant immediately. A wrong exponent-generalized
    bias correction would make the update drift across steps."""
    torch.manual_seed(0)
    p = nn.Parameter(torch.randn(8, 6))
    opt = SOAPFullPower([p], lr=1e-2, precond_power=alpha, damping=1e-3,
                        relative_damping=True, precondition_frequency=10000,
                        weight_decay=0.0, correct_bias=True)
    const_grad = torch.randn(8, 6)
    deltas = []
    prev = p.detach().clone()
    # first call seeds basis & skips; collect deltas over the next several steps
    for i in range(10):
        p.grad = const_grad.clone()
        opt.step()
        if i >= 1:  # after the init-skip step
            deltas.append((p.detach() - prev).clone())
        prev = p.detach().clone()
    deltas = torch.stack(deltas[1:])  # drop first real step (EMA still ramping is corrected, but be safe)
    # all deltas equal -> std across steps negligible relative to magnitude
    rng = deltas.std(dim=0).max().item()
    mag = deltas.abs().mean().item()
    assert rng <= 1e-6 * max(mag, 1e-8) + 1e-7, f"update drifts: std={rng:.3e} mag={mag:.3e}"


# --------------------------------------------------- demographic noise OFF == identical (§2.0c)
def test_demographic_noise_off_is_identical():
    """With demographic_noise=False (default) the trajectory is bit-identical to the
    no-hook code path (seeded)."""
    torch.manual_seed(0)
    shapes = [(12, 10)]
    base = [torch.randn(s) for s in shapes]
    def run(demo):
        ps = [nn.Parameter(b.clone()) for b in base]
        gen = torch.Generator().manual_seed(123)
        opt = SOAPFullPower(ps, lr=1e-3, precond_power=1.0, damping=1e-2,
                            precondition_frequency=10000,
                            demographic_noise=demo, demographic_temperature=0.0,
                            demographic_generator=gen)
        gg = torch.Generator().manual_seed(5)
        for _ in range(6):
            for p in ps:
                p.grad = torch.randn(p.shape, generator=gg)
            opt.step()
        return torch.cat([p.detach().flatten() for p in ps])
    a = run(False)
    b = run(True)  # demo flag True but temperature 0 -> guard keeps it off
    assert torch.equal(a, b), "demographic_noise with T=0 changed the trajectory"


# ------------------------------------------------------- alpha sweep, no NaNs (§2.0 DoD)
@pytest.mark.parametrize("alpha", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_alpha_sweep_no_nan(alpha):
    """End-to-end training on a 2-layer MLP must not NaN for any alpha, with relative
    damping ~1e-2 (essential at alpha=1)."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(16, 32), nn.Tanh(), nn.Linear(32, 4))
    opt = SOAPFullPower(model.parameters(), lr=1e-3, precond_power=alpha,
                        damping=1e-2, relative_damping=True, precondition_frequency=10,
                        precondition_1d=False)
    g = torch.Generator().manual_seed(1)
    x = torch.randn(64, 16, generator=g)
    y = torch.randn(64, 4, generator=g)
    loss_fn = nn.MSELoss()
    losses = []
    for _ in range(50):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert all(torch.isfinite(p).all() for p in model.parameters()), f"NaN params at alpha={alpha}"
    assert all(map(lambda v: v == v, losses)), f"NaN loss at alpha={alpha}"
    assert losses[-1] < losses[0] + 1e-6, f"loss increased at alpha={alpha}: {losses[0]:.4f}->{losses[-1]:.4f}"

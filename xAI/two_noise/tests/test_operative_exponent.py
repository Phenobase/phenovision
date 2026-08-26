"""The operative-exponent factor probe (curvature/operative_exponent.py): measures the realized
preconditioner exponent α (G ≈ C^{-α}). Validated (a) on a known G=C^{-α}, and (b) on a trained
optimizer mid-training (whiten→0.5, inverse→1.0) -- which also guards the optimizer's NS
convergence (the inverse mode must actually reach C^{-1})."""
import math
import warnings

import numpy as np
import torch
import torch.nn as nn

from optim.riccati_precond import RiccatiPrecond
from curvature.operative_exponent import operative_exponent_factors, operative_exponent_lanczos


def test_probe_recovers_known_exponent():
    # feed a known G = C^{-alpha} and check the probe returns alpha
    def probe(alpha):
        W = nn.Parameter(torch.zeros(8, 24))
        opt = RiccatiPrecond([W], precond="inverse")
        st = opt.state[W]; st["use_kron"] = True
        for Ck, Gk, d in (("CL", "GL", 8), ("CR", "GR", 24)):
            Q, _ = torch.linalg.qr(torch.randn(d, d)); e = torch.logspace(-1, 1, d)
            C = (Q * e) @ Q.t(); w, V = torch.linalg.eigh(C)
            st[Ck] = C.float(); st[Gk] = ((V * (w ** (-alpha))) @ V.t()).float()
        return operative_exponent_factors(opt, min_spread=0.3)[0]
    for a in (0.0, 0.5, 1.0):
        assert abs(probe(a) - a) < 0.05, (a, probe(a))


def test_trained_whiten_half_inverse_one():
    # mid-training, the whiten mode must read ~0.5 and the inverse mode ~1.0 (the latter guards
    # that the NS inverse actually converges to C^{-1} -- it silently did not before the basin fix).
    warnings.filterwarnings("ignore")
    torch.manual_seed(0)
    n, m, B = 32, 8, 512
    X = torch.randn(B, n) * torch.logspace(-0.8, 0.8, n)
    Y = X @ torch.randn(m, n).t() + 0.5 * torch.randn(B, m)

    def trained(mode, **kw):
        torch.manual_seed(1); W = nn.Linear(n, m, bias=False)
        opt = RiccatiPrecond(W.parameters(), lr=2e-3, precond=mode, precond_every=1,
                             inner_steps=4, damping=1e-6, **kw)
        for _ in range(40):
            opt.zero_grad(); ((W(X) - Y) ** 2).mean().backward(); opt.step()
        return operative_exponent_factors(opt, min_spread=0.4)[0]

    assert abs(trained("whiten") - 0.5) < 0.12
    assert abs(trained("inverse", shrink=0.0) - 1.0) < 0.15
    # shrinkage reduces the operative exponent (the O3 law) on a real-ish model
    assert trained("inverse", shrink=0.7) < trained("inverse", shrink=0.0) - 0.2


def test_lanczos_probe_recovers_known_exponent():
    # (ii) ground-truth probe: with a NON-degenerate Hessian (single output) and the preconditioner
    # set EXACTLY to G = H^{-alpha}, the true-Hessian gain probe must return alpha.
    warnings.filterwarnings("ignore")
    torch.manual_seed(0)
    n, B = 24, 4096
    X = torch.randn(B, n) * torch.logspace(-0.7, 0.7, n)
    Sig = (X.t() @ X) / B                                   # true Hessian of ½ mean (Wx)²

    def probe(alpha):
        W = nn.Linear(n, 1, bias=False)
        opt = RiccatiPrecond(W.parameters(), precond="inverse")
        params = list(W.parameters()); st = opt.state[params[0]]; st["use_kron"] = True
        w, V = torch.linalg.eigh(Sig)
        st["CR"] = Sig.float(); st["GR"] = ((V * (w ** (-alpha))) @ V.t()).float()
        st["CL"] = torch.ones(1, 1); st["GL"] = torch.ones(1, 1)
        lf = lambda: 0.5 * (W(X) ** 2).mean()
        return operative_exponent_lanczos(lf, opt, params, k=20, n_iter=60,
                                          generator=torch.Generator().manual_seed(5))[0]
    for a in (0.0, 0.5, 1.0):
        assert abs(probe(a) - a) < 0.05, (a, probe(a))

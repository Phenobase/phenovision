"""Does SOAP evolve-M's operative exponent show the two-phase dynamics the theory predicts:
M -> A (exponent toward 0) during loss DESCENT, then M -> A^-1 (exponent toward 1) once the
optimizer reaches the noisy EQUILIBRIUM (the optimum)?

On real ViT evolve-M sat at ~0.5 but never reached equilibrium and the meta-loop was signal-starved
(whitening flattens the per-direction update energy). Two fixes under test here:
  1. INJECT NOISE into M each step (m_noise) -- the single-trajectory analog of the batch-sampling
     noise that lets C be estimated; gives the rank-mu meta-loop variation to select on at the
     optimum, where the descent signal (dL) has vanished.
  2. RUN TO / PAST CONVERGENCE on a stochastic (minibatch) problem, so the equilibrium phase is
     actually reached and we can see whether the exponent FLIPS up toward A^-1.

CPU, fast. Run:  python -m toy.evolve_m_climb
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from optim.soap_full_power import SOAPFullPower
from curvature.operative_exponent import operative_exponent_soap


def make_problem(n=24, m=8, N=4096, cond=50.0, noise=0.05, seed=0):
    g = torch.Generator().manual_seed(seed)
    scales = torch.logspace(0, np.log10(cond), n) ** 0.5
    X = torch.randn(N, n, generator=g) * scales
    Wstar = torch.randn(m, n, generator=g)
    Y = X @ Wstar.t() + noise * torch.randn(N, m, generator=g)
    return X, Y


def run(X, Y, steps=2500, batch=128, lr=3e-3, mode="evolve", eta_m=0.1, meta_every=4,
        m_noise=0.0, m_load=0.0, m_nstar=1e9, seed=1):
    """Stochastic (minibatch) training; returns per-step loss and (step, exponent) samples."""
    N, n = X.shape; m = Y.shape[1]
    torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed + 99)
    W = nn.Linear(n, m, bias=False)
    evolve = mode == "evolve"
    power = 1.0 if mode == "inverse" else 0.5
    opt = SOAPFullPower(W.parameters(), lr=lr, precond_power=power, evolve_m=evolve,
                        evolve_m_weighted=True, eta_m=eta_m, meta_every=meta_every,
                        m_noise=m_noise, m_load=m_load, m_nstar=m_nstar,
                        precondition_frequency=1, damping=1e-7, relative_damping=True)
    losses, exps = [], []
    for t in range(steps):
        idx = torch.randint(0, N, (batch,), generator=gen)
        xb, yb = X[idx], Y[idx]
        opt.zero_grad(); loss = ((W(xb) - yb) ** 2).mean(); loss.backward()
        opt.step(closure=lambda: loss.detach())
        losses.append(float(loss))
        if t % 50 == 0 or t == steps - 1:
            exps.append((t, operative_exponent_soap(opt, min_spread=0.3)[0]))
    return losses, exps, opt, W


def _conv_step(losses, tol=0.02):
    """First step where the loss is within tol of its final plateau (rough convergence marker)."""
    final = np.median(losses[-200:])
    for i, L in enumerate(losses):
        if L <= final * (1 + tol):
            return i
    return len(losses)


def main():
    X, Y = make_problem()
    print("=== fixed baselines (does the inverse help on this problem?) ===")
    for mode in ("whiten", "inverse"):
        L, _, _, _ = run(X, Y, mode=mode)
        print(f"  {mode:8s} final loss = {np.median(L[-200:]):.4f}")
    print()
    print("=== L_eff load (theory sqrt(a*m), kappa(N*)) : does the SETTLING EXPONENT move with N*? ===")
    print("   (theory prediction: NO -- the exploration fixed point is m ~ 1/a (exponent 1) for any N*;")
    print("    kappa only sets the scale. If the exponent is ~constant across N*, that's confirmed.)")
    for nstar in (2.0, 16.0, 128.0, 1024.0):
        L, exps, opt, W = run(X, Y, mode="evolve", m_load=8.0, m_nstar=nstar)
        st = opt.state[W.weight]
        aniso = (st["m_L"].max() / st["m_L"].min().clamp_min(1e-12)).item()
        late = [e for t, e in exps if t > 1500 and np.isfinite(e)]
        late_exp = float(np.median(late)) if late else float("nan")
        print(f"  N*={nstar:7.1f} (kappa={8.0*(0.5+1/(4*nstar)):.2f}): late_exp={late_exp:5.2f}  "
              f"final_loss={np.median(L[-200:]):8.3f}  m_aniso={aniso:.2e}")


if __name__ == "__main__":
    main()

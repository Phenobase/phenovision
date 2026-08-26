"""GATE 0 — CUDA/autocast smoke for optim/riccati_precond.py.

The likeliest PyTorch-port bug is the fp16 Newton-Schulz / matrix_norm(ord=2) under
autocast (the reference was numpy-only). This runs a tiny 2D-weight model on CUDA with
AMP for whiten, inverse, and evolve_M modes and asserts everything stays finite.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from optim.riccati_precond import RiccatiPrecond  # noqa: E402


def run(mode, lr=2e-3, **kw):
    # bf16 autocast (fp32 exponent range -> no overflow, no GradScaler) is the clean AMP
    # path for Riccati: it lets us pass the loss closure directly (needed by evolve_M)
    # without the scaler bypassing our step(). This mirrors how O2/O4 will run on GPU.
    dev = torch.device("cuda")
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 10)).to(dev)
    opt = RiccatiPrecond(model.parameters(), lr=lr, precond=mode, **kw)
    x = torch.randn(32, 64, device=dev)
    y = torch.randint(0, 10, (32,), device=dev)
    losses = []
    for _ in range(40):
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = F.cross_entropy(model(x), y)
        loss.backward()                       # grads are fp32 under bf16 autocast
        opt.step(closure=lambda: loss.detach())
        losses.append(float(loss))
    finite = all(torch.isfinite(p).all() for p in model.parameters())
    assert finite, f"{mode}: non-finite params"
    assert all(l == l for l in losses), f"{mode}: NaN loss"
    print(f"[smoke] mode={mode:8s} kw={kw} loss {losses[0]:.3f} -> {losses[-1]:.3f}  finite=True")


def main():
    assert torch.cuda.is_available(), "no CUDA device"
    print("device:", torch.cuda.get_device_name(0))
    run("whiten")
    run("inverse", lr=1e-3, shrink=0.5, damping=1e-2)
    run("inverse", lr=1e-3, shrink=0.3, damping=1e-2, evolve_M=True, eta_M=1e-2, meta_every=10)
    print("[smoke] RiccatiPrecond CUDA/autocast smoke PASSED")


if __name__ == "__main__":
    main()

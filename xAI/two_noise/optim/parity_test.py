"""§2.0(a) parity check: SOAPFullPower(vanilla_soap_config) == official SOAP.

With ``precond_power=0.5``, ``damping=0``, ``relative_damping=False``,
``max_update_norm=0``, ``weight_decay=0`` and ``parity_mode=True``, our optimizer must
reproduce vanilla SOAP to numerical tolerance.

Design notes (each one is a real subtlety we verified empirically):

1. **Dense, full-rank gradients.** At step 1 Adam's update is ``sign(g_proj)·const``
   (``v = g²`` ⇒ ``g/√v = sign(g)``). A low-rank MLP-layer gradient has many *near-zero*
   projected entries whose sign flips between ``matmul`` and ``tensordot`` roundoff, producing
   large spurious update differences that are SOAP's own behavior, not ours. We therefore
   inject i.i.d. dense gradients directly into raw parameters so every projected coordinate is
   O(1) and ``sign`` is stable.

2. **Shared initial basis.** The initial preconditioner is rank-1 (one gradient), so its
   eigenbasis has a degenerate null space whose eigenvectors are arbitrary (roundoff-sensitive).
   Both optimizers build the basis from the same gradient; we additionally copy official SOAP's
   basis into ours after init so any null-space arbitrariness is shared. (Our projection
   convention matches official's exactly given the same Q; verified separately.)

3. **No basis refresh in-window.** ``precondition_frequency`` is set above ``n_steps`` so the
   inner-loop math is isolated from the refresh path.

The Adam-fallback path (no Kronecker) matches bit-exactly without any of this (separate test).

Run directly: ``python -m optim.parity_test`` (after activating the two_noise env).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from optim.soap_full_power import SOAPFullPower  # noqa: E402
from optim._reference.soap import SOAP  # noqa: E402


def _make_params(seed: int, shapes, dtype):
    g = torch.Generator().manual_seed(seed)
    return [torch.nn.Parameter(torch.randn(s, generator=g, dtype=dtype) * 0.1) for s in shapes]


def run_parity(seed: int = 0, n_steps: int = 8, lr: float = 3e-3,
               dtype=torch.float32, shapes=((16, 32), (32, 8)), verbose: bool = True):
    """Return (max_abs_diff_over_all_steps, per_step_diffs).

    Two sets of raw parameters are stepped in lockstep on identical dense random gradients;
    parameters are compared after every step.
    """
    precond_freq = 10_000  # >> n_steps: no basis refresh in-window

    p_ref = _make_params(seed, shapes, dtype)
    p_ours = [torch.nn.Parameter(p.detach().clone()) for p in p_ref]

    common = dict(lr=lr, betas=(0.95, 0.95), eps=1e-8,
                  precondition_frequency=precond_freq, correct_bias=True)
    opt_ref = SOAP(p_ref, weight_decay=0.0, merge_dims=False,
                   precondition_1d=False, normalize_grads=False, max_precond_dim=10000, **common)
    ours_kwargs = dict(common)
    ours_kwargs.update(SOAPFullPower.vanilla_soap_config())
    opt_ours = SOAPFullPower(p_ours, **ours_kwargs)

    ggen = torch.Generator().manual_seed(seed + 100)

    def grads_for_step():
        # one dense gradient per parameter, identical for both optimizers this step
        return [torch.randn(p.shape, generator=ggen, dtype=dtype) for p in p_ref]

    def apply(opt, params, grads):
        for p, gr in zip(params, grads):
            p.grad = gr.clone()
        opt.step()

    # --- init step (both build basis and skip the update) ---
    g0 = grads_for_step()
    apply(opt_ref, p_ref, g0)
    apply(opt_ours, p_ours, g0)
    # share official's (rank-1, null-space-arbitrary) basis with ours
    for pr, po in zip(p_ref, p_ours):
        sref, sour = opt_ref.state[pr], opt_ours.state[po]
        if "Q" in sref and sour.get("use_precond"):
            Q = sref["Q"]
            sour["QL"] = Q[0] if (len(Q) > 0 and len(Q[0]) > 0) else None
            sour["QR"] = Q[1] if (len(Q) > 1 and len(Q[1]) > 0) else None

    per_step = []
    for step in range(n_steps):
        grads = grads_for_step()
        apply(opt_ref, p_ref, [g.clone() for g in grads])
        apply(opt_ours, p_ours, [g.clone() for g in grads])
        d = max((pr - po).abs().max().item() for pr, po in zip(p_ref, p_ours))
        per_step.append(d)
        if verbose:
            print(f"step {step:2d}  max|Δparam| = {d:.3e}")

    return max(per_step), per_step


if __name__ == "__main__":
    tol = 1e-5  # float32 native to SOAP
    worst, _ = run_parity(dtype=torch.float32)
    print(f"\nworst max|Δparam| over all steps: {worst:.3e}  (tol {tol:.0e})")
    print("PARITY PASS" if worst < tol else "PARITY FAIL")
    sys.exit(0 if worst < tol else 1)

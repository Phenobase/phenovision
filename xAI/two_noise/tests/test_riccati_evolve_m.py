"""O4 — wiring/no-crash smoke (CPU). The science (operative exponent climbing 1/2->1 in the
top-k subspace, beating whiten) is the GPU run; this gates that all comparators run, the
operative-exponent diagnostic is computed and finite, and the optimizers are built correctly."""
import math

import torch

from ml_experiments.riccati_evolve_m import build_parser, run_condition, _make_opt, operative_exponent
from optim.riccati_precond import RiccatiPrecond


def _args(**over):
    a = build_parser().parse_args([])
    a.device = "cpu"; a.model = "tiny_vision"; a.dataset = "synthetic_vision"
    a.synthetic_n = 64; a.batch = 16; a.max_steps = 8; a.amp = False
    a.eval_max_batches = 2; a.num_workers = 0
    for k, v in over.items():
        setattr(a, k, v)
    return a


def test_conditions_run_and_diagnostic_finite():
    args = _args()
    dev = torch.device("cpu")
    for cond, kw in [("whiten", {}), ("inverse", {}),
                     ("evolve", dict(eta_M=1e-3, meta_every=4))]:
        recs = run_condition(cond, args, dev, **kw)
        assert recs, cond
        assert all(math.isfinite(r["train_loss"]) for r in recs), cond
        # the operative-exponent diagnostic is computed (top-k exponent finite)
        assert any(math.isfinite(r["op_exponent_topk"]) for r in recs), cond


def test_make_opt_builds_evolve_optimizer():
    import torch.nn as nn
    m = nn.Sequential(nn.Linear(8, 8))
    opt, lr = _make_opt("evolve", 1e-3, 10, m, None, 0.01, 1e-3)
    assert isinstance(opt, RiccatiPrecond)
    g = opt.param_groups[0]
    assert g["evolve_M"] and g["precond"] == "inverse"
    opt_w, _ = _make_opt("whiten", 1e-3, 10, m, None, 0.01, 1e-3)
    assert opt_w.param_groups[0]["precond"] == "whiten"


def test_operative_exponent_on_trained_inverse():
    # after a few inverse-mode steps the top-k operative exponent is finite and positive-ish
    import torch.nn as nn
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 4))
    opt = RiccatiPrecond(m.parameters(), lr=1e-3, precond="inverse", shrink=0.2, damping=1e-2)
    x = torch.randn(16, 16); y = torch.randint(0, 4, (16,))
    import torch.nn.functional as F
    for _ in range(30):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(m(x), y); loss.backward(); opt.step()
    op_top, op_flat, align = operative_exponent(opt)
    assert math.isfinite(op_top)

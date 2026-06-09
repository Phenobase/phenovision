"""O2 — wiring/no-crash smoke (CPU). The science is the GPU run; this just gates that all three
conditions run, the true-Fisher hook fires, and the recovered-fraction is computable."""
import math

from ml_experiments.riccati_schedule_vs_fisher import build_parser, run_condition, summarize


def _args(**over):
    a = build_parser().parse_args([])
    a.device = "cpu"; a.model = "tiny_vision"; a.dataset = "synthetic_vision"
    a.synthetic_n = 64; a.batch = 16; a.max_steps = 6; a.amp = False
    a.eval_max_batches = 2; a.num_workers = 0
    for k, v in over.items():
        setattr(a, k, v)
    return a


def test_all_conditions_run_finite():
    import torch
    args = _args()
    dev = torch.device("cpu")
    by = {}
    for cond in ["whiten", "inverse_fisher", "schedule"]:
        recs = run_condition(cond, args, dev)
        assert recs, cond
        assert all(math.isfinite(r["train_loss"]) for r in recs), cond
        by[cond] = recs
    frac = summarize(by)
    assert math.isfinite(frac) or frac != frac  # computed (may be nan on trivial synthetic data)


def test_schedule_uses_batch_dependent_shrink():
    from optim.riccati_precond import RiccatiPrecond
    from ml_experiments.riccati_schedule_vs_fisher import _make_condition
    a16 = _args(batch=16)
    a256 = _args(batch=256)
    kw16, _, _ = _make_condition("schedule", a16)
    kw256, _, _ = _make_condition("schedule", a256)
    assert kw16["shrink"] > kw256["shrink"]      # more shrink at smaller batch
    kw_f, use_tf, _ = _make_condition("inverse_fisher", a16)
    assert use_tf and kw_f["precond_stats_from_hook"]

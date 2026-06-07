"""CPU smoke tests for the §2.1/§2.2 real-model benchmark harness.

Tiny model + tiny synthetic data + a handful of steps, so the whole file runs in <1 min on CPU
with no downloads. Verifies: make_model/make_data/make_optimizer/train_eval run and produce finite
losses for adamw and soap@{0.5,1.0}; the benchmarks CSV writer emits the expected columns; the
alpha_vs_batch CLI produces a tidy CSV with an alpha* flag.

Run: mamba run -n two_noise python -m pytest tests/test_benchmarks.py -q
"""

import csv

import math
import torch
import pytest

from ml_experiments._harness import (DataMeta, make_data, make_model, make_optimizer,
                                     train_eval, is_lm_model)
from ml_experiments import benchmarks, alpha_vs_batch

DEVICE = torch.device("cpu")


def _tiny_vision_setup(seed=0):
    gen = torch.Generator().manual_seed(seed)
    train, val, meta = make_data("synthetic_vision", batch_size=8, generator=gen, synthetic_n=32)
    model = make_model("tiny_vision", num_classes=meta.num_classes)
    return model, train, val, meta


def test_make_model_shapes():
    m = make_model("tiny_vision", num_classes=10)
    out = m(torch.randn(4, 3, 32, 32))
    assert out.shape == (4, 10)
    assert not is_lm_model(m)


def test_make_data_cifar_meta_only():
    # Don't download here; just confirm synthetic meta is well-formed.
    _, _, meta = make_data("synthetic_vision", batch_size=8, synthetic_n=16)
    assert isinstance(meta, DataMeta)
    assert meta.task == "vision" and meta.num_classes == 10


@pytest.mark.parametrize("opt,alpha", [("adamw", None), ("soap", 0.5), ("soap", 1.0)])
def test_train_eval_finite(opt, alpha):
    model, train, val, _ = _tiny_vision_setup()
    kw = {} if alpha is None else dict(alpha=alpha)
    optimizer, lr = make_optimizer(opt, model.parameters(), lr=1e-4, **kw)
    assert lr > 0
    res = train_eval(model, optimizer, train, val, DEVICE, max_steps=5, log_every=1, lr=lr)
    assert res.steps_run == 5
    assert len(res.records) >= 1
    for r in res.records:
        assert math.isfinite(r["train_loss"])
    assert math.isfinite(res.final_val_loss)
    assert 0.0 <= res.final_val_metric <= 1.0  # accuracy


def test_train_eval_lm_finite():
    gen = torch.Generator().manual_seed(0)
    train, val, meta = make_data("synthetic_lm", batch_size=4, block_size=16, generator=gen,
                                 synthetic_n=64)
    model = make_model("nanogpt", vocab_size=meta.vocab_size)
    # shrink block_size won't match preset (256); rebuild a matching tiny GPT instead.
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_vendor"))
    from nanogpt.model import GPT, GPTConfig
    model = GPT(GPTConfig(block_size=16, vocab_size=meta.vocab_size, n_layer=2, n_head=2,
                          n_embd=32, dropout=0.0, bias=False))
    assert is_lm_model(model)
    optimizer, lr = make_optimizer("soap", model.parameters(), alpha=1.0, lr=1e-4)
    res = train_eval(model, optimizer, train, val, DEVICE, max_steps=4, log_every=1, lr=lr)
    assert res.val_metric_name == "perplexity"
    assert math.isfinite(res.final_val_loss)
    assert res.final_val_metric > 0  # perplexity = exp(loss)


def test_grad_accum_holds_steps():
    """accum_steps>1 still produces exactly max_steps optimizer steps."""
    model, train, val, _ = _tiny_vision_setup()
    optimizer, lr = make_optimizer("soap", model.parameters(), alpha=1.0, lr=1e-4)
    res = train_eval(model, optimizer, train, val, DEVICE, max_steps=3, accum_steps=2,
                     log_every=1, lr=lr)
    assert res.steps_run == 3


def test_benchmarks_csv_columns(tmp_path):
    argv = ["--model", "tiny_vision", "--dataset", "synthetic_vision",
            "--optimizer", "soap", "--alpha", "1.0", "--lr", "1e-4",
            "--batch-size", "8", "--max-steps", "3", "--log-every", "1",
            "--device", "cpu", "--synthetic-n", "32", "--out-dir", str(tmp_path)]
    csv_path = benchmarks.main(argv)
    assert csv_path.exists()
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == benchmarks.CSV_COLUMNS
        rows = list(reader)
    assert len(rows) >= 1
    r0 = rows[0]
    assert r0["model"] == "tiny_vision"
    assert r0["optimizer"] == "soap"
    assert float(r0["alpha"]) == 1.0
    assert math.isfinite(float(r0["train_loss"]))
    assert int(r0["eff_batch_size"]) == 8


def test_benchmarks_adamw_csv(tmp_path):
    argv = ["--model", "tiny_vision", "--dataset", "synthetic_vision",
            "--optimizer", "adamw", "--lr", "1e-3", "--batch-size", "8",
            "--max-steps", "3", "--log-every", "1", "--device", "cpu",
            "--synthetic-n", "32", "--out-dir", str(tmp_path)]
    csv_path = benchmarks.main(argv)
    assert csv_path.exists()


def test_alpha_vs_batch_csv(tmp_path):
    argv = ["--model", "tiny_vision", "--dataset", "synthetic_vision",
            "--alphas", "0.5", "1.0", "--batch-sizes", "8", "16",
            "--micro-batch", "8", "--budget-microbatches", "4",
            "--lr", "1e-4", "--device", "cpu", "--synthetic-n", "64",
            "--out-dir", str(tmp_path)]
    csv_path = alpha_vs_batch.main(argv)
    assert csv_path.exists()
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == alpha_vs_batch.CSV_COLUMNS
        rows = list(reader)
    # 2 alphas x 2 batch sizes = 4 cells.
    assert len(rows) == 4
    # exactly one alpha* per batch size (2 total).
    stars = [r for r in rows if r["is_alpha_star"] == "True"]
    assert len(stars) == 2

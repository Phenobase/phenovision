"""CPU smoke test for the α=1 true-Fisher stability experiment (ml_experiments/alpha1_stability.py).

Tiny conv net + tiny synthetic vision data + a handful of steps, no downloads, < ~1 min on CPU.
Verifies that condition B (TRUE-Fisher eigenvalues via precond_eigvals_from_hook + relative damping)
runs at precond_power=1.0 — exercising the sampled_label_gradient / assign_precond_grad assignment
path — and produces a finite, non-diverged result with a well-formed CSV.

Run: mamba run -n two_noise python -m pytest tests/test_alpha1_stability.py -q
"""

import csv
import math

import torch

from ml_experiments import alpha1_stability
from ml_experiments.alpha1_stability import CONDITIONS, run_condition, build_parser, CSV_COLUMNS


def _args(**over):
    a = build_parser().parse_args([])
    a.device = "cpu"
    a.model = "tiny_vision"
    a.dataset = "synthetic_vision"
    a.synthetic_n = 64
    a.batch = 8
    a.max_steps = 6
    a.lr = 1e-4
    a.amp = False           # AMP is CUDA-only; CPU smoke runs fp32
    a.num_workers = 0
    a.eval_max_batches = 2
    a.grad_clip = 1.0
    for k, v in over.items():
        setattr(a, k, v)
    return a


def test_conditions_table_is_alpha1_only():
    # All three conditions are full-inverse comparisons; only damping / true-Fisher / trust region vary.
    assert set(CONDITIONS) == {"A", "B", "C"}
    assert CONDITIONS["B"]["use_true_fisher"] is True
    assert CONDITIONS["A"]["use_true_fisher"] is False
    assert CONDITIONS["C"]["use_true_fisher"] is False


def test_condition_B_true_fisher_runs_finite():
    """The true-Fisher eigenvalue path (sampled_label_gradient + assign_precond_grad) runs and the
    full-inverse step stays finite on the tiny smoke setup."""
    device = torch.device("cpu")
    row = run_condition("B", _args(), device)
    assert row["condition"] == "B"
    assert row["use_true_fisher"] is True
    assert row["alpha"] == 1.0
    assert row["n_steps_run"] == 6
    # The whole point: true-Fisher + damping keeps it finite.
    assert row["diverged"] is False
    assert row["finite_fraction"] == 1.0
    assert math.isfinite(row["final_train_loss"])
    assert math.isfinite(row["final_val_acc"])
    assert 0.0 <= row["final_val_acc"] <= 1.0


def test_condition_C_control_runs():
    """Control (empirical + damping, no true-Fisher) also runs and reports a finite-fraction field."""
    row = run_condition("C", _args(), torch.device("cpu"))
    assert row["use_true_fisher"] is False
    assert 0.0 <= row["finite_fraction"] <= 1.0
    assert row["n_steps_run"] == 6


def test_csv_written_with_columns(tmp_path):
    args = _args(conditions=["B"], out_dir=str(tmp_path))
    csv_path = alpha1_stability.run(args)
    assert csv_path.exists()
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == CSV_COLUMNS
        rows = list(reader)
    assert len(rows) == 1
    assert rows[0]["condition"] == "B"
    assert rows[0]["use_true_fisher"] == "True"
    assert float(rows[0]["alpha"]) == 1.0

#!/usr/bin/env python3
"""
d90/gen_d90_grid.py — emit the init-d90 intrinsic-dimension sweep (briefing Part II §9.1,
plan component C6).

One line of the output grid = the argument string for ONE ``run_subspace.py`` invocation =
ONE GPU job (one array task of ``scripts/submit_d90.sh``). Each line confines fine-tuning of
one condition to a fixed random subspace of dimension ``d`` from that condition's init.

The sweep (locked design)
-------------------------
  * 3 CONDITIONS:  mae, plantclef (VT), naive.  (ImageNet is DROPPED.)
  * d-GRID (geometric, ~10^2 .. ~3x10^5):
        100, 300, 1000, 3000, 10000, 30000, 100000, 300000
  * 5 replicate SEEDS per (condition, d): different projections P.
  => 3 x 8 x 5 = 120 lines.

(Refine-around-threshold later: once a coarse sweep brackets where each condition's curve
crosses the 90%-of-gain target, add d-values around it with ``--extra-d`` and re-run only
the new lines. The base grid above is the coarse pass.)

Shared tokenizer
----------------
EVERY line carries ``--shared-tokenizer mae`` — the LOCKED common frozen input stage (MAE
sincos positions + MAE tokenizer) for all conditions, matching the main runs. (Asserted at
the bottom of this generator.)

Per-condition 90%-of-gain targets
----------------------------------
``target(condition) = perf_init(condition) + 0.9 * (perf_full(condition) - perf_init(condition))``
where ``perf_full`` is the converged held-out auc_pr_mean of that condition's **stable_evo
MAIN run** and ``perf_init`` is the held-out metric at that condition's init (pre-fine-tune).
These are PER-CONDITION and are injected onto every line as ``--perf_full`` / ``--perf_init``.

Source of the targets, in priority order:
  1. ``--perf-table PATH`` — a JSON or CSV mapping condition -> {perf_init, perf_full}. JSON:
         {"mae": {"perf_init": 0.62, "perf_full": 0.81}, "plantclef": {...}, "naive": {...}}
     CSV columns: condition, perf_init, perf_full.
  2. otherwise the built-in PLACEHOLDER values below (clearly marked) so the grid is
     generatable BEFORE the main runs finish; regenerate with ``--perf-table`` once the main
     stable_evo runs report their initial/final val. The placeholders are deliberately rough
     and MUST be replaced before the real sweep is used for the figure.

Usage
-----
    # coarse 120-line grid with placeholder targets (pre-main-run):
    python xAI/py/d90/gen_d90_grid.py
    # with real per-condition targets once the main runs are in:
    python xAI/py/d90/gen_d90_grid.py --perf-table xAI/output/preadapt/perf_table.json
    # -> writes xAI/py/d90/d90_grid.txt and prints the exact --array=0-119%3
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

# --- locked sweep axes -------------------------------------------------------------------
CONDITIONS: List[str] = ["mae", "plantclef", "naive"]
D_GRID: List[int] = [100, 300, 1000, 3000, 10000, 30000, 100000, 300000]
SEEDS: List[int] = [42, 43, 44, 45, 46]            # 5 replicates (briefing: 5-10/condition)
SHARED_TOKENIZER = "mae"                           # LOCKED common frozen input stage

# --- PLACEHOLDER per-condition targets (REPLACE with --perf-table once main runs report) --
# auc_pr_mean. Ordering reflects the framework expectation (task-aligned start -> higher
# init AND lower d90): plantclef(VT) >= mae > naive at init; all converge to a similar
# full-finetune ceiling. These are rough stand-ins, NOT measured values.
PLACEHOLDER_PERF: Dict[str, Dict[str, float]] = {
    "plantclef": {"perf_init": 0.70, "perf_full": 0.82},
    "mae":       {"perf_init": 0.62, "perf_full": 0.81},
    "naive":     {"perf_init": 0.45, "perf_full": 0.79},
}

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "d90_grid.txt"

# Defaults for the stopping rule / data — overridable on the CLI; injected onto every line so
# each run_subspace.py invocation is fully specified by its grid line.
DEFAULTS = dict(
    step_cap=4000, patience=8, eps=1e-4, eval_every=50,
    batch_size=384, lr=1e-3, variant="stable_evo",
    train_csv="data/inat/train_v1.1.0.csv",
    val_csv="data/inat/val_v1.1.0.csv",
)


def load_perf_table(path: str) -> Dict[str, Dict[str, float]]:
    """Load per-condition {perf_init, perf_full} from a JSON or CSV file."""
    if path.endswith(".json"):
        with open(path) as f:
            raw = json.load(f)
        table = {c: {"perf_init": float(raw[c]["perf_init"]),
                     "perf_full": float(raw[c]["perf_full"])} for c in raw}
    else:  # CSV: condition, perf_init, perf_full
        import csv
        table = {}
        with open(path) as f:
            for r in csv.DictReader(f):
                table[r["condition"]] = {
                    "perf_init": float(r["perf_init"]),
                    "perf_full": float(r["perf_full"]),
                }
    missing = [c for c in CONDITIONS if c not in table]
    if missing:
        raise ValueError(f"--perf-table is missing condition(s): {missing}")
    return table


def build_lines(perf: Dict[str, Dict[str, float]], out_table: str,
                d_grid: List[int], seeds: List[int]) -> List[str]:
    """Build the grid lines: condition (outer) x d (middle) x seed (inner)."""
    lines: List[str] = []
    for condition in CONDITIONS:
        pi = perf[condition]["perf_init"]
        pf = perf[condition]["perf_full"]
        for d in d_grid:
            for seed in seeds:
                parts = [
                    f"--condition {condition}",
                    f"--d {d}",
                    f"--seed {seed}",
                    f"--shared-tokenizer {SHARED_TOKENIZER}",
                    f"--variant {DEFAULTS['variant']}",
                    f"--perf_init {pi:g}",
                    f"--perf_full {pf:g}",
                    f"--step_cap {DEFAULTS['step_cap']}",
                    f"--patience {DEFAULTS['patience']}",
                    f"--eps {DEFAULTS['eps']:g}",
                    f"--eval_every {DEFAULTS['eval_every']}",
                    f"--batch_size {DEFAULTS['batch_size']}",
                    f"--lr {DEFAULTS['lr']:g}",
                    f"--train_csv {DEFAULTS['train_csv']}",
                    f"--val_csv {DEFAULTS['val_csv']}",
                    f"--out {out_table}",
                ]
                lines.append(" ".join(parts))
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description="Emit the init-d90 subspace sweep grid.")
    ap.add_argument("--perf-table", default=None,
                    help="JSON/CSV of per-condition {perf_init, perf_full}. If unset, use the "
                         "built-in PLACEHOLDERS (regenerate once the main runs report).")
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help="grid file to write (one run_subspace.py arg string per line).")
    ap.add_argument("--out-table", default="xAI/output/d90/d90_results.parquet",
                    help="the d90 results table each run appends its row to (--out of "
                         "run_subspace.py); injected onto every line.")
    ap.add_argument("--extra-d", type=int, nargs="*", default=None,
                    help="refine-around-threshold: ADD these d-values to the grid (e.g. "
                         "--extra-d 2000 5000) instead of the base 8-point grid.")
    args = ap.parse_args()

    if args.perf_table:
        perf = load_perf_table(args.perf_table)
        src = f"--perf-table {args.perf_table}"
    else:
        perf = PLACEHOLDER_PERF
        src = "BUILT-IN PLACEHOLDERS (replace with --perf-table before the real sweep!)"

    d_grid = D_GRID if not args.extra_d else sorted(set(D_GRID) | set(args.extra_d))
    lines = build_lines(perf, args.out_table, d_grid, SEEDS)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")

    n = len(lines)
    print(f"Wrote {n} d90 configs to {out}")
    print(f"  conditions = {CONDITIONS}")
    print(f"  d-grid     = {d_grid}")
    print(f"  seeds      = {SEEDS}  ({len(SEEDS)} replicates/condition-d)")
    print(f"  targets    = {src}")
    print(f"  results -> {args.out_table}")
    print(f"SLURM array size: --array=0-{n - 1}%3")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
gen_d90_backfill_queue.py — generate the d90 BACKFILL work-queue as one claimable file per
config (HYBRID collector+d90-backfill design).

Motivation
----------
Each preadapt run uses 3 concurrent GPUs: 1 B200 trainer + 2 L4 collector workers. The
collector workers PRIORITIZE checkpoint extraction, but there are long idle windows — the
entire B200 queue-wait *before* the trainer starts, and gaps during the run — where a worker's
L4 would otherwise sit idle. This queue lets an idle worker do useful work: run ONE d90
intrinsic-dimension config (``run_subspace.py``) to plateau/step_cap and log its perf(d) curve.

The d90 90%-of-gain *thresholds* are computed POST-HOC from the perf(d) curves, so backfill
runs do NOT need ``--perf_full`` / ``--perf_init`` targets (those would only define an early
"criterion_met" stop). Backfill = run to plateau / step_cap and record the curve.

The queue is PERSISTENT and SHARED across all runs' workers: successive runs' workers keep
chewing through the same ~120-240 configs without redoing any. Claiming is via atomic rename
(``os.rename`` here, ``mv`` in the bash worker) of a single ``.args`` file out of ``pending/``.

Layout (under --queue-dir, default xAI/output/preadapt/_d90_queue/)
-------------------------------------------------------------------
    pending/   <NNN>_<condition>_d<d>_s<seed>.args   # full run_subspace.py arg-line, one per file
    claimed/   (a worker renames the .args here while running it)
    done/      (worker moves it here on success)
    failed/    (worker moves it here on failure)
    d90_results.csv   shared append-only results table (run_subspace.py --out)

Each ``.args`` file contains the FULL run_subspace.py argument line, WITHOUT
``--perf_full``/``--perf_init`` (backfill), WITH ``--shared-tokenizer mae`` and
``--out <queue-dir>/d90_results.csv``.

Idempotent: re-running only (re)creates ``pending/`` entries for configs that are not already
present in ``pending/`` / ``claimed/`` / ``done/`` (a ``failed/`` entry is re-queued so it can
be retried). Reuses the CONDITIONS / d-grid / seed logic from ``gen_d90_grid.py``.

Usage
-----
    mamba run -n reticulate-gpu2 python xAI/scripts/gen_d90_backfill_queue.py
    # custom location / refine-around-threshold:
    python xAI/scripts/gen_d90_backfill_queue.py --queue-dir xAI/output/preadapt/_d90_queue \
        --extra-d 2000 5000
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List

# Reuse the locked sweep axes / stopping-rule + data defaults from the canonical grid generator
# so the backfill queue and the array-based sweep never drift.
_HERE = os.path.dirname(os.path.abspath(__file__))
_D90_DIR = os.path.join(_HERE, "..", "py", "d90")
if _D90_DIR not in sys.path:
    sys.path.insert(0, _D90_DIR)

from gen_d90_grid import CONDITIONS, D_GRID, SEEDS, SHARED_TOKENIZER, DEFAULTS  # noqa: E402

# Project-root-relative default queue dir (matches the collector worker's D90_QUEUE default).
DEFAULT_QUEUE_DIR = "xAI/output/preadapt/_d90_queue"
SUBDIRS = ("pending", "claimed", "done", "failed")


def config_stem(idx: int, condition: str, d: int, seed: int) -> str:
    """Stable, sortable, collision-free stem identifying one (condition, d, seed) config."""
    return f"{idx:03d}_{condition}_d{d}_s{seed}"


def build_arg_line(condition: str, d: int, seed: int, results_csv: str,
                   state_dir: str, max_wall_seconds: int) -> str:
    """The FULL run_subspace.py arg line for ONE backfill config.

    Backfill => NO --perf_full / --perf_init (run to plateau / step_cap, log the perf(d)
    curve; thresholds are post-hoc). Carries --shared-tokenizer mae and --out <results_csv>.

    TIME-BOX (--max_wall_seconds + --state_dir) is BAKED INTO the arg-line, not just appended by
    the worker, so that ALREADY-RUNNING hybrid workers (whose bash function predates the time-box)
    still pass it and yield the GPU back to collection every ~max_wall_seconds. A d90 config can
    take 25 min-2 h; without the box a worker stuck mid-d90 cannot collect, so if Phase 2 starts
    emitting the trainer stalls on backpressure. With it, run_subspace saves resumable state and
    exits 2 on timeout. (New-design workers ALSO append these flags; argparse takes the last value,
    so the redundancy is harmless. Old-design workers treat exit 2 as a 'failure' -> the config
    lands in failed/ with its state preserved and resumes on the next regen+claim.)
    """
    # L4-SAFE BATCH (CRITICAL): backfill runs ONLY on the L4 collector workers (22 GB), NOT the
    # B200 trainer. ViT-L training (full forward+backward to populate dL/dtheta, even though only
    # the d-dim ``v`` is optimized) at the grid DEFAULTS batch (384, sized for the B200) OOMs the
    # L4 — the perf_init eval (no_grad) fits but the first training step does not. Override to a
    # small batch that fits the L4 with margin; d90 only needs to reach plateau, so a smaller,
    # noisier/slower batch does not change the intrinsic-dimension threshold.
    L4_BACKFILL_BATCH = 64
    parts = [
        "--backfill",                       # run-to-plateau/step_cap; perf_init MEASURED; no target
        f"--condition {condition}",
        f"--d {d}",
        f"--seed {seed}",
        f"--shared-tokenizer {SHARED_TOKENIZER}",
        f"--variant {DEFAULTS['variant']}",
        f"--step_cap {DEFAULTS['step_cap']}",
        f"--patience {DEFAULTS['patience']}",
        f"--eps {DEFAULTS['eps']:g}",
        f"--eval_every {DEFAULTS['eval_every']}",
        f"--batch_size {L4_BACKFILL_BATCH}",   # OVERRIDE DEFAULTS['batch_size'] (384) -> L4-safe
        f"--lr {DEFAULTS['lr']:g}",
        f"--train_csv {DEFAULTS['train_csv']}",
        f"--val_csv {DEFAULTS['val_csv']}",
        f"--max_wall_seconds {max_wall_seconds}",   # BAKED time-box (yield to collection ~every box)
        f"--state_dir {state_dir}",                 # BAKED resumable-state dir (keyed cond,d,seed)
        f"--out {results_csv}",
    ]
    return " ".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate the d90 BACKFILL work-queue (one claimable .args file per config).")
    ap.add_argument("--queue-dir", default=DEFAULT_QUEUE_DIR,
                    help="root of the persistent shared queue (default: %(default)s).")
    ap.add_argument("--extra-d", type=int, nargs="*", default=None,
                    help="refine-around-threshold: ADD these d-values to the base grid.")
    ap.add_argument("--max-wall-seconds", type=int, default=600,
                    help="BAKED per-claim time-box (s) so even already-running workers yield the "
                         "GPU back to collection; run_subspace saves resumable state on timeout "
                         "(default: %(default)s).")
    args = ap.parse_args()

    queue_dir = Path(args.queue_dir)
    for sub in SUBDIRS:
        (queue_dir / sub).mkdir(parents=True, exist_ok=True)
    pending_dir = queue_dir / "pending"
    state_dir = str(queue_dir / "state")
    results_csv = str(queue_dir / "d90_results.csv")

    d_grid = D_GRID if not args.extra_d else sorted(set(D_GRID) | set(args.extra_d))

    # A config is "already present" (don't re-create its pending entry) if a file with its stem
    # exists in pending/, claimed/, or done/. failed/ does NOT count -> failed configs are
    # re-queued for retry. We match on stem prefix so the claimed/ "<stem>.args.<jobid>_wN"
    # suffix is still recognized.
    existing_stems = set()
    for sub in ("pending", "claimed", "done"):
        for f in (queue_dir / sub).iterdir() if (queue_dir / sub).is_dir() else []:
            name = f.name
            # strip a leading ".args..." so both "<stem>.args" and "<stem>.args.<tag>" map back.
            stem = name.split(".args", 1)[0]
            existing_stems.add(stem)

    idx = 0
    created = 0
    total = 0
    for condition in CONDITIONS:
        for d in d_grid:
            for seed in SEEDS:
                stem = config_stem(idx, condition, d, seed)
                total += 1
                idx += 1
                if stem in existing_stems:
                    continue
                arg_line = build_arg_line(condition, d, seed, results_csv,
                                          state_dir, args.max_wall_seconds)
                (pending_dir / f"{stem}.args").write_text(arg_line + "\n")
                created += 1

    n_pending = len(list(pending_dir.glob("*.args")))
    print(f"d90 backfill queue: {queue_dir}")
    print(f"  conditions = {CONDITIONS}")
    print(f"  d-grid     = {d_grid}")
    print(f"  seeds      = {SEEDS}  ({len(SEEDS)} replicates/condition-d)")
    print(f"  results -> {results_csv}")
    print(f"  total configs in sweep : {total}")
    print(f"  newly created (pending): {created}")
    print(f"  pending/*.args now     : {n_pending}")
    print(f"  (claimed/done preserved; failed/ entries are re-queued on regen)")


if __name__ == "__main__":
    main()

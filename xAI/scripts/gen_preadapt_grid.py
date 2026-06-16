#!/usr/bin/env python
"""Generate the preadapt trainer arg-grids (REDESIGN: one trainer per run, runs sequential).

LOCKED DESIGN (do not change without group OK):
  * 3 CONDITIONS only:  mae, plantclef (=VT), naive.   ImageNet is DROPPED.
  * 2 VARIANTS:         adamw (baseline) and stable_evo (StableEvolutionSOAP + tuning-free
                        demographic noise).
  * Common frozen input stage = MAE for ALL conditions  =>  every line carries
    ``--shared-tokenizer mae``.
  * Seed default 42  =>  every line carries ``--seed 42``.
  => 3 conditions x 2 variants = 6 MAIN RUNS.

Each emitted line is the full argument string for ONE
``python xAI/py/preadapt_train.py <args>`` invocation (one condition x one variant x one seed).
The trainer derives run_id = "{condition}__{variant}__s{seed}" and the output dir
xAI/output/preadapt/<run_id> from these args, so we do NOT pass --output_dir here.

GRID-FILE LAYOUT (one line per condition x variant)
----------------------------------------------------
We emit ONE grid file PER VARIANT:

    configs/experiment/preadapt_grid_adamw.txt        (3 lines: mae, plantclef, naive)
    configs/experiment/preadapt_grid_stable_evo.txt   (3 lines: mae, plantclef, naive)

REDESIGN ORCHESTRATION (collector-throughput-limited; do NOT use the old %2 array): each RUN is
launched INDIVIDUALLY by run_preadapt_run.sh as 1 trainer (B200) + 2 collector workers (2x L4) = 3
GPUs, and the 6 runs (3 conditions x 2 variants) run STRICTLY SEQUENTIALLY (run_preadapt_all.sh
chains them with afterany dependencies) so only one run -- and thus at most 3 GPUs -- is ever
active. The grid files are the SOURCE of the per-run trainer arg-line: run_preadapt_run.sh selects
the line for <condition> from preadapt_grid_<variant>.txt (1-based line = condition index + 1 in
the order mae, plantclef, naive) and launches a SINGLE trainer with it (NOT an array). The old
per-variant `--array=0-2%2` wave model is superseded; these grids are still per-variant only so the
selection (variant -> file, condition -> line) stays trivial.

The shared hyperparameters below are passed EXPLICITLY so the launched command is self-documenting
and a later change to a trainer default cannot silently move the experiment. Override only the knobs
the locked + REDESIGN design requires (--shared-tokenizer mae, --seed 42, the dense-early sampler
cadence --sampler_min_step 5 / --sampler_delta 0.02 / --sampler_max_step 200, and the backpressure
marks --backpressure-high 6 / --backpressure-low 2). NOTE: --heavy-every is a COLLECTOR arg (set in
run_preadapt_run.sh / submit_preadapt_collector2.sh), NOT a trainer arg, so it is NOT emitted here.

Usage:
    python xAI/scripts/gen_preadapt_grid.py
    # -> writes both grid files (3 lines each) carrying the dense-early + backpressure flags.
"""
from __future__ import annotations

import argparse
from pathlib import Path

# .../xAI/scripts/gen_preadapt_grid.py -> parents[1] == .../xAI
XAI_ROOT = Path(__file__).resolve().parents[1]
GRID_DIR = XAI_ROOT / "two_noise" / "configs" / "experiment"

# --- LOCKED design knobs -----------------------------------------------------------------
CONDITIONS = ("mae", "plantclef", "naive")   # ImageNet DROPPED; mae/plantclef(VT)/naive only
VARIANTS = ("adamw", "stable_evo")
SHARED_TOKENIZER = "mae"                      # common frozen MAE input stage for ALL conditions
SEED = 42

# --- shared hyperparameters (trainer defaults, passed explicitly; see preadapt_train.parse_args) ---
# Keep these aligned with preadapt_train.py defaults unless the design clearly needs an override.
#
# REDESIGN (collector-throughput-limited, dense-early): the COLLECTOR is now the bottleneck, NOT
# the trainer. Instead of widening the sampler so a single collector keeps up, we run TWO collectors
# per run and let the trainer PAUSE emission/training (backpressure) whenever the work queue gets
# deep. The sampler is therefore tuned for DENSE-EARLY sampling — many checkpoints in Phase-2 epoch
# 1 while weights move fast, slowing as displacement slows — and the backpressure high/low marks
# bound the on-disk work-queue depth so the collectors never fall irrecoverably behind.
#
# Knobs below that DELIBERATELY OVERRIDE the trainer defaults:
#   * max_train_samples = 214000  -> per-run train set is the fixed 214k subset (trainer default is
#     None == full set). Both variants carry it. With phase2_batch_size 384 this is ~557 Phase-2
#     steps/epoch x ~14 Phase-2 epochs ~= 7.8k Phase-2 steps total.
#   * sampler_min_step = 5, sampler_delta = 0.02, sampler_max_step = 200  (trainer defaults 20 /
#     0.10 / 500). DENSE-EARLY cadence: a low floor (5) + a small displacement trigger (0.02) emit
#     MANY checkpoints early (target >= 25 in Phase-2 epoch 1 while weights change fast), tapering
#     naturally as the per-layer relative displacement shrinks; the 200-step max-step cap guarantees
#     a regular backbone for the time series even once motion stalls. The collector keep-up problem
#     this would otherwise create is handled by TWO collectors + trainer backpressure, NOT by
#     widening the cadence.
#   * backpressure_high = 6, backpressure_low = 2  -> the trainer PAUSES checkpoint emission (and
#     training) once PENDING checkpoints in the watch dir exceed 6, resuming only when the queue
#     drains below 2 (hysteresis). This makes COLLECTOR THROUGHPUT the limiter: training waits for
#     the 2 collectors to catch up rather than letting the work queue / disk grow unbounded.
SHARED_HPARAMS = {
    "num_epochs": 15,
    "batch_size": 768,            # Phase-1 (frozen backbone)
    "phase2_batch_size": 384,     # Phase-2 (full model)
    "blr": 5e-4,                  # base lr (effective Phase-2 lr = blr * eff_batch / 256)
    "weight_decay": 0.05,
    "max_train_samples": 214000,  # OVERRIDE: fixed 214k-subset train set per run (default None)
    "sampler_delta": 0.02,        # OVERRIDE (default 0.10): small trigger => dense-early sampling.
    "sampler_min_step": 5,        # OVERRIDE (default 20): low floor => >= 25 ckpts in P2 epoch 1.
    "sampler_max_step": 200,      # OVERRIDE (default 500): forced-emission cap (regular backbone).
    "backpressure_high": 6,       # trainer PAUSES emission+training when PENDING > 6 (high-water).
    "backpressure_low": 2,        # ...resumes when PENDING < 2 (low-water; hysteresis).
    "demo_kappa": 1.0,
    "demo_warmup": 200,
    "val_every_n": 100,
    "projection_dim": 8192,
    "retention_latest_keep": 2,
}

# Hyperparameter keys whose argparse flag uses DASHES rather than underscores. Most trainer flags
# are ``--snake_case``, but the REDESIGN backpressure flags are ``--backpressure-high`` /
# ``--backpressure-low`` (see preadapt_train.parse_args). We map those keys when emitting the line.
DASH_FLAG_KEYS = {
    "backpressure_high": "backpressure-high",
    "backpressure_low": "backpressure-low",
}

# --- stable_evo-only hyperparameters ------------------------------------------------------
# Knobs that only apply to the StableEvolutionSOAP path. Emitted ONLY on stable_evo lines
# (AdamW does not have a Kronecker preconditioner). --max_precond_dim caps the Kronecker
# factor size so full-ViT-L optimizer-state checkpoints stay small (~GBs, not ~15 GB) and the
# checkpoint EXTRACTOR's host-RAM footprint stays bounded; over-cap factors fall back to
# diagonal Adam in the optimizer.
STABLE_EVO_HPARAMS = {
    "max_precond_dim": 2048,
}


def line_for(condition: str, variant: str) -> str:
    """Build the full preadapt_train.py arg-line for one (condition x variant) run.

    ALWAYS includes --shared-tokenizer mae and --seed 42 (the locked design); --variant routes
    to the AdamW baseline or StableEvolutionSOAP. demo_* knobs are emitted for both variants
    (harmless / ignored by the adamw path) so the two grids are structurally identical. The
    stable_evo-only knobs in STABLE_EVO_HPARAMS (e.g. --max_precond_dim) are emitted ONLY on
    stable_evo lines.
    """
    parts = [
        f"--condition {condition}",
        f"--variant {variant}",
        f"--shared-tokenizer {SHARED_TOKENIZER}",
        f"--seed {SEED}",
    ]
    hparams = dict(SHARED_HPARAMS)
    if variant == "stable_evo":
        hparams.update(STABLE_EVO_HPARAMS)
    for k, v in hparams.items():
        flag = DASH_FLAG_KEYS.get(k, k)  # backpressure flags use dashes; the rest are snake_case
        parts.append(f"--{flag} {v:g}" if isinstance(v, float) else f"--{flag} {v}")
    return " ".join(parts)


def grid_path(variant: str) -> Path:
    return GRID_DIR / f"preadapt_grid_{variant}.txt"


def main() -> None:
    ap = argparse.ArgumentParser(description="Emit the preadapt trainer arg-grids (per variant).")
    ap.add_argument("--grid-dir", default=str(GRID_DIR),
                    help="directory to write the per-variant grid files into")
    args = ap.parse_args()

    out_dir = Path(args.grid_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for variant in VARIANTS:
        lines = [line_for(cond, variant) for cond in CONDITIONS]
        path = out_dir / f"preadapt_grid_{variant}.txt"
        path.write_text("\n".join(lines) + "\n")
        total += len(lines)
        print(f"Wrote {len(lines)} configs ({', '.join(CONDITIONS)}) to {path}")
        for i, cond in enumerate(CONDITIONS):
            print(f"  line {i + 1}: {cond:<10s} (run_preadapt_run.sh {cond} {variant})")

    print()
    print(f"TOTAL: {total} runs ({len(CONDITIONS)} conditions x {len(VARIANTS)} variants).")
    print("REDESIGN orchestration (collector-throughput-limited): each run = 1 trainer (B200) + 2 "
          "collector workers (2x L4) = 3 GPUs; the 6 runs are launched SEQUENTIALLY by "
          "run_preadapt_all.sh (afterany chain) so <= 3 concurrent GPUs at all times.")
    print("Dense-early cadence: --sampler_min_step 5 --sampler_delta 0.02 --sampler_max_step 200; "
          "backpressure: --backpressure-high 6 --backpressure-low 2; heavy_every=3 (collector arg).")


if __name__ == "__main__":
    main()

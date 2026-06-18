#!/usr/bin/env python
"""Generate the preadapt v2 trainer arg-grid (RESTART: StableEvo-only, 2 seeds, replicated).

This is the V2 SIBLING of gen_preadapt_grid.py. The original is KEPT (it emits per-variant
adamw + stable_evo grids and will be used for the adamw arm later); this generator is a NEW
file so the two do not clash. Differences from v1:

V2 LOCKED DESIGN (do not change without group OK):
  * StableEvo ONLY:  the v2 restart compares conditions, not optimizers, so there is exactly
    ONE variant ``stable_evo``. The AdamW code path stays runnable (use gen_preadapt_grid.py
    for it); it is simply not part of this grid.
  * 3 CONDITIONS:    mae, plantclef (=VT), naive.   ImageNet is DROPPED (as in v1).
  * 2 SEEDS:         42 and 43  =>  two replicates per condition.
  * Common frozen input stage = MAE for ALL conditions  =>  every line carries
    ``--shared-tokenizer mae``.
  => 1 variant x 3 conditions x 2 seeds = 6 MAIN RUNS.

V2 trainer-config flags carried on EVERY line (the restart's distinguishing knobs):
  * ``--lr_schedule fixed``    (v1 used the cosine default)
  * ``--beta1 0.0``            (override; the v2 demographic-noise restart runs beta1 = 0)
  * ``--weight_decay 0.0``     (override; v1 used 0.05)
  * ``--max_train_samples 107000``  (override; v1 used 214000 -- the v2 restart halves it)
  * ``--phase2_early_stop``    (bare flag; enable Phase-2 plateau early stop)

ORDERING (REPLICATE-MAJOR, condition-minor): all FIRST replicates before any SECOND replicate:
    round 1: (mae s42, plantclef s42, naive s42)
    round 2: (mae s43, plantclef s43, naive s43)
so the 6 lines are 1:mae/s42  2:plantclef/s42  3:naive/s42  4:mae/s43  5:plantclef/s43
6:naive/s43. This lets orchestration finish one full replicate of every condition before
investing GPU time in a second replicate.

Each emitted line is the full argument string for ONE
``python xAI/py/preadapt_train.py <args>`` invocation (one condition x one seed; variant is
always stable_evo). The trainer derives run_id = "{condition}__{variant}__s{seed}" from these
args. We do NOT pass --output_dir here -- orchestration sets it to the V2 root
xAI/output/preadapt_v2/<run_id> (distinct from the old preadapt/ so v1 runs are preserved).

The shared hyperparameters below are passed EXPLICITLY so the launched command is
self-documenting and a later change to a trainer default cannot silently move the experiment.

Usage:
    python xAI/scripts/gen_preadapt_grid_v2.py
    # -> writes ONE grid file: configs/experiment/preadapt_v2_grid.txt (6 lines).
"""
from __future__ import annotations

import argparse
from pathlib import Path

# .../xAI/scripts/gen_preadapt_grid_v2.py -> parents[1] == .../xAI
XAI_ROOT = Path(__file__).resolve().parents[1]
GRID_DIR = XAI_ROOT / "two_noise" / "configs" / "experiment"
GRID_FILE = GRID_DIR / "preadapt_v2_grid.txt"

# --- LOCKED v2 design knobs --------------------------------------------------------------
CONDITIONS = ("mae", "plantclef", "naive")   # ImageNet DROPPED; mae/plantclef(VT)/naive only
VARIANT = "stable_evo"                        # v2 is StableEvo-only (AdamW via gen_preadapt_grid.py)
SHARED_TOKENIZER = "mae"                      # common frozen MAE input stage for ALL conditions
SEEDS = (42, 43)                              # 2 replicates per condition

# --- shared hyperparameters (trainer defaults, passed explicitly; see preadapt_train.parse_args) ---
# Knobs below that DELIBERATELY OVERRIDE the trainer defaults for the v2 restart:
#   * lr_schedule = "fixed"     (default "cosine"): v2 holds the Phase-2 lr flat.
#   * beta1 = 0.0               (default None): v2 demographic-noise restart runs beta1 = 0.
#   * weight_decay = 0.0        (default 0.05): v2 disables weight decay.
#   * max_train_samples = 107000  (v1 used 214000): v2 restart per-run train subset.
#   * sampler_delta/min/max = 0.01 / 5 / 200  (defaults 0.10 / 20 / 500): dense-early cadence
#     (many checkpoints while weights move fast), matching v1's collector-throughput design.
#   * backpressure_high/low = 6 / 2: trainer PAUSES emission+training when PENDING > 6, resumes
#     below 2 (hysteresis) so collector throughput is the limiter.
# Note: --phase2_early_stop is a bare store_true flag (no value); it is appended separately.
SHARED_HPARAMS = {
    "num_epochs": 25,             # backstop only; train-loss-plateau early-stop is the real
                                  # terminator. Raised 15->25 (2026-06-17): with FIXED LR (no cosine
                                  # forcing convergence) the neutral-manifold cruise can run long; v2
                                  # uses HALF data so 25 half-epochs ~= 12.5 full-data-epochs of
                                  # compute (< v1's 15 full). Generous backstop + early-stop avoids
                                  # truncating basin-settling while wasting no compute.
    "batch_size": 768,            # Phase-1 (frozen backbone)
    "phase2_batch_size": 384,     # Phase-2 (full model)
    "blr": 5e-4,                  # base lr (effective Phase-2 lr = blr * eff_batch / 256)
    "lr_schedule": "fixed",       # OVERRIDE (default "cosine"): v2 flat Phase-2 lr.
    "beta1": 0.0,                 # OVERRIDE (default None): v2 restart beta1 = 0.
    "weight_decay": 0.0,          # OVERRIDE (default 0.05): v2 disables weight decay.
    "max_train_samples": 107000,  # OVERRIDE (v1 214000): v2 per-run train subset.
    "sampler_delta": 0.01,        # OVERRIDE (default 0.10): dense-early; ~31 emits/P2 epoch 1.
    "sampler_min_step": 4,        # OVERRIDE (default 20): low floor => many early ckpts. 5->4
                                  # (2026-06-17) for denser unfreeze/metric-reorganization capture.
    "sampler_max_step": 200,      # OVERRIDE (default 500): forced-emission cap (regular backbone).
    "backpressure_high": 6,       # trainer PAUSES emission+training when PENDING > 6 (high-water).
    "backpressure_low": 2,        # ...resumes when PENDING < 2 (low-water; hysteresis).
    "demo_kappa": 1.0,
    "demo_warmup": 200,
    "val_every_n": 100,
    "projection_dim": 8192,
    "retention_latest_keep": 2,
    "max_precond_dim": 2048,      # stable_evo-only: caps Kronecker factor size (small ckpts).
    "num_workers": 4,             # 8->4 (2026-06-18): the DataLoader workers each fork a ~7.6 GB
                                  # copy of the dataset object; at 8 workers (x multiple loaders) the
                                  # trainer peaked ~155 GB host RSS, blocking the GPU collector from
                                  # getting the ~207 GB it needs for heavy checkpoints under the 437
                                  # GB QOS. 4 workers drops the trainer to ~91 GB (req 104), freeing
                                  # room so the collector fits heavy without thrash. GPU-bound
                                  # training, so 4 workers keep the GPU fed.
}

# Bare store_true flags (no value) emitted AFTER the key=value hparams.
STORE_TRUE_FLAGS = ("phase2_early_stop",)

# Hyperparameter keys whose argparse flag uses DASHES rather than underscores. Most trainer
# flags are ``--snake_case``, but the backpressure flags are ``--backpressure-high`` /
# ``--backpressure-low`` (see preadapt_train.parse_args).
DASH_FLAG_KEYS = {
    "backpressure_high": "backpressure-high",
    "backpressure_low": "backpressure-low",
}


def line_for(condition: str, seed: int) -> str:
    """Build the full preadapt_train.py arg-line for one (condition x seed) v2 run.

    ALWAYS includes --variant stable_evo, --shared-tokenizer mae, the v2 override flags, and
    --phase2_early_stop. We do NOT emit --output_dir (orchestration sets preadapt_v2/<run_id>).
    """
    parts = [
        f"--condition {condition}",
        f"--variant {VARIANT}",
        f"--shared-tokenizer {SHARED_TOKENIZER}",
        f"--seed {seed}",
    ]
    for k, v in SHARED_HPARAMS.items():
        flag = DASH_FLAG_KEYS.get(k, k)  # backpressure flags use dashes; the rest are snake_case
        if isinstance(v, bool):
            # No bool-typed hparams here, but guard against int(bool) ambiguity.
            parts.append(f"--{flag} {int(v)}")
        elif isinstance(v, float):
            parts.append(f"--{flag} {v:g}")
        else:
            parts.append(f"--{flag} {v}")
    for flag in STORE_TRUE_FLAGS:
        parts.append(f"--{flag}")
    return " ".join(parts)


def build_grid() -> list[tuple[str, int, str]]:
    """Return the 6 (condition, seed, arg_line) tuples in REPLICATE-MAJOR order.

    Outer loop = seed (replicate), inner loop = condition  =>  all first replicates
    (s42: mae, plantclef, naive) precede all second replicates (s43: mae, plantclef, naive).
    """
    grid: list[tuple[str, int, str]] = []
    for seed in SEEDS:
        for cond in CONDITIONS:
            grid.append((cond, seed, line_for(cond, seed)))
    return grid


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Emit the preadapt v2 trainer arg-grid (StableEvo-only, 2 seeds)."
    )
    ap.add_argument("--grid-dir", default=str(GRID_DIR),
                    help="directory to write the v2 grid file into")
    args = ap.parse_args()

    out_dir = Path(args.grid_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "preadapt_v2_grid.txt"

    grid = build_grid()
    out_path.write_text("\n".join(line for _, _, line in grid) + "\n")

    print(f"Wrote {len(grid)} v2 configs to {out_path}")
    print(f"  variant = {VARIANT} (StableEvo-only); conditions = {', '.join(CONDITIONS)}; "
          f"seeds = {', '.join(str(s) for s in SEEDS)}")
    print("Ordering (replicate-major: all first replicates, then all second replicates):")
    for i, (cond, seed, _) in enumerate(grid, start=1):
        run_id = f"{cond}__{VARIANT}__s{seed}"
        replicate = "round 1" if seed == SEEDS[0] else "round 2"
        print(f"  line {i}: {cond:<10s} s{seed}  ({replicate})  -> run_id {run_id} "
              f"-> output_dir preadapt_v2/{run_id}")

    print()
    print(f"TOTAL: {len(grid)} runs ({len(CONDITIONS)} conditions x {len(SEEDS)} seeds, "
          f"variant {VARIANT}).")
    print("v2 override flags on every line: --variant stable_evo --shared-tokenizer mae "
          "--lr_schedule fixed --beta1 0.0 --weight_decay 0.0 --max_train_samples 107000 "
          "--phase2_early_stop.")
    print("Dense-early cadence: --sampler_min_step 4 --sampler_delta 0.01 --sampler_max_step 200; "
          "backpressure: --backpressure-high 6 --backpressure-low 2. "
          "--output_dir is NOT emitted (orchestration sets preadapt_v2/<run_id>).")


if __name__ == "__main__":
    main()

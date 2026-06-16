#!/usr/bin/env python
"""Generate the preadapt trainer arg-grids for the SLURM array (plan component C5).

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

GRID-FILE LAYOUT (chosen for the cleanest two-wave-per-variant schedule)
------------------------------------------------------------------------
We emit ONE grid file PER VARIANT:

    configs/experiment/preadapt_grid_adamw.txt        (3 lines: mae, plantclef, naive)
    configs/experiment/preadapt_grid_stable_evo.txt   (3 lines: mae, plantclef, naive)

Why per-variant (not one combined file)? The orchestration runs ONE variant at a time
(submit_preadapt_train.sh dispatches a single grid file as a `--array=0-2%2` array of 3
condition-trainers, throttled to 2 concurrent GPUs; one collector watches that variant's 3 run
dirs). A per-variant file maps 1:1 onto "one array == one variant == 3 conditions throttled %2"
with no variant column to filter on, and the collector's --watch-dirs are exactly the run dirs of
the lines in that one file. Variant 2 is launched AFTER variant 1 (sequentially or with
`sbatch --dependency=afterany:<jobid>`), so GPU usage never exceeds 2 trainers + 1 collector = 3.

The shared hyperparameters below are the trainer's own defaults (read from preadapt_train.py
parse_args) — we pass them EXPLICITLY so the launched command is self-documenting and a later
change to a trainer default cannot silently move the experiment. Override only the few knobs that
the locked design requires (--shared-tokenizer mae, --seed 42).

Usage:
    python xAI/scripts/gen_preadapt_grid.py
    # -> writes both grid files and prints the exact "--array=0-2%2" spec to use.
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
# Three knobs below DELIBERATELY OVERRIDE the trainer defaults (locked design + collector tuning):
#   * max_train_samples = 214000  -> per-run train set is the fixed 214k subset (trainer default is
#     None == full set). Both variants carry it. With phase2_batch_size 384 this is ~557 Phase-2
#     steps/epoch x ~14 Phase-2 epochs ~= 7.8k Phase-2 steps total.
#   * sampler_min_step / sampler_max_step = 300 / 800  -> PROVISIONAL conservative sampler cadence
#     (trainer defaults are 20 / 500). Widened so the checkpoint collector can keep up with the
#     214k-subset runs: at min_step 300 the trainer emits at most one ladder checkpoint every 300
#     Phase-2 steps (~26 emissions over the ~7.8k-step run), and at most every 800 if the
#     displacement trigger is quiet. These two values are PROVISIONAL pending the §2.3 collector
#     utilization pilot: the FINAL min_step is set from the measured collector per-checkpoint
#     extraction time T_ex, with the per-trainer emission interval required to be >= 2 * T_ex so a
#     single collector never falls behind two concurrent trainers.
SHARED_HPARAMS = {
    "num_epochs": 15,
    "batch_size": 768,            # Phase-1 (frozen backbone)
    "phase2_batch_size": 384,     # Phase-2 (full model)
    "blr": 5e-4,                  # base lr (effective Phase-2 lr = blr * eff_batch / 256)
    "weight_decay": 0.05,
    "max_train_samples": 214000,  # OVERRIDE: fixed 214k-subset train set per run (default None)
    "sampler_delta": 0.10,
    "sampler_min_step": 500,      # OVERRIDE (default 20): collector-keep-up floor. Set from measured T_ex: collector
                                  #   ~3min/ckpt amortized (cheap ~2min, heavy ~5min L4, heavy_every=3); min_step 500 over a
                                  #   214k run (~7.8k Phase-2 steps) => ~16 emissions/trainer so 1 collector keeps up with 2 trainers.
    "sampler_max_step": 1000,     # OVERRIDE (default 500): forced-emission cap (regular backbone for time-series).
    "demo_kappa": 1.0,
    "demo_warmup": 200,
    "val_every_n": 100,
    "projection_dim": 8192,
    "retention_latest_keep": 2,
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
        parts.append(f"--{k} {v:g}" if isinstance(v, float) else f"--{k} {v}")
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
        print(f"  --array=0-{len(lines) - 1}%2   # 3 conditions, throttled to 2 concurrent trainers")

    print()
    print(f"TOTAL: {total} runs ({len(CONDITIONS)} conditions x {len(VARIANTS)} variants).")
    print("Per variant: sbatch --array=0-2%2 (2 trainers) + 1 collector = 3 GPUs (NEVER raise without group OK).")


if __name__ == "__main__":
    main()

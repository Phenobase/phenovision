#!/usr/bin/env python
"""Generate the §2.1 benchmark grid (one config per line) for the SLURM array.

Reads configs/experiment/benchmark.yaml and writes configs/experiment/benchmark_grid.txt, where
each line is the argument string for one `python -m ml_experiments.benchmarks` invocation. The
SLURM array (scripts/submit_benchmark.sh) picks the line for its SLURM_ARRAY_TASK_ID.

By default only the auto-downloadable pairs (cifar100) are emitted, so the launch is self-contained.
Pass --include-manual to also emit tiny_imagenet / tinystories configs once their data is provisioned.

Usage:
    python scripts/gen_benchmark_grid.py                # cifar100 pairs only
    python scripts/gen_benchmark_grid.py --include-manual
"""
import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "configs" / "experiment" / "benchmark.yaml"
OUT = ROOT / "configs" / "experiment" / "benchmark_grid.txt"

AUTO_DATASETS = {"cifar100"}


def lr_grid_for(spec, optimizer, alpha):
    if optimizer == "adamw":
        return spec["lr_grid"]["adamw"]
    return spec["lr_grid"]["soap_lo"] if (alpha is None or alpha <= 0.5) else spec["lr_grid"]["soap_hi"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-manual", action="store_true",
                    help="also emit tiny_imagenet/tinystories (require manual data setup)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    spec = yaml.safe_load(SPEC.read_text())
    b = spec["budget"]
    lines = []
    for pair in spec["model_data_pairs"]:
        if pair["dataset"] not in AUTO_DATASETS and not args.include_manual:
            continue
        for opt in spec["optimizers"]:
            alpha = opt.get("alpha")
            for lr in lr_grid_for(spec, opt["optimizer"], alpha):
                parts = [
                    f"--model {pair['model']}", f"--dataset {pair['dataset']}",
                    f"--optimizer {opt['optimizer']}",
                    f"--lr {lr:g}", f"--batch-size {b['batch_size']}",
                    f"--max-steps {b['max_steps']}", f"--eval-every {b['eval_every']}",
                    f"--log-every 25", f"--seed {b['seed']}",
                ]
                if opt["optimizer"] == "soap":
                    parts.insert(3, f"--alpha {alpha:g}")
                if b.get("amp"):
                    parts.append("--amp")
                lines.append(" ".join(parts))

    out = Path(args.out)
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(lines)} configs to {out}")
    print(f"SLURM array size: --array=0-{len(lines) - 1}%3")


if __name__ == "__main__":
    main()

"""§2.1 standard benchmarks — run ONE (model, dataset, optimizer/alpha, lr, batch, budget) config.

This is the per-config CLI a SLURM array task calls. One array task = one config = one tidy CSV
under runs/benchmarks/. Keeping it per-config (rather than looping the whole matrix in-process)
lets the array parallelize the matrix across GPUs while staying inside the 3-GPU cap (see
scripts/submit_benchmark.sh).

Objective (analysis_plan.md §2.1): establish that full power is competitive and characterize the
step-efficiency vs wall-clock split honestly. We record train loss vs step AND vs wallclock, final
val metric, per-step time, and peak memory — everything the §2.1 figures need.

Matrix this CLI is meant to cover (driven by the SLURM array / configs/experiment/benchmark.yaml):
    optimizers: {adamw} ∪ {soap @ alpha in {0, 0.25, 0.5, 0.75, 1.0}}
    a small lr grid per alpha (alpha=1 needs ~10x smaller lr + relative damping ~1e-2)
    models/datasets: at least 2 model/data pairs (e.g. vit_s+cifar100, nanogpt+tinystories)

Example (single config, CPU smoke):
    python -m ml_experiments.benchmarks --model tiny_vision --dataset synthetic_vision \
        --optimizer soap --alpha 1.0 --lr 1e-4 --batch-size 16 --max-steps 5 --device cpu
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml_experiments._harness import (RUNS_DIR, default_lr, make_data, make_model,
                                     make_optimizer, train_eval)

# Tidy CSV schema (one row per logged step). Stable column order across all configs so the
# per-config CSVs concatenate cleanly for the §2.1 figures.
CSV_COLUMNS = [
    "model", "dataset", "optimizer", "alpha", "lr", "batch_size", "accum_steps",
    "eff_batch_size", "step", "train_loss", "wallclock_s", "lr_actual", "val_metric",
    "val_metric_name", "val_loss", "peak_mem_mb", "step_time_ms", "seed",
    "precond_mode", "shrink", "evolve_m", "mean_exponent",
    "exp_std", "exp_max", "exp_frac_high", "exp_frac_floor", "demo_temp", "variant",
    # SWA / iterate-averaging eval (MAP estimate) + gradient noise scale (McCandlish B_simple)
    "val_metric_swa", "val_loss_swa", "noise_scale", "grad_norm_sq", "tr_sigma",
]


def config_name(args, lr_actual) -> str:
    """Filesystem-safe config id used for the CSV filename."""
    if args.optimizer in ("adamw", "sgd"):
        opt = args.optimizer
    elif args.optimizer == "stable_evo":
        opt = "stable_evo" + ("_seloff" if args.selection_off else "")
    elif args.optimizer == "soap":
        opt = f"soap_a{args.alpha:g}"
    else:  # riccati
        mode = args.precond_mode or ("whiten" if args.alpha <= 0.5 else "inverse")
        opt = f"riccati_{mode}_rho{args.shrink:g}" + ("_evM" if args.evolve_m else "")
    if getattr(args, "demographic_noise", False) and args.demographic_temperature > 0:
        opt += f"_demoT{args.demographic_temperature:g}"      # pSGLD posterior-sampling variant
    if getattr(args, "label_suffix", ""):
        opt += f"_{args.label_suffix}"                        # e.g. soaplr (stable_evo at soap's lr)
    return (f"{args.model}__{args.dataset}__{opt}__lr{lr_actual:g}"
            f"__bs{args.batch_size}x{args.accum_steps}__s{args.seed}")


def write_csv(path: Path, rows, static):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            out = {k: static.get(k, "") for k in CSV_COLUMNS}
            out.update({k: v for k, v in r.items() if k in CSV_COLUMNS})
            w.writerow(out)


def run(args) -> Path:
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    train_loader, val_loader, meta = make_data(
        args.dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        block_size=args.block_size, generator=gen, synthetic_n=args.synthetic_n,
    )

    if meta.task == "lm":
        model = make_model(args.model, vocab_size=meta.vocab_size)
    else:
        model = make_model(args.model, num_classes=meta.num_classes)

    lr = args.lr if args.lr is not None else default_lr(args.optimizer, args.alpha, args.base_lr)
    # max_update_norm is an UPDATE-space trust region (clips the preconditioned step), distinct
    # from --grad-clip which clips the raw gradient. The full inverse (alpha->1) amplifies even a
    # clipped gradient in flat directions, so only the update-norm clip keeps it finite. 0 = off.
    opt_kw = {}
    if args.optimizer in ("soap", "stable_evo") and args.max_update_norm > 0:
        opt_kw["max_update_norm"] = args.max_update_norm
    if args.optimizer == "stable_evo":
        # selection-driven exponent; recapitulation = selection_off + alpha_max=1.0.
        opt_kw["selection_off"] = args.selection_off
        opt_kw["alpha_max"] = args.alpha_max
    # demographic-noise (pSGLD) injection — supported by soap and stable_evo (both run Adam in the
    # Kronecker eigenbasis). Turns convergence into posterior sampling; we measure how it moves
    # the realized alpha, final accuracy, and speed.
    if args.demographic_noise and args.optimizer in ("soap", "stable_evo"):
        opt_kw["demographic_noise"] = True
        opt_kw["demographic_temperature"] = args.demographic_temperature
        opt_kw["demographic_warmup"] = args.demographic_warmup
        opt_kw["demographic_generator"] = torch.Generator().manual_seed(args.seed + 9973)
        if args.optimizer == "stable_evo" and args.demographic_shape_exp is not None:
            opt_kw["demographic_shape_exp"] = args.demographic_shape_exp   # experiment B noise shape
    if args.optimizer == "riccati":
        # Riccati uses shrink/safeguard as the stabilizer, NOT max_update_norm (which
        # clips the step and breaks FDT). precond_mode overrides the alpha->mode default.
        if args.precond_mode:
            opt_kw["precond_mode"] = args.precond_mode
        opt_kw["shrink"] = args.shrink
        opt_kw["inner_steps"] = args.inner_steps
        if args.evolve_m:
            opt_kw["evolve_m"] = True
            opt_kw["eta_m"] = args.eta_m
            opt_kw["meta_every"] = args.meta_every
    optimizer, lr_actual = make_optimizer(
        args.optimizer, model.parameters(), alpha=args.alpha, lr=lr,
        weight_decay=args.weight_decay, base_lr=args.base_lr, **opt_kw,
    )

    eff_bs = args.batch_size * args.accum_steps
    static = dict(
        model=args.model, dataset=args.dataset,
        optimizer=args.optimizer,
        alpha=args.alpha if args.optimizer in ("soap", "riccati") else "",
        lr=lr_actual, lr_actual=lr_actual, batch_size=args.batch_size,
        accum_steps=args.accum_steps, eff_batch_size=eff_bs,
        val_metric_name="perplexity" if meta.task == "lm" else "accuracy",
        seed=args.seed,
        precond_mode=(args.precond_mode or ("whiten" if args.alpha <= 0.5 else "inverse"))
        if args.optimizer == "riccati" else "",
        shrink=args.shrink if args.optimizer == "riccati" else "",
        evolve_m=int(args.evolve_m) if args.optimizer == "riccati" else "",
        demo_temp=(args.demographic_temperature
                   if (args.demographic_noise and args.demographic_temperature > 0) else ""),
        variant=getattr(args, "label_suffix", "") or "",
    )

    out_dir = Path(args.out_dir) if args.out_dir else (RUNS_DIR / "benchmarks")
    csv_path = out_dir / f"{config_name(args, lr_actual)}.csv"
    # flush partial results to the CSV at every eval (within-run visibility + preemption safety)
    result = train_eval(
        model, optimizer, train_loader, val_loader, device,
        max_steps=args.max_steps, epochs=args.epochs, accum_steps=args.accum_steps,
        log_every=args.log_every, eval_every=args.eval_every,
        eval_max_batches=args.eval_max_batches, grad_clip=args.grad_clip,
        amp=args.amp, lr=lr_actual, extra_record_fields=static,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        eval_hook=lambda res: write_csv(csv_path, res.records, static),
        swa_start_frac=args.swa_start_frac,
        noise_scale_every=args.noise_scale_every,
        noise_scale_k=args.noise_scale_k,
    )
    static["val_metric_name"] = result.val_metric_name
    write_csv(csv_path, result.records, static)

    print(f"[benchmarks] {csv_path}")
    print(f"[benchmarks] final {result.val_metric_name}={result.final_val_metric:.4f} "
          f"val_loss={result.final_val_loss:.4f} steps={result.steps_run} "
          f"peak_mem_mb={result.peak_mem_mb:.0f} mean_step_ms={result.mean_step_time_ms:.1f}")
    return csv_path


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True,
                   help="vit_s|vit_b|nanogpt|nanogpt_m|tiny_vision")
    p.add_argument("--dataset", required=True,
                   help="cifar100|tiny_imagenet|tinystories|synthetic_vision|synthetic_lm")
    p.add_argument("--optimizer", default="adamw",
                   choices=["adamw", "sgd", "soap", "stable_evo", "riccati"])
    p.add_argument("--alpha", type=float, default=0.5, help="precond_power for soap (ignored for adamw)")
    # --- stable_evo-only knobs (selection-driven exponent; logs realized mean_exponent per step) ---
    p.add_argument("--selection-off", action="store_true",
                   help="stable_evo: pin alpha=alpha_max (the biological-recapitulation mode)")
    p.add_argument("--alpha-max", type=float, default=0.9,
                   help="stable_evo: per-coordinate exponent ceiling (1.0 for recapitulation)")
    # --- demographic-noise (pSGLD) injection; soap/stable_evo only ---
    p.add_argument("--demographic-noise", action="store_true",
                   help="inject FDT-correct pSGLD noise (covariance ∝ applied preconditioner)")
    p.add_argument("--demographic-temperature", type=float, default=0.0,
                   help="pSGLD temperature T (T∝lr; T=1/(2 Ne_eff) in the biological reading)")
    p.add_argument("--demographic-warmup", type=int, default=0,
                   help="skip noise injection for the first N steps (preconditioner warmup)")
    p.add_argument("--demographic-shape-exp", type=float, default=None,
                   help="stable_evo noise SHAPE (experiment B): variance ∝ v̂**β, trace-matched to "
                        "the pSGLD preconditioner shape so only the SHAPE varies at matched "
                        "temperature. Default (unset)=pSGLD/FDT (∝P). β=1 ~ Fisher/curvature-aligned "
                        "(like minibatch noise); β=0 isotropic.")
    p.add_argument("--label-suffix", default="",
                   help="tag appended to the config name + a 'variant' column, to distinguish "
                        "otherwise-identical optimizers (e.g. 'soaplr' = stable_evo run at soap's lr)")
    # --- SWA / iterate-averaging eval (MAP estimate) + gradient-noise-scale probe ---
    p.add_argument("--swa-start-frac", type=float, default=0.0,
                   help="average params over the final (1-frac) of training and eval that "
                        "'posterior-mean' model alongside the raw iterate (0 = off; try 0.75)")
    p.add_argument("--noise-scale-every", type=int, default=0,
                   help="measure the gradient noise scale tr(Σ)/|g|² every N opt-steps (0 = off)")
    p.add_argument("--noise-scale-k", type=int, default=4,
                   help="number of independent micro-batch grads per noise-scale estimate")
    # --- riccati-only knobs (matrix-free Newton-Schulz preconditioner) ---
    p.add_argument("--precond-mode", default="", choices=["", "whiten", "inverse"],
                   help="riccati base mode; default derived from alpha (<=0.5 whiten, else inverse)")
    p.add_argument("--shrink", type=float, default=0.0,
                   help="riccati shrinkage rho toward isotropy (effective-exponent reducer / stabilizer)")
    p.add_argument("--inner-steps", type=int, default=2, help="riccati Newton-Schulz inner steps")
    p.add_argument("--evolve-m", action="store_true", help="riccati evolving-M meta-loop (O4)")
    p.add_argument("--eta-m", type=float, default=1e-3, help="evolve-M meta learning rate")
    p.add_argument("--meta-every", type=int, default=20, help="evolve-M accumulation window")
    p.add_argument("--lr", type=float, default=None, help="explicit lr; default derived from optimizer/alpha")
    p.add_argument("--base-lr", type=float, default=1e-3, help="reference lr for the default_lr schedule")
    p.add_argument("--batch-size", type=int, default=128, help="micro-batch size")
    p.add_argument("--accum-steps", type=int, default=1, help="grad-accum micro-batches per opt step")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-steps", type=int, default=0, help="optimizer steps (takes precedence over epochs)")
    p.add_argument("--epochs", type=int, default=0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=0, help="val eval every N steps (0 = end only)")
    p.add_argument("--eval-max-batches", type=int, default=0, help="cap val batches (0 = full)")
    p.add_argument("--early-stop-patience", type=int, default=0,
                   help="stop if val loss has not improved for N consecutive evals (0=off). Bounds "
                        "to-convergence runs so large batches don't over-train.")
    p.add_argument("--early-stop-min-delta", type=float, default=0.0,
                   help="minimum val-loss improvement to reset the early-stop patience counter")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--max-update-norm", type=float, default=0.0,
                   help="SOAP update-space trust region (clips preconditioned step); 0=off. "
                        "Needed to keep alpha->1 finite at small batch.")
    p.add_argument("--amp", action="store_true", help="autocast (cuda only)")
    p.add_argument("--block-size", type=int, default=256, help="LM sequence length")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--synthetic-n", type=int, default=64, help="size of synthetic_* datasets")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="", help="cpu|cuda (default: cuda if available)")
    p.add_argument("--out-dir", default="", help="override CSV output dir (default runs/benchmarks)")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.max_steps <= 0 and args.epochs <= 0:
        args.epochs = 1
    return run(args)


if __name__ == "__main__":
    main()

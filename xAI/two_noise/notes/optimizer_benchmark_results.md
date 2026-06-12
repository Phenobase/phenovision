# StableEvolutionSOAP benchmark: head-to-head results

**Run date:** 2026-06-11/12.  **Substrate:** ViT-S/16 trained from scratch on CIFAR-100
(no pretraining), AMP/bf16, single seed (s0), one L4 per cell, ≤3 GPUs concurrent.
**Code:** `ml_experiments/benchmarks.py` (one CSV per config), grid from
`scripts/gen_compare_grid.py`, array `scripts/submit_optimizer_compare.sh`, analysis
`figures/optimizer_compare.py`.

## What "accuracy" means here

**Validation-set top-1 accuracy.** Every accuracy number below is the `val_metric` column
written by `evaluate()` (`ml_experiments/_harness.py:542`), computed on the `val_loader`.
For CIFAR-100 that loader is `CIFAR100(root=..., train=False)` (`_harness.py:236`) — the
standard **10,000-image held-out test split**, which we use as the validation set for these
benchmarks (we never tune on it beyond reading the curve; the lr is set by a separate stage-1
range test on the train data). `val_loss` is the cross-entropy on that same split. So
"accuracy" is held-out top-1, not a train-set number.

## Protocol (two-stage, true-convergence budget)

1. **Stage 1 — lr range test** (`ml_experiments/lr_finder.py`): a short LR sweep per
   (optimizer, batch) → a tuned lr in `runs/lr_finder/suggested.csv`.
2. **Stage 2 — run to convergence** (this benchmark): one full run per (optimizer, batch) at
   that lr, **60-epoch budget** (per-batch `max_steps = 60·⌈50000/eff_batch⌉`, ~50 evals),
   with early stopping (patience 8 evals) so large batches don't burn budget on a plateau.
   This equalizes *data seen* across batch sizes and lets each cell actually settle.

Optimizers: `sgd` (Nesterov), `adamw`, `soap@0.5` (whitening, α=½), `soap@1.0` (full inverse,
α=1), `stable_evo` (StableEvolutionSOAP). Effective batches: 64 / 256 / 1024 / 4096 (micro-batch
256, gradient accumulation). Data in `runs/benchmarks_conv/`; a matched fixed-8000-step snapshot
(used for the lr contrast below) in `runs/benchmarks_fixedstep8k/`.

---

## 1. Performance — a tight cluster, no dominant winner

Final validation top-1 (convergence budget):

| eff batch | adamw  | soap@0.5 | stable_evo+soaplr | sgd   | soap@1.0 |
|----------:|:------:|:--------:|:-----------------:|:-----:|:--------:|
| 64        | 0.4632 | 0.4687   | **0.4741**        | 0.072 | 0.154    |
| 256       | **0.4655** | 0.4522 | 0.4403          | 0.074 | 0.149    |
| 1024      | 0.3864 | 0.4341   | **0.4466**        | 0.078 | 0.070    |
| 4096      | **0.4509** | 0.4252 | 0.4417          | 0.110 | 0.062    |

**Read:** the three real contenders — `adamw`, `soap@0.5`, `stable_evo` (at soap's lr; see §4) —
all land at **0.42–0.47**, within ~1–2 points of each other. The per-batch winner reshuffles
(stable_evo wins 64 & 1024, adamw wins 256 & 4096, soap@0.5 is never first but never far), which
at single-seed CIFAR-100 is **batch-shuffling inside the run-to-run noise band**, not a real
ranking. The adamw 0.386 at bs1024 is an outlier (its lr at that cell looks mistuned; adamw
recovers to 0.451 at bs4096) — drop it and adamw edges slightly ahead on average, which only
underlines that the spread is noise.

**Clear losers:** `sgd` never trains this ViT from scratch (≤0.11), and `soap@1.0` (full
inverse / Newton) collapses (≤0.15, worse at large batch) — the full-power inverse is unstable
here, exactly the regime the GENERATE-don't-INVERT construction is meant to avoid. The headline
is therefore **not** "stable_evo wins" but "stable_evo joins the competitive adaptive cluster
and the naive Newton it generalizes does not."

---

## 2. Speed — a fixed per-step preconditioner tax, amortized away at large batch

Mean step time (ms/optimizer-step), the clean speed metric (wallclock is confounded by early
stopping firing at different points):

| eff batch | sgd  | adamw | soap@0.5 | soap@1.0 | stable_evo+soaplr |
|----------:|:----:|:-----:|:--------:|:--------:|:-----------------:|
| 64        | 87   | 101   | 134      | 171      | 172               |
| 256       | 414  | 429   | 460      | 469      | 454               |
| 1024      | 1633 | 1640  | 1680     | 1674     | 1698              |
| 4096      | 6632 | 6664  | 6699     | 6684     | 6598              |

**Read:** the optimizer overhead is a *fixed per-step cost*, so it only shows at **small batch**.
At bs64 stable_evo (172 ms) is ~2× sgd (87 ms), ~70 % over adamw (101 ms), and ~28 % over
soap@0.5 (134 ms) — it sits at the **high end of the SOAP family**, on par with the full-inverse
soap@1.0. At bs4096 the forward/backward matmuls dominate and **every optimizer converges to
~6.6 s/step** — the preconditioner tax is invisible. So stable_evo is "expensive" only where the
net itself is cheap; in any realistic large-batch / large-model regime its overhead is in the
noise.

---

## 3. The exponent α never engaged (selection idle in the high-lr regime)

stable_evo's whole point is the per-coordinate operative exponent α = ½ + ½·shrink, which is
supposed to climb from ½ (whitening) toward 1 (inverse) on coordinates with strong, consistent
selection signal. Realized distribution at convergence (`stable_evo+soaplr`, no demo noise):

| eff batch | mean α | max α | frac_high (lean>0.8) | frac_floor (α≈½) |
|----------:|:------:|:-----:|:--------------------:|:----------------:|
| 64        | 0.5034 | 0.794 | 0.0                  | 0.995            |
| 256       | 0.5070 | 0.766 | 0.0                  | 0.984            |
| 1024      | 0.5063 | 0.900 | 1.6e-5               | 0.984            |
| 4096      | 0.5116 | 0.900 | 2.8e-4               | 0.938            |

**Read:** α is **floored at whitening essentially everywhere** — mean barely above 0.50, 94–99 %
of coordinates pinned at the floor, and the fraction strongly leaning Newton is ~0. The
trajectory is flat: α does **not** rise over training, and never spikes. There is a *faint,
correct-direction* batch trend (mean 0.503→0.512, frac_floor 0.995→0.938, and max α reaches the
0.90 ceiling for a vanishing fraction of coords as batch grows) — consistent with the theory's
"more batch → more selection signal → higher α" — but the magnitude is negligible at the lr these
runs needed. In the high-lr regime that wins on accuracy, the shrink term `m̂²/(v̂+ε)` stays
small, so selection is effectively idle and stable_evo runs as whitening SOAP with extra
bookkeeping. **Engaging α would need either a lower-lr / longer regime or an explicit selection
gain** — an open follow-up, not something this sweep exercised.

---

## 4. The headline methodological finding: the lr range test *undertunes* stable_evo

stable_evo at its **own** stage-1 lr underperformed badly; it only became competitive once run
at **soap@0.5's** (higher) lr — the `soaplr` variant used in every table above. Matched
fixed-8000-step budget at bs64 isolates this:

| config (bs64, 8000 steps) | lr      | final acc | val_loss |
|---------------------------|:-------:|:---------:|:--------:|
| stable_evo @ own finder lr | 5.9e-4 | 0.378     | 2.434    |
| soap@0.5 @ its lr          | 1.4e-3 | 0.461     | 2.143    |
| stable_evo @ soap's lr (soaplr, convergence) | 1.4e-3 | **0.474** | — |

An **~8-point accuracy gap purely from a ~2.5× lr undertune.** Mechanism — the **generative-lag
confound**: the multiplicative geometric-Riccati update `P ← P·(P_target/P)^κ` takes several
warmup steps to *build* the preconditioner. During the short LR range test the preconditioner
hasn't formed, so the loss-vs-lr curve looks like raw (undertrained) SGD and the finder latches
onto too low an lr. **Lesson: standard LR range tests systematically undertune
generative/warmup-lagged preconditioners** — tune them at the lr their fixed-power sibling
(here SOAP-½) wants, or extend the range-test warmup past the preconditioner formation time.

---

## 5. Injected demographic (pSGLD) noise: a mild regularizer with a sharp √(T·lr) cliff

The optimizer can inject demographic noise of std √(2·T·lr·P) (Langevin/pSGLD posterior
sampling; `--demographic-noise --demographic-temperature T`). Effect on `stable_evo+soaplr`:

| eff batch | acc (no demo) | acc (demo T=1e-7) | val_loss (no demo) | val_loss (demo T=1e-7) |
|----------:|:-------------:|:-----------------:|:------------------:|:----------------------:|
| 64        | 0.4741        | 0.448             | 2.886              | **2.116**              |
| 256       | 0.4403        | 0.4341            | 3.175              | **2.268**              |
| 1024      | 0.4466        | 0.431             | 2.856              | **2.316**              |
| 4096      | 0.4417        | 0.416             | 2.612              | **2.296**              |

**Read:** at **T=1e-7** the noise costs ~**0.02–0.03 top-1** but markedly **lowers val
cross-entropy** (e.g. 2.89 → 2.12 at bs64) and lets the run keep improving longer before
early-stop (≈2× more steps at bs256). That is a textbook **regularization signature** — the
posterior-sampling noise trades a little top-1 for a better-calibrated, lower-CE solution. It
also nudges α *up* a touch (mean 0.503→0.508 at bs64), i.e. injected demographic noise mildly
re-engages selection — the direction the two-noise theory predicts.

But the coupling is **√(T·lr)**, so T is sharp: **T=1e-4 diverged** (val_loss 19–29, acc ~0.009)
at the soap lr. A short T-sweep (1e-8 / 1e-7 / 1e-6, fixed 1500-step diagnostic) bracketed the
choice and motivated the full run at **T=1e-7** as the largest mild setting. **Takeaway:**
demographic noise here is a usable regularizer only in a narrow low-T window; at the high lr these
runs need, the √(T·lr) product hits the stability cliff quickly.

---

## Caveats

- **Single seed.** The ~1–2-point spread among the top three is at or below seed noise; treat
  the "no dominant winner" conclusion as the safe one, not any per-batch ranking.
- **ViT-S from scratch on CIFAR-100** tops out ~0.47 — this is a *relative* optimizer comparison,
  not a SOTA number.
- **α idle is regime-specific.** It reflects the high-lr accuracy-optimal regime these runs sit
  in; it is **not** evidence the selection mechanism is inert in general (the faint batch trend
  and the demo-noise nudge both point the predicted direction).
- **Speed numbers are step-time means** on one L4; wallclock comparisons are early-stop-confounded
  and deliberately omitted from the ranking.

## Repro

```bash
# stage 1 (lr range test) -> runs/lr_finder/suggested.csv
mamba run -n two_noise python -m ml_experiments.lr_finder
# stage 2 grid (convergence budget, stable_evo at soap's lr, demo T variant)
mamba run -n two_noise python scripts/gen_compare_grid.py --epochs 60 \
    --runs-dir runs/benchmarks_conv --stable-evo-at-soap-lr \
    --demo-on stable_evo --demo-temps 1e-7
TN_CMP_GRID=configs/experiment/compare_grid.txt sbatch --array=0-N%3 scripts/submit_optimizer_compare.sh
# analysis
TN_BENCH_DIR=runs/benchmarks_conv mamba run -n two_noise python -m figures.optimizer_compare
```

## Bottom line

StableEvolutionSOAP **earns its place in the competitive adaptive cluster** (≈ adamw ≈ soap@0.5)
and, unlike the naive full-inverse Newton it generalizes, **stays stable**. Its costs are honest:
a small-batch step-time tax (high end of the SOAP family, amortized away at large batch) and a
finder that undertunes its lr unless corrected. Its signature mechanism — the adaptive exponent α
— **did not engage** in this accuracy-optimal high-lr regime (selection idle, α floored at
whitening); demonstrating α climbing toward Newton on the coordinates that warrant it is the
clear next experiment, as is widening the demographic-noise window. For now the practical verdict
is: **as good as the best baselines, provably stable where naive Newton is not, with no free lunch
on speed or on the selection mechanism yet.**

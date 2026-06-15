# Guide: using `StableEvolutionSOAP` for the PhenoVision pretraining-start experiments

**Audience:** agents running the PhenoVision "different starting points" experiment —
fine-tuning the same ViT-L/16 from three pretrained initializations (ImageNet-pretrained,
PlantCLEF-pretrained, tokenizer-only-pretrained) and comparing how they converge.

**What you're being handed:** a drop-in PyTorch optimizer (`StableEvolutionSOAP`) plus an
optional, **tuning-free** "demographic-noise" mode. This guide tells you exactly how to wire it
into the PhenoVision finetune loop, what settings to use, what to log, and how to read the
results. You do **not** need to understand the derivation to run it correctly — but §1 gives the
one-paragraph "why" so the experiment has a hypothesis.

> Provenance: the optimizer and the demographic-noise mechanism were developed and validated in
> the `two_noise` subproject (branch `two_noise-build`). It has been validated **only** on
> ViT-S/CIFAR-100 trained from scratch. Applying it to ViT-L/16 *fine-tuning* from pretrained
> checkpoints on PhenoVision's multi-label BCE task is a new regime — see §8 (Caveats). Treat
> this as a controlled experiment, not a known-good recipe swap.

---

## 1. Why this optimizer, for this experiment (the hypothesis)

The framing is **pretraining-as-preadaptation** crossed with the **two-noise** decomposition of
SGD (see `notes/demo_noise_mechanism.md` and the xAI subproject):

- Each pretrained start is a different "preadapted ancestral state." The question the experiment
  asks is *how the starting basin shapes where fine-tuning lands.*
- SGD carries two noises. **Gradient/sampling noise** self-anneals (∝ loss → 0 near a fit).
  **Demographic/drift noise** is a separate, constant term that a finite population (finite
  batch) always carries. `StableEvolutionSOAP` is a preconditioner built from the
  breeder's-equation / CMA-ES analogy (GENERATE-don't-INVERT; selection sets a per-coordinate
  exponent), and it can optionally inject the **FDT-correct demographic-noise term** that plain
  Adam/SGD drops.
- We validated (CIFAR-100, in `notes/demo_noise_mechanism.md`) that the **tuning-free**
  demographic-noise mode *self-anneals* and lands in the **same basin** as the no-noise run — it
  does **no harm** (matched train/val to baseline, loss-tax → ~0). What it has **not** yet been
  shown to do is **help**. That is exactly what this experiment can test: does adding the
  principled drift term change which basin each *preadapted* start settles into, and does the
  effect depend on how preadapted the start is (e.g. tokenizer-only, least preadapted, may have
  the most exploration to gain; PlantCLEF, most preadapted to plants, may already sit in a good
  basin)?

So the deliverable is a **3 starts × 3 optimizer arms** grid (§7), read through PhenoVision's
real metric (PPV / thresholded val), with the optimizer's internal diagnostics logged.

---

## 2. The optimizer in one paragraph

`StableEvolutionSOAP` is a minimal modification of SOAP (Adam in the eigenbasis of the
Kronecker-factored curvature). Two changes: (1) it **generates** the preconditioner by a
multiplicative recursion toward `(v̂+damping)^(-α)` instead of inverting a noisy curvature, so it
stays bounded; (2) **selection** sets a per-coordinate exponent `α = ½ + ½·shrink ∈ [½, α_max]`,
where `shrink = m̂²/(v̂+eps)` is the gradient signal fraction — clean directions lean toward Newton
(α→1), noisy ones stay at whitening (α=½). The operative exponent therefore **rises with batch
size automatically** (read it back with `mean_exponent()`). Optionally it injects demographic
noise with FDT-correct covariance ∝ P; in the tuning-free mode the temperature is set so the
injected noise trace matches a fraction κ of the minibatch gradient-noise trace, which makes it
self-anneal as the fit improves. No `precond_power`/α to choose — it's dynamic.

---

## 3. Recommended config (the validated, tuning-free arm)

Use these settings for the demographic-noise arm. **Do not** set a temperature by hand — the
`match_grad` mode computes it every step from the gradient statistics.

```python
from optim.stable_evolution_optimizer import StableEvolutionSOAP

EFF_BATCH = batch_size * accum_steps          # the "population size" N_e

optimizer = StableEvolutionSOAP(
    param_groups,                              # same layer-decayed param groups as AdamW
    lr=LR,                                     # see §3.1 — near-whitening lr, tune around AdamW's
    betas=(0.95, 0.95),
    weight_decay=0.05,                         # match your PhenoVision finetune WD
    # --- preconditioner (leave at defaults; these are the validated values) ---
    alpha_max=0.9, alpha_min=0.5, kappa=0.4,
    damping=1e-2, precondition_frequency=10, max_update_norm=1.0,
    # --- tuning-free demographic noise (THE candidate arm) ---
    demographic_noise=True,
    demographic_match_grad=True,               # principled, self-annealing temperature
    demographic_kappa=1.0,                      # N_e == sampling population (1× grad-noise trace)
    demographic_batch=EFF_BATCH,               # the population size
    demographic_warmup=200,                     # inject only after curvature estimate settles
    demographic_generator=torch.Generator().manual_seed(seed + 9973),
)
```

The three arms you actually run (§7) are:
- **Arm A — baseline:** the existing PhenoVision recipe, `torch.optim.AdamW` (unchanged).
- **Arm B — preconditioner only:** `StableEvolutionSOAP` with `demographic_noise=False`.
- **Arm C — + demographic noise:** the block above (`demographic_noise=True,
  demographic_match_grad=True`).

Arm B isolates "is the GENERATE-don't-INVERT preconditioner itself better/worse?" from Arm C's
"does the drift term add anything?".

### 3.1 Learning rate

The realized exponent sits **near whitening** (α≈0.5–0.7), so the lr behaves more like a
preconditioned/SOAP-whitening lr than a raw-SGD lr — **not** the same scale as AdamW on raw
gradients. Start from your AdamW finetune lr and do a short 3-point sweep (×0.3, ×1, ×3) per
start; pick by early val. Keep lr **identical across the three arms** within a start so the arm
comparison is clean. (In the CIFAR validation the stable_evo arm shared SOAP's whitening lr.)

### 3.2 Don't hand-tune the temperature

`demographic_match_grad=True` ignores `demographic_temperature`. The whole point is that there is
**no temperature to tune** — `demographic_kappa=1.0` is the principled default (injected drift
trace = 1× the minibatch gradient-noise trace, i.e. N_e = batch). If you want a sensitivity check,
sweep `demographic_kappa ∈ {0.5, 1, 2}` — higher κ = more drift. Leave `demographic_match_raw`
off for the default (preconditioned-trace match); the raw variant is a secondary check.

---

## 4. Wiring it into `PlantCLEF2022/main_finetune.py`

The finetune script builds its optimizer at **`PlantCLEF2022/main_finetune.py:296–300`**:

```python
param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay, ...)
optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
```

Replace that with an arm switch (add a `--optimizer` / `--demo-noise` arg). The optimizer lives
in the `two_noise` subproject, so put its root on `sys.path` first:

```python
import sys
sys.path.insert(0, "/blue/guralnick/share/r.dinnage/Projects/phenovision/xAI/two_noise")

param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay, ...)

if args.optimizer == "stable_evo":
    from optim.stable_evolution_optimizer import StableEvolutionSOAP
    eff_batch = args.batch_size * args.accum_iter   # × world_size if distributed
    se_kw = dict(lr=args.lr, betas=(0.95, 0.95), weight_decay=args.weight_decay,
                 alpha_max=0.9, alpha_min=0.5, kappa=0.4, damping=1e-2,
                 max_update_norm=1.0, precondition_frequency=10)
    if args.demo_noise:
        se_kw.update(demographic_noise=True, demographic_match_grad=True,
                     demographic_kappa=args.demo_kappa, demographic_batch=eff_batch,
                     demographic_warmup=args.demo_warmup,
                     demographic_generator=torch.Generator().manual_seed(args.seed + 9973))
    optimizer = StableEvolutionSOAP(param_groups, **se_kw)
else:
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
```

The optimizer is a standard `torch.optim.Optimizer` — `zero_grad()` / `step()` / `state_dict()`
work as usual, so the rest of the loop and checkpointing are untouched. **It is not
distributed-aware**: it preconditions on each rank's local gradients. For the pretraining-start
experiment, prefer **single-GPU per run** (one start × one arm = one GPU job; that also respects
the ≤3-GPU group cap). If you must go multi-GPU/DDP, treat that as untested and verify gradients
are all-reduced before `step()`.

### 4.1 CRITICAL: precision / GradScaler

`main_finetune.py` uses `NativeScaler()` — i.e. **fp16 autocast with a loss-scaling
`GradScaler`**. A generative preconditioner that reads gradient *second moments* (and the
grad-noise matching) is distorted by fp16 loss-scaling unless the gradients are **unscaled before
`optimizer.step()`**. This exact failure sank an earlier optimizer (the RiccatiPrecond fp16
GradScaler gotcha — see project memory). **Use bf16 autocast with no GradScaler** for the
stable_evo arms (`torch.autocast(device_type="cuda", dtype=torch.bfloat16)` and call
`optimizer.step()` directly, no scaler). The L4/B200 GPUs all support bf16. If you keep
`NativeScaler`, you **must** confirm `scaler.unscale_(optimizer)` runs before `scaler.step(...)`
every iteration; bf16-no-scaler is simpler and is what the validation used. Keep the **baseline
AdamW arm on its existing precision** so it's a fair "current recipe" control — or, cleaner, run
all three arms in bf16-no-scaler so precision isn't a confound.

---

## 5. What to log (the optimizer exposes diagnostics)

Past the warmup, log these per eval step alongside train/val metrics. They're cheap reads on the
optimizer object:

| Reader | Meaning | What to watch |
|---|---|---|
| `optimizer.mean_exponent()` | size-weighted realized α | sits ~0.5–0.7; **rises with batch**; report per start |
| `optimizer.demo_T()` | demographic temperature used last step (auto-matched) | should **decline** toward 0 as the fit improves (self-annealing) |
| `optimizer.demo_trace()` | total injected noise variance Σvarᵢ last step | the actual amount of drift; → 0 near convergence |
| `optimizer.loss_tax()` | ½·tr(H·Σ) loss-cost rate of the injection | should be ~0 by end; a persistent floor means drift isn't annealing (flag it) |

Healthy tuning-free run (the no-harm signature from CIFAR): `loss_tax` and `demo_T` decay toward
0, final train/val match the no-noise arm, no loss floor. If instead train loss **floors** well
above the baseline and `loss_tax` stays positive, you've reproduced the constant-T drift-load
floor — that's a real (interesting) result, not a bug, but it means the noise isn't self-annealing
in this regime and should be reported as such.

---

## 6. Smoke test before any real run (required)

Per the subproject convention, smoke-test in your env first. This is the exact check that passed
when this guide was written (CPU, a few seconds):

```bash
cd /blue/guralnick/share/r.dinnage/Projects/phenovision/xAI/two_noise
mamba run -n two_noise python - <<'PY'
import torch
from optim.stable_evolution_optimizer import StableEvolutionSOAP
m = torch.nn.Sequential(torch.nn.Linear(32,64), torch.nn.GELU(), torch.nn.Linear(64,10))
opt = StableEvolutionSOAP(m.parameters(), lr=1e-3,
        demographic_noise=True, demographic_match_grad=True,
        demographic_kappa=1.0, demographic_batch=64, demographic_warmup=2,
        demographic_generator=torch.Generator().manual_seed(0))
x, y = torch.randn(64,32), torch.randint(0,10,(64,))
for _ in range(6):
    opt.zero_grad(); loss = torch.nn.functional.cross_entropy(m(x), y)
    loss.backward(); opt.step()
print("ok | loss=%.3f mean_exp=%.4f demo_T=%.2e loss_tax=%.2e"
      % (loss.item(), opt.mean_exponent(), opt.demo_T(), opt.loss_tax()))
PY
```

Expected: prints `ok | ...` with a finite `mean_exp` in [0.5, 0.9] and a small positive
`demo_T`. Then run **one short real PhenoVision finetune** (a few hundred steps, one start) with
each arm to confirm: no NaNs, loss decreasing, `mean_exponent()` finite, and (Arm C) `demo_T`
non-zero after the warmup. The preconditioner does an eigendecomposition every
`precondition_frequency` steps on each weight matrix — ViT-L/16's largest 2-D params are
1024×4096 (< `max_precond_dim=10000`, so fully preconditioned); confirm the per-step time and GPU
memory are acceptable on an L4 before committing to full runs. If memory is tight, lower
`max_precond_dim` (large matrices then fall back to a cheap diagonal preconditioner).

---

## 7. Experiment design (the grid to run)

For **each** of the three starts (ImageNet, PlantCLEF, tokenizer-only), run the **three arms**:

| | Arm A (baseline) | Arm B (precond only) | Arm C (+ demo noise) |
|---|---|---|---|
| Optimizer | AdamW | StableEvolutionSOAP | StableEvolutionSOAP |
| `demographic_noise` | — | False | **True** (`match_grad`, κ=1) |
| Purpose | current recipe control | isolate the preconditioner | the candidate / hypothesis |

Hold **everything else identical within a start**: data, schedule/epochs, weight decay, seed,
precision (prefer bf16-no-scaler for all three, see §4.1), and lr (one tuned lr per start, shared
across arms — §3.1).

**Record per run:** PhenoVision's real validation metric (PPV / thresholded; this is the
discriminator — not just raw val loss), train acc/loss, val acc/loss, and the four optimizer
diagnostics from §5 as time series. Save final weights (`torch.save(model.state_dict(), ...)`) so
the basin question can be probed.

**Reads to produce:**
1. Per start, does Arm C beat / match / lose to Arm A on val PPV? Does the **sign depend on the
   start** (the preadaptation hypothesis — e.g. biggest gain for tokenizer-only)?
2. Does `mean_exponent()` differ across starts (a more-preadapted start may have cleaner
   gradients → higher operative exponent)?
3. Does `demo_T`/`loss_tax` self-anneal (no-harm signature) or floor (drift-load) in this
   fine-tuning regime?
4. *(Optional, high-value)* **Basin probe via LMC.** `ml_experiments/lmc.py` interpolates two
   saved checkpoints and reports whether the path's accuracy collapses (distinct basins) or holds
   (same basin). Valid only when the two endpoints **share initialization and data order** — so
   compare Arm B vs Arm C **within the same start and seed** (they branch only by the injected
   noise). That directly answers "did the demographic noise move this start into a different
   basin?" Do **not** LMC across different starts (different init → naive LMC barriers
   meaninglessly; see the header of `lmc.py`).

GPU discipline: one start×arm = one single-GPU `hpg-turin gpu:l4:1` job; **≤3 concurrent GPUs**
(group fairness cap — see user/project memory). Queue the rest with `--dependency=afterany` or a
throttled array (`%3`). Log to `logs/`.

---

## 8. Caveats — what is and isn't established

- **Validated regime:** ViT-S/CIFAR-100 **from scratch**, softmax CE. **This experiment** is
  ViT-L/16 **fine-tuning** from pretrained, multi-label **BCE**, larger model. The optimizer is
  loss-agnostic (it only sees gradients), so BCE is fine, but the *fine-tuning-from-pretrained*
  and *scale* axes are untested. Watch the first runs closely.
- **No-harm, not yet benefit.** On CIFAR the tuning-free demographic noise matched baseline (it
  didn't hurt) but was not shown to *help*. A null result here (Arm C ≈ Arm A) is a legitimate,
  expected outcome — the interesting find would be a start-dependent gain.
- **Not distributed-aware.** Preconditions local gradients; prefer single-GPU runs (§4).
- **fp16 GradScaler will silently corrupt it** (§4.1). Use bf16-no-scaler.
- **Cost.** Eigendecomposition every `precondition_frequency=10` steps adds compute/memory vs
  AdamW. Budget for it; tune `max_precond_dim` / `precondition_frequency` if needed (changing
  these changes the optimizer, so keep them fixed across arms).
- **Warmup matters.** Inject demographic noise only after `demographic_warmup` steps so the
  curvature estimate has settled; 200 was used on CIFAR — scale to your step budget.

---

## 9. Reference — constructor arguments

`StableEvolutionSOAP(params, ...)`, defaults shown; only the ones you'd touch are flagged.

| Arg | Default | Notes |
|---|---|---|
| `lr` | `3e-3` | **set it** — near-whitening lr, tune around AdamW's (§3.1) |
| `betas` | `(0.95, 0.95)` | keep |
| `weight_decay` | `0.01` | **set** to your PhenoVision finetune WD |
| `alpha_max` | `0.9` | exponent ceiling (margin below the α=1 instability); keep |
| `alpha_min` | `0.5` | whitening floor / stability boundary; **keep at 0.5** |
| `kappa` | `0.4` | generative-recursion step; keep |
| `damping` | `1e-2` | relative LM damping; keep |
| `precondition_frequency` | `10` | eigh refresh cadence; raise to cut cost (keep fixed across arms) |
| `max_precond_dim` | `10000` | bigger matrices → diagonal fallback; lower if OOM |
| `max_update_norm` | `1.0` | per-step trust-region clip; keep |
| `selection_off` | `False` | `True` = recapitulation mode (α pinned to `alpha_max`); **leave False** |
| `demographic_noise` | `False` | **`True`** for Arm C |
| `demographic_match_grad` | `False` | **`True`** for the tuning-free temperature |
| `demographic_kappa` | `1.0` | drift-to-grad-noise trace ratio; sensitivity sweep {0.5,1,2} |
| `demographic_batch` | `0` | **set** to effective batch = `batch_size*accum_steps` |
| `demographic_warmup` | `0` | **set** (~200) — inject only after curvature settles |
| `demographic_match_raw` | `False` | secondary variant (match raw ∝loss trace); leave False |
| `demographic_temperature` | `0.0` | **ignored** when `match_grad=True` — don't set |
| `demographic_generator` | `None` | **set** a seeded `torch.Generator` for reproducibility |

Readers: `mean_exponent()`, `demo_T()`, `demo_trace()`, `loss_tax()` (§5);
`set_loss_scale(s)` is only for the loss-scaled-annealing variant (not used here).

---

## 10. Pointers

- Optimizer: `xAI/two_noise/optim/stable_evolution_optimizer.py` (deps: `optim/demographic_noise.py`)
- Mechanism writeup (why the noise self-anneals; the basin result): `xAI/two_noise/notes/demo_noise_mechanism.md`
- Reference harness usage (CLI flags mirrored in §3): `xAI/two_noise/ml_experiments/benchmarks.py`, `ml_experiments/_harness.py::make_optimizer`
- LMC basin probe: `xAI/two_noise/ml_experiments/lmc.py`
- Conventions (envs, RNG threading, smoke-test rule): `xAI/two_noise/CONVENTIONS.md`
- PhenoVision finetune entry point to patch: `PlantCLEF2022/main_finetune.py:296–300`

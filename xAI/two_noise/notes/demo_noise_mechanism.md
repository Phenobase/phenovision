# Why injected demographic (pSGLD) noise hurts — mechanism investigation

**Question:** the StableEvolutionSOAP optimizer can inject demographic/pSGLD noise (std
`√(2·T·lr·P)`, covariance ∝ the applied preconditioner `P`). In the benchmark it *lowered*
accuracy. Why? Minibatch (gradient) noise doesn't hurt — often helps — so what's different about
the injected noise? Is it the noise *magnitude*, its *shape*, or something else?

Two sweeps on ViT-S/16 / CIFAR-100, fixed lr=2.5e-3, single seed:
- **`tn_mech`** (`runs/benchmarks_mechanism/`): disentangle + SWA + gradient-noise-scale. Batches
  {64,256,1024,4096} × {base, demoT1e-7, demoT1e-6}, 30 epochs, no early-stop.
- **Experiment B** (`runs/benchmarks_shape/`): the noise-SHAPE test. Batches {4096,64} ×
  {base, psgld, fisher, iso, anneal} at T=1e-6, 1500-step floor, with `loss_tax` logging.

---

## Findings

### 1. The penalty is a temperature floor / drift load (not gradient-noise headroom)

Demo penalty (Δacc vs base) is **~batch-independent**: mild T=1e-7 ≈ −0.055, strong T=1e-6 ≈
−0.20 across bs64/256/1024 (bs4096 undertrained). No clean batch trend → the penalty is a
property of the *injected* noise, not of how much gradient noise it competes with.

Mechanistically the demo runs **underfit**: base nearly interpolates the train set (min train
loss ~0.08 at small batch) while the demo runs floor at 1.3 (T1e-7) / 2.2 (T1e-6) and can't
descend further. That floor is the OU/Langevin stationary excess loss `½·tr(D)` (D = injected
diffusion): the optimizer descends until the gradient pull balances the constant noise kicks,
then sits at finite excess loss above the minimum — it samples the posterior `exp(−L/T)` instead
of minimizing. Higher T → higher floor → more underfitting.

**Why minibatch noise doesn't do this:** it *self-anneals*. Near interpolation, every per-example
residual → 0, so the gradient-noise covariance `Σ ≈ (loss)·H` → 0 (Gauss-Newton). The injected
constant-T noise has no `loss` factor, so it stays full strength at the bottom and holds the floor.

**Biological reading:** this *is* drift load. Constant-N demographic noise = genetic drift; near
a fitness peak the directional selection (gradient) vanishes but drift doesn't, so the population
sits at a mutation–selection–drift balance a fixed deficit below the peak (nearly-neutral theory:
coordinates with `|g| < ` noise go effectively neutral). The deficit scales with the noise
temperature (`1/N`). "The model can't fit the training set" = "a finite population can't climb to
its fitness peak."

### 2. SWA / iterate-averaging does not rescue the penalty

SWA lifts *all* configs ~+0.05 (the usual variance-reduction win) but **does not close the demo
gap**, and its gain *shrinks* as T rises (bs64: +0.048 base / +0.042 T1e-7 / +0.026 T1e-6). Averaging
converges to the posterior *mean*, which equals the minimizer only for a *symmetric* (quadratic)
basin. In an anharmonic basin the mean is displaced from the mode toward the soft side, and the
skew grows with T — so the hotter the chain, the less averaging recovers. (Also: pSGLD noise is
loudest in the flat/low-curvature directions, which are the most anharmonic, so averaging is
especially poorly suited.) The principled fix is annealing `T→0`, not averaging.

### 3. The calibration twist

Demo runs have *lower* validation cross-entropy than base despite *lower* top-1 (e.g. bs64 base
val-loss 3.14 at acc 0.49 vs T1e-7 val-loss 2.15 at acc 0.44). The noise regularizes toward a
better-calibrated (less overconfident) solution at a top-1 cost — a real accuracy↔calibration
trade-off with a sweet spot around the mildest T. (This is calibration, **not** better
generalization in the accuracy sense.)

### 4. The gradient-noise-scale probe shows the signal collapse

McCandlish `B_simple = tr(Σ)/|g|²` (logged from K independent micro-batch grads) shows the
injected noise drives the optimizer into low-coherent-signal regions: as T rises, the measured
true-gradient norm |g|² collapses while the per-example variance tr(Σ) stays ~flat, so B blows up.
Caveat: at a fixed *step* the base-vs-demo B comparison is convergence-stage-confounded (base is
"further along"); the trustworthy comparison is at matched train loss, where B is comparable.

### 5. Experiment B — the SHAPE verdict (inverts the naive hypothesis)

The injected-noise covariance was re-shaped at **matched trace** (so only the shape varies):
pSGLD (∝ P ≈ A^{−½}, noise in *flat* directions), iso (∝ 1), Fisher (∝ v̂ ≈ A, noise in *steep*
directions — the shape minibatch noise has). Final bs64 (base interpolates, decisive):

| shape | min train loss | val acc | SWA |
|---|--:|--:|--:|
| base | 0.079 | 0.492 | 0.540 |
| **pSGLD** (flat) | **2.223** | 0.271 | 0.297 |
| iso | 2.417 | 0.268 | 0.287 |
| **Fisher** (steep) | **2.886** | 0.174 | 0.195 |

**Underfitting is ordered by curvature-alignment: Fisher (the minibatch-like shape) underfits the
MOST, pSGLD the least, iso between** — and Fisher generalizes worst too (no flat-minima
compensation). So the naive hypothesis ("injected noise hurts because it has the *wrong* shape vs
minibatch noise") is **inverted**: imposed at constant T, the minibatch-*like* (Fisher) shape is
the most harmful. Minibatch noise is benign **despite** its Fisher shape — via self-annealing
(`∝loss`), not because of the shape. Noise in steep directions costs the most loss (`½·tr(H·Σ)`),
so curvature-aligned constant-T noise is the worst for fitting.

### 6. The `√shrink` "anneal" arm is confounded — no clean annealing test (yet)

The `anneal` arm scaled noise by `√shrink` (shrink = m̂²/v̂ = gradient signal fraction). It looked
like it "broke the floor" at bs64 (→base) but **only because it under-injects**: its `loss_tax` is
uniformly tiny (~36× below pSGLD at bs64, ~10× at bs4096) and *flat over training*, not
high-then-declining. `shrink` is already floored near 0 in this high-lr regime (that's why α stayed
at ½), and lower at small batch — so `√shrink` just barely injects, batch-dependently. It does
**not** test whether genuine annealing relieves the drift load. → needs the loss-scaled variant.

---

## Theory threads (with the user)

- **`Σ ≈ (loss)·H`** is the Gauss-Newton/interpolation cartoon; it unifies the self-annealing
  (the `loss` factor → 0 at interpolation) and the curvature shape (the `H` factor). NOT in the
  Mandt–Hoffman–Blei constant-noise SDE, which linearizes around a fixed nonzero-loss minimum —
  the `loss` factor is specific to the overparameterized/interpolation regime, and annealing is
  *gradual/proportional*, never reaching exactly zero.
- **`tr(D)`**: for a quadratic, trace-matched shapes have *identical* equilibrium excess loss
  `½·tr(D)` — so any shape effect is non-equilibrium / anharmonic / generalization. **Caveat: not
  cleanly tested here** — the finite-budget bs64 separation (Fisher slowest) is the transient
  descent-rate effect; the bs4096 "tie" was all-heavily-underfit. Equilibrium-floor-trace-only
  remains a theory claim.
- **Two-floor structure:** both noises floor, symmetrically — selection/gradient noise floors at
  `Σ_w(z*)/S` (the compromise/segregation load over sample size; zero only if fitness components
  are concordant), demographic noise floors at `1/(2K)` (carrying-capacity drift barrier). Each is
  an inverse-sample-size term reducible but not eliminable.
- **Eco-evolutionary annealing = Wright's shifting balance.** Let N depend on mean fitness
  (logistic, cap K): demographic noise self-cools as the population adapts (phase 1 drift at
  low-fitness/small-N → exploration; phase 2 selection at high-fitness/large-N → exploitation) but
  bottoms out at the `1/(2K)` drift barrier (Lynch), and runs away (meltdown) at low fitness — an
  *adaptive thermostat*. SDE cousins: Wright–Fisher/Kimura (base), Barton–Rouhani (peak-shift /
  barrier crossing), Lande–Engen–Sæther (coupled demography-evolution), and **Nosé–Hoover / SGNHT**
  (the thermostat-variable ML analog — the closest recipe if we ever build the `T(loss)` optimizer).

---

## Next step: the `√(train_loss)` annealing variant (v2)

The clean test of "does annealing relieve the drift load" — scale the injected variance by the
*current training loss* (the true `∝loss` self-annealing law), not by the already-floored `shrink`.
Implemented as `demographic_anneal_loss`: the harness feeds the optimizer a normalized loss scale
each step (`loss/loss_ref`, clamped ≤1), and the injection scales variance by it. Unlike `√shrink`,
its `loss_tax` should be *high early and decline over training* (the signature of genuine
annealing). If `annealL` fits close to base **and** its tax declines (not uniformly tiny), annealing
is the lever; if it still floors, the drift load is irreducible at constant trace. Run: `annealL`
arm in `scripts/gen_shape_grid.py`.

---

# Part II — convergence runs, a tuning-free cure, and the basin verdict

Everything above used finite-budget runs. Part II runs to **true convergence** (train-loss
plateau stop), adds a **train-accuracy** probe (separates *interpolation* from *underfitting* —
the per-step `train_loss` alone can't), and settles the basin question with **linear mode
connectivity (LMC)**. Testbed unchanged: ViT-S/16 / CIFAR-100, `stable_evo`, lr 2.5e-3, bs64,
single seed. Reference points: base val-acc **0.50** (train interpolates, train_acc ~0.98).

### 7. A tuning-free cure: set the injected temperature to the gradient-noise scale

The drift-load floor comes from a *constant* T that doesn't anneal. The principled fix needs no
per-problem tuning: **match the injected noise trace to the minibatch-gradient-noise trace**,
`T = κ·lr·tr(P²v̂)/(2·B·tr(P))` with κ=1 and N_e = effective batch (biologically: the drift
temperature equals the sampling temperature). Because `v̂→0` near interpolation, the matched T
**self-anneals to zero** — no floor, by construction. Two variants:
- `demographic_match_grad` (preconditioned, ∝`tr(P²v̂)`),
- `demographic_match_raw` (raw, ∝`tr(v̂)` — the directly self-annealing form).

Both **reach base** at convergence (val **0.50** / **0.496**), train interpolates (train_acc
~0.97), and the noise cost stays negligible the whole way (`loss_tax` ~3e-8 / ~5e-14, flat). So
tying the drift temperature to the sampling-noise scale removes the floor *without* a schedule —
the self-annealing is automatic. This is the usable optimizer recipe the whole investigation was
after. (Biological reading: let N_e track the sampling effort, and drift cools itself as the
population climbs — no externally-imposed cooling schedule.)

### 8. Train accuracy: the floor is an *inability to memorize*, not just high loss

Adding a clean (un-augmented) train-set eval (`--train-eval`) sharpens what the constant-T floor
*is*. At the T=1e-6 floor the model fits only **~28%** of the training set (train_acc 0.28) while
base is at **~0.95–0.98**. So the drift load isn't merely "loss sits high" — the finite population
*cannot memorize the training data*. This is the ML face of "a finite population can't climb to
its fitness peak": directional selection (gradient) is overwhelmed by drift on most coordinates.

### 9. Two-phase escapability: the floor is escapable, but cooling abruptly lands a *worse basin*

Two-phase test (`--two-phase-noise-off`): run constant-T to its plateau (floor train_loss 2.224,
val 0.28, train_acc 0.28), then **turn the noise OFF** and continue to convergence.
- **Fitting fully recovers:** train_loss 2.224 → ~0.05, train_acc 0.28 → ~0.95 (interpolates).
  So the floor is *escapable* — it was a noise-held equilibrium, not a hard trap.
- **But generalization lands short:** val-acc settles at **0.42**, ~8 points below base's 0.50
  (and below the gradually-annealed matchgrad's 0.50). val_loss *rises* as it interpolates
  (overconfidence), exactly like base — but in a worse-generalizing solution.

So **switch-off recovers the *fit* but not the *basin*.** Sustained high-T diffusion in a
high-loss shell delocalizes the iterate where the noise amplitude exceeds basin-depth differences;
when you cool, it drops into whichever basin it's nearest — generically worse than the one the
clean trajectory funnels into. **Gradual annealing (matchgrad) never leaves that good trajectory,
so it keeps both the fit and the basin.** Argument for *gradual* over *switch-off*: a hot phase
costs you the basin, and late cooling can't buy it back.

### 10. The basin verdict — linear mode connectivity (LMC)

The decisive weight-space test. Interpolate `θ(λ)=(1−λ)·A+λ·B` and watch the loss along the path
(`ml_experiments/lmc.py`); a barrier ⇒ distinct basins. All four endpoints converged + checkpointed
(`--save-weights`); the three demo runs share **seed 0** (identical init *and* data order), so a
base-vs-{matchgrad,cooled} comparison isolates the *noise regime* — only the control uses a
different seed.

| comparison | shares base init+data | train-loss barrier | midpoint train_acc | verdict |
|---|---|--:|--:|---|
| control (base₀–base₁) | no (diff seed) | **+6.25** | 0.034 | collapses (generic permutation barrier) |
| cooled (base₀–cooled₀) | yes | **+7.09** | 0.014 | **distinct, worse basin** |
| matchgrad (base₀–matchgrad₀) | yes | **+0.17** | **0.934** | **same basin (linearly connected)** |

**matchgrad is linearly mode-connected to base** — the path holds train_acc 0.92–0.98 throughout
(val even *rises* to 0.52 mid-path: one wide connected basin). **cooled barriers hard and
collapses** (train_acc → 1.4% at the midpoint) despite sharing base's exact init+data — only the
noise differs. So:
- **Gradual annealing keeps you in base's basin; a sustained hot phase strands you in a different,
  worse one — irreversibly by late cooling.** This is the biological "shifting balance done wrong":
  drift must be *transient and released near a good peak*, not held high.
- **Caveat — naive LMC is a blunt instrument, but self-validating here.** Independent (different-
  init) nets barrier generically via permutation symmetry — the control's +6.25 confirms this, so
  a barrier *alone* isn't decisive. But matchgrad (same init, gentle noise) coming back **flat**
  proves the probe *detects* genuine connectivity when present; therefore cooled's barrier (same
  init!) is a real basin change, not an artifact. (`lmc.py`'s built-in verdict mislabels matchgrad
  "distinct" because +0.17 just edges its 0.15 threshold — but +0.17 with zero accuracy collapse is
  *connected*; the threshold is too tight.) For an init-independent confirmation, the rigorous
  follow-up is a shared-prefix spawn (train base to a stability point, branch clean-vs-noise from
  there) and/or permutation-aligned (Git Re-Basin) LMC.

### Summary of Part II

`T = κ·T_grad` (self-tuned, κ=1, N_e=batch) is a **tuning-free, schedule-free cure** for the drift
load: it self-anneals to base in both preconditioned and raw forms (no floor). The constant-T floor
is a genuine memorization failure (~28% train fit), escapable by cooling — but **abrupt** cooling
recovers the *fit* while landing in a distinct, ~8-pt-worse basin, whereas **gradual** annealing
stays in base's basin (LMC-flat) and matches base on every axis. *How* you cool determines the
basin, not just whether you cool.

### Bug fixed during these runs

`_harness.py` two-phase logic had an unconditional `break` (`:867`) that exited the loop the
instant it flipped the noise off, so phase-2 settling never ran (the run ended at the floor). Fixed
to fall through; the escapability result above is from the corrected code.

### Tooling added (all tested)

`stable_evolution_optimizer.py`: `demographic_match_grad` / `demographic_match_raw` (+ `_kappa`,
`_batch`), `demo_trace()` / `demo_T()` readers. `_harness.py`: `two_phase_noise_off`, train-loss
plateau stop, `train_eval_loader` (clean train-acc), `make_train_eval_loader()`, the `:867` fix.
`benchmarks.py`: the matching CLI flags, `--train-eval`, `--save-weights`, `train_metric` columns.
New: `ml_experiments/lmc.py` (linear mode connectivity). Runs: `runs/benchmarks_lmc/` (checkpoints),
`runs/lmc/` (the three LMC CSVs).

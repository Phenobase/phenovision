# Two-Noise Phase 2 — Execution Report

**To:** the author of the SDE-validation plan / Riccati-optimizer doc.
**From:** the team that executed the plan against the real repo + GPUs.
**Assumes:** you know the theory you wrote (the three timescales; `ℒ_eff(M)=κ·tr(A·Ĝ(M))−ℬ(M)`;
the canalization / tracking / exploration regimes; `M→A⁻¹`; the Riccati `G←G+η_p(M−GCG)`
construction). It assumes you have **not** seen any of the empirical execution, the Phase-1
results, or the bugs we hit. This fills you in.

> **One-paragraph version.** We built and ran the whole optimizer arm. The headline biological
> idea — that `M` evolving to `A⁻¹` is what an ML optimizer should do — turns out to be the wrong
> objective *for an optimizer*, and we can now say precisely why. Separately, the matrix-free
> Riccati had three real bugs (all fixed), `α=1` is fundamentally unstable on ViTs (reproduced
> cleanly), `α=½` whitening is the stable/performant point, and SOAP (exact eigendecomposition)
> beats the matrix-free path at ViT-S scale. We also derived a clean, testable result about the
> two noise sources. Details below.

---

## 1. What was built (infrastructure, all tested)

- **`SOAPFullPower`** (the eigendecomposition optimizer from Phase 1) gained: a `shrink` knob
  (Ledoit-Wolf toward isotropy = effective-exponent reduction), an `evolve_m` meta-loop (the SOAP
  analog of your `evolve_M`), `m_noise` (exploration noise on M), `m_load`/`m_nstar` (mutation-load
  cost). `relative_damping=True` is now the default. ~tests green.
- **`RiccatiPrecond`** (the matrix-free port of your `riccati_precond.py`): ported, fixed (see §3),
  now converges on real ViTs.
- **Experiments:** O2 (schedule-vs-true-Fisher), O4 (evolve-M), a convergence diagnostic, and a
  CPU positive-control toy (`toy/evolve_m_climb.py`). The operative-exponent probes
  (`curvature/operative_exponent.py`) measure the realized `α` from either the optimizer's factors
  or the true Hessian (Lanczos).

The plan's overall structure survived; what changed is the *interpretation* of the headline.

---

## 2. Optimizer-side empirical results (ViT-S / CIFAR-100, SOAP, fixed 4000 steps)

**`α=½` (whitening) wins; `α=1` (full inverse) fails — at both batch 64 and 256.**

| condition | batch 64 (val acc) | batch 256 (val acc) | operative exp |
|---|---|---|---|
| whiten (α=½) | **0.444** | **0.463** | 0.50 (measured exactly) |
| adamw | 0.294 | 0.424 | — |
| inverse (α=1, naive) | 0.02–0.03 | 0.035 | 0.60–0.64 |
| inverse + true-Fisher (2×) | — | 0.087 | — |
| inverse + shrink schedule (1×) | — | 0.025 | — |

Two consequences for the plan:

- **The O2 headline ("the cheap shrink schedule recovers most of true-Fisher's α=1 gain at 1×
  cost") does not hold on this benchmark** — there is *no α=1 gain to recover*. True-Fisher does
  not rescue the inverse, and shrink does not either; whitening dominates at every batch we tried.
  This is consistent with Phase 1's finding that full-power α=1 is fundamentally unstable on ViTs.
- **The gap does not visibly shrink with batch** in these fixed-step runs. NB the *old*
  `alpha_vs_batch` data (which appears to show α=1 improving relative to α=½ at small batch) used a
  fixed *compute budget*, so larger batches got far fewer steps (b16=8000 → b4096=31) — it's
  step-confounded and cannot test the α*(batch) prediction. **A clean fixed-steps, lr-tuned
  α-sweep across batch is still missing** and is the experiment needed to demonstrate the ML half
  of the shared α*(N*) prediction. (The biology IBM and the analytic SDE already show it; the ML
  side was never cleanly shown.)

**Cost:** in SOAP the exponent is free — whiten ≈ inverse ≈ evolve ≈ 210 ms/step (b64) — because
the eigendecomposition dominates and the meta-loop adds only ~3%. adamw is cheaper (~134). This
matters for the matrix-free comparison (§3).

---

## 3. The matrix-free Riccati had three real bugs (all fixed) — and what they taught us

Your `riccati_precond.py` was numpy-validated on small well-conditioned matrices. On real ViT
factors (dim ~384, condition ~1e4–1e6, *small-scale* eigenvalues ~1e-4–1e-2) it silently failed.
Three independent bugs, each with a clean diagnosis:

1. **Whiten over-normalized.** `_ns_inv_sqrt` normalized the Higham iteration by `trace(C)`, which
   is ~`d×` looser than `λ_max` in high dim → the iterates start with all eigenvalues ≪1 → 10
   steps leave `G≈I`. Symptom: `‖GCG−I‖≈1`, operative exponent ≈0, whiten never whitens. Fix:
   normalize by the spectral norm + early-stop with a higher iteration cap.
2. **Whiten needed more iterations.** Even spectral-normed, real conditioning needs ~20–40 Higham
   steps (10 is far too few). Fix: cap 60 + early-stop.
3. **Inverse's absolute damping swamped the curvature.** This was the subtle one. Real ViT factors
   are small-scale; an *absolute* damping of 1e-2 swamps eigenvalues that are themselves ~1e-4 →
   `(C+εI)⁻¹ ≈ ε⁻¹I` → operative exponent ≈0 (behaves like SGD), *not* an NS-convergence failure.
   Fix: **relative damping** (`ε·λ_max`), now the default everywhere. (SOAP already used relative
   damping — which is *why* SOAP's inverse partially worked while the matrix-free one didn't.)

After the fixes, the matrix-free path **does converge on real ViT** (verified): whiten → operative
exponent 0.50, `NSres≈1e-6`; inverse → operative exponent **0.98**, `NSres≈1e-2`. So the port is
now correct.

**But two findings temper the matrix-free story:**

- **It reproduces the α=1 instability cleanly and beautifully.** The verified inverse run reaches
  operative exponent 0.98 at step ~83, then **diverges to NaN by step 166.** So once `G` actually
  *is* `C⁻¹`, training blows up — the gradient-noise term `aᵢ^{1−2α}` amplifies the flat
  directions. There is **no stable operating point for the naive full inverse on a noisy ViT**:
  heavy damping → exponent 0 (stable but it's SGD); light damping → exponent 1 (the true inverse,
  but it diverges). You can have stability XOR the inverse, not both. This is the single cleanest
  confirmation of the Part-III α=½-stability-boundary argument.
- **Matrix-free is slower than SOAP at this scale.** At the inner-step / refresh settings needed to
  *actually* converge (inner_steps≥10, refresh every step), the matrix-free inverse runs 663
  (k=10) to 1162 (k=20) ms/step vs SOAP's 356. The whole point of matrix-free (avoiding the O(d³)
  eigendecomposition) doesn't pay off at ViT-S (d≈384) — the eigh is cheap. Report matrix-free as
  the **large-d** option, not a universal speedup.

---

## 4. The central result: `evolve_M` does not do for an optimizer what it does in biology

This is the part you most need to see, because it revises the headline.

**Setup.** We ported `evolve_M` to SOAP. In the curvature eigenbasis, the matrix source `M`
collapses to a **diagonal** — per-axis gains `m_L, m_R` — so the applied preconditioner is
`h_ij = √(m_L,i·m_R,j)·v_ij^{−½}` (whitening × a learned per-direction gain). `m∝1/v` ⇔ `M∝A⁻¹` ⇔
operative exponent 1; `m∝v` ⇔ `M∝A` ⇔ exponent 0; `m=const` ⇔ exponent ½. The biological
exploration prediction (`M→A⁻¹`) corresponds to the optimizer climbing to exponent 1.

**On real ViT, evolve-M sat at ≈½** (matching whiten exactly, lr-for-lr, at zero extra cost). The
question was whether it *correctly declined* the unstable inverse or was simply *inert*.

**A CPU positive-control toy settled it, and the answer is layered:**

1. **It is not inert.** evolve-M evolves `M` *hard* (anisotropy → 10¹⁰).
2. **The two-phase dynamics you'd expect are real.** The operative exponent stays ≈½ during
   descent (`M≈A`-aligned) and **climbs toward 1 at the noisy equilibrium** — i.e. once the
   loss-improvement signal vanishes, the productive-move signal favours the flat directions and
   `M` heads toward `A⁻¹`. Importantly this climb is driven by the **minibatch (batch) noise**, not
   by any injected noise — the single-trajectory analog of "C is estimable because batch sampling
   injects variation each step." So the diversity concern (whitening flattens the per-direction
   update energy: we measured `cv(whitened)=1.27` vs `cv(raw)=7.11`) is real at the *whitening
   fixed point* but the *batch noise at equilibrium* supplies enough variation to drive the climb.
3. **Without a cost term it runs away** — exponent → 2.4+, anisotropy → 10¹⁰, loss diverges. Your
   `ℒ_eff` has a mutation-load cost `κ·tr(A·Ĝ)`; our first implementation had only the benefit.
4. **We added the cost in a log-space, SPD-safe update** (`ℓ=log m`; benefit, cost, and noise all
   additive on `ℓ`; `m=exp ℓ` ⇒ positive by construction — the diagonal specialization of the
   matrix-log parameterization; the IBM's Cholesky encoding is the full-matrix analog). The cost
   **bounds the runaway** (κ tuned: exponent settles instead of diverging).
5. **But the settling exponent does not depend on N\*.** We swept `N*` in `κ(N*)=0.5+1/(4N*)` and
   the late-time operative exponent is *constant* across `N*∈{2,…,1024}` — confirmed both
   analytically and empirically. `κ(N*)` only rescales `M`; it does not change the *shape*
   (exponent). The exploration fixed point is `m∝1/a` (exponent 1) **for any N\***.

**The conclusion (the important one):**

> **The `evolve_M` objective — whether driven by the bare productive-energy signal or by the full
> `ℒ_eff` exploration gradient — targets `M∝A⁻¹` (operative exponent 1), independent of batch. The
> optimizer's optimum is `α*(batch) < 1`. These are different objectives.**

Why: `α*(batch)` is an **optimization-dynamics** quantity — the balance between convergence speed
(favours high α) and stationary noise (favours low α) *at finite effective sample size*. It lives
in the *transient/descent*. `ℒ_eff` is a property of the *stationary distribution* (the M-load
surface), whose exploration optimum is `A⁻¹` regardless of N*. Biology's `M`-evolution optimizes
**evolvability / exploration** — "keep variation for an uncertain future, never stop adapting."
An optimizer wants the opposite — **converge and stop.** So `evolve_M` faithfully transfers the
*biological* objective, which is the *wrong* objective for an optimizer. Its correct home in ML is
**posterior sampling / the equilibrium regime** (your O6), where exponent 1 *is* the goal and the
instability is handled by the FDT-Langevin machinery — not optimization, where it's fatal.

This does **not** touch the project's core shared-prediction result (α* rises with N* across
substrates) — that's about the *optimum*, validated by the SDE and the IBM. It specifically retires
the "evolve-M as a self-tuning optimizer" sub-claim as posed.

---

## 5. A clean standalone result: the two noises are separable except at the Newton limit

In A's eigenbasis with `G=A^{−α}`: demographic noise covariance `∝ G = a^{−α}`; gradient/sampling
noise covariance `∝ GAG = a^{1−2α}`. Their ratio is `a^{1−α}` — **identical only at α=1** (`G=A⁻¹`),
your point exactly. We quantified "how different" via the Gaussian divergences (shape-normalized):

> **symKL(demographic ‖ gradient) = c·Σᵢ[cosh((1−α)·ℓᵢ) − 1]**, `ℓᵢ = log aᵢ − mean log a`;
> near the Newton limit ≈ **K(κ)·(1−α)²** with `K = ½·Σℓᵢ² ∝ d·(ln κ)²`.

(The `(1−α)²` is the leading-order behaviour near α=1, **not exact** — `cosh` is convex, so the true
divergence grows faster than quadratic away from α=1, e.g. ~0.82× the quadratic extrapolation at
α=½, κ=100. `K` is the spectral-anisotropy constant ≈ ½·d·Var(log curvature) = the divergence in the
SGD limit α=0.) At **α=½** the gradient noise becomes
**white** (`GAG=A^{−½}·A·A^{−½}=I` — whitening literally whitens the sampling noise) while
demographic stays **colored** `∝A^{−½}` — their log-shapes are *orthogonal* (cosine 0), and for
realistic conditioning they barely overlap (Bhattacharyya 3.5 at κ=100 → ~3% overlap; 7.5 at
κ=1000 → ~0.05%). So at the stable operating point the two noises are about as different *in
character* as possible.

This is a genuine, testable prediction the in-silico system can verify (dial α, measure the two
fluctuation covariances separately, check the `(1−α)²` collapse) and one nature won't let you run.
It also reframes the instability: α=1 is unstable precisely *because* `a^{1−2α}` blows up in flat
directions — the same point at which the two noises merge. Stability and noise-merger are mutually
exclusive.

---

## 6. The honest verdict on "ML training as in-silico evolution"

- **First-order: robust.** Selection = gradient descent; the G-matrix = the preconditioner; the
  breeder's equation `Δz̄=Gβ` *is* a preconditioned step; the α*(N*) noise-tradeoff transfers. The
  geometry of adaptation (how covariance channels response to selection, canalization, G–A
  alignment, the speed/robustness frontier) is genuinely shared and studyable in silico.
- **Second-order (evolvability): they diverge, and the divergence is principled.** Optimization
  converges; evolution doesn't. `evolve_M` targets `A⁻¹` because that's evolution's logic
  (exploration), which is wrong for an optimizer. The slogan we landed on: *optimization is the
  convergent special case of evolution; the gap between them is evolvability.*
- **The regime problem (raised by the user, and correct):** vanilla minibatch ML is **pure
  gradient noise, zero demographic noise** — "a large population with wildly noisy selection
  estimation," a corner biology rarely occupies (biology is often drift-dominated). So vanilla ML
  is *not* a faithful proxy for the typical evolutionary regime.
- **But it's fixable by construction, and the machinery exists.** The two noises are the *same two
  terms* in both substrates (`∝G`, `∝GAG`) — ML is just missing the drift term and over-weighting
  sampling. Recipe to span the biological regime space: (a) raise batch to lower the gradient-noise
  floor below the target; (b) inject **FDT-correct `∝G`** demographic noise at strength `1/Nₑ`.
  The project already built the FDT-correct injection (and already learned the trap: the naive
  `G^{1/2}` proxy gives covariance `∝G²`, wrong shape — `posterior_sampling.py` / the `langevin_fdt`
  path fixes it). Faithful ML-as-evolution therefore requires **active two-noise engineering**, not
  vanilla training — which validates the skepticism, but the engineering is well-defined.

---

## 7. What this means for the plan, concretely

- **Retire / reframe** the "evolve-M self-tuning optimizer" headline (O4 as posed). Its honest
  forms: (i) **sampling** mechanism (O6, where `A⁻¹` is correct); (ii) a biological-analog
  *demonstration* that `M` evolves toward `A⁻¹` (matching the IBM), with the optimizer story
  resting on fixed-α `α*(batch)`; (iii) a genuinely different self-tuning objective
  (hypergradient/validation-loss meta-learning of the exponent — honest to label as *not* the
  biology analog).
- **O2's recovered-fraction headline does not hold** on ViT-S/CIFAR (no α=1 gain to recover). Keep
  the *small-batch α*→½ result and the α=1-instability reproduction, which are clean.
- **The ML α*(batch) demonstration is still owed** — a fixed-steps, per-α lr-tuned α-sweep across
  batch (b64/256/512/1024). The old `alpha_vs_batch` is budget-confounded and cannot serve.
- **Matrix-free Riccati** is correct now but **slower than SOAP at ViT-S**; position it as the
  large-d option. SOAP is the working vehicle for everything else.
- **Two genuinely new, defensible results** came out of this that weren't in the plan: the
  `(1−α)²` two-noise separability law (§5), and the optimization-vs-evolvability boundary (§4/§6).
  Both are arguably stronger paper material than the self-tuning optimizer.

---

## 8. Open problems / next steps the user wants to keep pursuing

1. **A correct self-tuning-α objective.** The crux: `α*(batch)` is an optimization-dynamics
   quantity, not an M-load quantity, so neither the productive-energy benefit nor `ℒ_eff` finds it.
   Candidate: an online estimator of the convergence-vs-noise tradeoff (per-direction SNR /
   stationary-variance estimate) driving the per-eigendirection exponent — still single-trajectory,
   still SPD-safe via the log-space update we built. The user is *not* giving up on this.
2. **Validate the two-noise injection recipe** — inject FDT-correct `∝G` drift at chosen `1/Nₑ`,
   confirm the realized fluctuation covariance matches the biological two-noise target, and that the
   FDT path beats the `G²` proxy. This is the load-bearing claim under "ML *can be made* an
   evolutionary proxy."
3. **The `(1−α)²` two-noise figure + writeup** as a standalone result.
4. **The clean ML `α*(batch)` sweep** (item in §7).

## Artifacts (for reference)
- Optimizers: `optim/soap_full_power.py` (shrink, evolve_m, m_noise, m_load, m_nstar, log-space M),
  `optim/riccati_precond.py` (3 fixes + relative damping).
- Experiments: `ml_experiments/riccati_schedule_vs_fisher.py` (O2, `--optimizer {riccati,soap}`),
  `ml_experiments/riccati_evolve_m.py` (O4, `--optimizer soap` supported),
  `ml_experiments/riccati_precond_convergence.py` (diagnostic), `toy/evolve_m_climb.py` (the
  positive-control toy: two-phase dynamics, cost term, N* sweep).
- Diagnostics: `curvature/operative_exponent.py` (`operative_exponent_soap` added).
- Conventions/decisions recorded in `xAI/two_noise/CONVENTIONS.md` (§3: shrink, evolve-M-in-SOAP,
  matrix-free vs exact, relative damping).

# A dynamically-stable, evolution-like optimizer

*Companion to `riccati_optimizer_and_plan.md` and the Phase-2 reports. Answers the
design question directly: build an optimizer that works the way biological evolution
does — generating its preconditioner from selected variation rather than inverting a
noisy curvature estimate — and that is therefore dynamically stable (bounded, SPD,
convergent) where a fixed full-inverse preconditioner is not. Deliverables:
`stable_evolution_optimizer.py` (PyTorch, a minimal diff of `soap_full_power.py`),
`stable_evolution_reference.py` (verified NumPy reference + experiments).*

---

## 1. The stability principle: GENERATE, don't INVERT

The Phase-2 arms agree on a sharp fact. The fast (z̄) and medium (G) timescales of the
two-noise system validate cleanly on both substrates; the place the optimizer breaks is
the preconditioner exponent. At α = 1 (full inverse / Newton in the eigenbasis), the
ViT run NaNs and the matrix-free Riccati reproduces the blow-up — once G actually equals
C⁻¹, the gradient-noise term, which scales as aᵢ^(1−2α) per direction, amplifies the
flat directions without bound. Whitening (α = ½) is the stable boundary.

Biology never hits this, and neither does CMA-ES. The evolution arm's Sim A reaches
G → A^(−0.84) **stably**, and Sim C shows CMA-ES reaching C → A⁻¹ **stably**. The reason
is structural and is the whole design: **they GENERATE covariance by accumulating
selected variation; they never INVERT a noisy curvature matrix.** Evolution builds G by
residence-time accumulation of standing variation; CMA-ES builds C by a rank-μ sum of
the steps selection kept. The α = 1 NaN is a pathology of the *inversion* path
specifically — and both SOAP's eigen-whitening and the matrix-free Riccati `G ← G +
η_p(M − GCG)` with a fixed source are inversion-flavoured (they target the inverse and,
pushed to it, inherit its noise sensitivity).

So the optimizer should construct its preconditioner the way evolution does. Two pieces
realize that, plus a third for safety.

### 1a. The generative recursion is *multiplicative* (SPD-preserving)

The natural-gradient view says G lives on the manifold of covariances, where the metric
is multiplicative; the correct generative update is therefore geometric (log-space), not
additive. Per curvature-eigendirection with eigenvalue `c` and target input `M`,

$$P \leftarrow P\cdot\Big(\tfrac{P_{\text{target}}}{P}\Big)^{\kappa},\qquad
P_{\text{target}}=(c+\text{damping})^{-\alpha},\quad \kappa\in(0,\tfrac12],$$

is the diagonal-in-eigenbasis form of the breeder/Riccati flow `Ġ = M − GCG` whose fixed
point is the matrix geometric mean A⁻¹ # M. It stays strictly positive for any noisy
`c`, and it never divides by a small noisy curvature value.

**A correction to the prior session.** The *additive* Riccati `G ← G + η_p(M − GCG)` was
described earlier as robustly bounded under curvature noise. It is not, at a fixed step:
its contraction bound is per-mode, `η_p < 1/√(Mc)`, which the steep directions violate,
and under 50% curvature noise it diverges (re-confirmed numerically). The multiplicative
form has no such per-mode step constraint and is bounded under heavy noise. Reference
result [4a]: under log-normal curvature noise of 30/50/100%, the generated preconditioner
stays near its target with a maximum-ever excursion of 3.1/4.2/8.7, while the direct
inverse 1/c on a flat direction has a heavy upper tail (median 4.0, 99.9th pct 19, max
33). This is the quantitative content of "generate, don't invert."

### 1b. SELECTION sets the per-coordinate exponent (the loss-weighting fix to `evolve_M`)

The `evolve_M` meta-loop already *is* a generative accumulator, which is why it is stable
in shape. Its problem, established in Phase-2, is that it accumulates the **unweighted**
outer product s·sᵀ, so it climbs to M ∝ A⁻¹ (operative exponent 1, batch-independent) —
the evolvability/exploration optimum, and the divergent target for an optimizer. The fix
is the one CMA-ES uses: **weight the accumulation by selection.**

In the SOAP eigenbasis this costs nothing extra, because the gradient *signal fraction*
is already available from the moments:

$$\text{shrink}_i=\frac{\hat m_i^{\,2}}{\hat v_i+\varepsilon}\in[0,1],
\qquad \alpha_i=\tfrac12+\tfrac12\,\text{shrink}_i\in[\tfrac12,\alpha_{\max}].$$

A direction with clean, consistent gradient (shrink → 1) leans toward Newton (fast); a
noise-dominated direction (shrink → 0) stays at whitening (α = ½, the stability
boundary, where update-noise amplification aᵢ^(1−2α) is flat). The signal fraction rises
with batch size — less per-step gradient noise pulls m̂² toward v̂ — so the operative
exponent increases with batch *automatically*, which is the framework's cross-substrate
prediction realized for free. Near a minimum the gradient becomes noise, shrink → 0, and
the optimizer retreats to whitening and converges: it cannot get stuck amplifying
flat-direction noise. This is the loss-weighted `evolve_M`, made per-coordinate.

### 1c. Bounded response

A relative spectral floor (Levenberg–Marquardt damping inside the target) and a global
update-norm clip cap the per-step change — the analog of evolution's bounded
per-generation response Δz̄ = G β. On by default.

---

## 2. The two framings, one mechanism

The same code serves both of Russell's asks, switched by one flag:

- **(i) Working optimizer** (default): selection on, `α_max < 1`. Sits at/near whitening,
  leans to Newton only where signal is clean and bounded below the α = 1 instability.
- **(ii) Biological recapitulation** (`selection_off=True`, `α_max=1.0`): the *unweighted*
  generative accumulation drives the operative exponent toward 1 — preconditioner → A⁻¹ —
  reproducing the evolvability optimum the evolution arm and CMA-ES reach. That this is the
  divergent target for optimization is exactly why selection (framing i) is what makes it
  work.

The slogan the Phase-2 work converged on holds precisely here: **optimization is the
convergent special case of evolution; the gap is evolvability.** Evolution, facing an
ever-renewing fitness gradient (effectively infinite horizon), keeps M growing toward
A⁻¹ to maximize future adaptability. An optimizer has a finite horizon and a fixed target,
so it wants the transient exponent α*(batch) < 1 and then to stop — which selection +
the load-cost contraction deliver.

---

## 3. What the numerics establish (`stable_evolution_reference.py`)

The testbed is stochastic linear regression, the regime where SGD-as-OU is exact, so the
exponent claims are testable against ground truth.

1. **The generative core reaches whitening and is lr-stable.** Forcing M = I, the
   geometric recursion converges to g = c^(−1/2) (the whitening preconditioner) and is
   stable across every learning rate, matching algebraic whitening — but reached
   generatively, without inverting. [Result 1]
2. **α = ½ is the stability boundary.** At a fixed learning rate (cond = 5000, label
   noise 2.0), stationary loss climbs monotonically as the exponent passes ½:
   0.76 → 2.1e3 → 7.6e3 → 8.5e3 → 8.6e3 for α = 0.5, 0.625, 0.75, 0.875, 1.0. This is the
   aᵢ^(1−2α) flat-direction amplification. [Result 2]
3. **The viable-exponent ceiling rises with batch.** lr-tuned, fixed budget: at B = 8 the
   largest exponent that stays competitive is ≈ 0.5 (α ≥ 0.6 degrades sharply); at
   B ≥ 32 the ceiling rises to ≈ 0.7. Small batches force lower exponents — the
   cross-substrate prediction. [Result 3]
4. **The selective optimizer is dynamically stable and convergent where fixed-Newton is
   catastrophic.** Across batches its best-over-lr loss is the same order as whitening and
   never blows up, while fixed α = 1 diverges or sticks at high loss. In this
   noise-dominated testbed the signal fraction stays low, so selection correctly holds the
   exponent near whitening (mean α ≈ 0.51–0.54) — the safe, conservative behavior; it
   leans toward Newton only where the per-step noise is low. [Results 4a, 4b]

---

## 4. Scope ledger

**Established:** the generative (multiplicative) preconditioner is bounded and SPD under
heavy curvature noise where direct inversion is not; α = ½ is the stationary-variance
stability boundary; the viable exponent ceiling rises with batch; the selective optimizer
is dynamically stable and convergent and never catastrophic.

**Modeling choices:** the signal fraction m̂²/v̂ as the selection statistic (the natural,
zero-cost Wiener factor from the moments SOAP already tracks; other estimators — e.g.
m̂²/(v̂ − m̂²) explicitly discounting only the gradient-noise part — are reasonable and
untested); `α_max = 0.9` and `κ = 0.4` as defaults; spectral-floor damping form.

**Open / honest caveats:**
- In the bounded-condition, shape-normalized setting the α = 1 failure is a trapped
  high-stationary-loss state (the monotone climb), not a literal NaN; the NaN requires
  genuinely flat directions (unbounded condition) or a nonlinearity that compounds the
  flat-direction excursions, as in the ViT. The SDE signature (variance ∝ aᵢ^(1−2α)) is
  the common cause.
- The *clean argmin* exponent α*(batch) is flat and within-noise among viable exponents
  in this linear testbed; the robust signal is the viable-exponent **ceiling**. A clean
  α*(batch) curve needs a richer problem (the owed ViT/CIFAR sweep below).
- The selective optimizer matches whitening here rather than beating it because every
  direction is noise-dominated near the optimum; its advantage — exploiting high-SNR
  directions with a Newton lean — needs a problem that *has* such directions (low-rank
  signal, larger batch, lower label noise).

---

## 5. Next steps

1. **Smoke-test `stable_evolution_optimizer.py`** in the PyTorch env (it is a minimal diff
   of the working `soap_full_power.py`; the inner loop mirrors the verified reference).
2. **The owed clean α*(batch) sweep on a real net** (ViT-S/CIFAR-100): fixed steps,
   per-α lr-tuned, batch ∈ {64, 256, 512, 1024}, fixed exponent — to get a clean α*(batch)
   curve where the signal is not uniformly noise-dominated. Then overlay the selective
   optimizer's realized mean exponent.
3. **Cross-substrate figure:** this α*(batch) against the evolution arm's disaster-severity
   sweep (see `notes_landscape_disaster_benefit.md`, tests T1/T6) on one α*(noise) axis.
4. **Recapitulation run** (framing ii): confirm `selection_off=True, α_max=1.0` drives the
   preconditioner to A⁻¹ on a controlled problem, matching the IBM and CMA-ES.

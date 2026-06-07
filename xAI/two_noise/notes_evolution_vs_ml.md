# Evolution vs ML: what transfers, what doesn't, and why C⁻¹ is stable in one and not the other

Synthesis of what the experiments (Sim A, Sim B, Sim C, the α-sweep, §2.4) have taught us about
the correspondence. For the paper's §4 ("the shared prediction" + "what doesn't transfer").

## 1. The instability IS the unification (the headline)

ML full-inverse (α=1) ran **fine at batch 256** but **NaN'd at batch 16**; Sim A reached G∝A⁻¹
(α≈0.84) **stably**; Sim B showed M-anisotropy **compresses toward isotropy (α→½) as N\* falls**.
Together: **as effective sample size drops, both substrates are pushed off the full inverse (α=1)
toward whitening (α=½)** — in ML as *instability* (α=1 explodes → forces damping / lower α), in
biology as *M-anisotropy compression* (Force 2, attenuation bias). Biology reaching α≈1 "stably"
is not a different rule: census populations make the gradient-noise term 1/N\* small, i.e. biology
runs at *large effective batch*, where ML α=1 is also stable. Shrink N\* in the sim and biology
destabilizes off α=1 too (Sim B). One law (the noise-dependent α*), two substrates — the §5 figure.

## 2. Why evolution gets G∝A⁻¹ with no blow-up: GENERATE vs INVERT

**ML computes H≈C⁻¹ by inverting a noisy estimate.** A near-singular flat direction (small
eigenvalue of the *estimated* C, dominated by minibatch estimation noise) → near-infinite
eigenvalue of H → you divide by noise → the α=1 NaN. Hence LM damping `(C+λI)⁻¹`.

**Evolution never inverts.** G∝A⁻¹ *emerges* from a generative equilibrium: weakly-selected
directions (small A) accumulate more standing variance because selection removes variants there
slowly (residence time ∝ 1/s ∝ 1/curvature). Self-limiting accumulation, not explosive
amplification. Three intrinsic stabilizers, each a concrete ML import:

1. **Built-in variance ceiling** — G is bounded by mutation supply vs selection removal
   (mutation–selection–drift balance); flat directions get more variance but still bounded by V_m.
   → **LM damping is the artificial version of biology's mutation-limited ceiling.**
2. **Natural trust region** — the response Δz̄ = G·β is gated by *available variance G* (bounded),
   not by the selection gradient's magnitude. ML's step H·∇L has an unbounded ∇L.
   → **`max_update_norm` (the trust region that fixed the avb α=1 crash) = variance-gated response.**
3. **Heavy averaging → low-noise metric** — G is averaged over a whole population and drifts
   slowly over generations (quasi-static); ML's C is a minibatch snapshot. Inverting a low-noise
   matrix is stable; inverting a noisy one is not. → Predicts (and matches) α=1 failing at batch 16
   but not 256; fix = average C longer (slower preconditioner timescale / larger effective batch).

## 3. The cleanest import: CMA-ES is evolution's *stable* route to C⁻¹

Sim C: CMA-ES reaches C→A⁻¹ stably (cosine >0.9, eig-slope ≈ −1) precisely because it **generates**
the covariance by selection on a population rather than inverting an estimate — the biological
mechanism ported to ML. So "can ML get C⁻¹ stably?" → *yes, the way evolution does*: maintain a
covariance that converges to A⁻¹ through a generative/population update, never an explicit inverse.
The genuinely stable full-inverse optimizer is closer to a population/ES method (or a slowly
accumulated covariance) than to inverting a minibatch Fisher.

## 4. True Fisher = importing biology's low-noise, correctly-specified curvature (CONFIRMED)

The §2.4 α=1 over-dispersion is the micro version: SOAP's *empirical* Fisher mis-estimates
flat-direction curvature → 1/denom too large there → over-amplification. Biology's "Fisher" is
effectively the *true* Fisher (selection acts on the actual fitness curvature, sampled over the
whole population). Importing the true Fisher into the preconditioner *eigenvalues* (not just the
basis; `precond_eigvals_from_hook`) **fixed it**: SOAP-NG (α=1) + demographic + true-Fisher then
recovers the posterior best — shape (cos 0.86, slope 1.11) AND magnitude (total_var 4.91 ≈
posterior 4.98), beating whitening (α=0.5). (notes_posterior_sampling.md.)

**A sharp sub-finding on which stabilizer is FDT-safe.** Evolution's two intrinsic stabilizers are
NOT interchangeable for *sampling*:
- **Variance ceiling (damping)** floors the denominator in BOTH drift and noise, so it cancels in
  the stationary `V=T/a` → FDT/posterior preserved while the Newton step is bounded. **FDT-safe.**
- **Variance-gated trust region (`max_update_norm`)** clips the drift but NOT the injected noise →
  restoring force weakens relative to noise → runaway over-dispersion (total_var ~4e4 vs ~5).
  **Breaks FDT.** Fine for the deterministic optimizer (§2.1, where it fixed the α=1 NaN), wrong
  for the sampler.

So the biology→ML import is more precise than "evolution bounds its steps": *which* bound matters.
The ceiling (mutation-limited variance) is the one that preserves the Gibbs/posterior structure;
the response-gating bound is an optimizer device, not a sampler device. This mirrors evolution:
the mutation-selection variance ceiling is what keeps G at a well-defined equilibrium *distribution*
(not just a bounded point estimate).

## 5. What does NOT transfer (so far)
- **M-evolution's timescale** — biology evolving its *mutational architecture* M on a slow
  second-order signal has no ML analog; Sim A confirmed it's too slow to even observe at accessible
  compute. (The framework's "second-order architecture evolution" remains a conjecture, not a result.)
- **Recombination** resists the clean SDE / measure-valued treatment.
- **The demographic term** — biology has it for free (finite-N transmission); ML must inject it
  (pSGLD), and getting the temperature/shape right is delicate (§2.4).

## 5b. Sim B compression: confounded by lag-load, then RESOLVED with incoherent noise

The spatial-correlation N\* knob (sim_b_phase.py) did NOT show compression: lowering N\* via
correlation makes the per-generation optimum perturbation *coherent* across individuals → a
per-generation shift (lag-load, Force 3) that *enhances* anisotropy and fights the gradient-noise
attenuation (Force 2). Compounded by measuring M, which barely evolves anisotropy (the slow
M-timescale, Sim A) so there is little to compress.

**Resolved** (sim/sim_b_compression.py) with two fixes, both diagnosed correctly:
1. **Incoherent perturbation** — independent per-individual optimum noise δ_i ~ N(0, amp² I)
   (identity kernel in run_sim_perturbed): no coherent shift → no lag-load, isolating attenuation.
2. **Measure G, not M** — G (standing genetic covariance) responds on the standing-variance
   timescale; M needs the slow architectural timescale.

Result (Ne large, A_eigratio 6): G-anisotropy compresses MONOTONICALLY toward isotropy as the
noise rises — 1.51 → 1.44 → 1.32 → 1.17 → 1.06 for amp 0→0.4→0.8→1.5→2.5 — with G–A alignment
flat (<0.65°) until extreme noise. So **Force-2 compression is real and demonstrable**; the
earlier failure was a mechanism/observable confound, not a wrong prediction. (Test: 1 pass.)

**Biological interpretation of the incoherent noise = errors-in-variables = environmental
variance.** Perturbing each individual's optimum independently is identical (fitness depends on
z−θ_i) to independent ENVIRONMENTAL/DEVELOPMENTAL variance in the expressed phenotype (P = G + E):
selection acts on a noisy read of the breeding value → regression dilution → low heritability
(R = h²S; E is exactly what makes h²<1 attenuate the response). It attenuates anisotropically
(stronger in sharp-A directions) → compresses the anisotropy ratio. Magnitude-of-E inflates the
gradient covariance C, which is equivalent to lowering the effective N\* in the gradient-noise term
C/N\*: **σ²_E ↑ ≡ inflate C ≡ lower effective N\* ≡ Force-2 compression.** The spatial-correlation
ρ̄ route is the framework's *headline* (limited-dispersal/niche-construction) version; the
errors-in-variables route is the classical (environmental-variance) version — both drive Force 2.

**Cross-substrate symmetry of errors-in-variables.** Biology: E blurs selection's read on the
genotype → attenuated/compressed response; remedy = higher heritability / larger effective sample.
ML: minibatch/empirical-Fisher noise blurs the *curvature* estimate → the §2.4 α=1 over-dispersion;
remedy = true Fisher / larger batch. "True Fisher fixes ML α=1" and "high heritability / large N\*
keeps biology at α≈1" are the SAME statement about errors-in-variables in the curvature/selection
signal — the cleanest single bridge we have.

**Also robustly shown in Sim B:** flat M–A alignment (0.0° everywhere — the no-eigenvalue-order-
inversion correction) and, in the strong-perturbation regime, Ne→replicate-variance inflation.

## 6. Directional summary of imports
- **biology → ML:** damping = mutation ceiling; trust region = variance-gated response; the stable
  route to C⁻¹ is *generative* (CMA-ES / slow covariance accumulation), not inversion; true-Fisher
  = low-noise curvature.
- **ML → biology:** the noise-dependent α* law; the natural-gradient/Fisher reading of G; CMA-ES
  theory (Akimoto 2012) as the analytic backbone for G→A⁻¹.

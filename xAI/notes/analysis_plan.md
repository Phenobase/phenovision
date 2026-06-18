# Analysis Plan: Two-Noise Evolutionary SDE, the Breeder's Equation–SGD Equivalence, and Full-Power SOAP

## Purpose and scope

This document specifies the analyses, simulations, reference implementations, and paper structure for the flagship cross-disciplinary paper. It is a companion to `two_noise_framework_summary.md` (the theory), `sgd_breeders_equation_derivation.md` (the derivation), and `soap_full_power.py` (the reference optimizer). It is written to be handed to Claude Code: each experiment has a setup, the quantities to vary, the quantities to measure with their estimators, an explicit prediction, a definition-of-done, and a reference implementation to build on.

The paper straddles evolutionary biology and ML with the weight on evolution. arXiv first (`q-bio.PE` cross-listed `cs.LG`), journal decided on results.

**A framing decision baked into this plan.** Do not stake the paper on "full-power SOAP trains faster." Morwani et al. (2024) already compared the inverse square root against the full inverse from the Hessian-approximation angle and found that the exponent change *does not manifest in optimization speed*, even though it changes the curvature-approximation quality. Closer still, **Lin et al. (2024, ICML) explicitly removed the root** from adaptive methods and found the square-root-free variants *close the generalization gap to SGD on convolutional architectures while staying roughly neutral on transformers* — so root removal is done, with public code (§1.1), and its motivation was numerical (root-free methods are stable in half precision), not a noise law. That CNN-vs-transformer split is itself a clue the best exponent may be noise- or architecture-dependent, which is exactly hypothesis (i). The exponent knob itself is also not new — it is the configurable root in Distributed Shampoo, and Eschenhagen et al. (2025) decompose Shampoo precisely along the eigenvalue-exponent axis. The genuinely unclaimed ground, and where the evolutionary story does real work, is three things: (i) a *derivation of the optimal exponent from a noise argument* — that $\alpha^\*$ falls from 1 toward $\tfrac12$ as gradient noise rises; (ii) the *posterior-sampling consequence* of restoring the demographic-noise term SGD lacks; and (iii) the framing of the exponent as the *single dial shared with the biological $G$–$A$ exponent*. Every experiment below is pointed at one of these, not at raw speed.

---

## The organizing spine

One prediction, tested in two substrates. The linearized SDE gives, in both systems, stationary variance $V_{\text{stat}}\propto(\text{curvature})^{-\alpha}$, drift coefficient $C^{1-\alpha}$, and diffusion $C^{1-2\alpha}/S$. The exponent $\alpha$ is `precond_power` in ML and the $G$–$A$ exponent in biology ($G\propto A^{-\alpha}$).

- **Biology:** evolved $M$ anisotropy (hence the realized $G$–$A$ exponent) decreases as $N^\*$ decreases. More gradient noise pushes toward whitening, $\alpha\to\tfrac12$.
- **ML:** the optimal `precond_power` decreases as effective batch size decreases. Same mechanism, same direction.

The deliverable that makes this one phenomenon rather than two stapled analogies is a single figure (§5): $\alpha^\*$-vs-$N^\*$ from the optimizer beside $M$-anisotropy-vs-$N^\*$ from the simulation, with the SDE curve overlaid on both, sharing an "effective sample size" x-axis. Build everything to feed that figure.

---

## 1. Reference implementations and dependencies

### 1.1 Optimizer base and the exponent lineage (Strand 1)

| What | Link | Use |
|---|---|---|
| **Our optimizer** | `/mnt/project/soap_full_power.py` | The `SOAPFullPower` class. `precond_power` is $\alpha$; LM damping, relative damping, update-norm clip, and the optional true-Fisher / demographic-noise hook are already scaffolded. Start here. |
| SOAP (official, PyTorch) | https://github.com/nikhilvyas/SOAP | The `soap.py` we modified. Reference for the two-timescale eigenbasis machinery and default hyperparameters (`lr=3e-3`, `betas=(.95,.95)`, `precondition_frequency=10`). |
| SOAP (JAX, unofficial) | https://github.com/haydn-jones/SOAP_JAX | If the sim stack ends up in JAX and a JAX optimizer is wanted for a unified codebase. |
| **PyTorch Distributed Shampoo (Meta)** | https://github.com/facebookresearch/optimizers | Production-grade; the inverse *root* is a configurable parameter — the canonical place the exponent already lives. Reference for correctness and for the LR-grafting machinery SOAP removes. |
| Morwani et al. 2024, *A New Perspective on Shampoo's Preconditioner* | https://arxiv.org/abs/2406.17748 | **The key exponent citation.** Shows $H^{1/2}$ vs full inverse changes Hessian-approx quality but not speed; studies empirical-Fisher vs sampled-label (true-Fisher) effect on the approximation — directly relevant to our true-Fisher hook. |
| **Lin et al. 2024, *Can We Remove the Square-Root in Adaptive Gradient Methods?* (ICML)** | paper https://arxiv.org/abs/2402.03496 · experiment code https://github.com/yorkerlin/remove-the-square-root | **The closest prior art — they already did the $\alpha:\tfrac12\to1$ move.** Root-free variants close the SGD generalization gap on CNNs, stay neutral on transformers; motivated numerically, not via a noise law. Cite and distinguish explicitly; their CNN/transformer split is a data point hypothesis (i) should explain. |
| Dangel et al., SIRFShampoo — structured inverse-/root-free Shampoo | https://github.com/f-dangel/sirfshampoo | Clean PyTorch reference for the root-free Shampoo inner loop (companion code to Lin et al. 2024). Template for the $\alpha=1$ update and half-precision stability without matrix-root decompositions. |
| Curated index of second-order / preconditioned optimizers | https://github.com/riverstone496/awesome-second-order-optimization | For Claude Code to discover further implementations of the exponent / Kronecker-preconditioner family. |
| Eschenhagen et al. 2025, *Purifying Shampoo* (NeurIPS) | https://arxiv.org/abs/2506.03595 | Decomposes Shampoo's heuristics along the eigenvalue-exponent axis; formalizes the eigenvalue-exponent mismatch and why per-step eigenvalue correction (SOAP) removes grafting. Position our $\alpha$ against their decomposition. |
| Liu et al. 2025, *Understanding SOAP via gradient whitening* | https://arxiv.org/abs/2509.22938 | Confirms vanilla SOAP is whitening ($\alpha=\tfrac12$). The interpretive anchor for "SOAP nails the eigenbasis but uses whitening eigenvalue scaling." |
| Lin et al. / An et al. 2025, *KL-Shampoo* | https://arxiv.org/abs/2509.03378 | KL-minimization view of the preconditioner; ties to the free-fitness / Bayesian thread. |
| *Clarifying Shampoo* (2026) | https://arxiv.org/abs/2602.09314 | Recent (post-cutoff — verify before citing): adapts spectral descent to stochasticity and the parameter trajectory. Likely the closest concurrent work; read for positioning. |
| KrADagrad | https://arxiv.org/abs/2305.19416 | Exponent $\alpha$ handling and inverse-root grouping; minor reference. |

### 1.2 Evolution strategies (Strand 2, Sim C bridge)

| What | Link | Use |
|---|---|---|
| **pycma** (Hansen) | https://github.com/CMA-ES/pycma | Reference CMA-ES. Run on the same quadratic as the QG sim to show $C\to A^{-1}$ by a third mechanism. |
| Akimoto et al. 2012 | (in `bibliography.md`) | Theory: rank-$\mu$ CMA-ES is natural-gradient ascent on the Gaussian manifold; covariance $\to$ inverse Hessian on quadratics. The analytic backing for Sim C. |

### 1.3 Posterior-sampling references (Strand 1, FDT payoff)

| What | Link | Use |
|---|---|---|
| Welling & Teh 2011, SGLD | (in `bibliography.md`) | The baseline noise-injection sampler. |
| Li et al. 2016, *Preconditioned SGLD* (AAAI) | search "preconditioned stochastic gradient Langevin dynamics Li Chen Carlson Carin" | The honest comparator: our demographic-noise injection is a structured pSGLD with a Kronecker preconditioner. Novelty is the biological derivation + exact FDT condition, not "noise injection." |
| Laplace library (`laplace-torch`) | https://github.com/aleximmer/Laplace | Cheap posterior reference for small nets (Laplace/GGN). |
| NumPyro | https://github.com/pyro-ppl/numpyro | HMC/NUTS gold-standard posterior for the toy Bayesian models. |

### 1.4 Curvature estimation (used across Strand 1)

| What | Link | Use |
|---|---|---|
| PyHessian | https://github.com/amirgholami/PyHessian | Top-$k$ Hessian eigenpairs via Lanczos; trace via Hutchinson. For the stationary-variance test and curvature tracking. |
| (true Fisher) | — | Sampled-label Fisher: backprop on labels drawn from the model's output distribution, per Morwani et al. §2.1.2. Feed via the `_soap_precond_grad` hook. |

### 1.5 Evolutionary simulation engine (Strand 2)

No public JAB repo exists; their code is available on request from the authors (cited in the methods of JAB 2003/2007/2014). Plan to reimplement the individual-based model from scratch — recommended in JAX for replicate/individual vectorization (see §6). `slimr` / SLiM is the wrong tool here (population-genomic, not quantitative-genetic IBM with an evolvable mutational architecture). The multilinear epistasis mechanism to reproduce is Hansen & Wagner (2001), as used by JAB 2014.

---

## 2. Strand 1 — Optimizer experiments

### 2.0 The optimizer (preliminary, before any experiment)

**Reference:** `/mnt/project/soap_full_power.py`.

Tasks: (a) unit-test against vanilla `soap.py` — with `precond_power=0.5`, `damping=0`, `relative_damping=False`, `max_update_norm=0`, the update must match SOAP to numerical tolerance on a fixed-seed step. (b) Verify bias correction generalizes correctly for arbitrary exponent. (c) Implement the **demographic-noise hook**: an optional per-step additive Gaussian with covariance $\propto H$ (shaping matrix $\sqrt H$, i.e. add noise in the eigenbasis with per-coordinate scale $\sqrt{V^{-\alpha}}$ matching the applied preconditioner power), temperature scaled by `lr`. This is the missing $\sqrt{G/N_e}\,dW_1$ term; it is off by default and only used in §2.4. (d) Expose `precond_power` as the swept hyperparameter throughout.

**Definition of done:** parity test passes; an `alpha` sweep runs end-to-end on a 2-layer MLP without NaNs at every $\alpha\in\{0,0.25,0.5,0.75,1.0\}$ given relative damping $\sim10^{-2}$ at $\alpha=1$.

### 2.1 Standard benchmarks (necessary, not the headline)

**Objective:** establish that full power is competitive, and characterize the step-efficiency vs wall-clock-efficiency split honestly.

**Setup.** Models: ViT-S/16 and ViT-B/16 (vision), plus a nanoGPT-scale LM (~10–125M params) so the result is not vision-only. Data: CIFAR-100 and Tiny-ImageNet or ImageNet-100 (vision); TinyStories or a C4 subset (LM). Baselines: AdamW, SOAP ($\alpha=0.5$), and the $\alpha$ sweep $\{0,0.25,0.5,0.75,1.0\}$ as a *continuous interpolation* (the continuity is part of the claim, not just two endpoints).

**Measure.** Train loss vs step; train loss vs wall-clock (report both — Newton-like full power makes them diverge); final val accuracy / perplexity; LR sensitivity (a small LR grid per $\alpha$). Record per-step time and memory.

**Prediction.** Competitive final performance across $\alpha$; full power needs ~10× smaller LR plus damping; no dramatic speed win (consistent with Morwani). Frame as "the exponent is not free to choose for speed, which is why the interesting question is what *governs* the choice" — segue to §2.2.

**Definition of done:** clean loss-vs-step and loss-vs-wallclock curves for all $\alpha$ on ≥2 model/data pairs, with LR-matched comparisons.

### 2.2 Centerpiece — the noise-dependent optimal exponent

**Hypothesis (the one to test):**
$$\alpha^\*(\text{effective batch})\ \text{is monotone increasing in batch size (decreasing in gradient noise),}$$
because the diffusion term $C^{1-2\alpha}/S$ over-amplifies low-curvature directions exactly when $\alpha\to1$ and $S$ is small. This is the ML image of the biological $N^\*$ prediction.

**Toy first (exact regime).** Anisotropic quadratic $L(\theta)=\tfrac12\theta^\top A\theta$ with known $A$ (controlled eigenvalue ratio, e.g. condition numbers $10^1$–$10^4$). Inject gradient noise with covariance $C$ at controllable magnitude $1/S$ (and optionally controllable *correlation* to decouple "batch size" from $N^\*$). Sweep $\alpha\in[0,1]$ (fine grid) × noise level. For each noise level, locate $\alpha^\*$ minimizing stationary expected loss (or time-to-target). **Validate the SDE prediction analytically here before touching ViTs.**

**Real models.** Cross the $\alpha$ sweep with batch size $\in\{16,64,256,1024,4096\}$ (use gradient accumulation / micro-batching to hold the optimizer's other state comparable). For each batch size fit $\alpha^\*$ = the $\alpha$ giving best val loss at a fixed compute budget. Plot $\alpha^\*$ vs batch size.

**Prediction.** Downward-sloping $\alpha^\*$-vs-noise (equivalently upward-sloping $\alpha^\*$-vs-batch). If flat, the flat-direction-noise term is not the binding constraint in practice — still publishable, and it sharpens the theory.

**Definition of done:** $\alpha^\*$ located for ≥5 noise/batch levels in the toy setting (with the analytic SDE curve overlaid) and ≥4 in at least one real model. This produces the left half of the §5 figure.

### 2.3 The stationary-variance / FDT test (the SDE made empirical)

**Objective:** confirm the SDE actually describes the optimizer by measuring its stationary fluctuation covariance and testing $V_{\text{stat}}\propto C^{-\alpha}$.

**Setup.** Near a minimum — fine-tune a converged model, or the toy quadratic where it is exact. Small fixed LR, no schedule, weight decay off (or matched). Run past a burn-in until the loss plateaus into its stationary fluctuation band.

**Measure.** Collect parameter snapshots $\{\theta_t\}$ after burn-in. Estimate the top-$k$ curvature eigenpairs of $A$ (Hessian or true Fisher via PyHessian/Lanczos; $k\sim20$–$50$). Project the centered snapshots onto those eigenvectors and form the empirical covariance $\hat V_{\text{stat}}$ of the projections. Regress $\log\lambda_i(\hat V_{\text{stat}})$ on $\log\lambda_i(C)$.

**Prediction.** Slope $\approx-\alpha$ across the sweep ($-0.5$ for vanilla SOAP, $-1$ for full power, intermediate for intermediate $\alpha$). A clean slope is direct evidence the stationary law is the power-reshaped Gibbs measure the framework predicts.

**Caveat to design around (state it in the paper).** $C$ drifts and empirical-Fisher $\neq$ true-Fisher in early training, so the law is cleanest near convergence and in *fine-tuning* — which is exactly the PhenoVision regime. Foreshadow the empirical paper here.

**Definition of done:** log-log slope estimates with CIs for ≥3 values of $\alpha$, in toy (exact) and ≥1 fine-tuning setting.

### 2.4 The posterior-sampling demonstration (the FDT payoff)

**Objective:** show the missing demographic term is what makes the stationary distribution the correct posterior — "biology gets the FDT-restoring term for free."

**Setup.** Known/approximable posterior. (a) Bayesian logistic regression (HMC reference via NumPyro). (b) Small MLP (Laplace via `laplace-torch`, HMC where feasible). Four samplers: SGD; SOAP $\alpha=0.5$; SOAP-NG $\alpha=1.0$; **SOAP-NG + demographic noise** (the §2.0 hook, covariance $\propto H$, temperature $\propto$ LR).

**Measure.** Posterior covariance recovery vs reference (matrix distance on the shared subspace); credible-interval coverage and posterior-predictive calibration (ECE); KL or 2-Wasserstein to the reference in the toy case.

**Prediction.** Only SOAP-NG + demographic noise recovers the posterior covariance; SGD/SOAP miss it because they sample $V_{\text{stat}}\propto C^{-\alpha}$, not the true $C^{-1}$ posterior scale.

**Honesty.** This is a *separate, larger* contribution than the one-line exponent change, and it is a structured pSGLD — compare directly to Li et al. 2016 and credit it. Novelty = the biological derivation and the exact FDT condition $C\propto G^{-1}$, plus the Kronecker structure.

**Definition of done:** calibration + covariance-recovery table across all four samplers on ≥2 toy posteriors, with the reference posterior shown.

### 2.5 Curvature tracking (the ML mirror of $M$–$A$ alignment)

**Objective:** the optimizer-side analog of measuring $M$–$A$ alignment in the sims, so both systems land on one axis.

**Setup.** During training, periodically (every $N$ steps) estimate top-$k$ eigenpairs of $A$ (Hessian/true Fisher) and read off the preconditioner's eigenstructure (SOAP's $Q_L,Q_R$ and the in-basis second moments).

**Measure.** Principal angles between the preconditioner eigenbasis and $A$'s top-$k$ eigenbasis (eigenvector alignment); log-log fit of preconditioner eigenvalues vs curvature eigenvalues (the realized exponent).

**Prediction.** Eigenbasis aligns; realized eigenvalue exponent tracks the set $\alpha$; under $\alpha=1$ with low noise, $C\to A^{-1}$.

**Definition of done:** alignment-angle and exponent-fit time series for the $\alpha$ endpoints on ≥1 real model.

---

## 3. Strand 2 — Evolutionary simulations

### 3.0 The simulation engine (build spec)

**Reimplement the JAB-style individual-based QG model.** Recommended stack: JAX (vmap over individuals and replicates; jit the generation loop), NumPy/Numba acceptable.

State and dynamics:
- **Genotype:** $L$ diploid loci, $n$ traits (start $n=2$ for visualizability and to match JAB figures). Per-locus allelic effects $y$.
- **Map (additive + multilinear epistasis, Hansen & Wagner 2001):** mutation of size $\alpha$ at locus $i$ contributes $\Delta X = \alpha\big(1+\sum_{j\neq i}\varepsilon(i,j)\,y(j)\big)$, with epistatic coefficients $\varepsilon(i,j)\sim\mathcal N(0,\sigma_\varepsilon^2)$ fixed per run. Set $\sigma_\varepsilon=0$ for the no-epistasis (fixed-$M$) arm.
- **Selection:** Gaussian stabilizing, $W(z)=\exp\!\big(-\tfrac12(z-\theta)^\top A\,(z-\theta)\big)$, with selection-surface curvature $A=\omega^{-1}$ (keep the $A$/$\omega$ distinction explicit in code and comments — it is a documented source of cross-paper confusion).
- **Reproduction:** fitness-weighted parent sampling, Mendelian segregation with recombination rate $r$, per-locus mutation rate $\mu$ and mutational increment variance.
- **Finite $N_e$** for demographic noise; environmental machinery in §3.2 for gradient noise.

Measured each generation: $G$ (additive-genetic covariance of breeding values), $M$ (covariance of mutational effects *entering* the population — average over the realized mutational input given the current background), trait means, mean fitness.

**Definition of done:** with $\sigma_\varepsilon=0$, isotropic $A$, static peak, the model reproduces the JAB 2003 baseline (no $G$ orientation; $G$ magnitude at mutation–selection–drift balance). Sanity figure matches JAB qualitative behavior.

### 3.1 Sim A — the $G$–$A$ exponent and the per-locus regime test

**Objective:** establish $G\to A^{-1}$ as a *measured exponent*, and discriminate Gaussian vs Zhang–Hill per-locus regimes — the eigenvalue-scaling test JAB never fitted (they showed eigenvector alignment via Flury CPC, not eigenvalue power laws).

**Setup.** Anisotropic $A$ (sweep $\omega$ eigenvalue ratio, e.g. 1.5–20×). Two arms: (i) **$M$ free** ($\sigma_\varepsilon>0$, modifier loci let $M$ evolve); (ii) **$M$ isotropic** ($\sigma_\varepsilon=0$, $M$ fixed isotropic). Vary per-locus mutation rate $\mu$ from high (Gaussian-like) to low (Zhang–Hill rare-allele) to move between closures.

**Measure.** Log-log slope of $G$ eigenvalues vs $A$ eigenvalues, across the $A$-anisotropy sweep, per arm and per $\mu$. Also Flury CPC eigenvector alignment as a replication/sanity check.

**Prediction.** Zhang–Hill (low $\mu$): slope $\approx1$ ($G\propto A^{-1}$) *even with isotropic $M$* (rare-allele cancellation). Gaussian (high $\mu$): isotropic $M$ gives slope $\approx\tfrac12$ ($G\propto A^{-1/2}$). $M$-free arm gives slope $\to1$ in both. This is the cleanest simulation test of which regime real biology occupies. **Lead Strand 2 with this.**

**Definition of done:** slope-vs-$\mu$ curves for both arms with CIs, plus an alignment-angle sanity panel.

### 3.2 Sim B — the $N^\*$ phase diagram and two-signature discrimination

**Objective:** demonstrate gradient noise is *real and distinguishable*, and that it matters at low $N^\*$ (the biological half of the §5 spine).

**Setup.** Add a *structured* within-generation environmental term: each individual sees a perturbed optimum $\theta_{i,t}=\theta_t+\delta_{i,t}$, with $\delta_{i,t}$ drawn with a spatial correlation length $\ell$ that controls $N^\*$. Operationalize $N^\*\approx N/\big(1+(N-1)\bar\rho\big)$ with $\bar\rho$ the mean pairwise environmental correlation. Regimes: independent ($N^\*=N$), spatially correlated (small $N^\*$), kin-correlated, niche-construction (feedback where behavior modifies $\ell$). Sweep $N^\*$ (via $\ell$) and $N_e$ **independently**.

**Measure.** (i) $M$ eigenvalue anisotropy ratio $\max/\min$ vs $N^\*$. (ii) $M$ eigenvector alignment angle to $A$'s axes vs $N^\*$. (iii) across-replicate variance of $M$'s structure vs $N_e$ (run many replicates per cell).

**Predictions (these discriminate the mechanism from a generic noise story).**
- Anisotropy *compresses* as $N^\*$ falls — the attenuation-bias signature of gradient noise.
- Eigenvector alignment *unchanged* across $N^\*$ — **no eigenvalue-order inversion** (the explicit correction to the earlier over-claim; show it).
- Gradient noise → systematic eigenvalue-ratio *bias*; demographic noise → inflated *across-replicate variance* around an unbiased mean (Langevin signature). The two fingerprints separate in the $(N_e,N^\*)$ plane.

The spatial-autocorrelation manipulation is the biologically legible knob — make limited dispersal / philopatry the headline mechanism.

**Definition of done:** a 2-D $(N_e,N^\*)$ phase diagram with anisotropy-bias and replicate-variance surfaces, plus the flat alignment-angle panel. The anisotropy-vs-$N^\*$ slice is the right half of the §5 figure.

### 3.3 Sim C — the natural-gradient identity and the CMA-ES bridge

**Objective:** show three mechanisms reach the same $C\propto A^{-1}$ attractor — the figure that puts evolution, ES, and SGD on one axis.

**Setup.** (a) In the QG sim, each generation compute the natural-gradient direction implied by the current $G$ and selection gradient, and compare to the realized $\Delta\bar z$. (b) Run pycma CMA-ES on the *same* quadratic landscape and track its covariance $C$.

**Measure.** Angle/ratio between natural-gradient and realized response over generations (QG); convergence of CMA-ES $C$ to $A^{-1}$ (proportional).

**Prediction.** Natural-gradient and realized response coincide as $G\to A^{-1}$; CMA-ES $C\to A^{-1}$ (Akimoto et al. 2012 backs this analytically, so the sim is the visual).

**Definition of done:** one three-panel figure — biological $M$-evolution, breeder's-equation response, rank-$\mu$ CMA-ES — converging to $C\propto A^{-1}$.

**Hold for the companion paper:** moving-peak tracking and full niche-construction dynamics (the moving-$\Omega_{\text{slow}}$ extensions). They are rich but dilute the spine; keep the flagship tight.

---

## 4. Other suggested analyses

- **The shared-prediction figure (insist on this).** One panel: $\alpha^\*$-vs-$N^\*$ (§2.2) beside $M$-anisotropy-vs-$N^\*$ (§3.2), SDE curve overlaid on both, common x-axis label. This *is* the argument that the two systems are one phenomenon.
- **A "what doesn't transfer" subsection.** State plainly: the $M$-evolution timescale has no current ML analog (the "second-order architecture evolution" prediction is a conjecture, not a result here); standard SGD lacks the demographic term entirely; recombination resists the measure-valued treatment. Stating boundaries separates this from the looser "evolution = learning" literature and matches the project's working style.

---

## 5. Preliminary paper structure (evolution-forward)

1. **Introduction.** The breeder's equation and the SGD update share a form, usually noted as analogy and dropped. We make it a derivation, extend it with a second noise source biology has and SGD lacks, identify the exact equivalence condition, and show the condition predicts both an evolutionary scaling law and an optimizer.
2. **The deterministic correspondence.** $\Delta\bar z=G\beta\equiv\Delta\theta=-H\nabla L$; $G$ as natural-gradient metric.
3. **The two-noise SDE.** Demographic noise $G/N_e$ (linear in $G$) and gradient noise $GCG/N^\*$ (quadratic in $G$); breeder's equation and MHB-SGD as the two single-finite-sample limits.
4. **The equivalence condition.** FDT requires $C\propto G^{-1}$; under the Fisher identity $G=A^{-1}$; the Gibbs/Sella–Hirsh stationary law; the $\alpha$-continuum $G\propto A^{-\alpha}$ and why $\alpha$ is the single dial.
5. **From the condition to an algorithm.** $G$'s dual role collapsing preconditioner and noise covariance; $M$-evolution under per-generation drag $\mathrm{tr}(AM)/2$ driving $\alpha\to1$; the `precond_power` knob; the missing demographic term and how to restore it.
6. **Evolutionary results.** Sim A (measured $G$–$A$ exponent + regime test); Sim B ($N^\*$ phase diagram + two-signature discrimination); Sim C (three-mechanism natural-gradient convergence).
7. **Optimizer results.** Benchmarks (honest, "competitive"); the noise-dependent $\alpha^\*$ law; the stationary-variance/FDT test; the posterior-sampling demonstration.
8. **The shared prediction.** The cross-substrate figure and what it licenses.
9. **Limits and non-transfers.**

**Venue.** arXiv `q-bio.PE` + `cs.LG`. Journal: the evolution-forward framing and dual empirical/theoretical payload fit *PNAS* or *eLife* better than a pure-ML venue. If the optimizer results stand alone, the posterior-sampling piece could spin into a NeurIPS/ICLR submission — but do not lead the flagship there, given the speed-neutrality finding a CS reviewer will fixate on.

---

## 6. Suggested build order and repo layout (for Claude Code)

**Build order (cheap/exact validations before expensive runs):**
1. §2.0 optimizer parity + demographic-noise hook.
2. §2.2 toy quadratic — validate the SDE $\alpha^\*$ prediction analytically.
3. §2.3 toy stationary-variance — validate $V_{\text{stat}}\propto C^{-\alpha}$ exactly.
4. §3.0 sim engine + JAB baseline replication.
5. §3.1 Sim A (regime test) — the Strand 2 lead result.
6. §2.4 posterior toy (logistic regression).
7. §3.2 Sim B phase diagram (most compute-heavy; many replicates).
8. §2.1 real-model benchmarks + §2.2 real-model $\alpha^\*$.
9. §2.5 curvature tracking, §3.3 Sim C, §4 shared figure.

**Suggested layout:**
```
two_noise/
  optim/
    soap_full_power.py        # from /mnt/project, extended
    demographic_noise.py      # the sqrt(H) injection hook
    parity_test.py            # §2.0
  toy/
    quadratic_sde.py          # §2.2, §2.3 exact validations
  ml_experiments/
    benchmarks.py             # §2.1
    alpha_vs_batch.py         # §2.2
    stationary_variance.py    # §2.3  (uses curvature/lanczos.py)
    posterior_sampling.py     # §2.4  (numpyro / laplace refs)
    curvature_tracking.py     # §2.5
  curvature/
    lanczos.py                # PyHessian-based top-k eigenpairs + true-Fisher
  sim/
    engine.py                 # §3.0  (JAX IBM: multilinear epistasis, Gaussian selection)
    sim_a_exponent.py         # §3.1
    sim_b_phase.py            # §3.2
    sim_c_naturalgrad.py      # §3.3  (pycma bridge)
  figures/
    shared_prediction.py      # §5 / §4 spine figure
  configs/                    # hydra/yaml per experiment
```

**Stack:** PyTorch for optim + ViT/LM (build on `soap.py`); JAX for the IBM sims (vmap replicates × individuals); pycma for CMA-ES; NumPyro + `laplace-torch` for posterior references; PyHessian (or custom Lanczos) for curvature. Keep the $A$/$\omega$ convention documented at the top of `sim/engine.py`.

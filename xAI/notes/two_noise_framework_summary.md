# Gradient Noise in Evolutionary Dynamics: A Two-Noise Framework

## A theoretical summary (revised — preconditioned noise terms, the scaling continuum, and the missing demographic term in SGD)

This document summarizes the theoretical work extending the SGD–breeder's-equation bridge to incorporate gradient noise as a first-class component of evolutionary dynamics. It supersedes the prior `two_noise_framework_summary` and folds in a set of corrections established by working through the SDE structure carefully. The companion document `three_matrices_explainer.md` defines and distinguishes the matrices A, C, and G; `m_vs_g_explainer.md` lays out the heritable-trait (M) versus population-statistic (G) distinction; `natural_gradient_explainer.md` develops the natural-gradient reading; `two_stage_analysis.md` gives the Analysis-2 → Analysis-1 derivation; `soap_full_power.py` is the reference optimizer implementation.

**Corrections incorporated in this revision** (relative to earlier drafts):

1. **The gradient-noise term carries G.** Earlier the gradient noise was written $\sqrt{C/N^*}\,dW_2$ (diffusion covariance $C/N^*$). This is wrong for the same reason the SGD gradient-noise term carries the preconditioner: the noise lives in the gradient estimate, and the gradient is reshaped by G before it moves the mean. The correct term is $\tfrac{1}{\sqrt{N^*}}GB\,dW_2$ with $BB^\top=C$, i.e. diffusion covariance $GCG/N^*$. Sections 6, 7, and 9 of the prior draft already used this corrected form; §1 was stale and is now harmonized.

2. **The FDT condition is $C \propto G^{-1}$, not $C \propto G$.** With the corrected term, total diffusion is $G/N_e + GCG/N^*$; this is proportional to G (the FDT requirement) iff $GCG \propto G$, i.e. $C \propto G^{-1}$ — the natural-gradient/Fisher-scoring configuration. Under the Fisher identity $C=A$ this is $G=A^{-1}$, at which both noise sources collapse to $G(1/N_e+1/N^*)$ and FDT is exact.

3. **The G–A scaling is a continuum indexed by a single exponent $\alpha$.** Earlier drafts treated $A^{-1/2}$ and $A^{-1}$ as two discrete regimes. They are the endpoints of a continuous family $G\propto A^{-\alpha}$, $\alpha\in[\tfrac12,1]$, with $\alpha=\tfrac{\gamma+1}{2}$ when $M\propto A^{-\gamma}$ under Gaussian closure. Partial M-evolution and gradient-noise compression of M's anisotropy produce intermediate exponents.

4. **Vanilla SOAP is whitening ($C^{-1/2}$), not the full inverse ($C^{-1}$).** Earlier the optimizer hierarchy placed SOAP at $H\approx C^{-1}$. Per `soap_full_power.py` and the gradient-whitening analysis (Liu et al. 2025), vanilla SOAP applies the inverse *square root* of the curvature estimate (the $\alpha=\tfrac12$ point); the full inverse $C^{-1}$ ($\alpha=1$) is the modified variant at `precond_power = 1.0`. SOAP nails the natural-gradient *eigenbasis* but uses *whitening eigenvalue scaling* — the eigenvector/eigenvalue conflation the framework otherwise insists on avoiding.

5. **The ML analog of biology's $A^{-1/2}$ regime is whitening ($C^{-1/2}$), not "anti-whitening" ($C^{+1/2}$).** A stray phrasing in `two_stage_analysis.md` is corrected in §9.

6. **SGD lacks biology's demographic-noise term, and what to do about it.** §9.5 (new) works out the consequence: standard SGD has no analog of $\sqrt{G/N_e}\,dW_1$, which is exactly why it does not sample the Gibbs/posterior; preconditioned SGLD restores it by injecting noise with covariance $\propto H$ (shaping matrix $\propto\sqrt H$); in the natural-gradient regime this is a pure temperature rescaling, which is why the correspondence survives the omission.

The framework's two standing claims are unchanged. First: the multivariate breeder's equation has, since Lande (1976), modeled only one of two finite-sampling noise sources operating in real populations; the other — gradient sampling noise, formally identical to minibatch noise in SGD — has been systematically underweighted, and the assumption that large census size makes it negligible fails once one tracks the effective sample size for *gradient estimation* rather than for transmission. Second: biological evolution and ML training are governed by the same family of stochastic differential equations, with the load functional being optimized selecting the configuration within the family. When two systems satisfy the same SDE they are the same dynamical system — trajectories, equilibria, fluctuations, and perturbation responses all agree distributionally — so AI-trained systems and biological populations are mutually accessible study systems wherever the SDE structures align. §7.5 makes this explicit.

---

## Corrected framework predictions (executive summary)

- **Eigenvector alignment of M with A's principal axes** is the robust prediction, driven primarily by per-generation mutation load $\tfrac12\mathrm{tr}(A\,M)$ under conditions where the multivariate Haldane–Muller principle is overcome (finite N, epistasis, developmental/network mechanisms). Support is methodological (Wagner 1989 established load analysis on M's structure; the framework uses per-generation drag where Wagner used equilibrium standing-variance load), simulation (JAB 2007, 2014; Hether & Hohenlohe 2014; Draghi & Whitlock 2012; Pavlicev et al. 2011; Chebib & Guillaume 2017, 2022; Milocco & Salazar-Ciudad 2020, 2022; do O & Whitlock 2023), and macroevolutionary-empirical (Houle et al. 2017; Rohner & Berger 2023, 2025; Holstad et al. 2024). Lande (1980) and Phillips & Arnold (1989) give the multivariate MSB equilibrium for G given fixed M and selection; they do *not* derive $M\propto A^{-1}$ as a prediction of M-evolution.

- **Quantitative macroevolutionary prediction:** the OU stationary variance from the Lyapunov equation is $V_{\text{stat}} = \tfrac{1}{2N_e}A^{-1} + \tfrac{1}{2N^*}\,GCG\big|_{\text{shape}}$, reducing in the natural-gradient regime to $\propto A^{-1}$. Connecting single-population $V_{\text{stat}}$ to the among-species R-matrix needs an additional assumption — common static optimum (Scenario A) or evolvability-limited stationary fluctuating selection (Scenario C, Holstad et al. 2024). Either route with $M\propto A^{-1}$ gives $R\propto A^{-1}\propto M$: log-log slope 1 with bounded magnitude, matching the data.

- **The G–A eigenvalue exponent $\alpha$ is set by the per-locus regime and M's structure.** Under Zhang–Hill/house-of-cards dynamics, $G\propto A^{-1}$ robustly across a wide range of M (rare-allele cancellation). Under Gaussian/Bulmer–Lande–Week dynamics, $G\propto A^{-\alpha}$ with $\alpha=\tfrac{\gamma+1}{2}$ for $M\propto A^{-\gamma}$: isotropic M ($\gamma=0$) gives $\alpha=\tfrac12$ (the BLW baseline), load-minimized M ($\gamma=1$) gives $\alpha=1$, and partial M-evolution gives intermediate $\alpha$. Intermediate exponents are a Gaussian-closure phenomenon and a genuinely novel prediction (no prior biological paper reports any G–A exponent outside $\{\tfrac12,1\}$).

- **Gradient noise compresses M's eigenvalue anisotropy** (regression-dilution/signal-to-noise attenuation), lowering the effective $\gamma$ toward 0 and thus $\alpha$ toward $\tfrac12$; **lag-load enhances anisotropy** toward $\Omega_{\text{slow}}$'s axes. The eigenvector alignment direction persists; only the eigenvalue spread moves.

- **The earlier eigenvalue-ordering-inversion claim was wrong** and is retracted. It conflated "the G selection would prefer if G were the optimization variable" with "the M that selection actually produces, given M is what's heritable." Under proper analysis the eigenvalue ordering does not invert; anisotropy compresses while eigenvector alignment holds.

---

## 1. The full two-noise SDE

The classical stochastic breeder's equation (Lande 1976) is

$$d\bar z = G\,\boldsymbol\beta(\bar z)\,dt + \sqrt{G/N_e}\,dW_1,$$

where the single noise source is Mendelian transmission drift in a finite-population reproduction step.

> *Attribution.* Lande (1976) derived the Fokker–Planck PDE for the density of the mean phenotype under stabilizing selection and drift; the SDE form is equivalent but not written explicitly there. The explicit multivariate Ornstein–Uhlenbeck SDE, with parameter-level identifications between the OU drift $G W$, diffusion $G/N_e$, and stationary variance $(2N_eW)^{-1}$, is **Hansen & Martins (1996)**, building on Lande (1976, 1979), in the context of phylogenetic comparative methods (their notation is "infinitesimal mean and variance" rather than Itô $dW$, but equivalent). The modern measure-valued martingale derivation is Week (2026). The framework's own contributions are the SGD/ML bridge (§9), the two-noise extension (§1), the natural-gradient reading, and the M-evolution and gradient-noise extensions.

The full two-noise version, which we argue is the correct biological model, is

$$\boxed{\,d\bar z = G\,\boldsymbol\beta(\bar z)\,dt + \sqrt{G/N_e}\,dW_1 + \tfrac{1}{\sqrt{N^*}}\,G B\,dW_2,\qquad BB^\top = C.\,}$$

The second source captures the across-individuals variance of the realized fitness gradient: each individual lives in a slightly different environmental realization and "sees" a slightly different fitness function; the population gradient is averaged across individuals; the average has residual covariance $C/N^*$ by the CLT. A noisy gradient does not move the mean directly — it moves it through the response map $\Delta\bar z = G\hat{\boldsymbol\beta}$ — so the injected noise is $G\boldsymbol\xi$ with covariance $GCG/N^*$. This is the term-for-term image of the MHB minibatch term $\tfrac{1}{\sqrt S}HB\,dW$ under $H\to G$, $S\to N^*$.

### Why the two noise terms carry different powers of G

The diffusion covariances are $\Sigma_1 = G/N_e$ (linear in G) and $\Sigma_2 = GCG/N^*$ (quadratic in G). The shaping matrices written against $dW$ are their square roots — $G^{1/2}/\sqrt{N_e}$ and $GB/\sqrt{N^*}$ with $BB^\top=C$ — which is why one shows $G^{1/2}$ and the other a whole G. The deeper reason is G's dual role:

- **Demographic noise is sampling noise born in breeding-value space.** Averaging the breeding values of $N_e$ parents, the per-individual covariance of a breeding value *is* G, so the variance of the mean is $G/N_e$. Here G appears once, as **the covariance of the quantity being sampled**; nothing maps it anywhere.
- **Gradient noise is sampling noise born in selection space.** The per-individual gradient contribution has covariance C, so the estimate has covariance $C/N^*$; the response map then transforms it as $\mathrm{Cov}(G\boldsymbol\xi)=G\,(C/N^*)\,G^\top$. Here G appears twice, as **the preconditioner/operator**, sandwiching C.

The drift $G\boldsymbol\beta$ uses G in the operator sense too, which is why drift and gradient noise share the "G multiplies a gradient-space object" structure, and only demographic noise uses G-as-covariance. This is precisely why the single-noise Lande equation gets FDT for free: drift coefficient G and diffusion covariance $G/N_e$ coincide *only because in biology the breeding-value covariance equals the preconditioner* — the same matrix wearing both hats. In ML these are separate matrices (H and C), there is no natural $H/N_e$ term, and the only intrinsic noise is the quadratic $HCH/S$ — the analog of the second term, never the first.

### Single-noise (collapsed) form

Because two independent Wiener increments with covariances $\Sigma_1\,dt$ and $\Sigma_2\,dt$ sum to one increment with covariance $(\Sigma_1+\Sigma_2)\,dt$, the SDE can be written with a single driving noise:

$$d\bar z = G\,\boldsymbol\beta(\bar z)\,dt + \sqrt D\;dW,\qquad D = \frac{G}{N_e} + \frac{GCG}{N^*} = G^{1/2}\!\left[\frac{1}{N_e}I + \frac{1}{N^*}G^{1/2}CG^{1/2}\right]G^{1/2},$$

with $\sqrt D$ any matrix square root ($\sqrt D\sqrt D^\top = D$; only the product enters). The factored form isolates the demographic isotropic floor $\tfrac{1}{N_e}I$ from the gradient noise rotated into G's eigenbasis. **Use D for computing the stationary distribution** — the Lyapunov equation only ever sees the total. **Keep the terms split for interpretation** — outside the natural-gradient regime the two sources have different shapes (one isotropic-in-G, one C-shaped), so there is no scalar effective temperature, and $\sqrt D$ buries exactly the distinction the M–A alignment program relies on (demographic noise as shape-preserving Langevin variance inflation, gradient noise as C-dependent signal-to-noise compression). The collapse assumes the two sources are independent; coupled kin/reproductive structure introduces a cross term $\propto \tfrac{1}{\sqrt{N_eN^*}}$ that the simple sum omits.

### Parameters, limits, and FDT

$N_e$ is the Wright effective size (independent transmission events per generation). $N^*$ is a *different* effective sample size — independent environmental samples contributing to the gradient estimate — given by the Kish correction $N^* = N/[1+(N-1)\langle\rho\rangle]$, where $\langle\rho\rangle$ is the average pairwise correlation in fitness-relevant environmental experience. $N$ is census size. These can differ by orders of magnitude in one species: a philopatric eusocial colony might have $N\approx10^5$, $N_e\approx10^2$ (reproduction through one queen), $N^*\approx10$ (workers share the colony microenvironment).

- **Classical Lande (1976):** $N\to\infty$, $N_e$ finite. Gradient noise vanishes; transmission drift remains.
- **Classical SGD (Mandt–Hoffman–Blei 2017):** $N_e\to\infty$, $N^*=S$ finite. Transmission noise vanishes; minibatch noise remains.
- **Biological reality:** both finite; both sources operate.

The breeder's equation and SGD are not parallel analogues but two limits of one process, distinguished by which finite-sampling step dominates.

**FDT.** The single-noise Lande equation has fluctuation-dissipation structure: drift coefficient (G) and diffusion covariance ($G/N_e$) share G, temperature $1/(2N_e)$, stationary distribution the Gibbs measure $\bar W^{2N_e}$ (Sella–Hirsh). Adding gradient noise gives total diffusion $G/N_e + GCG/N^*$, which is proportional to G — the FDT requirement — iff $GCG\propto G$, i.e. **$C\propto G^{-1}$**, the natural-gradient configuration. Under $C=A$ this is $G=A^{-1}$, at which the gradient term is $GCG=A^{-1}AA^{-1}=A^{-1}=G$, the demographic term is $G=A^{-1}$, and the total collapses to

$$D = \left(\frac{1}{N_e}+\frac{1}{N^*}\right)G,\qquad \frac{1}{N_{\text{eff}}} \equiv \frac{1}{N_e}+\frac{1}{N^*}.$$

FDT is then exact, with effective inverse temperature combining $N_e$ and $N^*$ harmonically. Away from $G=A^{-1}$ the two-noise SDE has a well-defined stationary distribution (it solves the Lyapunov equation) but no FDT-respecting Gibbs form. This is not a flaw; it is the correct structure of populations under environmental gradient noise, and it means the framework's G-evolution predictions are about which G minimizes fitness load in a generically non-FDT regime, not about which G preserves FDT.

---

## 2. Why gradient noise has been neglected, and why it should not be

The standard justification — census populations are large, the CLT correction is $1/N$, the noise is negligible — treats $N$ as the relevant parameter when the relevant parameter is $N^*$. Several mechanisms drive $N^*\ll N$:

- **Spatial autocorrelation.** Limited dispersal, philopatry, or fine-grained habitat structure makes neighbors experience similar environments.
- **Kin structure.** Siblings share parental environments; relatedness enters the average correlation of environmentally-induced effects.
- **Niche construction.** Organisms modifying their own environments make environmental contributions non-independent.
- **Within-generation temporal autocorrelation.** Shared temporal fluctuations affect all individuals at once.

In the small-correlation regime the effect is additive: $1/N^* \approx 1/N + \rho_{\text{spatial}} + \rho_{\text{kin}} + \rho_{\text{niche}} + \rho_{\text{temporal}}$. The single-noise approximation is appropriate for large, outcrossing, high-dispersal populations in temporally autocorrelated environments; it fails for kin-structured, philopatric, niche-constructing, or developmentally-buffered populations, where $N^*$ can be orders of magnitude below both $N$ and $N_e$.

---

## 3. The Fisher information identity

For a correctly-specified likelihood-type fitness model at the optimum, the per-sample gradient covariance equals the curvature: $C = A$ (true Fisher = Hessian). This identity is what lets "natural-gradient configuration" ($C\propto G^{-1}$) and "inverse-curvature G" ($G\propto A^{-1}$) be the same statement, and it underwrites the SOAP correspondence. It is an approximation: the *empirical* Fisher (per-sample squared gradients) differs from the true Fisher in general (Kunstner, Hennig & Balles 2019), and away from the optimum $\mathbb E[g]\neq 0$ so $\mathbb E[gg^\top]\neq C$. Where the framework's cleanest predictions are stated, $C=A$ is assumed; the mutation-load argument for M-evolution does *not* require it, but the ML-preconditioning connection becomes less clean without it (frequency- and density-dependent selection, adversarial environments).

---

## 4. Non-mathematical intuition: the offspring distribution picture

Each generation a parent produces offspring scattered in phenotype space; selection then reweights them by fitness. The breeder's response $\Delta\bar z = G\boldsymbol\beta$ is the shift of the population mean produced by this scatter-then-reweight. G sets the *shape* of the scatter (which trait combinations vary together); $\boldsymbol\beta$ sets the *direction* selection pushes. Demographic noise is the finite-sample lottery in which offspring actually got made; gradient noise is the finite-sample lottery in which environments the parents and offspring actually experienced, so that the reweighting itself is estimated from a noisy, environmentally-correlated sample.

---

## 5. Technical aside: matrix commutativity

Many results simplify when A and G (equivalently A and M) commute — i.e. share an eigenbasis — so that matrix expressions reduce to per-eigendirection scalar relations. Commutativity holds exactly in the single-trait case, when M and the selection surface $\omega$ share principal axes, and approximately when M has evolved to align with A. The scaling laws $G\propto A^{-\alpha}$ are eigenvalue statements *within the shared basis*; when A and G do not commute (colonizers, post-shift populations, novel niches) these become approximations and the eigenvector-alignment question separates from the eigenvalue-scaling question. The framework treats the commuting case as the analytically clean reference and flags non-commuting initial conditions as an open regime (§13).

---

## 5.5. What the G-evolution literature establishes

### Eigenvector alignment vs. eigenvalue proportionality

Two distinct claims are routinely blurred. **Eigenvector alignment** — M, G, A sharing principal axes — is well-supported (JAB 2007; many simulations; Houle/Rohner-Berger/Holstad empirical $R^2$). **Eigenvalue proportionality** — the specific power $G\propto A^{-\alpha}$ — is regime-dependent and much less settled. The framework keeps these strictly separate; collapsing them is the error that produced both the retracted eigenvalue-inversion claim and (in the ML direction) the mischaracterization of SOAP. The empirical macroevolutionary signal speaks mainly to alignment plus the slope-1 scaling between log variances; it does not by itself fix $\alpha$ without the OU-equilibrium interpretation.

### The Bulmer–Lande–Week square-root baseline

The canonical Gaussian-closure result for the equilibrium G under mutation-selection balance is $G\approx\sum_i\sqrt{\mu_i M_i V_s}$ (Lande 1980, restated as Chantepie & Chevin 2020 eq. 8), which in the commuting case is $g_i\propto\sqrt{m_i}\,A_i^{-1/2}$. With **isotropic M** ($m_i=m$) this is the famous $G\propto A^{-1/2}$ — the half-power baseline. It assumes Gaussian allelic effects (large mutational input), weak selection, additivity, infinite N, and commuting M and $\omega$. Turelli (1984, 1985) showed the Gaussian assumption typically fails because allele frequencies are small; the house-of-cards regime (Bulmer 1972; Bürger & Hofbauer 1994) gives $V_G\propto\mu/s$, i.e. $G_i\propto A_i^{-1}$ — twice the Gaussian exponent. Wagner (1989) showed the B-matrix is "almost neutral" under Gaussian stabilizing selection in an infinite population, anticipating Veller & Muralidhar (2026) and providing the analytical basis for slow drift of M's eigenstructure.

### M-evolution: what's established

Selection acts on M (the heritable mutational architecture); G emerges from M through mutation-selection-drift balance. The robust result across simulation and the multivariate house-of-cards algebra is that M evolves toward A-eigenvector alignment, driven by mutation load. JAB 2007 demonstrate $r_\mu\to r_s$ (M aligning with the selection surface) under Z-H closure — agreeing with Week (2026), CMA-ES theorems (Akimoto et al. 2012; Shir & Yehudayoff 2020), and JAB's own simulations. The JAB Gaussian-closure derivation predicting antialignment ($r_\mu\to-r_s$) is an outlier disagreeing with all other evidence including their own simulations, and is treated as a closure artifact. The JAB 2014 epistasis mechanism (multilinear epistasis, Hansen & Wagner 2001) is the best mechanistic account of how $M\propto A^{-1}$ emerges.

### The empirical R-matrix and $V_{\text{stat}}$

The OU stationary variance from the Lyapunov equation, in the commuting case, is

$$V_{\text{stat}} = \frac{1}{2N_e}A^{-1} + \frac{1}{2N^*}\,G_{\text{(grad-noise shape)}},$$

so $V_{\text{stat}}\propto A^{-1}$ in the natural-gradient regime, with bounded magnitude scaling as $1/N_e+1/N^*$. Connecting single-population $V_{\text{stat}}$ to the among-species divergence matrix R requires a macroevolutionary assumption:

- **Scenario A (common static optimum):** each species follows the OU dynamics to $\mathcal N(\theta, V_{\text{stat}})$, so $R = V_{\text{stat}}\propto A^{-1}$. Clean but unrealistic.
- **Scenario B (different static optima):** $R = \mathrm{Var}(\theta_p) + V_{\text{stat}}$; needs $\mathrm{Var}(\theta_p)\propto A^{-1}$ as an extra assumption.
- **Scenario C (stationary fluctuating selection; Holstad et al. 2024):** optima fluctuate stochastically; the joint $(\tilde x_p,\eta_p)$ is a 2-block OU system. In the evolvability-limited regime (fast environmental change relative to $GA$), tracking variance is bounded by G in each direction, so $G\propto A^{-1}$ gives $R\propto A^{-1}\propto M$. This is the most realistic route and is structurally identical to the framework's natural-gradient OU dynamics with stochastic optima.

**Slope-1 forces $M\propto A^{-1}$.** For $V_{\text{stat}}\propto M$ (the observed slope-1 condition) in any noise regime: drift-dominated needs $M\propto A^{-1}$; gradient-noise-dominated under generalized BLW ($G\propto\sqrt{M A^{-1}}$) needs $\sqrt{M A^{-1}}\propto M$, again $M\propto A^{-1}$; Z-H ($G\propto A^{-1}$) needs $M\propto A^{-1}$. For $M\propto A^{-\alpha}$ with $\alpha\neq1$ the predicted slope is $1/\alpha$ (drift) or $(\alpha+1)/(2\alpha)$ (gradient-noise) — at $\alpha=\tfrac12$, slopes 2 and 1.5, neither matching. The empirical slope $\approx1$ (Houle 2017; Rohner-Berger 2023, $b\approx1.27$; Holstad et al. 2024, 1000+ traits) is uniquely consistent with $M\propto A^{-1}$.

**Bounded magnitude.** The OU equilibrium gives a *stationary* (not Brownian-accumulating) variance, bounded by $V_{\text{stat}}=A^{-1}/(2N_e)+G/(2N^*)$ — orders of magnitude below neutral expectation over 40+ My, as Houle et al. note.

**Competing accounts.** Deleterious pleiotropy (Houle et al.'s preferred mechanism; needs implausibly uniform purging across directions), neutral-subset (Jiang & Zhang 2020; slope $\le1$), pleiotropic hitchhiking (alignment only along the size-shape direction). The framework's distinctive feature is that slope-1 *and* bounded magnitude co-emerge from one mechanism (M-evolution + OU at natural-gradient equilibrium), and the result connects to natural-gradient optimization. The framework is not uniquely supported, but is quantitatively consistent with the most striking observations, and the Holstad mechanism endorsed by leading workers is structurally its Scenario C.

**Both noise sources oppose alignment, yet alignment is observed.** JAB 2014 report that demographic noise (small $N_e$) disrupts alignment via drift interfering with the compensatory tuning of allelic effects against epistatic coefficients; gradient noise (small $N^*$) compresses M's eigenvalue anisotropy via attenuation. That alignment is nonetheless observed over 40–185 My is an indirect empirical constraint on both noise magnitudes in real populations.

### Gradient noise as attenuation bias, demographic noise as Langevin diffusion

The two noises are statistically distinct in how they act on M. Gradient noise attenuates M's eigenvalue spread via signal-to-noise compression — structurally an errors-in-variables (regression-dilution) attenuation bias, which shrinks the *anisotropy* without rotating the eigenvectors. Demographic noise acts as Langevin diffusion — variance inflation without directional bias. They are potentially discriminable in simulation precisely because one compresses spread while the other inflates variance.

---

## 6. M-evolution and emergent G structure: the four forces

Selection acts on M; G(M) follows from mutation-selection-drift balance (Analysis 2 in `two_stage_analysis.md`). Pure stabilizing selection at a static peak is M-neutral only under isotropic A; under anisotropic A, mutation load is the dominant force on M structure. Four forces contribute to the load functional minimized by selection on M:

**Force 1 — Mutation load** (most fundamental, operative under anisotropic A): $L_{\text{mut}} = \tfrac12\mathrm{tr}(A M) = \tfrac12\sum_i a_i m_i$. Under a non-collapse constraint (e.g. $\sum_i\ln m_i\ge -K$, preventing any $m_i\to0$), minimization gives $m_i\propto1/a_i$, i.e. **$M\propto A^{-1}$** in both eigenvector and eigenvalue senses. (The Lagrangian $a_i/2=\lambda/m_i$ gives $m_i=2\lambda/a_i$.) Standing-variance load $\tfrac12\mathrm{tr}(A V_G)$ — Wagner 1989's quantity — is M-independent under Z-H (the multivariate Haldane–Muller principle) and reinforces Force 1 under Gaussian closure, so per-generation drag is the cleaner, regime-robust functional.

**Force 2 — Gradient-noise load** (compresses M anisotropy toward the floor): $L_{\text{grad}} = \tfrac{1}{4N^*}\sum_i a_i g_i(M,A)$. Under Gaussian closure $g_i=\sqrt{m_i/(2a_i)}$ gives $L_{\text{grad}}\propto\tfrac{1}{N^*}\sum_i\sqrt{a_i m_i}$, with $\partial L_{\text{grad}}/\partial m_i\propto\sqrt{a_i/m_i}>0$ everywhere — unilaterally favoring smaller, less anisotropic M, more strongly in sharp-A directions. The signal-to-noise reading: small $N^*$ attenuates the effective M anisotropy.

**Force 3 — Lag-load** (enhances M anisotropy toward $\Omega_{\text{slow}}$): under a moving optimum with per-generation covariance $\Omega_{\text{slow}}$, $L_{\text{lag}}=\tfrac12\mathrm{tr}(\Omega_{\text{slow}}A^{-1}G^{-2})$, which under Gaussian closure ($g_i^2\propto m_i/a_i$) is $\propto\sum_i\omega_i/m_i$, minimized by pushing $m_i$ up where $\omega_i$ is large ($m_i\propto\sqrt{\omega_i}$ under a total-mutation constraint).

**Force 4 — Demographic-noise load:** a constant $\propto(v/n)d/4$, independent of M, exerting no selection on M structure.

The total load functional is

$$L_{\text{total}}(M) = \tfrac12\mathrm{tr}(AM) + \tfrac{1}{4N^*}\mathrm{tr}(A\,G(M)) + \tfrac12\mathrm{tr}(\Omega_{\text{slow}}A^{-1}G(M)^{-2}) + \tfrac{(v/n)d}{4}.$$

Its minimizer over M is a compromise: eigenvectors align with a weighted mix of A's and $\Omega_{\text{slow}}$'s axes; eigenvalue anisotropy is compressed by Force 2 and enhanced by Force 3. The framework's distinctive empirical signature is **how M's eigenvalue anisotropy varies with $N^*$ and $\Omega_{\text{slow}}$ across species** — not eigenvalue-ordering inversion.

A note on the moving-optimum case: when A itself fluctuates, G evolves toward $\langle A\rangle^{-1}$ (the time-averaged landscape over G's evolutionary timescale), not the instantaneous $A(t)^{-1}$; single-season snapshots can misestimate the alignment target. And the environmental spectrum splits high-pass/low-pass: within-generation uncorrelated variation feeds C (and hence gradient noise, pulling toward the $A^{-1/2}$/BLW end via M compression), across-generation correlated variation feeds $\Omega_{\text{slow}}$ (pulling toward $A^{-1}$ alignment). The ML analogues are exact: minibatch sampling within stationary data ↔ gradient noise; distribution shift / continual learning ↔ moving optimum.

---

## 7. Where biology sits relative to FDT, and the scaling continuum

### Closure regimes and the exponent

The FDT-preserving regime is $C\propto G^{-1}$ (equivalently $G\propto A^{-1}$ under Fisher). Whether biology sits there depends on the per-locus regime and M:

- **Zhang–Hill with any reasonable M:** $G\propto A^{-1}$ (rare-allele cancellation: residence time $\propto1/s\propto V_s/m$ cancels input rate $\propto m$, so per-locus contributions $\propto V_s\propto A^{-1}$ for any M not perversely anti-aligned). FDT preserved, no M-evolution required.
- **Gaussian with $M\propto A^{-1}$ (M-evolved):** $g_i=\sqrt{m_i V_{s,i}/2}\Rightarrow g_i\propto V_{s,i}$, so $G\propto A^{-1}$. FDT preserved.
- **Gaussian with isotropic M:** $g_i\propto\sqrt{V_{s,i}}$, so $G\propto A^{-1/2}$ — *not* natural-gradient ($CG\propto A^{1/2}\neq I$). Biology sits elsewhere in the SDE family.

### The continuous exponent $\alpha$

Generalizing the Gaussian-closure result: with $M\propto A^{-\gamma}$,

$$g_i\propto\sqrt{m_i V_{s,i}}\propto\sqrt{a_i^{-\gamma}a_i^{-1}} = a_i^{-(\gamma+1)/2}\quad\Longrightarrow\quad G\propto A^{-\alpha},\quad\boxed{\alpha=\tfrac{\gamma+1}{2}}.$$

As $\gamma$ sweeps $0\to1$, $\alpha$ sweeps $\tfrac12\to1$ continuously. The endpoints are the BLW baseline (isotropic M, $\alpha=\tfrac12$) and full M-evolution ($\alpha=1$). **Gradient noise sets $\gamma$**: small $N^*$ attenuates M's anisotropy (Force 2 / regression dilution), pulling $\gamma$ — and hence $\alpha$ — toward $\tfrac12$; strong lag-load pushes toward $\gamma\to1$. So the $N^*$/lag-load balance is a continuous dial on the realized exponent:

| Regime | $\alpha$ | Mechanism |
|---|---|---|
| Strong gradient-noise dominance (small $N^*$, isotropic M) | $\tfrac12$ | BLW baseline; M anisotropy fully compressed |
| Balanced | intermediate (e.g. $\tfrac23\leftrightarrow\gamma=\tfrac13$) | partial M-evolution / partial compression |
| Lag-load / mutation-load dominance (M-evolved) | $1$ | $M\propto A^{-1}$, natural gradient, FDT |

This is a **Gaussian-closure phenomenon**: under Zhang–Hill, $G\propto A^{-1}$ robustly regardless of M, so no intermediate exponents arise. Intermediate $\alpha$ is a genuinely novel prediction — the literature reports no biological G–A exponent outside $\{\tfrac12,1\}$ — and a specific value like $\tfrac23$ should be presented as an example pending a balance-condition derivation that pins $\gamma=\tfrac13$. (The load functional $\tfrac{1}{2N^*}\sum_i a_i^{1-\alpha}$ has no interior optimum in $\alpha$ without a constraint, which is the tell that $\alpha$ is fixed by the M-generating mechanism, not by free optimization.)

### Stationary variance under the exponent

In the gradient-noise-dominated limit with $G=A^{-\alpha}$, the per-eigendirection Lyapunov solution is

$$V_{\text{stat}}(a) = \frac{a^{1-2\alpha}/N^*}{2\,a^{1-\alpha}} = \frac{a^{-\alpha}}{2N^*}\quad\Longrightarrow\quad V_{\text{stat}}\propto A^{-\alpha}.$$

So the exponent is directly readable from the standing-variance/curvature relationship: $\alpha=1$ (natural gradient) gives $V_{\text{stat}}\propto A^{-1}$, $\alpha=\tfrac12$ (whitening/BLW) gives $V_{\text{stat}}\propto A^{-1/2}$. The two regimes share an eigenbasis and differ only in this exponent — the eigenvalue-scaling distinction the framework keeps separate from mere alignment, and the one the macroevolutionary R-matrix work can in principle measure.

### The clean conceptual claim

Biology produces $G\propto A^{-1}$ — natural-gradient, FDT-preserving — robustly under Zhang–Hill, and under Gaussian closure when M has evolved to $A^{-1}$. Long-evolved lineages should land at $\alpha=1$ either way, matching natural-gradient ML. The distinctive contributions are (i) a comparative test of per-locus regime — populations with non-evolved M show $\alpha=1$ if Zhang–Hill, $\alpha=\tfrac12$ if Gaussian; (ii) M anisotropy as a function of $N^*$ and $\Omega_{\text{slow}}$; (iii) the architectural-evolution timescale (M-evolution itself), which no current ML method implements.

---

## 7.5. AI-evolved systems as study systems for biological evolution

The correspondences are not analogies. Two systems satisfying the same SDE are the same dynamical system — trajectories, equilibria, fluctuation structure, and perturbation response all agree distributionally — regardless of whether reproduction is sexual or asexual, gradients come from minibatches or environmental heterogeneity, or the particles are organisms or weight vectors. Those are mechanism-level details integrated out at the SDE level. This is how phenomenology works: the Langevin equation applies identically to pollen, polymers, firing rates, and financial returns. Where the SDE structures align, an AI-trained system is an accessible *in silico* study system for evolutionary questions about the joint $(\bar z,G)$ dynamics, and biological observations are predictive tools for optimizer design. **PhenoVision** — a vision transformer on natural-history specimen images for plant phenology — is the framework's in silico testbed for evolvability and preadaptation, where loss trajectories, weight-displacement tracking, efficiency ratios, intrinsic dimensionality, and per-head gradient covariance stand in for the corresponding evolutionary quantities.

Cross-system study works when the SDE structures align (Gaussian-with-M-evolution or Z-H biology ↔ natural-gradient-class SGD) and fails where they diverge — most importantly at biology's slowest timescale (M-evolution), which has no current ML analog.

---

## 8. Comparative predictions

1. **G–A exponent across $N^*$ and $\Omega_{\text{slow}}$ regimes.** Niche-constructors (low $N^*$) should sit closer to $\alpha=\tfrac12$ (compressed M anisotropy); stable, high-dispersal lineages with evolved M closer to $\alpha=1$. This replaces the earlier (incorrect) eigenvalue-inversion signature with a continuous-exponent signature.
2. **Per-locus regime discrimination.** Populations with approximately isotropic M showing $\alpha=1$ implies Zhang–Hill biology; the same populations showing $\alpha=\tfrac12$ implies Gaussian biology.
3. **Alignment with $\langle A\rangle^{-1}$, not $A(t)^{-1}$.** G tracks the time-averaged landscape; snapshot A can mislead.
4. **Bounded, stationary among-species divergence** with slope-1 R–M scaling, as observed.
5. **Niche-construction signature:** compressed M eigenvalue anisotropy relative to closely-related non-niche-constructors — a specific empirical handle on a verbal EES claim.

These are more modest than the earlier eigenvalue-inversion claim but better grounded.

---

## 9. AI training as an evolutionary process: the optimizer arc

Every optimizer is $\theta_{t+1}=\theta_t - H\hat g_t$; H reshapes raw gradient into the update, playing G's role. The structure built into H corresponds to **M-matrix structure** in biology. Crucially, an optimizer's relation to curvature has *two* independent aspects — which eigenbasis it preconditions in (eigenvector alignment) and what power of the curvature eigenvalues it applies (eigenvalue scaling, the exponent $\alpha$). Conflating them is what made the earlier hierarchy place SOAP at $H\approx C^{-1}$.

### The optimizer hierarchy (corrected)

| Optimizer | Eigenbasis | Eigenvalue power $\alpha$ | Effective H | Biological analog |
|---|---|---|---|---|
| Vanilla SGD | none (isotropic) | $0$ | $\propto I$ | no structured G |
| AdaGrad / RMSProp / Adam | axis-aligned (diagonal) | $\tfrac12$ (divide by $\sqrt v$) | diagonal $C^{-1/2}$ | axis-aligned G, no off-diagonal correlation |
| K-FAC / Shampoo | Kronecker-factored curvature basis | $\approx\tfrac12$ | block $\approx C^{-1/2}$ | within-block structural correlations |
| **Vanilla SOAP** | **natural-gradient (Shampoo) eigenbasis** | $\tfrac12$ (Adam-in-basis = whitening) | $\boldsymbol{C^{-1/2}}$ | **Gaussian biology, isotropic M, $\alpha=\tfrac12$** |
| **SOAP-NG (`precond_power=1.0`)** | same eigenbasis | $1$ (full inverse) | $\boldsymbol{C^{-1}}$ | **biology with evolved M / Z-H, $\alpha=1$** |

The key correction: **vanilla SOAP gets the natural-gradient *eigenbasis* but uses *whitening* eigenvalue scaling** ($C^{-1/2}$, the $\alpha=\tfrac12$ point), as `soap_full_power.py` states explicitly ("SOAP divides by $\sqrt V$ — power $\tfrac12$ → whitening; dividing by $V$ — power $1$ → the full inverse $H^{-1}$") and as the gradient-whitening reading confirms (Liu et al. 2025). The full inverse $C^{-1}$ is the modified variant. So the preconditioner-power knob `precond_power` *is* the exponent $\alpha$, and biology's M-evolution trajectory from isotropic M ($\alpha=\tfrac12$) to load-minimized M ($\alpha=1$) is the same path as turning `precond_power` from $0.5$ to $1.0$.

### The SDE for intermediate $\alpha$

Mapping $G\leftrightarrow H$, $A\leftrightarrow C$ (Fisher), biology's $G\propto A^{-\alpha}$ is ML's $H\propto C^{-\alpha}$. The linearized gradient-noise SGD SDE with $H=C^{-\alpha}$, $A_{\text{loss}}=C$ is

$$d\theta = -\,C^{1-\alpha}(\theta-\theta^*)\,dt + \tfrac{1}{\sqrt S}\,C^{-\alpha}B\,dW,\qquad V_{\text{stat}}\propto C^{-\alpha},$$

with drift $C^{1-\alpha}$ and diffusion $C^{1-2\alpha}/S$. It is a well-defined stable OU process for every $\alpha$. The interpretation: $V_{\text{stat}}\propto C^{-\alpha}$ means precision $\propto C^{\alpha}$, so the stationary law is the Gibbs measure of a *power-reshaped* potential $\tfrac12(\theta-\theta^*)^\top C^{\alpha}(\theta-\theta^*)$; **only $\alpha=1$ recovers the Gibbs measure of the true loss (FDT exact)**, deformed otherwise. Endpoints: $\alpha=0$ vanilla SGD ($V_{\text{stat}}\propto I$), $\alpha=\tfrac12$ whitening/SOAP ($V_{\text{stat}}\propto C^{-1/2}$), $\alpha=1$ natural gradient/SOAP-NG ($V_{\text{stat}}\propto C^{-1}$). Both systems share $V_{\text{stat}}\propto(\text{curvature})^{-\alpha}$, so $\alpha$ is the single cross-system dial — `precond_power` in ML, the G–A exponent in biology.

The noise-robustness tradeoff governs the choice in both. The full inverse ($\alpha=1$) is drift-optimal (isotropic relaxation, $C^{1-\alpha}=I$) but over-amplifies flat-direction noise, which is precisely why `soap_full_power.py` must add Levenberg–Marquardt damping once the square root is removed ("flat directions would otherwise produce unbounded steps"). Whitening ($\alpha=\tfrac12$) is gentler and noise-robust. Intermediate $\alpha$ interpolates. The square root in vanilla SOAP does for free what the damping does by hand.

### Correction: whitening, not anti-whitening

The ML analog of biology's $A^{-1/2}$ regime is **whitening, $H\propto C^{-1/2}$ (vanilla SOAP)** — not "anti-whitening $H\propto\sqrt C$" as a stray phrasing in `two_stage_analysis.md` had it. Under the framework's own Fisher convention, $G\propto A^{-1/2}$ maps to $H\propto A_{\text{loss}}^{-1/2}=C^{-1/2}$, and `soap_full_power.py` labels power $0.5$ as whitening. The only defensible reading of "anti-whitening" is *relative*: $C^{-1/2}=C^{-1}\cdot C^{1/2}$, so the deviation from full natural gradient is an anti-whitening factor $C^{1/2}$. A literal $H\propto C^{+1/2}$ optimizer would invert the prediction and is a slip to fix before this lands in the talk.

### Noise structure clarification

- **$N_e$ is effectively infinite in standard ML.** No transmission step, so the $G/N_e$ term is absent (see §9.5). Population-based methods (CMA-ES, ES) partially restore it via population sampling.
- **$N^*$ is the effective batch size**, corrected for sample correlations.

The two-noise SDE is the common ancestor of the breeder's equation ($N^*\to\infty$) and MHB SGD ($N_e\to\infty$); the distinctive regime is both finite. CMA-ES is a third mechanism reaching $C\propto A^{-1}$ — Akimoto et al. (2012) show the rank-$\mu$ update is natural-gradient ascent on the Gaussian manifold, with covariance converging to the inverse Hessian on quadratics — by direct selection on the search distribution, structurally distinct from both biological routes. All converge on the same SDE attractor.

### Framework prediction for ML

The distinctive prediction is **second-order architecture evolution**: a method that uses second-order signals to reshape its preconditioner-generating *architecture* (not just preconditioner values) is the ML analog of M-evolution. No current method implements this; it should outperform fixed-architecture methods when the optimal preconditioner structure is unknown a priori, with architecture updates slower than value updates by a factor reflecting the second-order signal. (The earlier "$C^{-2}$ double-whitening" prediction was retracted: Z-H gives $G\propto A^{-1}$, not $A^{-2}$, so no biological regime corresponds to it.)

---

## 9.5. The missing demographic-noise term in SGD, and pSGLD

Standard SGD carries only the gradient-noise term, diffusion covariance $HCH/S$. It has no analog of biology's $\sqrt{G/N_e}\,dW_1$, because there is no finite-population reproduction step — the parameter vector is a single deterministic state pushed by gradients, not a finite sample resampled through a Mendelian lottery. **This is exactly why vanilla SGD does not sample the Bayesian posterior.** The demographic term is the one whose covariance is proportional to the preconditioner (G-as-covariance, the $\sqrt G$-shaped term); that proportionality is what FDT needs (drift coefficient and diffusion covariance sharing the matrix at one temperature). The gradient term $HCH$ is C-shaped, not H-shaped, so on its own it produces a stationary distribution matching no natural potential. Biology gets the FDT-restoring term for free; SGD lacks it.

**What to inject, and at which power.** Covariance proportional to $H$ — equivalently, shaping matrix proportional to $H^{1/2}=\sqrt H$. These describe one term (the covariance is $H$-linear; the matrix written against $dW$ is its square root), and it is the right one. This is preconditioned SGLD:

$$d\theta = -H\nabla L\,dt + \sqrt{2T\,H}\,dW,\qquad \pi(\theta)\propto e^{-L(\theta)/T},$$

whose stationary distribution is the Gibbs measure for *any* positive-definite H, precisely because the H in drift and the H in noise covariance match. Setting $T=1/(2N_e^{\text{eff}})$ makes this term-for-term biology's demographic term. Injecting covariance $\propto\sqrt H$ (shaping $\propto H^{1/4}$) would break the matching and lose the Gibbs form. So the $\sqrt{}$ in biology's term is the shaping matrix of an $H$-covariance noise, not an $\sqrt H$-covariance noise. (Welling & Teh 2011; preconditioned form Li et al. 2016.)

**Effect on the dynamics — regime-dependent.** Two effects:

- *Scale (always present, mild in shape).* Adding the term raises total diffusion to $H/N_e^{\text{eff}}+HCH/S$, broadening fluctuations. At the natural-gradient point $H=C^{-1}=A^{-1}$, both terms are $A^{-1}$-shaped ($HCH=A^{-1}$, $H=A^{-1}$), so the total is $A^{-1}(1/N_e^{\text{eff}}+1/S)$ — a **pure temperature rescaling** that broadens exploration uniformly and leaves the *shape* of $V_{\text{stat}}$ ($\propto A^{-1}$) untouched. Its eigenstructure (what the R-matrix work probes) is unchanged; only its magnitude moves.
- *Shape and sampling (appreciable, off natural gradient).* When $H\neq C^{-1}$ the $H$-shaped injection reshapes $V_{\text{stat}}$ toward H, and if it dominates the gradient noise the total diffusion becomes $\propto H$, FDT is restored, and SGD stops being an optimizer and becomes an MCMC sampler of the posterior — a qualitative change. (With non-negligible minibatch noise present, the known SGLD bias appears; exact sampling needs the injection to swamp or correct the gradient term.)

**Why this matters for the framework rather than being a footnote.** The breeder's-equation ↔ SGD correspondence survives SGD's missing demographic term *specifically because, in the natural-gradient regime where the correspondence is exact, that term would only rescale temperature, not reshape.* A biological population with finite $N_e$ runs at a higher effective temperature than an SGD trajectory with the same H, C, and small gradient noise, but they explore the same directions with the same relative magnitudes. **SGD is biology at the same shape but a colder (gradient-noise-only) temperature; pSGLD is the construction that adds the missing thermal contribution** — useful for sampling or when H is off the natural-gradient point, dynamically inert in shape when on it.

---

## 10. The universal form: stochastic optimization under finite resources

The two-noise SDE is the canonical form of noisy gradient ascent under finite-resource constraints. Many fields arrived at it independently:

| Field | Discovery | Optimality criterion |
|---|---|---|
| Brownian motion (Einstein 1905) | fluctuation–dissipation, Langevin | thermodynamic equilibrium |
| Shifting balance (Wright 1932) | drift-aided peak hopping | long-term metapopulation fitness |
| Population genetics (Lande 1976) | stochastic breeder's equation | mean fitness response |
| Free fitness (Iwasa 1988; Sella–Hirsh 2005) | Gibbs equilibrium | free-energy minimization |
| Simulated annealing (Kirkpatrick 1983) | temperature-scheduled SGD | global optimum |
| SGLD (Welling–Teh 2011) | explicit Langevin for posterior | posterior fidelity |
| MHB SGD (Mandt–Hoffman–Blei 2017) | SGD as approximate posterior sampling | posterior fidelity |
| Diffusion models (2019–2020) | score-based sampling | generated-distribution fidelity |
| SOAP / SOAP-NG (2024–2025) | two-timescale (whitening / natural gradient) | convergence under noise |

Each is the same Langevin SDE; the load functional sets the equilibrium location. SGLD is the entry that most directly mirrors biology's demographic term (explicit $H$-covariance injection); biology adds it for free through finite-$N_e$ transmission. Biology occupies the FDT-preserving natural-gradient region under M-evolution by mutation load (both Gaussian-with-evolved-M and Z-H give $G\propto A^{-1}$); gradient noise and lag-load shift it within (and the exponent $\alpha$ off) that region. The framework's contribution is threefold: identifying the universal form, distinguishing the load functionals that select configurations within it, and operationalizing cross-system study.

---

## 11. The simulation program

The core extension is to take the JAB 2014 epistasis-driven M-alignment simulation and add the two noise sources explicitly, so as to discriminate their distinct signatures on M–A alignment.

**Six concrete extensions** to discriminate signal-to-noise compression (gradient noise) from Langevin diffusion (demographic noise):

1. Vary $N^*$ at fixed $N_e$ and measure M's eigenvalue anisotropy — predict compression toward isotropy as $N^*$ falls (exponent $\alpha\to\tfrac12$ under Gaussian closure).
2. Vary $N_e$ at fixed $N^*$ and measure variance inflation without anisotropy change — predict Langevin diffusion signature (broadening without rotation).
3. Cross the two to confirm the effects are separable (compression vs inflation).
4. Track the realized G–A exponent $\alpha$ as a function of the $N^*$/$\Omega_{\text{slow}}$ balance — test $\alpha=\tfrac{\gamma+1}{2}$ under Gaussian closure and $\alpha\approx1$ robustness under Z-H.
5. Toggle the per-locus regime (Gaussian vs Zhang–Hill allelic effects) to confirm intermediate exponents appear only under Gaussian closure.
6. Measure $V_{\text{stat}}$ directly and verify $V_{\text{stat}}\propto A^{-\alpha}$ matches the independently-estimated exponent.

**Predictions to test:** eigenvector alignment of M with A is robust across all noise settings; eigenvalue anisotropy compresses with smaller $N^*$ and enhances with larger $\Omega_{\text{slow}}$; the G–A exponent is regime-dependent in the predicted way; demographic and gradient noise leave structurally distinct fingerprints (inflation vs attenuation).

**Where simulations could push back:** if anisotropy compression under small $N^*$ does not appear, Force 2 is mis-specified; if Z-H shows intermediate exponents, the rare-allele cancellation argument is incomplete; if demographic and gradient noise are not separable in M's spectrum, the attenuation/diffusion distinction collapses.

**Validation strategy:** match simulation outputs against the macroevolutionary slope-1 R–M signal and against natural-gradient ML training runs at varied `precond_power`, treating the latter as an in silico evolutionary system per §7.5.

---

## 12. Implications for the broader framework

The framework identifies mutation load as the primary force shaping M-evolution (consistent with JAB 2007 and do O & Whitlock 2023), quantifies an EES claim about niche construction (low $N^*$ → compressed M anisotropy → exponent toward $\tfrac12$) with a specific comparative signature, makes the constraint-vs-adaptation debate tractable (alignment involves A *and* C and depends on $N^*$), and identifies where standard single-noise theory works and breaks. It operationalizes cross-system study between biological evolution and AI training: when the SDE structures align (Gaussian-with-evolved-M or Z-H biology ↔ natural-gradient-class SGD), the two are the same dynamical system. And it predicts one new class of ML algorithm — second-order architecture evolution, the analog of biology's M-evolution timescale. (The earlier "double-whitening $C^{-2}$" prediction was retracted; the corrected ML correspondence places vanilla SOAP at the whitening end and SOAP-NG at the natural-gradient end of one shared exponent axis.)

---

## 13. Open questions and next steps

- **Derive intermediate exponents from a balance condition.** $\alpha=\tfrac{\gamma+1}{2}$ gives the mechanism, but pinning a specific value (e.g. $\tfrac23\leftrightarrow\gamma=\tfrac13$) needs a derivation of $\gamma$ from the $N^*$/$\Omega_{\text{slow}}$ balance under a stated M constraint. The load functional alone has no interior optimum, so the constraint structure is doing the work.
- **Test the whitening-vs-natural-gradient distinction empirically and in silico.** Measure $V_{\text{stat}}\propto A^{-\alpha}$ across lineages (macroevolutionary) and across SOAP `precond_power` settings (PhenoVision), and check the two agree.
- **Reconcile the "anti-whitening" phrasing** in `two_stage_analysis.md` with the corrected whitening reading before the talk.
- **Formal coupled three-timescale dynamics:** a joint SDE for $(\bar z, G, M)$ with all timescales explicit (`extended_joint_sde.md`; Week 2026 flags this).
- **The non-Fisher regime:** under frequency/density-dependent or adversarial selection, $C\neq A$; the mutation-load argument survives but the ML-preconditioning connection loosens.
- **Non-commuting initial conditions:** colonizers, post-shift populations, novel niches — the commuting-case scaling laws become approximations.
- **Drift's effect on G in finite/clonal populations:** Week 2026's drift-to-$\pm1$ correlation result complicates inference of selection from observed G; recombination's counteracting strength is open.
- **The asexuality caveat:** many analytical results inherit Week's asexual measure-valued setting; the ML bridge is unaffected (no recombination), but biological applications to sexual species need care.

---

*This document is the revised two-noise framework summary, incorporating the preconditioned-noise correction to §1, the $C\propto G^{-1}$ FDT condition, the continuous scaling exponent $\alpha=\tfrac{\gamma+1}{2}$ with intermediate-power predictions, the corrected SOAP-as-whitening characterization with the `precond_power` $=\alpha$ mapping, the whitening (not anti-whitening) ML analog, and the new §9.5 on the missing demographic-noise term and preconditioned SGLD. Companion documents: `three_matrices_explainer.md`, `m_vs_g_explainer.md`, `natural_gradient_explainer.md`, `extended_joint_sde.md`, `two_stage_analysis.md`, `soap_full_power.py`. The framework is a working theoretical platform, not a final formulation.*

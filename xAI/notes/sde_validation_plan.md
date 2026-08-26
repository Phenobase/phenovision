# Validating the (z̄, G, M) SDE Against Individual-Based Models

## A self-contained specification for an autonomous implementation agent

This is a standalone brief. The agent running it has **no access to the conversation or companion documents in which the theory was developed** — everything needed to understand and implement the validation is contained here. **Part 0** states the theory and defines every symbol from scratch; the remaining parts specify the tests. Citations point only to published literature the agent can look up; there are no live cross-references to unavailable documents.

The work sits inside a larger research program arguing that evolutionary quantitative genetics and machine-learning optimization are governed by the same stochastic dynamics. That program has two empirical arms; **this document is the evolutionary-simulation arm, specifically the part that checks whether the continuous stochastic differential equations (SDEs) the theory is built on actually approximate the discrete population genetics they claim to summarize.** The optimizer arm is not the agent's concern here.

---

# Part 0 — The theory being validated

## 0.1 The claim, in one paragraph

The theory summarizes a whole population by a few quantities: the vector of mean trait values **z̄**, the matrix of standing additive genetic variances and covariances **G**, and the matrix of per-generation mutational input **M**. It writes coupled SDEs for how these change under selection, drift, and mutation. Its central new claim is that when the genes controlling M are heritable — so M itself evolves — the **slow** evolution of M is approximated by a deterministic gradient descent,

$$\mathbb E[\dot M \mid M] \;=\; -\,\mathcal G_M\,\nabla_M \mathcal L_{\text{eff}}(M),$$

on a closed-form fitness-load surface $\mathcal L_{\text{eff}}(M)$, where $\mathcal G_M$ is the genetic variation available to reshape M. This closed form is derived under two approximations: **(a)** the population's phenotype distribution is multivariate Gaussian, and **(b)** the faster variables z̄ and G have relaxed to their conditional equilibria given the current M. Individual-based models (IBMs) — explicit simulations of finite populations of discrete individuals with genotypes, mutation, recombination and selection — make **none** of these approximations. The validation asks: do IBMs move M the way the SDE says, and where does the approximation break? The agent's job is to build the IBMs and run the comparison ladder in Parts V0–V6.

## 0.2 The setup and the three matrices

Each individual has a $d$-dimensional **phenotype** $z = g + e$, where $g$ is the heritable (additive genotypic) value summed over many loci and $e$ is a random environmental deviation. Track $d$ traits jointly (e.g. two beak dimensions of a finch; $d=2$ is fine for all tests). The relevant matrices, all $d\times d$ and symmetric:

- **$G$** — additive genetic covariance: the covariance of the genotypic values $g$ across the population. The *standing* heritable variation.
- **$M$** — mutational input covariance: the covariance of the increments that new mutations add to $g$ **per generation**. The raw material from which $G$ is built and replenished.
- **$A$** — curvature of the (log) fitness surface, i.e. the strength of stabilizing selection. Defined precisely in 0.3. Symmetric positive definite (SPD); large eigenvalues = steep = strong stabilizing selection, small eigenvalues = flat = weak selection.

Geometric reading (used throughout): an SPD matrix is an ellipsoid — its eigenvectors are the axes and its eigenvalues the squared axis lengths. Two SPD matrices are **aligned** (equivalently **commuting**) when they share eigenvectors; then each axis $i$ can be treated independently with scalar eigenvalues, and matrices like $G$, $A$, $M$ reduce to per-axis numbers $g_i, a_i, m_i$. Many formulas below have a clean "aligned" form and a general (non-commuting) matrix form; both matter, because misalignment drives one of the predictions (0.9).

## 0.3 Selection and the breeder's equation

Stabilizing selection is modeled by a Gaussian fitness function with optimum $\theta$:

$$W(z) \;\propto\; \exp\!\Big(-\tfrac12 (z-\theta)^\top A\,(z-\theta)\Big).$$

The matrix $A = -\nabla^2 \ln W$ is the curvature. The **selection gradient** is $\beta = \nabla_{\bar z}\ln \bar W$, the gradient of *mean* log-fitness with respect to the mean phenotype, where $\bar W$ is fitness averaged over the population's phenotype distribution. For a Gaussian phenotype distribution under this landscape, $\beta = -A(\bar z - \theta)$ — it points back toward the optimum, more strongly the steeper the landscape. The per-generation response of the mean is the **multivariate breeder's equation** (Lande 1979):

$$\Delta\bar z = G\,\beta.$$

Response = standing genetic variation × selection push. (In one trait this is $R = h^2 S$.)

## 0.4 Mutation–selection balance: the G-equation (a Riccati equation)

$G$ is maintained by a balance: mutation adds $M$ each generation; stabilizing selection removes variance at a rate set by $A$. The deterministic continuous-time dynamics are

$$\dot G = M - GAG - \tfrac{v}{n}G,$$

where $-\tfrac{v}{n}G$ is a small drift-contraction term (drop it for the leading picture). This is a **matrix Riccati equation** — the unknown $G$ appears *quadratically* in the term $-GAG$ (G multiplied by itself with $A$ in the middle). That quadratic term is why the equilibrium is an inverse-type object. Setting $\dot G = 0$ gives the **mutation–selection balance (MSB)** condition

$$GAG = M,$$

whose unique SPD solution is the **matrix geometric mean** of $A^{-1}$ and $M$:

$$\hat G = A^{-1}\#M = A^{-1/2}\big(A^{1/2} M A^{1/2}\big)^{1/2} A^{-1/2}.$$

(Verify by substitution: $\hat G A \hat G = M$, using $A^{-1/2}AA^{-1/2}=I$.) When $G$, $A$, $M$ are aligned this reduces to $\hat g_i = \sqrt{m_i/a_i}$ on each axis: more standing variation where the landscape is flat. This $\hat G(M)$ is the conditional equilibrium that the adiabatic reduction (0.7) substitutes for the medium-timescale variable.

## 0.5 Finiteness: the two noises and the stationary spread

Finite populations add randomness to the mean's dynamics, from two distinct sources with different shapes (this two-noise structure is specific to this framework and matters for the tests):

1. **Genetic drift (demographic noise).** Chance in which individuals reproduce makes $\bar z$ wander. Covariance $\propto G$, scaled by $1/N_e$ (written $v/n$: reproductive variance $v$ over population size $n$, which plays the role of $1/N_e$).
2. **Gradient (sampling) noise.** The selection gradient the population actually experiences is estimated from finitely many individuals/environments. Covariance $\propto GCG/N^*$, where $C$ is the covariance of the per-environment gradient and $N^*$ an effective *sampling* size. Near the optimum $C = A$ by the score/information identity ($\mathrm{Var}[\nabla\ln W] = -\mathbb E[\nabla^2\ln W] = A$).

The mean's SDE near the optimum (with $x=\bar z-\theta$, drift matrix $GA$):

$$dx = -GA\,x\,dt \;+\; \sqrt{v/n}\,\sqrt G\,dB_1 \;+\; \tfrac{1}{\sqrt{N^*}}\,G\sqrt{C}\,dB_2,$$

noise covariances $(v/n)G$ and $GCG/N^*$. The stationary covariance of $x$ solves the Lyapunov equation $(GA)\Sigma_\infty + \Sigma_\infty(GA)^\top = (v/n)G + GCG/N^*$, with **closed-form solution valid for general (non-commuting) $G$ and $A$**, using $C=A$:

$$\Sigma_\infty = \frac{v}{2n}A^{-1} + \frac{1}{2N^*}G.$$

(Verify: $(GA)\Sigma_\infty = \tfrac{v}{2n}G + \tfrac{1}{2N^*}GAG$, and $\Sigma_\infty(AG)$ gives the same, summing to the RHS.) Aligned: $\sigma_i = \tfrac{v}{n}\tfrac{1}{2a_i} + \tfrac{g_i}{2N^*}$. This process is reversible; its stationary distribution is the Gibbs form of an effective potential $\hat U = \Sigma_\infty^{-1}$, which equals the fitness curvature $A$ only as $N^*\to\infty$. ($G$ also fluctuates as a matrix; for these tests the fast-equilibrium predictions $\hat G$ and $\Sigma_\infty$ are what matter.)

## 0.6 Letting M evolve: second-order selection and the modifier metric

Mutation rates and the *pattern* of mutational effects are partly under genetic control, so $M$ can itself be heritable. Treat the independent entries of $M$ (its `vech`, the lower-triangular vectorization) as heritable traits. By the same breeder's-equation logic, M's expected change is

$$\dot M = \mathcal G_M\,\beta_M, \qquad \beta_M = \nabla_M \ln\bar W,$$

where $\beta_M$ is the **second-order selection gradient** (how a change in $M$ changes long-run mean fitness) and $\mathcal G_M$ is the additive genetic covariance of M's parameters — the genetic variation available to reshape $M$, which acts as a **metric** on the space of $M$'s. Critically, $\mathcal G_M$ is whatever the modifier genetics produce; it is **not** in general the "natural" (Fisher) metric.

Two facts the tests rely on. **(1)** The M-dynamics equal a *natural-gradient* flow (the geometrically canonical one) **if and only if** $\mathcal G_M \propto \tfrac12(M^{-1}\otimes M^{-1})$, the Fisher metric of the Gaussian family $\mathcal N(0,M)$; a sufficient biological condition is that modifier loci act **log-additively on M's eigenvalues with equal per-locus variance**. **(2)** The *fixed point* of the M-dynamics is **metric-independent** — it is wherever $\beta_M = 0$ — so the regime predictions of 0.8 hold regardless of $\mathcal G_M$; only the transient path and speed depend on it. This is why the drift-direction tests must control for $\mathcal G_M$ (see the metric-confound section after Part 0).

## 0.7 The adiabatic reduction: the effective load ℒ_eff(M)

$M$ evolves on a far slower timescale than $\bar z$ and $G$ (its genetic variance is small relative to the focal traits; call the timescale-separation ratio $\varepsilon \ll 1$). On M's timescale, $\bar z$ and $G$ sit at their conditional equilibria. Substituting them into mean log-fitness and collecting the M-dependent terms yields a closed effective load.

Derivation (near-optimum, quadratic approximation). The relevant fitness for slow second-order selection is the long-run growth rate, the stationary average of mean log-fitness $\langle\ln\bar W\rangle$. Near the optimum, $\ln\bar W \approx \text{const} - \tfrac12(\bar z-\theta)^\top A(\bar z-\theta) - \tfrac12\mathrm{tr}(AG)$ (mean-displacement load + standing-variance load). Averaging over the conditional stationary distribution (using $\langle(\bar z-\theta)^\top A(\bar z-\theta)\rangle = \mathrm{tr}(A\Sigma_\infty)$) and substituting $\hat G(M)=A^{-1}\#M$ and $\Sigma_\infty = \tfrac{v}{2n}A^{-1} + \tfrac{1}{2N^*}\hat G$:

$$\boxed{\;\mathcal L_{\text{eff}}(M) = \kappa\,\mathrm{tr}\!\big(A\,\hat G(M)\big) - \mathcal B(M), \qquad \kappa = \tfrac12 + \tfrac{1}{4N^*}\;}$$

so $\langle\ln\bar W\rangle = \text{const} - \mathcal L_{\text{eff}}(M)$, and the predicted slow drift is $\mathbb E[\dot M\mid M] = -\mathcal G_M\nabla_M\mathcal L_{\text{eff}}$. Here $\mathcal B(M)$ is a **benefit** term (0.8). Aligned explicit form for implementation:

$$\mathcal L_{\text{eff}} = \kappa\sum_i \sqrt{a_i m_i} - \mathcal B,\qquad \frac{\partial\mathcal L_{\text{eff}}}{\partial m_i} = \frac{\kappa}{2}\sqrt{a_i/m_i} - \frac{\partial\mathcal B}{\partial m_i}.$$

For general (non-aligned) $M$, compute $\mathcal L_{\text{eff}}(M)$ from the matrix formula (eigendecompose the $d\times d$ matrices to form $A^{-1}\#M$) and take the gradient in `vech` coordinates by finite differences or autodiff.

## 0.8 The three regimes (the predicted fixed points)

The benefit term $\mathcal B$, hence the M-fixed-point, depends on the environment. These predicted fixed points are what V1's potential minima and V3's drift zeros should match:

- **Static optimum, no premium on variation ($\mathcal B = 0$).** $\partial\mathcal L_{\text{eff}}/\partial m_i = \tfrac{\kappa}{2}\sqrt{a_i/m_i} > 0$ for all $i$: $M$ shrinks toward a mutational floor, fastest in steep (high-$a_i$) directions. **Canalization.** ($M\to$ lower bound; its *shape* transiently $\propto A^{-1}$.)
- **Tracking — optimum drifts with displacement covariance $\Omega$.** Benefit = reduced lag load $\tfrac14\sum_i\omega_i/g_i$. Fixed point $m_i^\dagger \propto \omega_i$, i.e. **$M\propto\Omega$**: the mutational architecture aligns with the *direction of environmental movement*, independent of the curvature $A$. Isotropic $\Omega$ → isotropic $M$ → standing $g_i\propto a_i^{-1/2}$.
- **Exploration — fluctuating/uncertain environment, variation has bet-hedging value.** Benefit = entropy term $\mathcal B = \tfrac{\lambda}{2}\log\det M$. Fixed point $m_i^\dagger\propto 1/a_i$, i.e. **$M\propto A^{-1}$** → standing $g_i\propto a_i^{-1}$. (The full inverse-curvature regime.)

A note for interpreting drift: canalization (shrinking the overall size $\mathrm{tr}\,M$) and inverse-curvature alignment (tuning the *shape* of $M$) are different operations — the magnitude knob and the shape knob — and can coexist (the size collapses while the shape aligns). Reporting the drift's eigenvalue-scaling part and eigenvector-rotation part separately (V3) keeps them distinct.

## 0.9 Gradient-flow structure and the curl prediction

The reduced slow flow $\dot M = -\mathcal G_M\nabla_M\mathcal L_{\text{eff}}$ is, to leading order, a **gradient flow**: it descends the scalar $\mathcal L_{\text{eff}}$ monotonically, since $\tfrac{d}{dt}\mathcal L_{\text{eff}} = -\nabla\mathcal L_{\text{eff}}:\mathcal G_M:\nabla\mathcal L_{\text{eff}} \le 0$ for any SPD $\mathcal G_M$. Two predictions to test:

1. **Curl-free at leading order, regardless of the metric.** A symmetric positive mobility $\mathcal G_M$ applied to a gradient force cannot produce rotation, so the choice of $\mathcal G_M$ changes the path but never introduces circulation.
2. **Any genuine non-conservative (curl) correction is higher-order and conditional.** It can arise only from irreversibility of the fast $(\bar z, G)$ subsystem, and the theory predicts it scales as $\varepsilon\times(\text{noise})^2$ (timescale separation × squared fluctuation amplitude, the latter $\propto 1/N_e + 1/N^*$), appears only when $A$ and $G$ are **misaligned** (non-commuting), and **vanishes when they share eigenvectors**. This is the population-genetic analog of a geometric-pumping / Berry-curvature force on a slow variable driven by an irreversible fast bath.

## 0.10 Symbol glossary

| Symbol | Meaning |
|---|---|
| $\bar z$ | vector of mean trait values ($d$-dim); $x=\bar z-\theta$ is the deviation from the optimum |
| $\theta$ | optimum trait vector (may be static, drifting, or fluctuating) |
| $G$ | $d\times d$ additive genetic covariance (standing heritable variation) |
| $M$ | $d\times d$ mutational input covariance (per generation); the slow evolving variable |
| $A$ | $d\times d$ fitness-surface curvature $=-\nabla^2\ln W$ (stabilizing-selection strength), SPD |
| $\beta$ | selection gradient $=\nabla_{\bar z}\ln\bar W = -A(\bar z-\theta)$ |
| $\hat G(M)$ | conditional equilibrium of $G$ given $M$: the geometric mean $A^{-1}\#M$ |
| $\Sigma_\infty$ | stationary covariance of the mean-deviation $x$; $=\tfrac{v}{2n}A^{-1}+\tfrac{1}{2N^*}G$ |
| $N_e$ | effective population size (drift); enters as $v/n$ |
| $v/n$ | demographic-noise coefficient (reproductive variance $v$ / population size $n$), $\sim 1/N_e$ |
| $N^*$ | effective environmental-sampling size (gradient-noise strength) |
| $C$ | covariance of the per-environment selection gradient; $C=A$ near the optimum |
| $\kappa$ | load prefactor $=\tfrac12 + \tfrac{1}{4N^*}$ |
| $\mathcal L_{\text{eff}}(M)$ | effective fitness-load surface that the slow M-flow descends |
| $\mathcal B(M)$ | benefit term: $0$ (static), lag-load reduction (tracking), or $\tfrac{\lambda}{2}\log\det M$ (exploration) |
| $\Omega$ | covariance of the optimum's per-generation displacement (tracking regime) |
| $\mathcal G_M$ | additive genetic covariance of M's parameters (metric on M-space); measured from modifier breeding values |
| $\varepsilon$ | timescale-separation ratio (modifier genetic variance relative to focal traits); $M$ slow when $\varepsilon\ll1$ |
| $A^{-1}\#M$ | matrix geometric mean, $A^{-1/2}(A^{1/2}MA^{1/2})^{1/2}A^{-1/2}$ |
| `vech`(M) | vector of independent entries of symmetric $M$ (lower triangle), e.g. $[m_{00},m_{01},m_{11}]$ for $d=2$ |
| $\langle\ln\bar W\rangle$ | stationary time-average of mean log-fitness = long-run growth rate (drives second-order selection) |

## 0.11 Building the individual-based models

The IBMs are standard multilocus quantitative-genetics simulations; references for the architecture: Bürger (2000), *The Mathematical Theory of Selection, Recombination and Mutation*; Jones, Arnold & Bürger (2003, 2004, 2007).

- **Genotype → phenotype.** Each individual carries allelic values at $L$ loci that sum additively to a $d$-dim genotypic value $g$ (pleiotropic loci: each locus contributes to several traits). Phenotype $z = g + e$, $e\sim\mathcal N(0,E)$ with fixed environmental covariance $E$. Diploid, sexual, with recombination (free recombination is simplest) each generation.
- **Selection.** Survival/reproduction probability $\propto W(z)$ from the Gaussian landscape in 0.3.
- **Mutation / M.** Each generation, mutations add increments to the allelic values; set per-locus mutation rates and effect distributions so the **total per-generation input covariance to $g$ equals $M$**. For **fixed-M** runs (V1, V2), $M$ is a fixed parameter.
- **Evolvable M (V3–V6).** $M$ must be encoded by heritable **modifier loci** — genes that control the mutational process of the focal-trait loci. Template: Jones, Arnold & Bürger (2007) made the mutational correlation $r_\mu$ heritable; generalize so modifier loci control the full mutational covariance $M$ (its eigenvalues and orientation). The modifier architecture sets $\mathcal G_M$:
  - For the **natural-gradient (clean) test** — metric-control design (ii) — make modifiers act **additively on the log-eigenvalues of $M$ with equal per-locus variance**, which makes $\mathcal G_M\propto$ Fisher, and compare drift to the natural gradient $-2M\nabla_M\mathcal L_{\text{eff}}M$.
  - For the **general test** — design (i) — use any architecture and **measure** $\mathcal G_M$ from the population.
- **Measuring the quantities.** $\langle\ln\bar W\rangle$ = time-average of $\ln$(population mean fitness) at stationarity; $G$ = sample covariance of genotypic values $g$; $\Sigma_\infty$ = sample covariance of the population **mean** $\bar z$ over time; $\mathcal G_M$ = sample covariance of modifier breeding values; the current $M$ = realized mutational covariance implied by the modifier genotypes.
- **The ensemble drift estimator (V3, V4).** Clone a population at a fixed $M$ with the fast variables equilibrated, run $R$ independent replicates forward $\tau$ generations, and estimate $\widehat{\mathbb E[\Delta M\mid M]} = \tfrac1R\sum_r (M_r(\tau)-M)/\tau$. Settings established in V0: $\tau$ above the fast correlation time, and the per-window travel small (so the population does not drift far during the window). Units/normalizations are free as long as they are consistent between the IBM measurements and the analytic predictions.

---

## The validation as a ladder

The theory is a chain of approximations (Gaussian closure → two-noise SDE for (z̄, G) → adiabatic reduction to ℒ_eff(M) → gradient-flow M-dynamics → higher-order curl). These links fail independently, so the validation is a **ladder** in which each test isolates one link, localizing any discrepancy rather than just flagging "something is wrong." A precursor (V0) needs no simulation and is already complete; V1–V2 need only fixed-M simulations; V3–V6 need the evolvable-M IBM. The central object tested throughout is the predicted slow drift $\mathbb E[\dot M\mid M] = -\mathcal G_M\nabla_M\mathcal L_{\text{eff}}(M)$.

---

## The metric confound (read before designing V3)

What an IBM's M actually moves along is $-\mathcal G_M\nabla\mathcal L_{\text{eff}}$, the load gradient **bent by the realized modifier genetic covariance** $\mathcal G_M$ — the breeder's equation applied to M's own parameters. It is *not* $-\nabla\mathcal L_{\text{eff}}$ unless $\mathcal G_M\propto$ Fisher (the Shahshahani condition). A naive comparison of measured $\Delta M$ against $-\nabla\mathcal L_{\text{eff}}$ can therefore show a spurious mismatch that is pure metric-bending, not a failure of the mechanism. Every drift-direction test must control for this by one of:

- **(i) Measure $\mathcal G_M$.** Estimate the additive genetic covariance of the M-encoding modifier loci from the IBM population (covariance of modifier breeding values), and compare measured drift to $-\mathcal G_M\nabla\mathcal L_{\text{eff}}$ using that estimate. Tests the full M-equation.
- **(ii) Engineer $\mathcal G_M\propto$ Fisher.** Build modifiers that act log-additively on M's eigenvalues with isotropic per-locus variance (the sufficient condition); then the prediction reduces to the natural gradient $-2M\nabla\mathcal L_{\text{eff}}M$ and can be compared directly. Tests the reduced theory in its cleanest form.

Run both: (ii) as the controlled validation of the natural-gradient claim, (i) as the test that the realized metric is what bends the drift.

---

## V0 — De-risk the measurement machinery (no simulation; complete)

Before building any IBM, the ensemble drift estimator that V3 and V4 depend on was validated on a Gaussian-closure surrogate where the true drift $b(M) = -\mathcal G_M\nabla\mathcal L_{\text{eff}}$ is *known by construction* — so this tests the estimator, not the theory. Implementation: `drift_estimator_prototype.py` (d = 2, M parameterized by its vech).

**Established:**
- *Recovery.* The estimator $\widehat{\mathbb E[\Delta M\mid M]} = \tfrac1R\sum_r (M_r(\tau)-M)/(\tau\eta_M)$ recovers the drift **direction** to $\cos>0.999$ with as few as $R\approx200$ replicates.
- *The metric correction is necessary and works.* Comparing measured drift to the bare gradient $-\nabla\mathcal L_{\text{eff}}$ gave $\cos = 0.70$ — a spurious "failure"; comparing to $-\mathcal G_M\nabla\mathcal L_{\text{eff}}$ gave $0.999$. Estimating $\mathcal G_M$ from as few as $N=50$ modifier breeding values already gave $0.99$. The realized-metric estimation V3 relies on is cheap and accurate.
- *The eigenvalue-scaling / eigenvector-rotation split is sound* — $\cos$ 0.999 / 1.000 in the two subspaces separately.
- *The V4 curl detector* reads $\approx 0.00$ for a pure-gradient drift and jumps when a rotational component is injected: a reliable presence/absence detector (it saturates at large curl, so treat it as a detector, not a precise magnitude gauge).

**Design lessons that feed forward into V3/V4:**
- *Window bias–variance (the magnitude trap).* The drift *magnitude* is biased by how far the population travels during the window: as $\eta_M\tau\,|b|$ grew from 0.045 to 1.12, the magnitude error grew 5% → 59% and the direction began to degrade. Keep slow steps small and the window short relative to the scale over which the drift field curves; lengthen $\tau$ only enough to average the fast fluctuations.
- *Window lower bound.* $\tau$ must exceed the fast correlation time — $\tau=3$ gave $\cos=0.985$ (fast jitter not averaged), $\tau=30$ gave $0.9995$.

**Consequence for the ladder:** the measurement layer is sound, so a low metric-corrected cosine against a real IBM (V1/V3) is a signal about the *theory* — the closure or the second-order-selection mechanism — not the estimator. Swapping the surrogate `increment()` for the evolvable-M IBM runs V3/V4 with the same harness.

---

## V1 — The effective potential, directly (fixed-M fitness scan)

The cheapest and most decisive first test; needs only standard Bürger-style fixed-M simulations, no modifier dynamics.

**Procedure.** On a grid of fixed M (vary eigenvalues and orientation relative to A), run each IBM to stationarity and measure the stationary long-run fitness $U_{\text{IBM}}(M) = \langle\ln\bar W\rangle$ (time-average of mean log-fitness). Compare the surface to $-\mathcal L_{\text{eff}}(M)$, and its finite-difference gradient $\nabla U_{\text{IBM}}$ to $-\nabla\mathcal L_{\text{eff}}$. The finite-difference gradient *is* the empirical second-order selection force — your "direction selection on M points" — obtained without M ever evolving.

**Closure diagnostic (the useful refinement).** Compute the prediction two ways: **(A)** $\mathcal L_{\text{eff}}$ using the analytical equilibrium $\hat G = A^{-1}\#M$ (tests closure + substitution + load functional jointly); **(B)** the load $\kappa\,\mathrm{tr}(A\hat G)$ using the IBM's *measured* conditional-equilibrium G at each M (tests only the load functional, given correct fast states). If (B) matches $U_{\text{IBM}}$ but (A) does not, the error is in the MSB closure $\hat G=A^{-1}\#M$, not the second-order-selection mechanism. If neither matches, the load functional itself is wrong.

**Done when:** $U_{\text{IBM}}(M)$ matches $-\mathcal L_{\text{eff}}(M)$ (version A or B) in shape, and $\cos(\nabla U_{\text{IBM}}, -\nabla\mathcal L_{\text{eff}}) > 0.9$ across the grid; the A-vs-B comparison attributes any mismatch.

---

## V2 — The fast equilibria and the Gaussian closure

Validates the substitution that V1's "version A" relies on.

**Procedure.** At fixed M, measure the IBM's realized equilibrium G and the stationary mean-displacement covariance $\Sigma_\infty$. Compare to $\hat G = A^{-1}\#M$ and the closed form $\Sigma_\infty = \tfrac{v}{2n}A^{-1} + \tfrac{1}{2N^*}G$. Do this for **commuting** A and G (aligned eigenbases) and **non-commuting** A and G, since the closed-form $\Sigma_\infty$ is claimed to hold in general. Separately, test the Gaussian closure directly: measure phenotype-distribution skew and excess kurtosis along principal axes.

**Done when:** measured $\hat G$ and $\Sigma_\infty$ match the predictions (relative error small) in both commuting and non-commuting cases; the closure's validity is quantified, with the house-of-cards regime (large per-locus mutational effects) identified as the expected breakdown.

---

## V3 — The M-drift direction (evolvable-M IBM)

The core test of the second-order-selection mechanism. Requires an IBM where M is heritable — modifier loci encoding the mutational architecture (template: Jones, Arnold & Bürger 2007 for $r_\mu$; extend to full M). The main implementation cost of this strand is here.

**Procedure (ensemble drift estimator).** Fix an initial M and let the fast variables (z̄, G) equilibrate. Run many replicate IBMs for a short window τ ≪ M's relaxation time; estimate the conditional drift $\widehat{\mathbb E[\Delta M\mid M]} = \tfrac1{R}\sum_r (M_r(\tau)-M)/\tau$ across $R$ replicates. Repeat at several M values spanning a region of interest (including off-equilibrium points where the drift is large and informative). Compare the drift direction to $-\mathcal G_M\nabla\mathcal L_{\text{eff}}$, controlling for $\mathcal G_M$ by (i) or (ii) above.

**Decompose the drift** into the part that rescales M's eigenvalues (the canalization/scale knob) and the part that rotates its eigenvectors (the alignment knob), and report agreement for each separately — the scaling and rotation predictions are distinct and the literature conflates them.

**Done when:** $\cos(\widehat{\mathbb E[\Delta M]},\, -\mathcal G_M\nabla\mathcal L_{\text{eff}}) > 0.9$ at multiple M, under both metric-control designs, with the eigenvalue-scaling and eigenvector-rotation components each validated; and the regime fixed points (§0.8) are recovered as the drift's zeros.

---

## V4 — Gradient flow versus curl

The definitive test of the gradient-flow structure (the prediction in §0.9).

**Procedure.** Over a grid of M values, assemble the measured drift field $v(M)=\mathbb E[\Delta M\mid M]$ (from V3's estimator). Premultiply by $\mathcal G_M^{-1}$ and Helmholtz-decompose: fit a scalar potential $\phi$ by least-squares ($\nabla\phi\approx \mathcal G_M^{-1}v$) and measure the curl fraction $\|\mathcal G_M^{-1}v - \nabla\phi\|/\|\mathcal G_M^{-1}v\|$. The theory predicts: curl-free to leading order ($\phi\approx\mathcal L_{\text{eff}}$), with a rotational residual that (a) scales as $\varepsilon\times\text{noise}^2$, (b) grows when A and G are misaligned, and (c) vanishes when they are co-diagonalized.

**Done when:** the curl fraction is small in the aligned/large-population regime, and its dependence on misalignment and on noise ($1/N_e$, $1/N^*$) matches the predicted scaling; the recovered potential $\phi$ matches $\mathcal L_{\text{eff}}$.

---

## V5 — Adiabaticity dialing

Validates the reduction's regime of validity and the $\varepsilon$-scaling of the curl.

**Procedure.** Vary the timescale-separation parameter $\varepsilon$ — set in the IBM by the modifier-trait genetic variance relative to the focal traits (modifier mutation rate / effect sizes). Measure (a) the agreement between the reduced SDE drift and the IBM drift (V3 statistic) and (b) the curl fraction (V4), as functions of $\varepsilon$.

**Done when:** the SDE approximation improves monotonically and the curl residual shrinks as $\varepsilon\to0$, with the curl scaling $\propto\varepsilon$ as predicted; the value of $\varepsilon$ at which the approximation degrades to a chosen tolerance is reported as the boundary of the adiabatic regime.

---

## V6 — Full trajectory / distribution match

The strongest and most expensive test; do last, once V1–V5 have validated the pieces.

**Procedure.** Solve the coupled (θ, z̄, G, M) SDE numerically (Euler–Maruyama, with $dt\leftrightarrow$ one generation and all noise covariances matched to the IBM's $N_e$, $N^*$). Compare M(t) and G(t) from the SDE ensemble to the IBM ensemble: marginal moments over time, and a distributional distance (e.g. sliced-Wasserstein on the M-eigenvalue trajectories). Report agreement in each of the three regimes (static, tracking, fluctuating).

**Done when:** SDE and IBM ensembles agree in mean and covariance of M(t), G(t) within sampling error across the regimes, in the adiabatic parameter range mapped by V5.

---

## What failure would tell us

- **V1 mismatch, but (B) matches:** the load functional is right; the MSB closure $\hat G=A^{-1}\#M$ is off — likely the Gaussian closure (check against V2's skew/kurtosis).
- **V1 mismatch, neither version:** the effective load functional itself is wrong; revisit the cost/benefit accounting in §0.7–0.8.
- **V3 mismatch after controlling for $\mathcal G_M$:** the second-order-selection mechanism (natural-gradient-on-M) is wrong, not merely metric-bent — the most consequential possible result.
- **V4 large curl in the aligned/large-N regime:** the gradient-flow claim fails where it shouldn't; the dynamics have a rotational component the reduction misses.
- **V2/V5 breakdown in the house-of-cards regime:** the theory is valid only in the Gaussian/clustered-continuum regime — a *bound* on applicability, not a refutation, and exactly where the $A^{-1/2}$-vs-$A^{-1}$ eigenvalue-scaling story is most delicate, so worth knowing precisely.

---

## Build order and connection to the spine

1. **V0** (estimator de-risk, no simulation) — **complete**: the drift estimator and the metric-correction machinery are validated, so any downstream discrepancy attributes to the theory rather than to the measurement.
2. **V1** (fixed-M scan) and **V2** (fast equilibria + closure) — cheap, decisive, need only standard simulations; run first and they immediately tell you whether the effective potential and the substitution are trustworthy.
3. **V3** (evolvable-M IBM, drift direction) — the core mechanism test; build the modifier architecture under both metric-control designs.
4. **V5** (adiabaticity dialing) — maps the regime of validity; cheap once the IBM exists.
5. **V4** (curl decomposition) — the distinctive structural test; needs the V3 drift field on a grid.
6. **V6** (full trajectory match) — last and strongest.

Connection to the larger program: its central claim is a single cross-substrate law — the optimal "preconditioner exponent" $\alpha$ (in $G\propto A^{-\alpha}$), equivalently the evolved shape of $M$, tracks the effective sample size ($N^*$ on the evolutionary side, minibatch size on the optimizer side). That law is only credible if the SDE carrying the prediction is itself shown to approximate the discrete dynamics it summarizes. V0–V6 are what license treating the evolutionary simulations as a faithful reading of the SDE.

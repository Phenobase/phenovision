# Landscape perturbation and the benefit to M: moving optima, random disasters, and the anisotropy flip

*A working note deriving the second-order-selection benefit on the mutational covariance $M$ from an
explicit generative model of how the fitness landscape changes in time. Extends
`notes_landscape_fluctuation_benefit.md` by (a) replacing the exogenous untrackable-fluctuation
covariance $\Sigma_x$ with a derived functional of a stochastic-process model of the optimum, and
(b) adding the **fixed-shape / time-perturbed** ("random disasters and boons") model, which the
earlier note treated only through a posited $\log\det M$ term. The central new result is a **sign
flip** in the evolved $M$ anisotropy — mild perturbations select $M$ concentrated in the steep
directions ($M\propto A^{+}$), severe/catastrophic ones select $M\propto A^{-1}$ — with the
crossover set by the objective (arithmetic vs geometric mean), which is set by perturbation
severity, tail, and population size. All closed forms below are checked numerically (`§7`).*

---

## 0. Scope and relation to prior work

Two generative models of "a changing landscape," sharing a fixed curvature $A$:

- **Model I — moving optimum.** The optimum $\theta_t$ wanders smoothly (autocorrelated). The mean
  $\bar z$ tracks the slow part; $M$ benefits only from the **untrackable** (fast) residual. This is
  the spectral-filter picture; the closed form for the residual is derived in `§2`.
- **Model II — fixed shape, perturbed in time.** $\theta_t = \theta_0 + \xi_t$ with $\xi_t$
  **i.i.d. across generations** (white): a fresh "disaster" (or boon — *disastertunity*) each
  generation, drawn with covariance $\Omega$. The population cannot track it at all. This is the
  $\phi\to\infty$ (zero-correlation-time) limit of Model I, and it is where the bet-hedging /
  $M\propto A^{-1}$ regime lives. Derived in `§3`.

**Prior art this must be positioned against (attribution).** That a fluctuating environment favours
standing genetic variation is **Lande & Shannon (1996)**; the moving/stochastic-optimum
quantitative-genetics tradition is Lynch & Lande (1993), Bürger & Lynch (1995), Bürger (1999), Tufto
(2000, 2015), Charlesworth (1993), reviewed for moving optima by Kopp & Matuszewski (2014); the
geometric-mean / bet-hedging machinery is Gillespie (1974), Frank & Slatkin (1990), and Lande's
stochastic-demography line (Lande, Engen & Sæther 2003). The OU-optimum object is identical to the
phylogenetic Ornstein–Uhlenbeck of Hansen & Martins (1996) (already in the project bibliography).
**What is not standard, and is the contribution here:** (i) treating adaptive tracking as a
**linear filter** whose bandwidth is set by the standing variance, so that $\Sigma_x$ becomes an
explicit functional of the optimum's spectrum and of $G$ (hence of $M$) — making the M-optimization
self-consistent rather than one-shot; (ii) the explicit **arithmetic→geometric anisotropy flip**
($M\propto A^{+}$ vs $M\propto A^{-1}$) and its mapping to the cross-substrate preconditioner
exponent; (iii) deriving *which* benefit functional ($\log\det M$ vs trace) the perturbation
statistics select. The basic "variation is favoured under fluctuation" result is **not** claimed as
novel.

---

## 1. Common setup

Population Gaussian, $z\sim\mathcal N(\bar z, G)$, with $G=\hat G(M)=A^{-1}\#M$ the fast-equilibrium
standing variance (matrix geometric mean; aligned $g_i=\sqrt{m_i/a_i}$). Landscape Gaussian,
$W_t(z)=\exp\!\big(-\tfrac12(z-\theta_t)^\top A\,(z-\theta_t)\big)$, $A$ fixed SPD. The
population-mean fitness is the Gaussian integral

$$\log\bar W_t = -\tfrac12\log\lvert I+GA\rvert \;-\; \tfrac12\,x_t^\top \tilde A\,x_t,
\qquad x_t \equiv \bar z-\theta_t,\quad \tilde A \equiv A(I+GA)^{-1}. \tag{1}$$

Two loads: the **variance load** $\tfrac12\log\lvert I+GA\rvert$ (cost of being spread; the
canalization pressure) and the **lag load** $\tfrac12 x_t^\top\tilde A x_t$ (cost of the mean being
off the displaced optimum). Note $\tilde A=A(I+GA)^{-1}$ is the *effective* curvature seen by the
lag, **reduced** by having more standing variance: a spread population is hurt less by a given lag,
because some individuals are always near the optimum. That reduction is the seed of every benefit
below.

Second-order selection on the slow modifier $M$ maximizes the long-run growth rate, the time-average
of $\log\bar W_t$:

$$\Lambda(M)=\big\langle \log\bar W_t\big\rangle_t. \tag{2}$$

We write everything as $\mathcal L_{\text{eff}}(M)=\text{const}-\Lambda(M)=\kappa\,\mathrm{tr}\!\big(A\hat G(M)\big)-\mathcal B(M)$
and the job is to **derive $\mathcal B(M)$** for each landscape model.

---

## 2. Model I (moving optimum): adaptation as a tracking filter

The mean follows the optimum by the breeder's equation $\dot{\bar z}=-GA\,(\bar z-\theta_t)$, so the
lag $x=\bar z-\theta$ obeys the first-order linear filter

$$\dot x + GA\,x = -\dot\theta. \tag{3}$$

In the aligned (commuting) case, per mode with loop gain $h_i\equiv g_i a_i$, the lag transfer
function is the **high-pass** $\lvert H_i(\omega)\rvert^2=\omega^2/(\omega^2+h_i^2)$, and the
stationary residual variance is the filtered spectrum of the optimum:

$$\sigma_{x,i}^2=\int \frac{\omega^2}{\omega^2+h_i^2}\,S_{\theta,i}(\omega)\,\frac{d\omega}{2\pi}.
\tag{4}$$

This is the formal version of "trackable vs untrackable": slow components ($\omega\ll h_i$) are
suppressed ($\lvert H\rvert^2\to0$, no lag, no benefit to $M$); fast components ($\omega\gg h_i$)
pass through fully. For an **OU optimum** (per-mode reversion rate $\phi_i$, stationary variance
$\varsigma_i^2$, spectrum $S_{\theta,i}=2\phi_i\varsigma_i^2/(\omega^2+\phi_i^2)$), (4) integrates in
closed form (confirmed exactly, `§7.1`):

$$\boxed{\;\sigma_{x,i}^2=\varsigma_i^2\,\frac{\phi_i}{\phi_i+h_i}=\varsigma_i^2\,\frac{\phi_i}{\phi_i+g_i a_i}\;}\tag{5}$$

The corner is the ratio of the optimum's wander rate to the loop gain. Fast optimum
($\phi_i\gg g_ia_i$): $\sigma_{x,i}^2\to\varsigma_i^2$ — the population eats the full variance of the
optimum as lag, **nothing is trackable**. Slow optimum ($\phi_i\ll g_ia_i$):
$\sigma_{x,i}^2\to\varsigma_i^2\phi_i/(g_ia_i)\to0$ — lag suppressed by the timescale ratio. The
dependence on $g_i$ (hence on $M$ through $\hat G$) makes the benefit **self-consistent**: more
standing variance widens the tracking band, which shrinks the residual it then has to absorb.

Substituting $\Sigma_x=\mathrm{diag}(\sigma_{x,i}^2)$ into (1)–(2) gives the Model-I growth rate

$$\Lambda(M)\approx-\tfrac12\log\lvert I+GA\rvert-\tfrac12\,\mathrm{tr}\!\big(\tilde A\,\Sigma_x(G)\big),\tag{6}$$

whose first-order optimization over $G$ is the earlier note's **Channel 1**: the standing variance
aligns with the untrackable-fluctuation covariance, $g_i^\star=\sigma_{x,i}^2-1/a_i$ (floored),
i.e. $M\propto\Sigma_x$ for an anisotropic untrackable spectrum. The new content is that
$\Sigma_x$ is now a derived functional (5) of the OU parameters and of $G$, not an input. **Model II
is the $\phi_i\to\infty$ limit**, $\sigma_{x,i}^2\to\varsigma_i^2$ (write $\Omega\equiv\mathrm{diag}(\varsigma_i^2)$):
a fully untrackable white perturbation. We treat that limit in detail next, because the bet-hedging
regime lives there.

---

## 3. Model II (random disasters / boons): the fixed-shape, time-perturbed landscape

$\theta_t=\theta_0+\xi_t$, $\xi_t\stackrel{\text{iid}}{\sim}$ a distribution with covariance
$\Omega$ (the **disaster covariance**: its shape is *which directions get hit*, its scale is
*severity*; "boons" are simply the favourable tail, and for a symmetric $\xi$ distribution the
analysis is unchanged — only the variance enters at leading order). Because $\xi_t$ is i.i.d.,
$\bar z$ cannot follow it; $\bar z$ sits at $\theta_0$ and the per-generation lag is $x_t=-\xi_t$
with covariance $\Omega$. Standing variance no longer helps by *tracking* — it helps only by
*coverage* (a spread cloud has individuals near wherever the optimum lands), through the
$\tilde A=A(I+GA)^{-1}$ reduction in (1).

### 3.1 The arithmetic-mean regime → $M\propto A^{+}$ (the surprise)

For relative fitness in a large population, the long-run growth rate is exactly (2) and, since
$\log\bar W_t$ is **linear** in the fluctuating quadratic $L_t=\tfrac12\xi_t^\top\tilde A\xi_t$,

$$\Lambda_{\text{arith}}(M)=-\tfrac12\log\lvert I+GA\rvert-\tfrac12\,\mathrm{tr}\!\big(\tilde A\,\Omega\big).\tag{7}$$

Optimizing per aligned mode, $f(g)=-\tfrac12\log(1+ga)-\tfrac12\,\omega\,a/(1+ga)$ with
$\partial f/\partial g=0$:

$$1+g_i a_i=\omega_i a_i \;\Longrightarrow\; g_i^\star=\omega_i-\frac1{a_i}\quad(\text{floored at the mutational floor}).\tag{8}$$

**Anisotropic $\Omega$:** $G$, and hence $M$, aligns with $\Omega$ — track the disaster direction.
**Isotropic $\Omega=\sigma^2 I$:** $g_i^\star=\sigma^2-1/a_i$ is *near-isotropic* standing variance.
But the **mutational input** that maintains it is not isotropic. From mutation–selection balance
$m_i=a_i g_i^{\,2}$ (the aligned form of $GAG=M$):

$$m_i^\star=a_i\big(\sigma^2-\tfrac1{a_i}\big)^2\;\xrightarrow[\ \sigma^2\ \text{large}\ ]{}\;a_i\sigma^4\;\propto\;a_i=A.\tag{9}$$

So **isotropic disasters select $M\propto A$** — mutation *concentrated in the steep directions*.
The mechanism is pure MSB bookkeeping: to hold a fixed standing variance against selection, which
erodes variance at rate $\propto a_i$ (faster where steeper), you must replenish faster where
steeper. The intuition "isotropic environment $\Rightarrow$ isotropic $M$" is wrong; isotropic
*standing variance* needs $\propto A$ *mutational input*. Numerically (`§7.2`) the slope of
$\log m$ vs $\log a$ is positive throughout, $\to+1$ in the strong-disaster regime; when disasters
are **weak** ($\sigma^2<1/a_i$ in the flat directions), those directions fall to the floor
(partial **canalization**) and the steep directions retain variance, so $M$ is even *more*
concentrated in the steep directions (slope $\gg1$).

This refines — and partly corrects — the earlier note. The earlier Channel-1/Channel-2 split sent
isotropic $\Omega$ to $M\propto A^{-1}$, but that used the bet-hedging objective implicitly. Under
the **same arithmetic objective**, isotropic $\Omega$ gives $M\propto A$. The isotropic case is
precisely where the two objectives *diverge*, and that divergence is the diagnostic of `§5`.

### 3.2 The second cumulant does not rescue $A^{-1}$

The geometric-mean correction is $\Lambda\approx\langle r\rangle-\tfrac12\mathrm{Var}_t(r)$ with
$r=\log\bar W_t$. Using $\mathrm{Var}(\xi^\top S\xi)=2\,\mathrm{tr}(S\Omega S\Omega)$ for
$\xi\sim\mathcal N(0,\Omega)$,

$$\mathrm{Var}_t(r)=\tfrac12\,\mathrm{tr}\!\big((\tilde A\Omega)^2\big),\qquad
\Lambda_{\text{geom}}(M)=-\tfrac12\log\lvert I+GA\rvert-\tfrac12\mathrm{tr}(\tilde A\Omega)-\tfrac14\mathrm{tr}\!\big((\tilde A\Omega)^2\big).\tag{10}$$

The variance term *does* reward more standing variance (it shrinks $\tilde A$, more valuably where
$a_i$ is large), but optimizing (10) per isotropic mode gives $u\equiv1+ga=\sigma^2 a\cdot\varphi$
with $\varphi=(1+\sqrt5)/2$ — still **near-isotropic $g$**, hence still $M\propto A$. The Gaussian
cumulant ladder, to any finite order, does not produce $M\propto A^{-1}$. That regime is
**non-perturbative**: it requires the geometric mean to be dominated by rare catastrophic
generations.

### 3.3 The catastrophe regime → $\log\det M$ benefit → $M\propto A^{-1}$

What flips the sign is a hard survival floor with **heavy-tailed, directionally-random** disasters
in a **finite population**. Model survival as viability: a disaster displaces the optimum by
$\delta=R\,n$ (direction $n$ isotropic on the sphere, magnitude $R$ heavy-tailed); the lineage
persists only if it has an offspring near $\theta_0+\delta$. With offspring cloud
$\mathcal N(\theta_0,\Sigma)$, the expected number of survivors is $\propto$ the cloud density at
$\delta$, so

$$\log(\text{surviving offspring}) \approx \text{const} -\tfrac12\log\det\Sigma -\tfrac12\,\delta^\top\Sigma^{-1}\delta.\tag{11}$$

The lineage's long-run growth is $\langle\log(\text{survivors})\rangle$, but with a floor: when the
expected count drops below $\sim1$ the lineage goes extinct (in finite $N$ a single near-zero
generation is unrecoverable). For **heavy-tailed** $R$ the geometric mean is dominated by the rare
huge displacements, and the controlling quantity is the worst-covered direction — the
$n$ maximizing $n^\top\Sigma^{-1}n$, i.e. the **smallest eigenvalue of $\Sigma$**. Survival then
demands *no direction be left uncovered*: a direction-blind premium on keeping every eigenvalue of
$\Sigma$ away from zero. The leading-order log-barrier enforcing that is $-\log\det\Sigma$, so the
benefit, expressed on the heritable injected spread $M$, is the **entropy / log-determinant** term

$$\boxed{\;\mathcal B(M)=\tfrac{\lambda}{2}\log\det M\;}\qquad
\lambda \;\sim\; (\text{disaster rate})\times(\text{tail heaviness})\times \tfrac1N.\tag{12}$$

Balanced against the mutation-load cost $\kappa\,\mathrm{tr}(A\hat G)=\kappa\sum_i\sqrt{a_i m_i}$:

$$\frac{\partial}{\partial m_i}\Big[\kappa\sqrt{a_i m_i}-\tfrac{\lambda}{2}\log m_i\Big]=0
\;\Longrightarrow\; m_i^\star=\frac{\lambda^2}{\kappa^2}\,\frac1{a_i}\;\propto\;A^{-1}.\tag{13}$$

This is **robust to the cost form** — the linear cost $\tfrac12\mathrm{tr}(AM)$ gives the identical
$M\propto A^{-1}$ — because the $\log\det$ derivative $1/m_i$ is scale-free (confirmed numerically,
`§7.2`, slope $-1.000$ for both costs). The $\log\det$ benefit is **the** ingredient that produces
the full inverse-curvature regime, and it is genuinely separate from anything the arithmetic or
finite-cumulant analysis produces.

**Why this is a third noise source.** This premium is not random *which individuals* die (drift,
$\propto1/N_e$) nor random *gradient estimation* (within-generation sampling, $\propto1/N^*$). It is
random *which trait combinations are unlucky this generation*, and it acts only at the
geometric-mean / lineage level. It is invisible to the first-order arithmetic analysis, which is
exactly why `§3.1` and `§3.3` give opposite answers.

### 3.4 Why the IBM lands on $A^{-1}$ — and a falsification

The evolvable-M IBM, with its "shared random-disaster challenge," reached $m_{\text{flat}}/m_{\text{steep}}\approx4$
for isotropic $\Omega$ (the $A^{-1}$ ratio for $A=\mathrm{diag}(1,4)$). By the analysis above, that
places the IBM's challenge **in the catastrophe / finite-$N$ regime** (`§3.3`), not the
arithmetic-mean regime (`§3.1`). The framework therefore makes a sharp, testable prediction: soften
the IBM's disaster toward **mild Gaussian fecundity selection** and raise $N$, and the isotropic
result should **flip** from $M\propto A^{-1}$ toward $M\propto A$ — the ratio $m_{\text{flat}}/m_{\text{steep}}$
should cross $1$ and invert. If it does not, the catastrophe mechanism for $A^{-1}$ is wrong, or the
IBM's $A^{-1}$ comes from somewhere the analysis misses. (Test T1/T2 in `§8`.)

---

## 4. The unified picture: a cumulant/severity ladder

One growth-rate functional, with terms switched on by the perturbation statistics:

| Term in $\Lambda(M)$ | Statistical origin | Pull on $M$ |
|---|---|---|
| $-\tfrac12\log\lvert I+GA\rvert$ | zeroth order (cost of spread) | $\to$ floor (**canalization**) |
| $-\tfrac12\mathrm{tr}(\tilde A\Sigma_x)$ | 1st cumulant (mean lag load) | $M\propto\Sigma_x$ (anisotropic) / $M\propto A$ (isotropic) — **track / maintain** |
| $-\tfrac14\mathrm{tr}((\tilde A\Sigma_x)^2)$ | 2nd cumulant (temporal fitness variance) | more standing variance in steep directions; does **not** reach $A^{-1}$ |
| $+\tfrac{\lambda}{2}\log\det M$ | tail / catastrophe (non-perturbative, finite $N$) | $M\propto A^{-1}$ — **bet-hedge** |

The **untrackable spectrum** sets $\Sigma_x$ (Model I, eq. 5; white limit $\Sigma_x=\Omega$); the
**directional structure** of $\Omega$ chooses anisotropic-track vs isotropic; the **severity/tail/$N$**
choose arithmetic ($A^{+}$) vs catastrophe ($A^{-1}$). The within-generation gradient-noise channel
(Force 2 / "Channel 0"), which **whitens** toward $g_i\propto a_i^{-1/2}$ ($M\propto A^{-1/2}$),
sits orthogonal to this axis and is handled separately (Sim B; it is an errors-in-variables
compression, not a landscape-fluctuation benefit, and its exponent is cost-form dependent here so it
is not re-derived from coverage).

The evolved exponent in $G\propto A^{-\alpha}$ (equivalently the $M$ anisotropy) therefore sweeps a
full range as the perturbation statistics change: **track/maintain ($\alpha<0$, $M\propto A$)** —
**canalization (floor)** — **whitening ($\alpha=\tfrac12$, within-gen noise)** — **bet-hedge
($\alpha=1$, $M\propto A^{-1}$)**.

---

## 5. The cross-substrate reading (why this is the same object as the optimizer's $\alpha^\ast$)

The exponent $\alpha$ is a readout of the dominant selective mechanism, which is a readout of the
noise statistics — the *same* statement on both substrates:

| regime | landscape statistics (biology) | optimizer analog |
|---|---|---|
| track / maintain ($M\propto A$) | mild, anisotropic, trackable perturbation | preconditioner emphasizes the clean signal directions (low effective $\alpha$) |
| canalization (floor) | static optimum, no premium on variation | converged; preconditioner stops moving |
| whitening ($A^{-1/2}$) | within-generation gradient noise | small-batch SGD; whitening is the stable point |
| bet-hedge ($A^{-1}$) | severe, heavy-tailed, directionless catastrophe; small $N$ | the exploration / posterior-sampling regime; full inverse is *correct* (and unstable as an optimizer) |

This is the mechanistic resolution of the project's organizing prediction — the optimal
preconditioner exponent decreases as effective sample size falls — into *named, separable* regimes,
and it tells us **which knob does what**: severity/tail/$N$ move you along the track $\to$ hedge
axis; within-generation sampling noise pulls orthogonally toward whitening. The biological
$\alpha^\ast(\text{statistics})$ surface this implies is the one to overlay on the optimizer's owed
$\alpha^\ast(\text{batch})$ sweep.

---

## 6. Scope ledger

**Derived (clean, numerically confirmed):**
- The OU tracking-filter residual $\sigma_{x,i}^2=\varsigma_i^2\,\phi_i/(\phi_i+g_ia_i)$, eq. (5).
- The arithmetic-mean optimum $g_i^\star=\omega_i-1/a_i$ and its consequence $M\propto A$ for
  isotropic $\Omega$, eqs. (8)–(9).
- The non-rescue by the 2nd cumulant, eq. (10).
- $\log\det M$ benefit $\Rightarrow M\propto A^{-1}$, robust to cost form, eqs. (12)–(13).

**Modeling choices (defensible but assumption-dependent):**
- The Gaussian population / Gaussian landscape closure (shared with the rest of the framework).
- Treating $\xi_t$ as symmetric (boons = the favourable tail; asymmetry enters only beyond leading
  order — worth a separate check if "disastertunities" are meant to be skewed).
- The viability/coverage survival model of `§3.3` (vs the fecundity/lag model of `§3.1`); the IBM
  result implies the catastrophe model is the operative one there, but this is an inference, T1/T2.

**Open (flagged for derivation or simulation):**
- The **functional form and coefficient $\lambda$** of the catastrophe benefit from a *specified*
  disaster tail. The endpoints are clean ($\log\det M\Rightarrow A^{-1}$); the interpolation from a
  light tail (no log-barrier; arithmetic regime, $A^{+}$) to a heavy tail (log-barrier; $A^{-1}$) is
  heuristic. A rigorous large-deviation derivation of the barrier for, e.g., a Student-$t$ or
  Pareto displacement magnitude is the missing piece.
- The finite-$N$ scaling of $\lambda$ (the $1/N$ in eq. 12 is asserted from the
  geometric-mean-with-demographic-variance heuristic, not derived).
- The **self-consistent** Model-I optimum: $\Sigma_x(G)$ depends on $G=\hat G(M)$, so the
  $M$-optimum of (6) with (5) substituted is a fixed point, not yet solved in closed form.

---

## 7. Numerical confirmations (already run)

**7.1 OU integral.** Direct quadrature of $\int|H_i|^2 S_{\theta,i}\,d\omega/2\pi$ against
$\varsigma^2\phi/(\phi+h)$ over $(\phi,\varsigma^2,h)\in\{(1,1,0.5),(2,3,4),(0.3,1,5),(5,2,0.2)\}$:
ratio numeric/predicted $=1.0000$ in every case. Eq. (5) is exact.

**7.2 Anisotropy by regime** (slope of $\log m$ vs $\log a$, $a\in[0.25,16]$, isotropic $\Omega$):
- **Lag / arithmetic** (optimize $g$, then $m=ag^2$): slope $+8.0$ at $\sigma^2=0.5$, $+6.8$ at
  $\sigma^2=2$, $+1.27$ at $\sigma^2=8$ — positive throughout, $\to+1$ as disasters strengthen,
  steeper under flooring. $M\propto A^{+}$ confirmed.
- **Bet-hedge** ($\tfrac{\lambda}{2}\log\det M$ minus cost): slope $-1.000$ for **both** the
  $\sqrt{a m}$ cost and the linear $a m$ cost. $M\propto A^{-1}$ confirmed, cost-form-robust.
- **Coverage** ($-\tfrac12\mathrm{tr}(M^{-1}\Omega)$ minus cost): slope $-0.500$ (linear cost),
  $-0.333$ ($\sqrt{}$ cost) — **cost-form dependent**, which is why $A^{-1/2}$ is attributed to the
  within-generation channel (Sim B), not derived here.

Verification script: `verify_landscape_disaster.py` (reproduces all of the above with
`numpy`/`scipy`).

---

## 8. Tests and analyses

All re-use the existing `evolvable_m.py` IBM and the `test_landscape_channels.py` harness; only the
challenge generator and sweep axes change. Each has a definite "done when."

**T1 — The anisotropy flip (the headline test).** Isotropic $\Omega$; sweep disaster **severity**
$\sigma$ (small$\to$large) crossed with **tail** (Gaussian fecundity vs hard-threshold heavy-tailed
viability) and **$N$**. *Predict:* $m_{\text{flat}}/m_{\text{steep}}$ moves from $>1$ ($M\propto A^{-1}$,
catastrophe regime) to $<1$ ($M\propto A$, arithmetic regime) as the challenge softens / $N$ grows.
*Done when:* the ratio crosses $1$ along the severity/tail/$N$ axis and the crossing location is
reported, with the arithmetic-regime slope $\to+1$ and the catastrophe-regime slope $\to-1$ recovered
at the extremes.

**T2 — Diagnose the existing $A^{-1}$ result.** Confirm the current IBM challenge is heavy-tailed /
threshold (read the mechanism in `evolvable_m.py`); then replace it with mild Gaussian fecundity
selection at large $N$ and check the isotropic result flips to $M\propto A$. *Done when:* the same
codepath produces $A^{-1}$ (catastrophe) and $A^{+}$ (mild) under the two challenge types — a direct
internal falsification of `§3.3`.

**T3 — Calibrate $\lambda$.** In the catastrophe regime, measure the realized $\log\det$-benefit
coefficient (fit the evolved $M$-spectrum to eq. 13) as a function of disaster rate, tail index, and
$N$. *Predict:* $\lambda$ increases with rate and tail-heaviness and with $1/N$. *Done when:* the
monotonic dependence on all three is established (a scaling-law fit, not necessarily the exact form).

**T4 — Model I $\to$ Model II as a correlation-time limit.** Drive the optimum as OU with
correlation time $\tau_c=1/\phi$; sweep $\tau_c$ from long (trackable) to $\to0$ (white). *Predict:*
the measured residual lag variance matches eq. (5) and $\to\Omega$ as $\tau_c\to0$; the evolved $M$
interpolates from canalization/track (slow optimum, the mean absorbs it) to the disaster regime.
*Done when:* eq. (5) is recovered across $\tau_c$ and the two models join continuously.

**T5 — The variance term.** Measure $\mathrm{Var}_t(\log\bar W_t)$ directly and check it tracks
$\tfrac12\mathrm{tr}((\tilde A\Omega)^2)$ and shrinks with $G$. *Done when:* the prediction matches
within sampling error across a $G$-range, confirming the 2nd-cumulant accounting of `§3.2`.

**T6 — Cross-substrate overlay.** Put the IBM severity sweep (T1) and the optimizer
$\alpha^\ast(\text{batch})$ sweep on one $\alpha^\ast(\text{noise})$ axis. *Done when:* both arms,
plus the SDE prediction, fall on a single curve (or the discrepancy is localized to a named regime).

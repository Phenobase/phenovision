# Annealing demographic noise in an evolution-inspired optimizer

*Self-contained theory note for an agent implementing the optimizer. Question it answers:
injected **demographic noise** impairs optimization (the loss plateaus), while **batch-sampling
gradient noise** does not — what is the principled, "evolutionary-robust" way to anneal the
demographic noise? Conclusion first, then derivations + numerics, then implementation. All numeric
claims are reproduced by `verify_demographic_noise.py`.*

---

## 0. Conclusion first (what to implement, and what not to)

Two facts decide everything.

1. **Gradient noise self-anneals; injected demographic noise does not.** The minibatch
   gradient-noise covariance is proportional to the loss (plus a small irreducible floor), so it
   vanishes as you converge. Demographic noise, injected at a fixed scale, has no loss dependence.
2. **The demographic-noise loss floor is $d/(4N_e)$ and is *independent of the preconditioner* $G$**
   (its scale and its shape). This is the framework's G-neutrality result, and it is not subtle —
   the noise and the selection response both scale with $G$, so $G$ cancels in the stationary loss.

Therefore:

- **Do NOT anneal demographic noise by canalizing/shrinking $G$ or decaying the learning rate.** The
  floor is G-neutral; shrinking $G$ merely *freezes* the iterate at (or above) the same floor rather
  than lowering it. Verified: decaying the $G$ scale leaves the floor at $\approx d/(4N_e)$ and a
  faster decay is slightly *worse* (it freezes earlier, at a less-converged point).
- **DO anneal it by lowering the temperature $T = 1/(2N_e)$**, by either:
  - **(A) loss-matched injection** — scale the injected demographic-noise *amplitude* $\propto
    \sqrt{\hat L(t)}$ (a running loss EMA). This reproduces, by hand, the self-annealing that makes
    gradient noise benign. Verified: drives the loss to $\approx 0$.
  - **(B) growing the effective population size $N_e$** over training (your N-coupling idea). This is
    the *self-annealing* ecological route, and it is already supported by Week's framework; in ML it
    is batch-size growth.
- **Keep canalization for what it actually is**: step-size / search-radius control (the scale of $G$
  is the learning rate), a separate and legitimate knob — just not the demographic-noise annealer.

---

## 1. The two noises (the SDE you are emulating)

The mean-trait / parameter SDE in the two-noise framework (extending Week 2026; see
`extended_joint_sde.md`) is

$$d\bar z = G\beta\,dt \;+\; \underbrace{\sqrt{\tfrac{v}{n}}\,\sqrt{G}\,dB}_{\text{demographic}}\;+\;\underbrace{\tfrac{1}{\sqrt{N^\ast}}\,G\sqrt{C}\,dW}_{\text{gradient}},\qquad \beta=-A(\bar z-\theta^\ast).$$

- **Demographic noise**: covariance $(v/n)\,G = G/N_e$ (with $N_e \equiv n/v$, the Wright effective
  size; $n$ = abundance, $v$ = reproductive variance). **Linear in $G$**, lives in breeding-value
  space. This is the multivariate Lande (1976) drift term. *This is the noise you are injecting.*
- **Gradient noise** (minibatch sampling): covariance $G\,C\,G/N^\ast$, with $C$ the per-example
  gradient covariance and $N^\ast$ the batch size. **Quadratic in $G$** (a sandwich), lives in
  selection space.

The two carry different powers of $G$ because demographic noise originates in breeding-value space
(linear) while gradient noise is reshaped by $G$ acting as an operator (quadratic). This is the
crux of everything below.

---

## 2. Why gradient noise self-anneals and demographic noise does not

**Gradient noise is loss-proportional.** In the locally-quadratic model the per-example gradient
covariance is

$$C(\delta) \;=\; \underbrace{2A\delta\delta^\top A + (\delta^\top A\delta)\,A}_{\text{loss-proportional: } \propto L(\delta)} \;+\; \underbrace{\sigma^2_\varepsilon A}_{\text{irreducible floor}},\qquad \delta=\bar z-\theta^\ast.$$

The first group scales with the loss $L=\tfrac12\delta^\top A\delta$ and $\to 0$ as you converge,
leaving only the small label-noise floor $\sigma^2_\varepsilon A$. So the gradient noise *shrinks
with the loss on its own*.

There is a second, independent reason it anneals: in a discretized optimizer the SGD/gradient noise
enters through the step-size-scaled update, so its stationary contribution scales with the learning
rate, $\propto \eta/N^\ast$ (the Mandt–Hoffman–Blei "SGD temperature"). Decaying $\eta$ anneals it.
So gradient noise has **two** annealing channels: loss-proportional $C$, and $\propto\eta$.

**Demographic noise has neither.** Its covariance $G/N_e$ has no $\delta$ or loss dependence, and as
a proper diffusion (noise variance $\propto dt = \eta$ per step, the same power as the drift) its
stationary contribution is **$\eta$-independent**. Injected at fixed $G,N_e$, it is a constant
temperature — exactly why it plateaus the loss where gradient noise would not.

---

## 3. The G-neutrality result (why canalization cannot anneal it)

Take demographic noise alone and solve the OU stationary variance. Drift $K=GA$ (aligned, $k_i=g_ia_i$),
noise covariance $G/N_e$ (per-mode $g_i/N_e$). The Lyapunov balance $K\Sigma+\Sigma K^\top = G/N_e$ gives

$$\Sigma_i=\frac{g_i/N_e}{2\,g_i a_i}=\frac{1}{2 N_e a_i}\quad\Longrightarrow\quad L_{\text{floor}}=\tfrac12\sum_i a_i\Sigma_i=\boxed{\;\frac{d}{4N_e}\;}$$

**$g_i$ cancels** — the drift ($g_ia_i$) and the noise ($g_i/N_e$) both carry $g_i$. So the
demographic loss floor is independent of $G$, in both scale and shape. (This is stated in the
framework as *"pure stabilizing selection at a static peak under demographic noise alone is
G-neutral"*, `extended_joint_sde.md` §7; it is also why JAB 2003's baseline shows no preferred $G$
orientation.) The stationary distribution is the Gibbs / Sella–Hirsh free-fitness form

$$\pi(\delta)\propto \bar W^{\,2N_e}=\exp(-2N_e\,L),\qquad T=\tfrac{1}{2N_e},$$

i.e. demographic noise is the **thermodynamically clean** noise — it samples $\exp(-L/T)$ at
temperature $1/(2N_e)$. You anneal it by lowering $T$, i.e. raising $N_e$.

**Numerical confirmation** (`verify_demographic_noise.py`, $d=16$, $N_e=50$, predict floor $0.08$):
- Floor vs $G$ scale $\in\{0.3,1,3\}$: $0.087, 0.085, 0.086$ — **flat** (G-neutral).
- Floor vs $N_e\in\{25,50,100,200\}$: $0.170, 0.085, 0.043, 0.021$ vs predicted $0.16,0.08,0.04,0.02$
  — clean $1/N_e$.
- **Decaying the $G$ scale does not help**: $0.085$ (no decay) $\to 0.087,0.092,0.102$ as the decay
  gets faster. Shrinking $G$ freezes the iterate at $\sim$the floor; it never goes below it, because
  the *quasi-stationary* variance the iterate is always relaxing toward is $d/(4N_e)$ regardless of
  $G$. Freezing early just locks in a worse, less-converged value.

So canalization / LR decay is the wrong lever for this floor, full stop.

---

## 4. The schedules that work

### 4a. Route A — loss-matched injection (lower $T$ directly; "evolutionary-robust")

Make the demographic temperature track the loss, $T(t)\propto \hat L(t)$, by scaling the injected
**amplitude** $\propto\sqrt{\hat L(t)}$ (variance $\propto \hat L$). Then the demographic noise
vanishes as the loss does — by construction the same self-annealing gradient noise gets for free.
This is "evolutionary-robust" in the sense that it makes the injected drift behave like the
sampling noise the dynamics already tolerates. Verified: loss-matched injection drives the loss to
$\approx 0$ where constant injection plateaus at $0.085$.

Cheap proxy if you do not want to evaluate the loss: amplitude $\propto \|g\|$ (gradient norm), or a
running EMA of the minibatch loss.

### 4b. Route B — grow $N_e$ with fitness (the ecological route; your idea, and it works)

$T=v/(2n)$. If abundance $n$ grows as the population adapts, $T$ self-anneals. Week's abundance SDE
is exactly this:

$$dn=\bar m\,n\,dt+\sqrt{v}\,n\,dB_n,$$

so a higher mean per-capita fitness $\bar m$ grows $n$, raising $N_e=n/v$ and lowering the
demographic temperature. **Caveat that matters:** at the eco-evolutionary equilibrium (carrying
capacity) $\bar m=0$ and $n=K$, so $T\to v/(2K)$ — the floor is then set by the carrying capacity.
For the floor to keep falling as you converge you need the *equilibrium* $N_e$ to rise with
adaptedness, i.e. a carrying capacity that increases as the loss drops, $K(\bar z)\uparrow$ as
$L\downarrow$ (e.g. $K\propto \bar W$ or $K\propto 1/L$). That is a *choice of the density-dependent
fitness* $m(\nu,z)$ within Week's framework, not new machinery.

In ML this is **batch-size growth** ("don't decay the learning rate, increase the batch size",
Smith et al. 2018): grow the effective sample size as the loss drops. It anneals *both* noises at
once ($\propto 1/N^\ast$ gradient, $\propto 1/N_e$ demographic).

---

## 5. Canalization's correct role (so you don't waste it on the wrong knob)

Canalization reduces $M$, hence the **scale** of $G$ (through $\hat G=A^{-1}\#M$). The scale of $G$
is the step size / learning rate; its shape is the preconditioner exponent. So canalizing the scale
$=$ LR decay $=$ search-radius / step-size annealing (the CMA-ES $\sigma$-control analog). That is
useful for convergence speed and locality — just not for the demographic-noise temperature.

If you want an *evolution-derived* step-size schedule, the no-benefit canalization rate (derived in
the canalization analysis: with the multiplicative/SPD-safe modifier the scale decays as
$m_i(t)=[m_i(0)^{-1/2}+\tfrac{V\kappa}{4}\sqrt{a_i}\,t]^{-2}$, so $G$ scale $\sim 1/t$, fastest in
the stiff directions) is available. Apply it as a step-size schedule, knowing it is orthogonal to
the noise temperature.

---

## 6. Is the joint $N$-modeling already in Week's framework?

**Yes for the machinery, no for the full coupling you want.**

- Week (2026) is explicitly **eco-evolutionary**: the state is $(n,\bar z,G)$, abundance $n$ has its
  own SDE, the fitness $m(\nu,z)$ is **density-dependent** (this is the logistic / carrying-capacity
  structure you'd want), and $N_e$ enters as $v/n$. So jointly modelling $N$ with $\bar z$ and $G$,
  with demographic noise $\propto G/N_e$, is **native to Week** — you are specializing it, not
  building it.
- What is **not** in Week: **$M$-evolution** ($M$ is fixed in Week and in the current two-noise
  extension). So the full $(n,\bar z,G,M)$ system you're imagining $=$ Week's eco-evolutionary
  $(n,\bar z,G)$ block $+$ the framework's slow-$M$ dynamics. Each piece exists; their combination,
  plus the adaptedness-dependent carrying capacity $K(\bar z)$, is the addition.
- Concretely to add it: keep Week's $dn=\bar m\,n\,dt+\sqrt v\,n\,dB_n$, make $\bar m$ (or $K$)
  increase with mean fitness so $n$ rises as $L$ falls, and append the slow-$M$ drift
  $\dot M=-\mathcal G_M\nabla\mathcal L_{\text{eff}}$. Expect a self-consistency loop ($n\!\to\!N_e\!\to$
  the $M$/$G$ equilibrium $\to$ fitness $\to n$) worth solving as a fixed point if you pursue this
  route seriously.

---

## 7. Implementation guidance (concrete)

For the agent coding the optimizer:

- **Do not** anneal demographic noise by shrinking $G$, decaying the LR, or canalizing $M$ — the
  floor is G-neutral; you will only freeze the iterate at it.
- **Inject demographic noise with covariance $\propto G$** (i.e. $\Sigma_{\text{demo}}=G/N_e$), not
  $\propto I$. The $\propto G$ shape is the FDT-clean one that samples $\exp(-L/T)$; an isotropic
  injection is mis-shaped *and* gets worse if you ever shrink $G$.
- **Pick one annealing route:**
  - *Route A (simplest):* $\Sigma_{\text{demo}}(t)=\dfrac{G}{N_0}\cdot\dfrac{\hat L(t)}{\hat L(0)}$
    with $\hat L$ a running loss EMA (or use $\|g\|^2$ as the proxy). Demographic noise vanishes as
    you converge.
  - *Route B (ecological / self-annealing):* maintain an effective sample size $N_e(t)$ that grows
    as fitness rises — e.g. ramp the batch, or an explicit logistic
    $N_e(t)=N_{\min}+(N_{\max}-N_{\min})\,\big(1-\hat L(t)/\hat L(0)\big)$, or couple to a carrying
    capacity $K(\bar z)$ rising with $-L$. Anneals both noises.
- **Keep a separate step-size/LR schedule** (optionally the $1/t$ canalization-derived one) for
  convergence speed — independent of the noise temperature.
- **Diagnostics:**
  - Estimate the demographic floor $d/(4N_e)$ and overlay it on the loss curve. If the loss plateaus
    at $\approx d/(4N_e)$, you are demographic-noise-floored: raise $N_e$ or turn on loss-matching —
    do *not* reach for LR decay.
  - Confirm the injected covariance is $\propto G$ (Gibbs-clean) vs $\propto I$ (mis-shaped).
  - Sanity check the two annealing channels separately: the gradient floor moves with LR and batch;
    the demographic floor moves only with $N_e$ / injection amplitude.

---

## 8. Scope ledger

**Derived / in-framework:** the two-noise structure and powers of $G$ (`extended_joint_sde.md`);
G-neutrality of the demographic floor $d/(4N_e)$ (framework §7, confirmed numerically here);
demographic Gibbs distribution $\bar W^{2N_e}$, $T=1/(2N_e)$; gradient self-annealing ($C\propto L$
+ floor; $\propto\eta$); abundance/eco-evolutionary dynamics native to Week (2026).

**Modeling choices:** the loss-matched injection schedule (Route A); the carrying-capacity coupling
$K(\bar z)$ (Route B); using the no-benefit canalization rate as the step-size schedule; injecting
$\propto G$ rather than $\propto I$.

**Open:** the full $(n,\bar z,G,M)$ joint SDE with adaptedness-dependent carrying capacity is not yet
written down (combine Week's $(n,\bar z,G)$ with slow-$M$ and a $K(\bar z)$ coupling); the
self-consistency fixed point of that loop; and the design decision of whether you *want* the
demographic Gibbs sampler retained (for Bayesian / ensemble use) or annealed away (for pure
optimization). Note also that Route A and Route B coincide in effect (both lower $T$) but differ
operationally — A scales the noise, B scales $N_e$; B additionally reduces gradient noise.

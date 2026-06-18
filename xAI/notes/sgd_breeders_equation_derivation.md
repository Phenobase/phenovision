# From the Breeder's Equation to Stochastic Gradient Descent: A Step-by-Step Derivation

*A pedagogical walkthrough of the formal correspondence between Lande's stochastic breeder's equation and the Mandt-Hoffman-Blei OU treatment of SGD, written for readers who want to follow the chain of reasoning without taking any step on faith.*

---

## Preface: What We're Trying to Show

There are two equations from two different fields that, on close inspection, are the same equation written in different notation.

**From quantitative genetics**, Lande's (1976, 1979) stochastic multivariate breeder's equation describes how the mean phenotype of a population changes from one generation to the next under combined natural selection and random genetic drift.

**From machine learning**, the continuous-time stochastic differential equation (SDE) limit of stochastic gradient descent — formalized by Mandt, Hoffman & Blei (2017) — describes how a neural network's parameters change from one training step to the next under combined gradient descent and minibatch sampling noise.

The claim is that these two SDEs have the same mathematical form, with each term in one having a precise counterpart in the other. The deeper claim is that this isn't a loose analogy: the correspondence is exact, and the well-developed theory of one process can be imported wholesale into the other.

This document walks through the correspondence one step at a time. We build up from the deterministic skeleton, add stochasticity in stages, and end at the stationary distribution and its connection to free-fitness theory.

---

## Part A: The Deterministic Skeleton

Before we worry about randomness, let's make sure the deterministic backbone of each equation is clearly understood.

### A.1: Lande's deterministic multivariate breeder's equation

The breeder's equation in its most familiar form (the one taught in introductory quantitative genetics) is:

$$R = h^2 S$$

where R is the response to selection (change in mean phenotype between generations), S is the selection differential (difference in mean between the selected parents and the population before selection), and h² is the heritability. This is for a single trait.

For multiple traits evolving together, Lande (1979) generalized this to:

$$\Delta \bar{z} = G \boldsymbol{\beta}$$

Let's unpack each symbol.

- **z̄** is a vector containing the **mean values of each trait** in the population. If you're tracking three traits — say, beak length, beak width, and body mass — then z̄ is a vector of three numbers, one per trait. The bar above the z denotes "averaged over the population."

- **Δz̄** is the **change** in that vector from one generation to the next. If the mean beak length increases by 0.2 mm and the other traits don't change, Δz̄ = (0.2, 0, 0).

- **β** (the **selection gradient**) is also a vector, one entry per trait. It captures how strongly natural selection is favoring increases (or decreases) in each trait. Crucially, it is defined as the gradient of the *log of mean population fitness* with respect to the mean phenotype:

  $$\boldsymbol{\beta} = \nabla \ln \bar{W}(\bar{z})$$

  Here W̄(z̄) is the mean fitness of the population when its mean phenotype is z̄. Think of W̄ as a *fitness landscape*: a function defined over the space of possible mean phenotypes that returns how well the population is doing. The gradient β points in the direction of *steepest increase* of fitness.

- **G** is the **additive genetic covariance matrix** (the "G-matrix"). It's a symmetric matrix whose entries describe how heritable variation in different traits is structured.
  - The diagonal entries G_ii are the additive genetic variances of each trait (how much heritable variation exists for trait i alone).
  - The off-diagonal entries G_ij are the additive genetic covariances between traits (when trait i tends to vary heritably, do you also see correlated heritable variation in trait j?).

  G describes the raw material that selection has to work with.

So Lande's equation says: **the change in mean phenotype is the genetic covariance matrix multiplied by the selection gradient.** Selection points in a direction of fitness increase (β); the population's response is reshaped by the genetic architecture available to respond (G).

#### A geometric interpretation

Imagine the fitness landscape as a tilted plane and β as the direction "uphill." If G were the identity matrix (each trait has unit variance, no covariances), Δz̄ would point straight uphill. But G generally isn't the identity. If trait 1 has lots of genetic variance and trait 2 has little, the population responds more in the trait-1 direction even when β points equally in both. If trait 1 and trait 2 are strongly genetically correlated, response in trait 1 drags response in trait 2 along with it. **G acts as a filter or lens between the selection direction and the actual evolutionary response.**

Mathematicians call this kind of object a *preconditioner* or, in differential geometry, a *Riemannian metric*. We'll see in a moment that this is exactly the role G plays in the SGD analogy.

#### A worked toy example

Suppose two traits, with G = [[2, 1], [1, 1]] and β = (1, 0) — selection favors trait 1, is neutral on trait 2.

$$\Delta \bar{z} = G\boldsymbol{\beta} = \begin{pmatrix} 2 & 1 \\ 1 & 1 \end{pmatrix} \begin{pmatrix} 1 \\ 0 \end{pmatrix} = \begin{pmatrix} 2 \\ 1 \end{pmatrix}$$

Even though selection is purely on trait 1, trait 2 evolves too — because of the genetic covariance between them. This is "correlated response" and it's the central reason multivariate quantitative genetics is harder than single-trait quantitative genetics.

### A.2: Stochastic gradient descent in plain language

Now let's set up SGD. The setting is supervised machine learning: we have a model with parameters θ (a long vector — could be billions of entries for a modern neural network), and we want to choose θ to minimize a loss function L(θ) that measures how badly the model performs.

The loss is typically a sum or average over training examples:

$$L(\theta) = \frac{1}{N} \sum_{n=1}^N \ell_n(\theta)$$

where ℓ_n(θ) is the loss on the nth example. With N = millions of examples, computing the full gradient ∇L(θ) every step is expensive. So instead we draw a small **minibatch** of S examples and use the average gradient over the minibatch as a stand-in:

$$\hat{g}_S(\theta) = \frac{1}{S} \sum_{n \in \mathcal{S}} \nabla \ell_n(\theta)$$

This is a noisy estimate of the true gradient — it equals the true gradient on average, but on any one minibatch it deviates from the true gradient by some random amount.

The SGD update rule is:

$$\theta_{t+1} = \theta_t - \eta \, \hat{g}_S(\theta_t)$$

where η (the **learning rate**) is a small positive number controlling step size.

In words: take a step in the negative direction of the noisy gradient estimate, scaled by the learning rate.

#### Adding a preconditioner

A more general form of SGD uses a **preconditioning matrix** H to reshape the gradient before stepping:

$$\theta_{t+1} = \theta_t - H \, \hat{g}_S(\theta_t)$$

The plain SGD case is H = ηI (identity matrix scaled by η). Adaptive methods like AdaGrad, RMSProp, and Adam use diagonal H. Natural gradient descent (Amari 1998) uses H = (Fisher information)⁻¹.

The role of H here is *exactly* the role G plays in the breeder's equation: it reshapes the raw gradient into the actual update direction. **H is to SGD as G is to Lande's equation.**

### A.3: Translation between the two

Now let's lay them side by side, matching variables.

| Quantitative genetics | Machine learning |
|---|---|
| Mean phenotype vector z̄ | Parameter vector θ |
| Selection gradient β = ∇ ln W̄ | Negative loss gradient -∇L |
| Mean log fitness ln W̄ | Negative loss -L |
| Genetic covariance G | Preconditioner H |
| Generation Δ | Learning rate × time step η · Δt |

Plug these into Lande's equation:

$$\Delta \bar{z} = G \boldsymbol{\beta}$$

becomes

$$\Delta \theta = H \cdot (-\nabla L) = -H \nabla L$$

which is exactly the preconditioned gradient descent update (in the deterministic limit where the gradient is computed exactly, not from a minibatch).

**At the deterministic level, Lande's multivariate breeder's equation is identical to preconditioned gradient descent.** The only thing that distinguishes them is the substrate: in biology the "gradient" is measured by selection on real organisms, while in ML it's computed from the loss function. But the equation is the same.

#### Continuous-time form

If we let the time step shrink to zero (Δt → 0), the discrete update becomes a continuous-time differential equation:

$$\frac{d\theta}{dt} = -H \nabla L(\theta)$$

This is called *gradient flow* — the trajectory of a particle sliding downhill on the loss landscape, with the slope reshaped by H. In quantitative genetics terms, this is the continuous-generations limit of Lande's equation:

$$\frac{d\bar{z}}{dt} = G \boldsymbol{\beta}(\bar{z}) = G \nabla \ln \bar{W}(\bar{z})$$

So far, no stochasticity. Just two equations describing the same deterministic geometric process.

---

## Part B: Adding Noise

Real evolution is not deterministic, and neither is SGD. Both pick up randomness as you scale down population size (in evolution) or minibatch size (in ML). The interesting work is figuring out exactly how the noise enters.

### B.1: The two distinct sources of noise in finite populations

When biologists say "random genetic drift," they actually conflate two conceptually different things, both of which are sources of stochastic change in mean phenotype. It's worth pulling them apart.

**Source 1: Mendelian transmission noise.** Even if selection acts identically on every individual, when those selected individuals reproduce, the alleles they pass to offspring are sampled randomly (which homologue from each chromosome pair, what crossover patterns, etc.). With N_e parents, this random sampling introduces a per-generation variance in mean breeding value of order **G/N_e** — proportional to the genetic covariance, inversely proportional to effective population size. This is what Lande (1976) modeled, and what evolutionary biologists usually mean by "drift."

**Source 2: Selection-gradient sampling noise.** The selection gradient β is a population-level quantity — strictly, it's the gradient of mean log fitness *averaged over individuals*. With a finite population, you're really estimating β from the actual individuals present, each of whom has a particular phenotype and fitness. Different individuals would give different gradient estimates. So the realized gradient is noisy in a second, distinct way: not because of random transmission, but because you're sampling the gradient itself from a population of individuals who each contribute differently.

In Lande's classical treatment, Source 2 vanishes (he assumes infinite-population selection acting on a finite-population transmission step). But for the SGD analogy, Source 2 is the more relevant one — minibatch noise is exactly gradient sampling.

For now, let's follow Lande and keep only Source 1.

### B.2: Lande's stochastic breeder's equation

Adding Mendelian transmission noise to the deterministic equation:

$$\Delta \bar{z} = G \boldsymbol{\beta} + \boldsymbol{\epsilon}, \qquad \boldsymbol{\epsilon} \sim \mathcal{N}(0, G/N_e)$$

In words: each generation, the mean phenotype changes by a deterministic part (G β) plus a Gaussian random part with mean zero and covariance G/N_e.

This is the discrete-generation version. In continuous time, it becomes a stochastic differential equation:

$$d\bar{z} = G \boldsymbol{\beta}(\bar{z}) \, dt + \sqrt{G/N_e} \, dW$$

where dW is a standard multivariate Wiener process (random Brownian motion in each direction), and √(G/N_e) is the matrix square root of G/N_e.

A note on the matrix square root: if Σ is a positive-definite covariance matrix, √Σ is any matrix M satisfying MM^T = Σ. The point is that if you write a noise term M dW, the resulting process has noise covariance Σ. The square root is just the technical operator that converts isotropic white noise (dW) into noise with the desired covariance shape.

This is the key SDE on the biology side. Lande himself didn't write it as an SDE — he wrote the equivalent Fokker-Planck PDE for the probability density — but the SDE form is mathematically equivalent and easier to compare with the SGD literature.

### B.3: SGD's gradient noise

Now for the SGD side. The minibatch gradient ĝ_S equals the true gradient g(θ) plus a random error:

$$\hat{g}_S(\theta) = g(\theta) + \frac{1}{\sqrt{S}} \Delta g(\theta), \qquad \Delta g \sim \mathcal{N}(0, C(\theta))$$

The Gaussianity follows from the central limit theorem (the minibatch is a sum of S independent samples), and the covariance C(θ) is the per-sample gradient covariance — the covariance of ∇ℓ_n(θ) across data points n. The 1/√S scaling comes from the fact that averaging S samples reduces the variance by a factor of S.

Substituting into the SGD update:

$$\theta_{t+1} - \theta_t = -\eta \left[ g(\theta_t) + \frac{1}{\sqrt{S}} \Delta g \right]$$

For the preconditioned version with matrix H instead of scalar η:

$$\theta_{t+1} - \theta_t = -H g(\theta_t) - \frac{1}{\sqrt{S}} H \Delta g$$

Taking the continuous-time limit (Mandt, Hoffman & Blei 2017), this becomes the SDE:

$$d\theta = -H g(\theta) \, dt + \sqrt{1/S} \, H B \, dW$$

where C = BB^T (B is the matrix square root of the gradient covariance C). Note: the original MHB paper uses ε for the learning rate; if we have absorbed it into H, the prefactor becomes simpler.

### B.4: Writing both SDEs side by side

The biology SDE (Lande):

$$d\bar{z} = G \boldsymbol{\beta}(\bar{z}) \, dt + \sqrt{G/N_e} \, dW$$

The ML SDE (preconditioned MHB):

$$d\theta = -H g(\theta) \, dt + \sqrt{1/S} \, H B \, dW$$

Now we can see exactly how the correspondence works.

| Component | Lande | MHB |
|---|---|---|
| State variable | z̄ (mean phenotype) | θ (parameters) |
| Drift coefficient | G | H |
| Driving "force" | β (selection gradient) | -g (negative loss gradient) |
| Per-individual covariance | G | C |
| Sample count | N_e (effective population size) | S (minibatch size) |
| Diffusion covariance | G/N_e | (1/S) HCH^T |
| Noise-shaping matrix | (G/N_e)^(1/2) | (1/S)^(1/2) HB, where BB^T = C |

A note on the noise-shaping matrices. B in the MHB formulation is the matrix square root of C: by definition, BB^T = C. It exists in the SDE notation purely because Itô SDEs require the noise term in the form M dW with dW being white noise of identity covariance. To produce noise with target covariance Σ, you need M satisfying MM^T = Σ — that operator is the matrix square root, which we call B (when factoring C) or (G/N_e)^(1/2) (when factoring G/N_e). It's a bookkeeping device, not a physical object. Lande's equation has its own implicit "B-equivalent" — the matrix square root of G/N_e — we just didn't bother naming it because only one matrix is involved.

The deeper observation hiding in this table: **G plays a dual role in Lande's equation that MHB splits into two separate matrices.** G is both the preconditioner (drift) and the per-individual covariance (noise source). MHB has H and C as independent matrices, which only collapse together in the Fisher-scoring case (H ∝ C⁻¹). Biology has them collapsed together by construction — there is no separate "per-individual gradient covariance" because the same G that reshapes selection response also describes the variance among individuals' breeding values. This is exactly why Lande's equation lands in the Fisher-scoring regime structurally: it has one fewer free matrix than generic preconditioned SGD, and that one matrix is forced to play both roles.

The drift terms match exactly under the identification G ↔ H and β ↔ -g.

The diffusion terms are where things get interesting.

---

## Part C: The Structural Alignment

The drift correspondence is straightforward. The diffusion correspondence requires care, because Lande's noise covariance is just G/N_e while MHB's is the "sandwich" HCH^T scaled by 1/S.

### C.1: When the diffusion terms match

The diffusion covariances become structurally identical when:

$$\frac{1}{S} H C H^T = \frac{1}{N_e} G$$

Identifying G ↔ H, this becomes:

$$\frac{1}{S} G C G^T = \frac{1}{N_e} G$$

Since G is symmetric (G = G^T), we can simplify GCG^T = GCG. For this to equal (S/N_e) G times a constant, we need:

$$G C G = \text{(constant)} \cdot G$$

Multiplying by G⁻¹ on both sides:

$$C \cdot G = \text{(constant)} \cdot I$$

In other words: **C and G must be inverses of each other (up to a scalar).**

This is the **Fisher scoring** condition — the optimal preconditioning identified in Mandt-Hoffman-Blei (2017, Theorem 5) and originally proposed by Ahn et al. (2012). It's the natural-gradient case in disguise: when H = C⁻¹, the SGD dynamics correspond to natural gradient descent on the manifold of distributions, with Fisher information playing the role of the metric.

**The structural conclusion: Lande's stochastic breeder's equation is mathematically equivalent to a natural-gradient Langevin SDE.** It's not a generic preconditioned SGD process — it's specifically the special case where the preconditioner equals the inverse of the gradient noise covariance.

This is a non-trivial observation about quantitative genetics that the SGD analogy makes visible. Lande never framed it this way, because the natural-gradient framework didn't exist in 1976. But the mathematics of his equation puts it squarely in that family.

### C.2: Why this matters: fluctuation-dissipation

In statistical physics, an SDE of the form

$$d x = -\Gamma \nabla U(x) \, dt + \sqrt{2 \Gamma T} \, dW$$

(with Γ a positive-definite "mobility" matrix and T a scalar "temperature") satisfies a fluctuation-dissipation relation. Its stationary distribution is the Gibbs measure:

$$p(x) \propto \exp(-U(x) / T)$$

Lande's SDE fits this template:

- Γ = G
- U = -ln W̄ (negative log mean fitness)
- T = 1/(2N_e)

So the stationary distribution of Lande's stochastic breeder's equation is:

$$p(\bar{z}) \propto \exp(2 N_e \ln \bar{W}(\bar{z})) = \bar{W}(\bar{z})^{2 N_e}$$

This is exactly the **Sella-Hirsh free-fitness result** (Sella & Hirsh 2005, building on Iwasa 1988). It's a known result in quantitative genetics, but the SGD framing reveals it as the Gibbs distribution of a natural-gradient Langevin process — which connects it to the broader thermodynamic literature on optimization.

### C.3: When the bridge breaks

For generic preconditioned SGD where C ≠ H⁻¹, the diffusion structure HCH^T doesn't reduce to a clean form, and the stationary distribution is generally not the Gibbs measure of L. The MHB analysis still gives you a Gaussian stationary distribution near a minimum (computed via the Lyapunov equation, see below), but it isn't proportional to exp(-L/T) for a fixed temperature T.

Why does this matter for the biology bridge? Because if real biological selection contains substantial gradient-sampling noise (Source 2 in section B.1), and if the gradient covariance C in the population isn't proportional to the inverse of G, then the dynamics depart from Lande's clean picture. Whether this happens in practice is an empirical question and an open one.

---

## Part D: The OU Process and Stationary Distribution

We've established that Lande's equation is structurally a Langevin SDE. To extract concrete predictions — like the variance of the population's mean around the optimum — we need to solve the SDE near a fitness peak. This is where the Ornstein-Uhlenbeck (OU) process enters.

### D.1: Linearizing around the optimum

Suppose the fitness landscape ln W̄ has a peak at z̄*. Near that peak, we can Taylor-expand:

$$\ln \bar{W}(\bar{z}) \approx \ln \bar{W}(\bar{z}^*) - \frac{1}{2} (\bar{z} - \bar{z}^*)^T A (\bar{z} - \bar{z}^*)$$

Here A is the **negative Hessian** of ln W̄ at the optimum — a matrix describing the curvature of the fitness surface. Sharp peaks correspond to large eigenvalues of A; broad, flat peaks to small ones.

Taking the gradient:

$$\boldsymbol{\beta}(\bar{z}) = \nabla \ln \bar{W} \approx -A (\bar{z} - \bar{z}^*)$$

So near the peak, the selection gradient pulls the population back toward z̄* with strength proportional to A.

Substituting into Lande's SDE (and shifting coordinates so z̄* = 0):

$$d\bar{z} = -G A \, \bar{z} \, dt + \sqrt{G/N_e} \, dW$$

This is the multivariate OU process: linear drift toward zero, additive Gaussian noise.

### D.2: The Lyapunov equation

The stationary distribution of a multivariate OU process is Gaussian: p(z̄) ∝ exp(-½ z̄^T Σ⁻¹ z̄). The covariance matrix Σ satisfies the **Lyapunov equation**:

$$(GA) \Sigma + \Sigma (GA)^T = G/N_e$$

This is an algebraic equation in Σ — once you know G, A, and N_e, you can solve for Σ.

In the special case where G and A commute (e.g., both are diagonal in the same basis), the solution simplifies dramatically:

$$\Sigma = \frac{1}{2 N_e} A^{-1}$$

The stationary covariance of the population's mean phenotype is the inverse of the fitness curvature matrix, scaled by 1/(2N_e).

### D.3: Reading the result

The stationary distribution is:

$$p(\bar{z}) \propto \exp\left(- N_e (\bar{z} - \bar{z}^*)^T A (\bar{z} - \bar{z}^*) \right)$$

Compare this to the linearized fitness:

$$\bar{W}(\bar{z})^{2 N_e} \propto \exp\left( -N_e (\bar{z} - \bar{z}^*)^T A (\bar{z} - \bar{z}^*) \right)$$

These match. So the stationary distribution is exactly W̄^(2N_e) — confirming the Sella-Hirsh free-fitness result we anticipated.

**The interpretation:** the mean phenotype undergoes a stochastic dance around the fitness peak. How tightly it's clustered depends on the sharpness of the peak (A) and the population size (N_e). Larger populations = tighter clustering (less drift); sharper peaks = tighter clustering (stronger restoring selection). The exponent 2N_e plays the role of an inverse temperature — high N_e is "cold" (selection dominates), low N_e is "hot" (drift dominates).

This is why population size shows up everywhere in molecular evolution: it's literally setting the temperature scale of the underlying Langevin dynamics.

### D.4: The MHB version

For SGD near a minimum (loss L ≈ ½ θ^T A θ), the Mandt-Hoffman-Blei stationary covariance solves:

$$(HA) \Sigma + \Sigma (HA)^T = \frac{1}{S} H C H^T$$

Same Lyapunov form, different right-hand side. With C = H⁻¹ (Fisher scoring case), this reduces to:

$$(HA) \Sigma + \Sigma (HA)^T = \frac{1}{S} H$$

which (when HA commutes appropriately) gives Σ = (1/2S) A⁻¹ — the same form as the biology result, with S replacing N_e.

**Population size in biology and minibatch size in ML play exactly the same role.** Both set the temperature of the Langevin dynamics.

---

## Part E: Fluctuation-Dissipation and the Unity of Langevin Systems

The result that Lande's equation in the Fisher-scoring regime has stationary distribution W̄^(2N_e) is more than a curiosity. It's a manifestation of one of the deepest principles in nonequilibrium statistical mechanics: the **fluctuation-dissipation theorem**. Once you see what's happening at the physical level, several things that initially felt magical become inevitable — and the bridge connects to an unexpectedly wide set of fields beyond biology and optimization.

### E.1: What fluctuation-dissipation is

In 1905 Einstein noticed something about a pollen grain bouncing around in water. The fluid's *viscosity* resists the grain's motion — friction. The fluid's *thermal agitation* pushes the grain around — random motion. These look like separate properties, but Einstein realized they must be linked. A fluid that strongly resists motion also strongly fluctuates: both reflect the same underlying coupling between the grain and the surrounding molecules. He derived:

$$D = k_B T / \gamma$$

where γ is friction, D is the diffusion coefficient (random-motion strength), and T is temperature. **Friction and fluctuation are two faces of the same coupling.** Neither exists without the other at thermal equilibrium.

This generalizes to multivariate stochastic systems. A Langevin equation with fluctuation-dissipation structure has the form:

$$dx = -\Gamma \nabla U(x) \, dt + \sqrt{2 \Gamma T} \, dW$$

where Γ is a positive-definite "mobility matrix," U is a potential, T is a scalar temperature, and dW is standard Wiener noise. The defining feature: **the same Γ appears in both the drift and the diffusion.** When this structure holds, the stationary distribution is automatically the Gibbs measure:

$$p(x) \propto \exp(-U(x) / T)$$

regardless of the microscopic noise mechanism. Once Γ appears in both places and a single temperature T is well-defined, the equilibrium depends only on U and T. The mechanism producing the noise is washed out.

### E.2: Lande's equation is a fluctuation-dissipation system

In the Fisher-scoring form, Lande's stochastic breeder's equation reads:

$$d\bar{z} = G \nabla \ln \bar{W} \, dt + \sqrt{G/N_e} \, dW$$

Reading off the Langevin template:
- Γ = G (genetic covariance acts as mobility)
- U = -ln W̄ (negative log mean fitness as energy)
- T = 1/(2N_e) (inverse effective population size as temperature)

The Gibbs measure then gives:

$$p(\bar{z}) \propto \exp(-U/T) = \exp(2 N_e \ln \bar{W}) = \bar{W}^{2N_e}$$

which is Sella-Hirsh's free-fitness distribution. **This isn't a separate derivation — it falls out automatically from recognizing Lande's equation as an FDT system.** The thermodynamic analogy isn't metaphorical or loose; the equation literally is a Langevin equation with N_e setting the inverse temperature.

### E.3: Why the noise source stops mattering

This resolves what felt like magic. We've discussed three structurally different noise sources:

1. **Mendelian transmission drift** (Lande's noise): random allele sampling at reproduction
2. **Minibatch gradient noise** (SGD): random data subsampling at each step
3. **Explicit Gaussian injection** (SGLD-style): deliberate white-noise addition

These are physically completely different. The per-step noise structure differs in each case — different correlation patterns, different scaling with system parameters, different microscopic origin. But **once FDT structure is enforced**, the stationary distribution depends only on U and T. The microscopic noise mechanism doesn't matter for equilibrium.

This is why "biology's noise" and "SGD's noise" can produce the same equilibrium even though they arise from completely different mechanisms. They don't need to be the same; they only need to produce the same effective temperature within an FDT-structured dynamics.

The further implication is that the Fisher-scoring regime — where preconditioner and noise are properly matched — isn't just a special case where the SGD-Lande bridge happens to work. It's the regime where **fluctuation-dissipation holds**, which is the regime where the system has a meaningful thermodynamic equilibrium at all. Outside this regime, the stationary distribution still exists (it solves the Lyapunov equation), but it isn't a simple function of U; it depends on the specific noise structure in a more complicated way. The "magic" of biology and SGD producing the same equilibrium *requires* FDT to hold.

### E.4: SGLD — biology's noise structure realized algorithmically

The thought experiment "infinite batch size SGD with explicit Gaussian noise added to the weights" turns out to be a published algorithm: **Stochastic Gradient Langevin Dynamics** (Welling & Teh 2011).

$$\theta_{t+1} = \theta_t - \frac{\eta}{2} H \nabla L(\theta_t) + \sqrt{\eta T} \, M \, \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, I)$$

with M chosen so MM^T = H (the noise covariance is the matrix square root of the preconditioner — explicit enforcement of FDT). In the Fisher-scoring case (H = C⁻¹), this is exactly Lande's equation structure with the identification N_e ↔ 1/T.

SGLD is used for Bayesian deep learning: run it to equilibrium, collect samples, and they're approximate draws from the posterior p(θ) ∝ exp(-L/T). The stationary distribution is the Bayesian posterior, by exact analogy with biology's W̄^(2N_e) being the population-genetic free-fitness equilibrium.

So the answer to "what's the equivalent of biology's noise structure in a hypothetical infinite-batch SGD?" is: **SGLD**. It's been a published algorithm since 2011, and it's doing what biology does with the temperature explicitly chosen rather than emerging from minibatch size.

### E.5: Diffusion generative models are the same SDE

The largest practical instance of FDT Langevin dynamics in modern technology is **diffusion generative models** — DDPMs, score-based models, and the technology behind Stable Diffusion, DALL-E, Midjourney, and most modern video generation systems.

A diffusion model learns a **score function** s(x) ≈ ∇ log p(x) — the gradient of log probability of the data distribution. Sampling from the trained model consists of running Langevin dynamics:

$$x_{t+1} = x_t + \frac{\eta}{2} \nabla \log p(x_t) + \sqrt{\eta} \, \varepsilon_t$$

This is literally Lande's equation with relabeled variables: the score takes the role of the selection gradient (∇ log p replacing ∇ ln W̄), the data distribution takes the role of the equilibrium distribution (p(x) replacing W̄^(2N_e)), and explicit Gaussian noise provides the dissipation-matched fluctuation.

**What diffusion models are doing when they generate images is mathematically the same operation as a biological population reaching mutation-selection-drift equilibrium.** The score function is the selection gradient. The samples are the equilibrium population. The denoising trajectory is the trajectory toward equilibrium.

This is not a metaphor. It's the same SDE in different vocabulary.

### E.6: A canonical structure across fields

The same Langevin SDE with FDT structure appears in:

1. Brownian motion (Einstein 1905)
2. Statistical mechanics generally (Gibbs measures, mid-20th century)
3. Mutation-selection-drift equilibrium (Iwasa 1988; Sella & Hirsh 2005)
4. Lande's quantitative genetic dynamics in Fisher-scoring form (Lande 1976)
5. Bayesian posterior sampling via SGLD (Welling & Teh 2011)
6. Approximate Bayesian inference via vanilla SGD (Mandt, Hoffman & Blei 2017)
7. Score-based diffusion generative models (Song & Ermon 2019; Ho et al. 2020)
8. Frontier optimization via natural-gradient methods (SOAP, Shampoo family, 2024–2026)

This recurrence isn't coincidence. The Langevin SDE with FDT structure is the canonical form of "optimal stochastic exploration of a high-dimensional landscape given finite resources." Any field that needs to balance descent toward optima with exploration to avoid getting trapped will rediscover it.

The order of discovery is itself instructive. Biological evolution has been implementing this dynamics for ~3.5 billion years. Physics formalized it in 1905. Population genetics encoded it (in Fokker-Planck PDE form) in 1976. Bayesian sampling adopted it explicitly in 2011. SGD-as-Langevin was formalized in 2017. Generative modeling reorganized around it from 2020 onward, now reshaping a major industry. Frontier optimization is converging on it in 2024–2026. **The same structure has been rediscovered repeatedly across fields, each time reframed in local vocabulary.** Recognizing the unifying form is itself a substantial contribution.

### E.7: What this means for the framework

The FDT framing sharpens what pretraining accomplishes. The standard intuition — "pretraining puts you in a good basin" — is incomplete. In FDT language, pretraining establishes a local Γ matrix that's well-shaped for downstream tasks. The G(x) = C(x) + V(x) decomposition is a decomposition of Γ: C captures the slowly-changing structural part of the metric, V captures the within-structural-frame component that responds to local task curvature.

The three components of "good pretraining" map onto three FDT-level objects:

| Component | FDT object |
|---|---|
| Ecological fitting (starting position) | Position in parameter space relative to U |
| Short-term evolvability (rate of response) | Local Γ matrix (Riemannian metric on the loss landscape) |
| Long-term evolvability (peak quality) | Global structure of U near the starting basin |

The Jones-Arnold-Bürger result on M-matrix evolution can now be read at a deeper level: **biology evolves not just the position in parameter space, but the metric itself.** Standard pretraining mostly evolves position (with the architecture fixing C); multi-task pretraining and meta-learning evolve V as well. The closest ML analog of "evolving the M-matrix" is meta-learning the optimizer or the architecture — explicitly evolving the metric on the loss landscape, not just the position within it. This is the formal version of the qualitative point that ML's "deep past" and biology's "deep past" play structurally identical roles.

---

## Part F: What This Looks Like in Practice — SOAP and Two-Timescale Structure

The derivation above establishes a clean formal correspondence between Lande's stochastic breeder's equation and a natural-gradient Langevin SDE. But there's an immediate practical worry: real machine learning doesn't actually run natural gradient descent. The Fisher matrix for a modern neural network has D² entries (with D ~ 10⁹ parameters), and inverting it costs O(D³) operations — completely infeasible. So if real ML doesn't operate in the natural-gradient regime, does the formal bridge to biology have practical content?

The answer is yes, and getting to it requires understanding what real ML actually does in 2024–2026 — which has changed substantially in the last two years.

### F.1: The historical default — Adam — is not natural gradient

For most of the last decade, the dominant optimizer for deep learning has been Adam (Kingma & Ba 2015). Adam maintains an exponential moving average of squared gradients per parameter, v_t, and uses √v_t (not v_t itself) to rescale each parameter's update. It's computationally cheap — O(D) storage, O(D) work per step — but it's a poor approximation of natural gradient in two distinct ways:

1. It's **diagonal**, ignoring all parameter-parameter correlations. Real Fisher matrices in deep networks have substantial off-diagonal structure (parameters within the same layer are highly correlated). Adam throws all of that away.

2. It uses the **square root** of v rather than v⁻¹. Natural gradient calls for the inverse of the Fisher, not the inverse square root. The square root form comes from AdaGrad's historical origins and is structurally different from natural gradient.

There's also a more subtle issue: what v_t actually approximates the diagonal of is the *empirical Fisher* (per-sample squared gradients), which is known to differ from the true Fisher in ways that matter (Kunstner, Hennig & Balles 2019).

The upshot is that Adam is far from the Fisher-scoring regime where the bridge to Lande's equation is exact. If Adam were the only show in town, the bridge would be a formal curiosity with limited practical traction.

### F.2: The two-timescale insight

Adam is not the only show in town. A line of work culminating in SOAP (Vyas et al., ICLR 2025), with predecessors in K-FAC (Martens & Grosse 2015), Shampoo (Gupta, Koren & Singer 2018), and E-KFAC (George et al. 2018), has discovered something important about how to do natural-gradient-style preconditioning at scale.

The insight is that you don't need to recompute the full Fisher every step. You can decompose the natural-gradient operation into two pieces:

1. A **slow component**: a rotation matrix Q that captures the structural directions of curvature in parameter space — essentially the eigenbasis of an approximate Fisher matrix. Q is updated infrequently, every 10–100 optimization steps.

2. A **fast component**: cheap diagonal Adam-style preconditioning *within* the rotated basis. This runs every step.

Mathematically, if Q is the (Kronecker-factored) eigenbasis of the gradient curvature, then a preconditioner of the form Q D Q^T (with D diagonal) approximates the full inverse-Fisher operation. The slow rotation handles the structural part — which directions are coupled. The fast diagonal handles within-direction scaling.

SOAP runs ordinary Adam in this rotated basis. To process each gradient, it rotates into the eigenbasis (Q^T g), applies diagonal Adam, then rotates back (Q · update). The eigenbasis itself is updated periodically using Shampoo's Kronecker factorization, which avoids ever materializing or inverting the full Fisher.

This is now the leading non-diagonal optimizer for large-scale model training. A Shampoo variant closely related to SOAP won the AlgoPerf optimization competition in 2025. The frontier of ML optimization has, in the last two years, moved decisively in this direction.

### F.3: Biology has been doing this all along

The two-timescale structure of SOAP is not an algorithmic novelty when viewed from quantitative genetics — it's exactly how biological evolution operates. Consider the timescales of the relevant biological quantities:

- **Slow timescale**: The M-matrix (mutational covariance), the modular structure of pleiotropy, and the overall shape of the G-matrix evolve over many generations under stabilizing and correlational selection. Jones, Arnold & Bürger (2007) showed in simulation that the M-matrix evolves toward alignment with the adaptive landscape; Arnold et al. (2008) reviewed the broader convergence of G toward A⁻¹ under stabilizing + correlational selection. These are the structural features of variation.

- **Fast timescale**: Within any given generation, the population responds to current selection through Lande's equation Δz̄ = Gβ. This is the per-direction scaling within the current structural backbone.

The slow process determines which directions are coupled in phenotypic space — which traits tend to vary together, which are independent, which are constrained by shared developmental machinery. The fast process determines how strongly the population responds in each currently-available direction.

This is the same two-timescale separation SOAP implements: structural learning of the curvature directions (slow Q), within-structure response (fast diagonal updates within the rotated basis).

In the notation of the G(x) = C(x) + V(x) framework from the project notes:

- **C(x)** corresponds to the slowly-evolving structural component — modular pleiotropy, body-plan-level constraints, the parts of architecture that don't change on adaptation timescales. Algorithmically analogous to Q in SOAP.
- **V(x)** corresponds to the within-module evolvable variation — the directions within the current structural backbone where the population can respond to selection. Algorithmically analogous to the diagonal Adam updates in the rotated basis.

The G(x) = C(x) + V(x) split is the natural-gradient regime under two-timescale structure. It's not a special case or an idealization; it's the structure both biology and current frontier optimization converge on.

### F.4: A revised correspondence

The corrected parameter table:

| Component | Lande / biology | SOAP / modern ML |
|---|---|---|
| Slow structural learning | M-matrix evolution; modular pleiotropy under correlational selection | Periodic eigenbasis update Q (Shampoo-style Kronecker factorization) |
| Fast within-structure response | Δz̄ = Gβ per generation | Diagonal Adam preconditioning within rotated basis |
| Timescale separation | Many generations (slow) vs. one generation (fast) | Every 10–100 steps (slow) vs. every step (fast) |
| Structural attractor | G ∝ A⁻¹ (natural-gradient regime) | Q Q^T approximates Fisher⁻¹ (natural-gradient regime) |
| What evolves | M-matrix structure, then G eigenstructure, then mean phenotype | Q rotation, then diagonal scaling, then parameters |

### F.5: Why this changes the framing of the bridge

Three things shift when you look at the bridge through SOAP rather than through pure natural gradient or pure Adam.

First, **the Fisher-scoring case isn't an idealization to apologize for**. It's the regime that both biology and the current frontier of optimization theory are pulled toward. The bridge isn't telling us biology is doing something rare or special; it's telling us biology has been doing what good optimizers have belatedly figured out how to do at scale.

Second, **the two-timescale structure of evolution is not biologically peculiar**. Slow architectural learning and fast within-architecture response is a structural feature of natural-gradient methods that are computationally feasible. The slow/fast separation is required by tractability, not optional.

Third, **the G(x) = C(x) + V(x) decomposition gains concrete computational content**. The split between "structural backbone you can't easily change" (C) and "evolvable variation within the structure" (V) is exactly the split that SOAP implements algorithmically. Biology and SOAP are using the same architectural strategy because it's the right strategy for natural-gradient optimization in high dimensions.

### F.6: Implications for the empirical work

This suggests a methodological refinement for any empirical test of the framework. Standard practice in fine-tuning vision transformers (and most other modern models) is to use AdamW — which is in the diagonal-with-square-root regime, far from the natural-gradient regime where the formal bridge is tight. If you fine-tune with SOAP instead, you operate in a regime structurally aligned with biological evolution: slow structural learning + fast within-structure response.

Two testable predictions follow:

1. The decomposition into ecological fitting / short-term evolvability / long-term evolvability should be **cleaner under SOAP than under AdamW**. The theoretical predictions assume natural-gradient dynamics; SOAP is the closest practical approximation. Measurements should align more tightly with theory.

2. The gap between SOAP and AdamW performance should be **larger for randomly-initialized models than for pretrained ones**. Pretrained models are already in a good basin where any reasonable preconditioning works — ecological fitting reduces optimizer sensitivity. Random-init models have more to gain from better preconditioning. This is directly testable and would speak to the operational distinction between ecological fitting and evolvability: if pretrained models are optimizer-insensitive while random-init models are optimizer-sensitive, that's mechanistic evidence that pretraining shifts the basin (ecological fitting) rather than just the geometry within the basin.

---

## Part G: What Just Happened

Let's review the chain of reasoning.

1. **Step 1**: Lande's deterministic multivariate breeder's equation Δz̄ = Gβ is preconditioned gradient descent in disguise. The genetic covariance G plays the role of the preconditioning matrix; the selection gradient β plays the role of the negative loss gradient.

2. **Step 2**: Adding Mendelian transmission noise (Lande's drift), the breeder's equation becomes a stochastic differential equation with drift G β and noise covariance G/N_e.

3. **Step 3**: Adding minibatch noise to SGD gives an analogous SDE (Mandt, Hoffman & Blei 2017), with drift -Hg and noise covariance (1/S) HCH^T.

4. **Step 4**: The two SDEs have identical structure when G ↔ H, β ↔ -g, and the noise covariances match. Matching the noise covariances requires C ∝ G⁻¹ — the Fisher scoring case, equivalent to natural gradient descent.

5. **Step 5**: This means Lande's stochastic breeder's equation is structurally a natural-gradient Langevin SDE — a non-generic, highly structured object from the optimization literature.

6. **Step 6**: Linearizing around a fitness optimum reduces both SDEs to Ornstein-Uhlenbeck processes whose stationary distributions are Gaussian. The variance scales as 1/N_e (biology) or 1/S (ML), and the shape is the inverse fitness Hessian A⁻¹.

7. **Step 7**: The full stationary distribution of Lande's process is exactly W̄^(2N_e) — recovering the Sella-Hirsh free-fitness result and revealing it as the Gibbs measure of a Langevin-like process. The exponent 2N_e is the inverse temperature of the dynamics.

8. **Step 8**: This isn't coincidence — Lande's equation in the Fisher-scoring regime is a fluctuation-dissipation system. The same Γ appearing in drift and diffusion, with a single temperature, is the structural signature of equilibrium statistical mechanics. The stationary distribution depends only on U and T, not on the microscopic noise source. This is why Mendelian drift, minibatch sampling, and explicit Gaussian injection all produce the same equilibrium under proper preconditioning — and why the same Langevin SDE appears in Brownian motion, statistical mechanics, population genetics, Bayesian sampling, SGLD, diffusion generative models, and frontier optimization.

9. **Step 9**: The bridge implies that machinery developed for one process can be imported to the other. From CS to biology: continuous-time limits, Lyapunov-equation tools, natural-gradient theory, information geometry. From biology to CS: the multi-timescale evolution of G itself (G-matrix evolution literature), the role of architectural constraints, the empirical literature on standing variation.

10. **Step 10**: Real ML, after a decade of Adam dominance, has in 2024–2026 converged on two-timescale natural-gradient methods (SOAP, K-FAC, Shampoo) that mirror the structure biology has used all along — slow Kronecker-factored preconditioning + fast within-basis updates. The Fisher-scoring regime where the bridge is exact is the regime both fields are pulled toward. The G(x) = C(x) + V(x) decomposition has direct algorithmic content: C is the slow rotation, V is the fast within-basis variation.

### What's not yet shown

This derivation assumes:

- Selection is "infinite-population" (no gradient sampling noise)
- The G-matrix is approximately constant over the timescales of phenotypic evolution (relaxed in Part F, where G evolves on a slower timescale)
- The fitness landscape is locally quadratic
- Mendelian drift is the only stochastic source

Each of these is a place where the bridge could be extended. The most interesting extension theoretically is replacing Mendelian drift with gradient-sampling noise — letting "individual organism = sample of the gradient" become a rigorous claim and directly mirroring SGD's minibatch structure rather than an analog of it.

A complementary extension is making the two-timescale structure of Part F formal. Currently the slow evolution of G is invoked qualitatively (M-matrix tends to align with the adaptive landscape under correlational selection). The formal version would be a coupled system: a fast SDE for z̄ on the current G, plus a slow SDE for G itself driven by the time-averaged selection gradient outer product. This would close the loop between Lande's stochastic breeder's equation and the G-matrix evolution literature, with SOAP as the algorithmic touchstone.

### Open empirical question

Is real biological evolution close enough to the natural-gradient regime that the bridge predicts measurable quantities accurately? The G-matrix evolution literature (Jones, Arnold & Bürger 2007; Arnold et al. 2008) suggests it should be: populations evolve G-matrices that align with the curvature of the adaptive landscape, which is the structural prerequisite for natural-gradient-like dynamics. Independent confirmation comes from the evolution strategies literature (Shir & Yehudayoff 2020; Akimoto et al. 2012), which proves the covariance of selected individuals converges to the inverse Hessian.

For empirical work on neural networks, the parallel question is whether real fine-tuning (especially with SOAP rather than AdamW) produces dynamics that align with the natural-gradient Langevin predictions. If yes, the bridge becomes a working framework for interpreting fine-tuning dynamics in evolutionary terms — ecological fitting, short-term evolvability, long-term evolvability — and the PhenoVision empirical decomposition becomes a test of theoretical predictions, not just a metaphorical mapping.

---

*This document accompanies a working theoretical project on the formal correspondence between AI foundation-model training and quantitative genetic evolution. It is intended to make the chain of reasoning explicit and inspectable, so that subsequent claims (about evolvability, intrinsic dimensionality, free-energy of complexity, etc.) can be built on a solid mathematical foundation.*

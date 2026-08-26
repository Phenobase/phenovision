# Predicting the benefit to M from the statistics of landscape variation

*A working note deriving the form of second-order selection on the mutational covariance M from the
distribution of fitness-landscape fluctuations. Connects the four "forces" on M (framework summary
§6) to a single functional, and identifies two distinct benefit channels that pick out M ∝ Ω vs
M ∝ A⁻¹.*

## 1. The object selection maximizes

Second-order selection on the mutational architecture M maximizes the **long-run (geometric-mean)
growth rate** — the time-average of log mean-fitness:

$$\Lambda(M) = \big\langle \log \bar W_t \big\rangle_t, \qquad \bar W_t = \mathbb E_{z\sim\text{pop}}[\,W_t(z)\,].$$

Population is Gaussian, N(z̄, G), with G = Ĝ(M) = A⁻¹#M the fast-equilibrium standing variance.
Landscape is Gaussian, W_t(z) = exp(−½(z−θ_t)ᵀ A (z−θ_t)), with a fixed curvature A and an optimum
θ_t that varies stochastically in time. The Gaussian integral gives

$$\log \bar W_t = -\tfrac12\log|I+GA| \;-\; \tfrac12\,(z̄-θ_t)^\top \tilde A\,(z̄-θ_t), \qquad \tilde A \equiv A(I+GA)^{-1}.$$

Two terms:
- **Variance load** `V = ½ log|I+GA| ≈ ½ tr(GA)` — the cost of being spread out (∝ A, costly in steep
  directions). On its own this is the **canalization** force (shrinks G/M; framework Force 1 cost).
- **Lag/mismatch load** `L_t = ½ (z̄-θ_t)ᵀ Ã (z̄-θ_t)` — the cost of the mean being off the optimum.

## 2. The mean tracks; M absorbs the residual (the spectral split)

z̄ follows θ_t by the breeder's equation Δz̄ = −GA(z̄−θ_t): a **low-pass filter** with corner rate
~GA. Decompose the optimum's variation by timescale relative to GA:

- **Trackable (slow)** components: z̄ follows them → no residual lag → **no benefit to M.** This is the
  *predictable* part — the mean handles it.
- **Untrackable (fast)** components: z̄ cannot follow → residual lag with stationary covariance Σ_x
  (the high-pass of θ_t at the tracking corner) → **M must absorb them.**

So only the **untrackable** part of the landscape variation creates a benefit for M, and

$$\Lambda(M) \approx -\tfrac12\log|I+GA| - \tfrac12\,\mathrm{tr}\!\big(\tilde A\,\Sigma_x\big), \qquad \Sigma_x = \text{untrackable-fluctuation covariance.}$$

## 3. Two benefit channels

### Channel 1 — population-mean lag absorption (arithmetic mean)
Maximizing the first-order Λ above over G, in the aligned (commuting) case, gives per eigendirection

$$g_i^\star = \sigma_{x,i}^2 - \frac{1}{a_i} \quad\Longrightarrow\quad \boxed{M \propto \Sigma_x \;(=\Omega_{\text{fast}})}$$

(floored at the mutational floor where σ²_{x,i} < 1/a_i). The standing variance **matches the
untrackable fluctuation covariance**, with a −1/aᵢ correction that demands *more* fluctuation to be
worth tracking where selection is weak. Derivation: ∂Λ/∂g_i = 0 ⇒ a_i/(1+g_i a_i) = σ²_{x,i}
a_i²/(1+g_i a_i)² ⇒ 1+g_i a_i = σ²_{x,i} a_i. This is framework **Force 3** (lag-load), generalized to
the untrackable spectrum.

### Channel 2 — lineage wipeout avoidance (geometric mean)
The quadratic lag load misses the catastrophe: when a *clustered* lineage lands far from θ_t,
W̄_t → 0 and log W̄_t → −∞. Because Λ is a **log-mean**, those near-extinction generations dominate.
Avoiding wipeout is a **direction-blind** premium — a lineage is wiped if it is far in *any*
direction, so diversity *everywhere* insures — i.e. the **entropy / log det M** term
ℬ = ½λ log det M. Balanced against the anisotropic variance cost ½ tr(AM):

$$\min_M \tfrac12\mathrm{tr}(AM) - \tfrac{\lambda}{2}\log\det M \;\Longrightarrow\; \tfrac12 a_i = \frac{\lambda}{2 m_i} \;\Longrightarrow\; \boxed{M \propto A^{-1}}.$$

This is framework **Force 1 + non-collapse**, and it is the **third evolutionary noise source**:
not random *which individuals* die (drift, ∝1/Nₑ), not random *gradient estimation* (sampling noise,
∝1/N\*), but random *which trait combinations* are unlucky this generation. It only appears at the
*geometric-mean / lineage* level and is invisible to the first-order arithmetic-mean analysis.

### Channel 0 — within-generation noise (for completeness)
Variation *within* a generation (each individual mis-experiencing the landscape) is **gradient
(sampling) noise** — errors-in-variables on the selection gradient. It *compresses* M's eigenvalue
anisotropy toward isotropy (framework Force 2), pulling toward the **A^{−1/2}** (whitening) end. Sim B.

## 4. The regime map (the answer to "predict the form from the distribution")

The optimal M is predictable from **three features of the landscape's variation**:
(i) its **spectrum** — only the untrackable part counts; (ii) its **covariance** Σ_x — Channel 1
aligns M with it; (iii) its **amplitude relative to the selection width 1/A** — large amplitude
triggers Channel 2.

| Landscape variation | Channel | Optimal M |
|---|---|---|
| fixed (Σ_x = 0) | — (cost only) | M → floor (**canalization**) |
| within-generation noise | 0 (gradient noise) | M compressed → **A^{−1/2}** |
| untrackable fluctuation, **anisotropic Ω** | 1 (lag) | **M ∝ Σ_x (= Ω_fast)** |
| untrackable fluctuation, **isotropic / directionless** | 2 (bet-hedging) | **M ∝ A⁻¹** |

**The discriminator (IBM-confirmed, §5) is the DIRECTIONAL STRUCTURE of the untrackable fluctuation,
not its amplitude per se:**
- **Anisotropic Ω → M ∝ Ω** (Channel 1): the fluctuation has a *predictable direction*, so M tracks
  it; the alignment sharpens as amplitude rises.
- **Isotropic Ω → M ∝ A⁻¹** (Channel 2): there is *no direction to track*, so the only predictable
  structure left to insure against is the **fixed landscape A**, and M hedges to its inverse.
- **Weak challenge (any Ω) → canalization** dominates: M collapses with the transient A⁻¹ shape.

This is the precise sense of "M converges to match the fixed landscape, the only predictable part of
it" — it holds exactly when the perturbation is *directionless*.

## 5. IBM confirmation

A = diag(1, 4); challenge optimum Ω favoring the STEEP direction (aniso = [0.5, 2]); evolved-M ratio
m_flat/m_steep (A⁻¹ target = 4, Ω target < 1):

| σ (amplitude) | ratio m_flat/m_steep | reading |
|---|---|---|
| 0.4 (weak)   | 1.53 | canalization dominates (A⁻¹-shaped collapse) |
| 1.5          | 0.31 | **M follows Ω (steep)** — Channel 1 |
| 2.5 (strong) | 0.04 | **M ∝ Ω (steep), strongly** — Channel 1 |

Isotropic Ω at the same strength gives ratio ≈ 4 (M ∝ A⁻¹). So **M tracks an anisotropic fluctuation
and hedges to A⁻¹ for a directionless one** — both channels confirmed, discriminated by Ω's
directional structure.

## 5b. The remaining prediction (open)

Map the canalization → Channel-1/2 transition vs challenge amplitude and N\* (the evolutionary
α\*(noise) surface), to place beside the optimizer's α\*(batch).

### (archived first-guess predictions)

1. **Anisotropic, weak disaster ⇒ M ∝ Ω, not A⁻¹.** Give the random challenge an anisotropic,
   *weak* covariance Ω; at low amplitude the evolved M should align with Ω (Channel 1), even when Ω
   is anti-aligned with A⁻¹. `tests/test_landscape_channels.py` / `sim/three_regimes.py
   --challenge-aniso`.
2. **Amplitude crossover ⇒ M rotates from ∝Ω toward ∝A⁻¹** as the challenge amplitude rises and
   wipeout-risk turns on (Channel 1 → Channel 2).

## 6. The cross-substrate reading

α (in G ∝ A^{−α}) is set by *which* channel dominates, which is set by the *noise statistics*:
canalization α→floor; gradient noise α→½; lag α set by Ω; wipeout α→1. On the optimizer side the
identical knob is set by batch/curvature-noise statistics — the noise-dependent α\* law. The benefit
to M (evolution) and the optimal preconditioner exponent (optimization) are the *same functional of
the same noise statistics*.

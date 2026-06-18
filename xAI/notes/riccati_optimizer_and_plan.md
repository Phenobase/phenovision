# The Riccati Preconditioner: Theory, Optimizer, and Analysis Plan

A self-tuning preconditioned optimizer derived from the quantitative-genetics / SGD correspondence, the biological theory behind it, an honest account of what is new versus assembled, a stability analysis that explains the full-power-SOAP instability from first principles, and an analysis plan spanning evolutionary simulation and optimizer benchmarks.

---

## Is this a novel optimizer? An honest accounting

Fair characterization: **mostly assembled from components that already exist and have each been tried, unified for the first time by a single theory, plus a small number of genuinely new pieces.** Stated plainly so it can go in a paper without overclaiming.

*Already exists (the deterministic core and its parts):*
- Matrix-free inverse-root iteration instead of eigendecomposition — Shampoo's coupled-Newton option; Muon's Newton–Schulz orthogonalization.
- Kronecker-factored curvature — Shampoo, SOAP, K-FAC.
- Lazy / amortized (warm-started) preconditioner updates — SOAP, distributed Shampoo.
- A tunable preconditioner exponent (removing the square root) — Lin et al. (2024); the earlier `SOAPFullPower`.
- Preconditioned Langevin / SGLD — exists in the Bayesian-DL literature.
- Damping / Levenberg–Marquardt floors, and shrinkage estimation of precision matrices (Ledoit–Wolf) — standard, though shrinkage is rarely framed as an optimizer knob.
- CMA-ES rank-μ covariance adaptation — Hansen.

*Genuinely new (the ties and the new pieces):*
1. **The unifying object.** The single discrete matrix Riccati update $G \leftarrow G + \eta_p(M - GCG)$ *is* the mutation–selection-balance equation for the genetic covariance **and** a preconditioner update, with the source $M$ setting the exponent. Whitening, Newton, Shampoo-via-Newton–Schulz, Muon, and damping all become special cases of one equation with one biological meaning. The unification is the contribution, not any single component.
2. **A derived, noise-dependent exponent schedule.** The two-noise structure predicts a stability boundary at exponent $\alpha = 1/2$ and an optimal exponent $\alpha^*$ that *decreases monotonically as effective batch size falls* — actionable, falsifiable, and the same cross-substrate law tested on the evolutionary side. Tying the empirical instability of full-power SOAP to the regime theory (Part III) is new.
3. **The evolving-M meta-loop.** A slow second optimizer that adapts the preconditioner's *target structure* — the source $M$ — under a second-order objective, via CMA-ES-style accumulation. No standard optimizer adapts the preconditioner's target shape on a separate slow timescale. This is biology's third timescale made algorithmic, and it is the piece with no direct precedent.
4. **Framing the damping floor as shrinkage = effective-exponent reduction**, which both stabilizes and interpolates the exponent, and is the cheap alternative to the true-Fisher fix (Part III).

The honest one-line version: *a known matrix-free Kronecker preconditioner, re-derived from evolutionary theory, equipped with a principled exponent schedule and an optional slow meta-loop that learns the preconditioner's target shape.*

---

## Part I — The biological theory, condensed

The multivariate breeder's equation, $\Delta\bar z = G\beta$, says a population's mean traits respond to selection in proportion to the standing genetic covariance $G$ times the selection gradient $\beta$. Under a Gaussian fitness landscape with curvature $A$ (stabilizing selection), $\beta = -A(\bar z-\theta)$, and three matrices govern everything: $G$ (standing genetic variation), $M$ (per-generation mutational input), and $A$ (landscape curvature).

Two facts make this an optimizer. First, $\Delta\bar z = G\beta$ is exactly a **natural-gradient** ascent: the Fisher metric of the trait distribution's mean is $G^{-1}$, so $G\beta$ is the metric-corrected gradient. Second, the standing variation obeys a matrix **Riccati** equation,

$$\dot G = \underbrace{M}_{\text{mutation}} - \underbrace{GAG}_{\text{selection}},$$

whose equilibrium $GAG = M$ is the matrix geometric mean $\hat G = A^{-1}\#M$. Aligned, $g_i = \sqrt{m_i/a_i}$: more variation where the landscape is flat.

Letting the mutational architecture $M$ itself evolve (it is partly heritable) adds a slow third timescale. Solving for the $M$ that second-order selection favors gives three regimes:

| Environment | Evolved M | Standing G scaling | Reading |
|---|---|---|---|
| Static peak, no premium on variation | $M\to$ floor (collapse) | — | canalization |
| Optimum drifting in a direction $\Omega$ | $M \propto \Omega$ | depends on $\Omega$ | track the moving target |
| Fluctuating / uncertain (variation has value) | $M \propto A^{-1}$ | $g_i \propto a_i^{-1}$ | full inverse-curvature (CMA-ES regime) |

Finiteness adds two noises to the mean's dynamics: demographic (drift, $\propto G/N_e$) and **gradient noise** ($\propto GCG/N^*$, where $N^*$ is an effective sampling size and $C$ the gradient variability, with $C=A$ near the optimum by the score identity). The gradient noise is the crucial term for what follows. Full derivations: the primer and the `coupled_zgm_sde` / `adiabatic_reduction_M` documents.

---

## Part II — The optimizer

**The mapping.** Training is descent on a loss landscape: $A\leftrightarrow$ loss curvature (Hessian), $\beta\leftrightarrow -\nabla L$, $G\leftrightarrow$ preconditioner $H$, $N^*\leftrightarrow$ batch size $B$, gradient noise $\leftrightarrow$ minibatch noise. The breeder's equation and preconditioned SGD are the same SDE.

**The core update.** Two clocks: outer step $t$ (weights, curvature, source), inner relaxation $j$ (the Riccati). With warm-starting,

$$G_t^{(0)} = G_{t-1}, \qquad G_t^{(j+1)} = G_t^{(j)} + \eta_p\big(M_t - G_t^{(j)} C_t G_t^{(j)}\big), \qquad W_{t+1} = W_t - \eta\, G_{L,t}\, g_t\, G_{R,t},$$

with $C_t = \beta_c C_{t-1} + (1-\beta_c)\, g_t g_t^\top$ the curvature EMA (Kronecker-factored: $C_{L} \propto gg^\top$, $C_{R}\propto g^\top g$). The fixed point $G C G = M$ makes the **source the exponent**:

$$M = \varepsilon I \Rightarrow G\to C^{-1/2}\ (\alpha=\tfrac12,\ \text{whitening; the Muon/Adam regime}),\qquad M = G \Rightarrow G\to C^{-1}\ (\alpha=1,\ \text{Newton}).$$

In general, targeting $G = C^{-\alpha}$ needs source scaling $M = G^{(2\alpha-1)/\alpha}$. The two endpoints are clean (Newton–Schulz inverse and coupled Newton–Schulz inverse-sqrt); intermediate exponents are reached *in practice* by shrinking $C$ toward isotropy (the parameter $\rho$ below), which is simultaneously the stabilizer (Part III). No eigendecomposition, ever — only matmuls.

Validated numerically (numpy): the inverse iteration reproduces $C^{-1}$ to $2\times10^{-15}$, and the Kronecker preconditioned descent converges.

**How M is updated — three modes.**
- *(a) Fixed source* $M=\varepsilon I$: no update; $G\to C^{-1/2}$; $\varepsilon I$ doubles as the Levenberg–Marquardt floor and the biological mutational floor.
- *(b) Source slaved to G* $M=G$: Newton–Schulz inverse, $G\to C^{-1}$. $M$ is not an independent variable.
- *(c) Evolving M (the meta-loop, non-standard):* $M$ is a slow learned matrix updated by CMA-ES-style rank-μ accumulation of *productive* steps,
$$M_{t+1} = (1-\eta_M)M_t + \eta_M\sum_i w_i\, s_i s_i^\top,\quad s_i = \text{applied step},\ \ w_i\propto\max(0,\,\Delta L_i),\ \ \eta_M\ll\eta_p.$$
Directions that repeatedly reduce loss get amplified in the source; near a basin these are the low-curvature directions, driving $M\to C^{-1}$ structure — the exploration fixed point, reached by temporal averaging (variance-reduced) rather than per-step inversion.

**Efficiency.** Kronecker factoring (two small flows per layer); warm-starting (1–3 inner steps because $C$ drifts slowly — replaces $O(d^3)$ eigendecomposition with a few matmuls, the Muon argument); and an optional low-rank mode where $GCG$ needs only $r$ Hessian-vector products. Implementation: `riccati_precond.py`.

---

## Part III — Stability: why full-power SOAP exploded, and whether this version fixes it

**The empirical finding.** `SOAPFullPower` at exponent $\alpha=1$ (full inverse) blew up in flat directions, even with damping, worst at low batch; the fix was the true Fisher (clean curvature via sampled-label gradients) at ~2× forward/backward cost. The question: does the inversion-free Riccati form avoid this?

**Short answer: no, and it shouldn't, because the instability is in the *exponent* (the regime), not in the inversion method.** The Riccati iteration toward $C^{-1}$ converges to the same unstable target as direct inversion, and computing it *more* accurately makes things *worse*. Here is the analysis with numbers.

**The $\alpha=1/2$ stability boundary, derived.** The preconditioned update noise is $-\eta\, C^{-\alpha}\xi$ with $\mathrm{Cov}(\xi)=\Sigma_\xi/B$. Near the optimum $\Sigma_\xi\approx A$ (score identity) and $C\approx A$, so the update-noise covariance is

$$\eta^2\, C^{-\alpha}\Sigma_\xi C^{-\alpha}/B \ \approx\ \eta^2\, A^{1-2\alpha}/B.$$

In a flat direction (small $a_i$) the variance scales as $a_i^{\,1-2\alpha}$. Therefore:

$$\alpha < \tfrac12:\ \text{flat-dir noise suppressed};\qquad \alpha=\tfrac12:\ \text{bounded};\qquad \alpha>\tfrac12:\ \text{amplified as } a_i^{\,1-2\alpha}.$$

At $\alpha=1$ the amplification is $a_i^{-1}$ — unbounded as curvature vanishes. This is exactly the framework's gradient-noise term $GCG/N^* \propto C^{1-2\alpha}$, and it is the first-principles explanation of the full-power instability. **$\alpha=1/2$ is the critical exponent; full inverse is on the wrong side of it.**

**Numerical confirmation (diagonal quadratic, curvatures spanning $10^{-3}$–$10^{1}$, A-shaped gradient noise):**

| α | batch | flat-direction wander $|x_{\text{flat}}|$ | steep-direction $|x_{\text{steep}}|$ |
|---|---|---|---|
| 0.5 | 256 | 0.11 | 0.045 |
| 0.5 | 16 | 0.21 | 0.034 |
| 0.75 | 16 | 0.79 | 0.034 |
| 1.0 | 256 | 2.39 | 0.077 |
| 1.0 | 16 | **4.36** | 0.033 |

Flat-direction wandering grows ~20× from $\alpha=0.5$ to $\alpha=1$ and is worse at low batch, while steep directions are unaffected — the exact "exploding flat directions" signature. (This stationary toy has a restoring force so it does not run to infinity; in a real network the non-stationary curvature and direction-starvation turn this relative amplification into hard divergence.)

**The inversion-free form does not help, and can hurt.** On an *anisotropic* least-squares problem (genuinely small Hessian eigenvalues), the $\alpha=1$ (inverse) mode **diverged even at full batch** (loss $\sim 10^5$–$10^6$), while $\alpha=1/2$ (whitening) stayed stable (loss 2.2). Crucially, increasing the inner Newton–Schulz steps from $k=2$ to $k=6$ made divergence **worse** (loss $\sim 10^6$ → $\sim 10^6$, larger): a more accurate $C^{-1}$ is a more faithfully unstable preconditioner. An under-converged warm-started iteration is *accidentally* regularizing; converging it removes that accident. The inversion-free iteration also carries a conditional-convergence failure mode — warm-starting a stale large $G$ once $C$ recovers can overshoot (Newton–Schulz diverges outside $\|CG_0\|<2$), which the spectral safeguard in `riccati_precond.py` exists to catch.

What the relaxation *does* buy is mild low-pass filtering of high-frequency curvature noise (G moves toward the target at a finite rate, so single-step troughs in $C$ produce only partial spikes). That helps transient noise, not persistently low-signal directions. Net: still unstable at $\alpha=1$, low batch.

**The cheap fixes the theory points to (no extra passes).** Two distinct failure modes share the same cures: *noise amplification* (Russell's low-batch case; cured by clean curvature or lower $\alpha$) and *conditioning* (small true eigenvalues; cured by damping/shrinkage or lower $\alpha$). At the $\alpha=1$, $B=16$ corner:

| Fix | max\|x\| | loss |
|---|---|---|
| naive (damping $10^{-8}$) | 19.2 | 0.27 |
| heavy damping ($10^{-2}$) | 1.50 | 0.11 |
| shrinkage $\rho=0.3$ | 1.05 | **0.10** |
| shrinkage $\rho=0.7$ | 1.02 | 0.17 |

and $\alpha$-backoff is monotone (loss 0.27 → 0.10 → 0.064 → 0.050 as $\alpha: 1\to 0.75\to 0.6\to 0.5$). **Shrinkage toward isotropy is the cheapest, most robust lever** — it is Ledoit–Wolf precision shrinkage, it is the biological mutational floor, and it is an effective-exponent reduction in flat directions, all the same operation. The true-Fisher fix works because it makes the curvature clean enough that $C^{-1}$ is well-conditioned and $\alpha=1$ becomes safe; the framework offers the cheaper route of **not demanding the full inverse where the data cannot support it** — the noise-dependent $\alpha^*$ schedule (`RiccatiPrecond.shrink_from_batch`).

**Recommendation.** Default to $\alpha=1/2$ (whitening) for stability. Push the effective exponent toward 1 *only* in directions/regimes with clean signal — via the $\rho$ schedule (more shrinkage at small batch), or via the evolving-M meta-loop, which sharpens only directions with consistent productive signal and leaves genuinely flat directions at the floor. The meta-loop is the principled way to get inverse-curvature *structure* in the well-sampled subspace without the global $\alpha=1$ instability, and is the first thing to test (O4 below).

---

## Part IV — Analysis plan

Organizing spine: a single cross-substrate prediction — **the optimal preconditioner exponent decreases monotonically as effective sample size decreases** — tested on both substrates. On the evolutionary side the "sample size" is $N^*$ (and the regime is set by environmental structure); on the optimizer side it is batch $B$.

### Strand A — Evolutionary simulation (individual-based; NumPy prototype, JAX for sweeps)

Prototype the generation loop in NumPy; port to JAX with `vmap` over replicate ensembles for the sweeps (use Numba instead if per-individual branching dominates). Track $\bar z$, the realized $G$, and the evolving $M$ over $\sim10^4$–$10^5$ generations.

- **E1 — Regime discrimination.** Three environments: static optimum; optimum drifting with covariance $\Omega$; optimum fluctuating (stationary, variance-bounded). *Predict:* $M\to$ floor (canalization) static; $M\propto\Omega$ tracking; $M\propto A^{-1}$ fluctuating. *Done when:* the evolved $M$ eigenstructure matches the predicted target (cosine alignment $>0.9$) in each regime across $\geq 20$ replicates.
- **E2 — Eigenvalue scaling.** Measure the exponent $\alpha$ in $G\propto A^{-\alpha}$ with $M$ fixed-isotropic vs. evolving. *Predict:* $\alpha\approx 1/2$ (fixed isotropic $M$) rising to $\alpha\approx 1$ (evolved $M$, exploration regime). *Done when:* fitted $\alpha$ separates the two conditions with non-overlapping CIs, and **eigenvalue scaling is shown distinct from eigenvector alignment** (report both; the scaling is the novel claim).
- **E3 — Two-noise signatures on M–A alignment.** Separate gradient-noise vs demographic-noise dominated regimes. *Predict:* gradient noise acts as regression dilution (compresses the eigenvalue spread of $M$ relative to $A$); demographic noise acts as Langevin diffusion (inflates variance without bias). *Done when:* the two signatures are distinguishable in the $M$–$A$ alignment statistics.
- **E4 — Phase diagram.** Sweep $N^*$ and $\Omega$; locate the canalization↔exploration boundary and the inferred $\alpha^*(N^*)$. *Done when:* the boundary is mapped and $\alpha^*$ is shown to **decrease as $N^*$ decreases** — the evolutionary half of the cross-substrate prediction.

### Strand B — Optimizer experiments (PyTorch; `riccati_precond.py`)

Models: a small MLP and a small transformer (e.g. a ~10–50M GPT) on a standard text task; ablations also on CIFAR for speed. Baselines: AdamW, Shampoo/SOAP, Muon, and `SOAPFullPower` at $\alpha=1$ with true Fisher.

- **O1 — Stability boundary.** Sweep effective exponent (via $\rho$) × batch size; measure divergence rate and the flat-direction update-noise scaling. *Predict:* boundary near $\alpha=1/2$; flat-direction noise $\propto a^{1-2\alpha}$; $\alpha^*(B)$ decreasing in $1/B$. *Done when:* the $a^{1-2\alpha}$ scaling is recovered and $\alpha^*(B)$ is mapped.
- **O2 — Cheap schedule vs true Fisher.** Compare fixed $\alpha=1$+true-Fisher (2× cost) against $\alpha=1/2$, and against the $\rho$-schedule (`shrink_from_batch`) at 1× cost. *Predict:* the $\rho$-schedule recovers most of the true-Fisher benefit at ~1× cost. *Done when:* loss-vs-wallclock curves are plotted; report the fraction of the true-Fisher gain recovered.
- **O3 — Shrinkage = effective exponent.** Hold the base mode at `inverse`, sweep $\rho$; confirm $\rho$ both stabilizes and smoothly reduces the operative exponent (measure the spectrum of the realized preconditioner). *Done when:* realized-preconditioner exponent moves from ~1 toward ~1/2 as $\rho\to1$, with divergence vanishing.
- **O4 — Evolving-M meta-loop.** Turn on `evolve_M`; sweep the timescale ratio $\eta_M/\eta_p$ (and `meta_every`). *Predict:* the meta-loop reaches stable approximate-inverse-curvature *structure* in the well-sampled subspace where naive $\alpha=1$ diverges, beating $\alpha=1/2$ on convergence at equal stability. *Done when:* meta-loop matches or beats SOAP/Shampoo final loss at $\alpha=1/2$-level stability; identify the stable $\eta_M/\eta_p$ window (too fast → chases noise; too slow → never leaves $\alpha=1/2$).
- **O5 — Warm-start efficiency.** Wall-clock and final-loss vs Shampoo (eigendecomposition) and Muon at matched compute, sweeping inner steps $k\in\{1,2,3\}$. *Done when:* matmul-only warm-start matches Shampoo quality at lower step cost.
- **O6 (optional) — Langevin sampling.** Turn on `langevin`; check posterior calibration on a small Bayesian regression/classification task with the *learned, evolving* metric. *Done when:* calibration (e.g. ECE / coverage) is competitive with preconditioned SGLD.

### Build order
1. NumPy IBM → E1, E2 (establish the regimes and the scaling, the headline biological result).
2. `riccati_precond.py` smoke test → O1, O3 (stability boundary and the shrinkage knob — the immediately useful, low-risk wins that also retire the full-power instability question).
3. O2 (the practical payoff: cheap schedule vs true Fisher).
4. JAX port → E3, E4 (sweeps, the phase diagram, the evolutionary half of the cross-substrate law).
5. O4 (the novel meta-loop — highest-variance, highest-upside).
6. Cross-substrate figure: $\alpha^*$ vs effective sample size, evolution (E4) and optimizer (O1/O2) on one axis.

---

## Appendix — code

The optimizer is in `riccati_precond.py`. Two harnesses for the first experiments.

**O1 stability-boundary diagnostic** (diagonal surrogate; the cheapest reproduction of the instability and the $\alpha=1/2$ boundary):

```python
import numpy as np
def flat_dir_amplification(alpha, B, d=24, T=4000, eta=0.05, beta=0.95,
                           delta=1e-8, shrink=0.0, sigma2=1.0, seed=0):
    """Returns flat- and steep-direction wander for preconditioned SGD on a
    diagonal quadratic with A-shaped gradient noise. Demonstrates update-noise
    variance ~ a^{1-2 alpha}: alpha>1/2 amplifies flat directions, worse at low B."""
    r = np.random.default_rng(seed)
    a = np.logspace(-3, 1, d); x = np.ones(d); C = np.full(d, 1e-3)
    for _ in range(T):
        g = a * x + r.standard_normal(d) * np.sqrt(a * sigma2 / B)
        C = beta * C + (1 - beta) * g * g
        Cs = (1 - shrink) * C + shrink * C.mean()
        x = x - eta * (Cs + delta) ** (-alpha) * g
    return abs(x[0]), abs(x[-1])      # flat, steep
# sweep alpha x B; expect flat-dir wander to grow with alpha (>1/2) and 1/B,
# steep-dir wander flat. Then verify shrink=0.3 stabilizes alpha=1.
```

**O2 schedule-vs-Fisher experiment skeleton** (PyTorch, fill in model/data):

```python
import torch
from riccati_precond import RiccatiPrecond

def make_opt(kind, model, batch_size):
    if kind == "whiten":                       # alpha = 1/2, stable baseline
        return RiccatiPrecond(model.parameters(), lr=3e-3, precond="whiten")
    if kind == "inverse_fisher":               # alpha = 1 + clean curvature (2x cost)
        return RiccatiPrecond(model.parameters(), lr=3e-3, precond="inverse",
                              shrink=0.0)       # pair with a true-Fisher grad pass
    if kind == "schedule":                     # alpha interpolated by batch-dependent rho (1x cost)
        rho = RiccatiPrecond.shrink_from_batch(batch_size, ref_batch=512, rho_max=0.6)
        return RiccatiPrecond(model.parameters(), lr=3e-3, precond="inverse",
                              shrink=rho, inner_steps=2, safeguard=8.0)
    if kind == "evolve":                       # the meta-loop (O4)
        return RiccatiPrecond(model.parameters(), lr=3e-3, precond="inverse",
                              shrink=0.2, evolve_M=True, eta_M=1e-3, meta_every=20)

# true-Fisher gradient (for inverse_fisher): sample labels from the model's own
# predictive distribution, backprop the resulting loss -> clean curvature in C.
# This is the 2x-cost path the schedule is meant to approximate at 1x.
```

The comparison that settles the practical question: **does `schedule` (1× cost) recover most of `inverse_fisher`'s (2× cost) advantage over `whiten`?** If yes, the noise-dependent exponent is the cheap substitute for clean Fisher. If only `evolve` closes the gap, the meta-loop is the contribution that earns its keep.

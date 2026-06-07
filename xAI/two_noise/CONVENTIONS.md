# `two_noise` — Conventions (single source of truth)

These conventions resolve the most dangerous cross-paper / cross-substrate ambiguities. They are
binding for all code in `xAI/two_noise/`. When in doubt, this file wins; cite it in docstrings.

## 1. `A` vs `ω` (selection surface)

- **`A`** = selection-surface **curvature** = the matrix in Gaussian stabilizing selection
  `W(z) = exp(-½ (z-θ)ᵀ A (z-θ))`. `A` is PSD. Larger eigenvalue ⇒ sharper selection ⇒ stronger
  constraint in that direction.
- **`ω`** = selection **width** matrix. `A = ω⁻¹`.
- **Rule:** use `A` (curvature) everywhere internally. If a value enters as a width `omega`, invert
  it to `A` *at the boundary* with an explicit comment. Never let `omega` propagate into the core.
- This is a documented source of confusion across the QG literature — the convention block is
  repeated at the top of `sim/engine.py`.

## 2. `A` vs `C` (the Fisher identity)

- **`A`** = curvature (Hessian of the loss / negative Hessian of log mean fitness).
- **`C`** = per-sample gradient (noise) covariance.
- Theory uses **`C = A`** (true Fisher = Hessian) *at the optimum, correctly specified*. This is an
  **approximation** (empirical Fisher ≠ true Fisher; `E[g]≠0` away from the optimum).
- **Rule:** code keeps `A` and `C` as **separate objects**. Assert `C == A` only inside tests, never
  silently in production paths. `curvature/true_fisher.py` computes both and compares — the
  empirical-vs-true-Fisher caveat is structural, not buried.

## 3. The exponent `α`

- `α` = `precond_power` in the optimizer (`optim/soap_full_power.py`): `0` SGD, `0.5` whitening
  (vanilla SOAP), `1.0` full inverse / natural gradient.
- `α` = the G–A eigenvalue exponent in biology: `G ∝ A^(-α)`.
- Same symbol, two substrates — that identity *is* the paper. The shared-prediction figure's x-axis
  is **effective sample size** (`N*` in biology / effective batch in ML); the y-axis is the realized
  `α` (or `M`-anisotropy).

## 4. Analytic targets (from `notes/two_noise_framework_summary.md`)

- α-parameterized OU SDE: `dθ = -C^(1-α)(θ-θ*) dt + (1/√S) C^(-α) B dW`, `BBᵀ = C`.
- Per-eigendirection stationary variance: **`V_stat(a) = a^(-α) / (2S)`** (drift `a^(1-α)`,
  diffusion `a^(1-2α)/S`). ⇒ log-log slope of `V_stat` vs curvature eigenvalue `= -α`.
- Demographic-noise / pSGLD: `dθ = -H∇L dt + √(2T·H) dW` ⇒ Gibbs `π ∝ e^(-L/T)`. Inject noise with
  **covariance ∝ H** (the *applied* preconditioner `V^(-α)` in the eigenbasis; per-coordinate std
  `V^(-α/2)`), temperature `T ∝ lr`. Injecting covariance `∝ √H` would **break** the Gibbs form.
- FDT condition: total diffusion `∝ G` iff `C ∝ G^(-1)` (⇒ `G = A^(-1)` under Fisher).

## 5. RNG discipline

- **PyTorch:** thread an explicit `torch.Generator` through every stochastic op. Parity and
  demographic-noise tests require bit-reproducibility; never rely on global RNG state.
- **JAX:** explicit `jax.random.PRNGKey` split lattice, never global state. This is also what makes
  `vmap`-over-replicates correct (each replicate gets an independent split key).

## 5.5. Conda environments (SPLIT — verified necessary)

PyTorch and JAX **cannot coexist** in one env here: torch 2.6+cu124 pins cudnn 9.1.0.70 while
jax 0.10.1 needs 9.23, and JAX GPU init fails (`dnn_support != nullptr`) when torch wins. So:

- **`two_noise`** — Strand 1 (PyTorch): optim/, toy/, curvature/, ml_experiments/. torch 2.6
  +cu124, timm, transformers, numpyro, pyhessian, cma, matplotlib, hydra, pytest.
- **`two_noise_jax`** — Strand 2 (JAX, GPU): sim/. jax[cuda12], numpy/scipy/pandas, cma, pytest.

Run with `mamba run -n two_noise ...` / `mamba run -n two_noise_jax ...`. The strands exchange
only tidy CSVs (then R draws figures), so the split is invisible to the science.

## 6. Units & matrix sign

- Loss `L = ½ θᵀ A θ` with `A` PSD; the optimizer's in-basis second moment `V` estimates curvature
  eigenvalues, so the applied preconditioner is `V^(-α)` (curvature to a negative power).
- All covariances are symmetric PSD; eigen-decompositions sort **descending** (column 0 = top).

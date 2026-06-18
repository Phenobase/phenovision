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
- **Shrinkage = effective-exponent reduction (`shrink`/`rho`).** Both optimizers take a `shrink`
  knob: shrink the curvature spectrum toward isotropy (`v → (1-ρ)v + ρ·mean(v)`) *before* the
  power. This yields a **spectrally non-uniform** effective exponent — ≈`α` in steep directions,
  →0 in flat ones — which is exactly the noise-aware preconditioner the SDE predicts (the α=1
  update-noise `aᵢ^(1-2α)` blows up in flat directions; shrinkage selectively tames it there).
  `SOAPFullPower` applies it **exactly** on the eigenvalues; `RiccatiPrecond` matrix-free via NS.
- **evolve-M lives in the eigenbasis as a diagonal (`evolve_m`).** Both optimizers learn the
  preconditioner's target shape `M` by loss-weighted CMA-ES rank-μ accumulation of productive steps.
  In SOAP the matrix `M` collapses to per-axis gains `m_L, m_R`: applied `h_ij = √(m_L,i m_R,j)·v_ij^{-power}`
  (whitening base × learned gain, `m` unit-mean-normalized so only anisotropy acts). Use
  `precond_power=0.5`. The operative exponent is read by `operative_exponent_soap` (validated:
  whiten→0.5, inverse→1.0 *exactly*, unlike the matrix-free path). Run via
  `riccati_evolve_m.py --optimizer soap`.
- **Matrix-free vs exact (important, 2026-06-10).** The matrix-free Riccati whiten
  *under-converged on real ViTs* (operative exponent ≈0, `‖GCG−I‖≈1`, G stuck near I). ROOT CAUSE
  (diagnosed via `ml_experiments/riccati_precond_convergence.py`): `_ns_inv_sqrt` normalized the
  Higham iteration by `trace(C)` instead of the spectral norm; for a ~384-dim ViT factor `trace`
  is ~`d×` looser than `λ_max`, so 10 iterates left `Z≈I`. FIXED: normalize by `_spec_norm(C)`
  (regression test `test_ns_inv_sqrt_high_dim_converges_in_few_steps`). The diagnostic was
  budget-independent (k2=k5=k10, all clamped ≥10) — the tell that it was normalization, not steps.
  → Still, for the shrinkage **science** (O2/O3) prefer `--optimizer soap` (`SOAPFullPower`, exact
  spectrum, zero convergence risk); the matrix-free path is the "and it's cheap" follow-on, now
  that whiten converges (re-verify inverse/evolve on GPU before trusting O4's matrix-free numbers).
- **Relative damping is the default EVERYWHERE (`relative_damping=True`, 2026-06-10).** Both
  optimizers floor the curvature as a FRACTION of its spectral norm (`eps·λ_max`), not an absolute
  `eps`. Real ViT factors are small-scale (eigenvalues ~1e-4..1e-2), so an *absolute* `1e-2` floor
  swamps them and the inverse collapses to `~eps⁻¹·I` (realized exponent ~0, behaves like SGD) —
  this, NOT NS non-convergence, was why the matrix-free *inverse* read op≈0 (whiten's bug was the
  separate trace-norm/iteration one). Diagnosis: `(C+εI)⁻¹` exponent 0.13 (abs) vs 0.87 (rel) on
  eigenvalues 1e-4..1e-2. `damping=1e-2` now means `1e-2·λ_max`. Toy unit tests that tuned an
  absolute floor pass `relative_damping=False` explicitly. SOAP already did this — which is why its
  inverse partially worked while matrix-free didn't.

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

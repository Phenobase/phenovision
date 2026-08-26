# Two-Noise Phase 2 — Evolution / IBM Arm Execution Report

**To:** the author of the SDE-validation plan.
**From:** the team that executed the plan against the real repo.
**Companion to:** `phase2_execution_report.md` (the optimizer arm). This one covers the **biology /
individual-based-model (IBM) arm** — the evolvable-M IBM, the V0–V6 validation ladder, Sim A/B/C,
the three-regime and landscape-channel tests, and the `ℒ_eff` theory module. Assumes you know the
theory (three timescales; `ℒ_eff`; canalization/tracking/exploration; the four forces; the
two-channel landscape-fluctuation note). Fills you in on what happened when it was simulated.

> **One-paragraph version.** The **fast (z̄) and medium (G) timescales validate cleanly and
> beautifully** — the matrix-geometric-mean closure `G=Ĝ(M)=A⁻¹#M`, `Σ_∞∝A⁻¹`, the Gaussian
> closure, the regime fixed points, and the two benefit channels (track-Ω vs hedge-to-A⁻¹) all
> hold, the last being the single nicest biological result. The **slow (M) timescale — the keystone
> "M dynamically evolves to A⁻¹" — does not directly validate as a *process*: M-architecture
> evolution is too slow to observe at accessible compute** (Sim A's own conclusion), the
> exploration-regime drift only weakly aligns with `−𝒢_M∇ℒ_eff` at pilot/burst scale (cos ≈ −0.3),
> and the adiabatic ε→0 sweep does not show the predicted clean improvement. So the IBM confirms
> the *theory's fixed points and the fast/medium dynamics* and the *static→exploration regime
> structure*, but the slow second-order M-flow remains **inferred, not watched.** This exactly
> parallels the optimizer arm, where `evolve_M` also failed to be the headline — in both substrates
> the third (M) timescale is the hard, unconfirmed part.

---

## 1. What was built (all in `sim/`, tested in `tests/`)

- **`engine.py`** — the JAX fixed-M IBM (untouched by Phase 2; the substrate).
- **`evolvable_m.py`** — the keystone: an IBM whose **mutational architecture M is itself heritable
  and evolves.** Three encodings (`eig_diag`, `eig_rot`, `chol`); anisotropic mutational injection;
  a genuine emergent bet-hedging mechanism (a shared random-disaster challenge each generation).
- **`theory_load_surface.py`** — the analytic `ℒ_eff(M)=κ(N*)·tr(A·Ĝ(M))−ℬ(M)`, `Ĝ=A⁻¹#M`,
  `Σ_∞`, the three regimes, `grad_L_eff`, natural gradient. The ground truth the ladder checks against.
- **`drift_estimator.py`** (V0), **`v1_fitness_scan.py` … `v6_trajectory.py`** (V1–V6),
  **`three_regimes.py`**, **`sim_a_exponent.py`**, **`sim_b_compression.py`** / `sim_b_phase.py`,
  **`directional_alignment.py`**, **`exploration_scaling.py`**.
- Two working notes: `notes_landscape_fluctuation_benefit.md` (the two-channel theory) and
  `notes_evolution_vs_ml.md` (the cross-substrate synthesis).

Run in the JAX env (`mamba run -n two_noise_jax`, `JAX_PLATFORMS=cpu`).

---

## 2. The theory module is exact (`ℒ_eff` identities — the de-risking gate)

`tests/test_theory_load_surface.py` checks the analytic spine to ~1e-9:

- `Ĝ A Ĝ = M` (matrix-geometric-mean / MSB identity), **commuting and non-commuting** A,M.
- Aligned closed form `Ĝ_i = √(m_i/a_i)` (and Ĝ diagonal at diagonal M).
- `Σ_∞` solves the Lyapunov equation; aligned grad == finite-difference grad.
- **Regime fixed points exact:** exploration `m_i* = (λ/κ)²/a_i` ⇒ `m_i a_i = const` ⇒ **M∝A⁻¹**;
  tracking `m_i ∝ ω_i` ⇒ **M∝Ω**; static gradient > 0 ⇒ **M→floor (canalization)**.

So the *predictions* the IBM must reproduce are themselves verified analytically first. This held.

---

## 3. The evolvable-M IBM (keystone) — mechanically correct

`tests/test_evolvable_m.py` — the keystone gate is **realized M == encoded M**: an IBM whose
modifiers encode a target M reproduces it to `atol 1e-6`. Plus:

- Anisotropic injection covariance matches the encoded M (`atol 0.05`).
- SPD always under the Cholesky design; determinism; vmap==loop; heritability (M moves off init and
  `tr(𝒢_M)` is positive, i.e. there is real second-order genetic variance to select on).
- Modifier param counts correct per design (eig_diag 2, eig_rot 3, chol 3 for d=2).

**Status: the machine works** — you *can* build a population whose mutational architecture is
heritable and whose realized M equals what the modifiers encode. This is the genuinely new
simulation capability the plan called for, and it is sound.

---

## 4. Fast (z̄) and medium (G) timescales — clean, with one honest constant

This is the strongest part of the arm.

**V0 — drift estimator** (`tests/test_drift_estimator.py`, on the Gaussian surrogate whose true
drift is `−𝒢_M∇ℒ_eff` by construction): recovery direction **cos > 0.999**; the **metric
correction is necessary** (corrected cos > 0.99 vs bare-gradient cos lower by >0.05); injected
metric recovered (cos > 0.99); scaling/rotation split isolates components (each frac > 0.99, other
< 0.05); curl detector ≈0 for gradient jumps, jumps for rotation. The estimator is trustworthy — so
a low cosine downstream indicts the *theory or the IBM*, not the measurement.

**V1 — fixed-M fitness scan** (`runs/v1_fitness_scan`): G is diagonal under diagonal A; the IBM
fitness `U_IBM(M)` aligns with `−ℒ_eff` (version-B cosine > 0.55); the closure is **right in shape**
(G/Ĝ ratio near-constant across the M grid) but the IBM variance **exceeds** the bare
mutation-selection-balance closure (ratio mean > 1.2). I.e. the *direction* is right; there is a
scale constant.

**V2 — fast equilibria + closure** (`runs/v2_closure/results.csv`, 8 cells, commuting +
non-commuting, with A-rotation): the headline closure result —
- **G aligns with Ĝ=A⁻¹#M: `shape_cos` = 0.991–1.000** (essentially exact in shape).
- **Σ_∞ ∝ A⁻¹: `sigma_shape_cos` = 0.957–1.000.**
- The shape/scale discrepancy is a **single near-constant factor: `scale_factor` ≈ 2.47–3.14,
  mean ≈ 2.9** (the calibratable Gaussian-approximation coefficient; CV < 0.2 across M / anisotropy
  / rotation). The IBM standing variance exceeds the bare MSB closure by this ~2.8–2.9×.
- **Gaussian closure holds:** phenotype skew < 0.4, excess kurtosis < 1.0 everywhere.

**Reading:** the `(z̄, G)` part of the three-timescale theory is correct up to a single calibratable
constant. That constant (~2.8–2.9) is a real, reportable result, not a failure — it's the
Gaussian-closure coefficient relating MSB variance to the realized IBM variance.

---

## 5. Slow (M) timescale — where it gets hard (the honest part)

**V3 — M-drift direction** (`runs/v3_drift/results.csv`): **static / canalization aligns well —
`cos_drift_vs_theory = 0.98`, pure scaling (scaling_frac 1.0, rotation_frac 0.0).** But the
**exploration regime does NOT align at this scale: `cos = −0.30`.** So the IBM's M-drift matches
`−𝒢_M∇ℒ_eff` for the canalization (shrink) direction but not for the exploration (climb-to-A⁻¹)
direction — the second-order signal there is too weak / noisy relative to the drift the estimator
can resolve at burst-array scale.

**V5 — adiabaticity** (`runs/v5_adiabatic/results.csv`): the prediction is that the reduced
gradient-flow approximation **improves as ε→0**. The production sweep shows the **opposite** —
`cos_drift_vs_theory` = 0.98 → 0.90 → −0.34 → −0.91 as ε descends (0.457 → 0.057). So the clean
adiabatic limit is **not** observed; at small ε the (exploration-regime) drift signal degrades into
noise, same failure mode as V3-exploration. (The pilot-config unit test asserts only a weak/robust
monotone and a small-config sanity bound, which is why it passes while the production sweep does
not show the clean story.)

**V4 — curl** (`runs/v4_curl`): aligned-case curl fraction is small (test bound < 0.35; finite,
well-formed) — consistent with the drift being approximately a gradient flow in the aligned regime.
**V6 — SDE vs IBM** (`runs/v6_trajectory`, static regime): the Euler–Maruyama reduced SDE matches
the IBM ensemble — sliced-Wasserstein on the M-eigenvalue cloud = **0.011** (near-zero), tr(M)
shrinks in both. So for the *static* regime the reduced SDE is a faithful surrogate; the
exploration/tracking trajectories at production scale inherit the V3/V5 weakness.

**The headline caveat (from `notes_evolution_vs_ml.md` §5, and consistent with the above):**
> **M-architecture evolution is too slow to observe at accessible compute.** Sim A's own conclusion
> is that the second-order signal driving M is too slow to watch directly. So the framework's
> "second-order architecture evolution" — the keystone biological claim that `M` *dynamically*
> evolves toward `A⁻¹` — **remains a conjecture supported by fixed-point + regime evidence, not a
> directly observed dynamical process.**

---

## 6. The regime structure and the two channels — the nicest biological result

Where the slow M-flow *direction* is hard to resolve, the **regime fixed points and the channel
discriminator are clean**, because they're equilibrium/structural rather than slow-rate signals.

**`three_regimes.py`** (`runs/three_regimes/results.csv`, A with a_steep/a_flat = 4 so the A⁻¹
target ratio = 4):

| regime | `ratio_flat_over_steep` | `dM_flat` | reading |
|---|---|---|---|
| canalization | 2.79 | −0.011 | M shrinks |
| **exploration** | **4.97** | **+0.043** | **flat grows → M∝A⁻¹ (target 4)** |
| tracking | 2.32 | −0.016 | leans per Ω |
| intermediate | 3.64 | +0.011 | between |

Exploration drives the **flat** direction up to ratio ≈ 5 against the A⁻¹ target of 4 — in the right
ballpark (test bound 2.5–8.0), and clearly distinguished from canalization. So the *emergent*
bet-hedging mechanism (a shared random-disaster challenge, no imposed cost term) **does** push M
toward A⁻¹ at equilibrium, even though the *rate* of the climb is what V3/V5 can't cleanly resolve.

**The two-channel landscape theory** (`notes_landscape_fluctuation_benefit.md`,
`tests/test_landscape_channels.py`) — the result I'd highlight. The note derives, from the
geometric-mean growth rate, that only the **untrackable** part of landscape fluctuation benefits M,
and that **two distinct channels** set its shape, discriminated by the *directional structure* of
the fluctuation, not its amplitude:
- **Channel 1 (arithmetic-mean lag), anisotropic Ω → M ∝ Ω** (track the fluctuation).
- **Channel 2 (geometric-mean wipeout / bet-hedging), isotropic Ω → M ∝ A⁻¹** (hedge to the fixed
  landscape — the only predictable structure left).

IBM confirmation (A = diag(1,4); A⁻¹ target ratio 4; steep-favoring Ω):

| challenge σ | ratio m_flat/m_steep | channel |
|---|---|---|
| 0.4 (weak) | 1.53 | canalization (A⁻¹-shaped collapse) |
| 1.5 | 0.31 | **M follows Ω (steep)** — Channel 1 |
| 2.5 (strong) | 0.04 | **M ∝ Ω (steep), strongly** — Channel 1 |
| isotropic Ω, same σ | ≈ 4 | **M ∝ A⁻¹** — Channel 2 |

The test asserts the separation (anisotropic-steep Ω → ratio < 1; isotropic Ω → ratio > 1.5). **Both
channels confirmed, cleanly discriminated by Ω's directional structure.** This is the sharpest "we
predicted the *form* of second-order selection from the statistics of landscape variation" result
in the arm, and it's the biological half of the noise-dependent-α* story.

---

## 7. Sim A / B / C — the G-level cross-substrate spine (validated earlier, summarized here)

- **Sim A (G–A exponent):** G→A⁻¹ emerges **stably** at **α ≈ 0.84** (eig-slope of G vs A;
  `runs/sim_a`). Stable because evolution **generates** the covariance (residence-time accumulation
  in weakly-selected directions), it never inverts a noisy estimate — the GENERATE-vs-INVERT
  distinction that explains why biology reaches α≈1 where ML α=1 NaNs.
- **Sim B (Force-2 compression):** the spatial-correlation N* route was confounded by lag-load and
  did *not* show compression; the **incoherent-noise (errors-in-variables) route resolved it** —
  `sim_b_compression.py`: G-anisotropy compresses **monotonically toward isotropy** as noise rises:
  **1.51 → 1.44 → 1.32 → 1.17 → 1.06** (env_amp 0 → 2.5), G–A alignment flat. Force-2 (gradient-noise
  → α→½) is real and demonstrable; the biological reading is environmental variance / regression
  dilution = the exact twin of ML's empirical-Fisher mis-estimation.
- **Sim C (CMA-ES):** CMA-ES reaches C→A⁻¹ **stably** (cos > 0.9, eig-slope ≈ −1) — evolution's
  *generative* route to the full inverse, the cleanest "stable C⁻¹ the way biology does it."

Together these are the medium-timescale (G) cross-substrate validation: the **noise-dependent α\***
law shows up in biology as M/G-anisotropy compression as N* falls, mirroring ML's α=1→½
destabilization — one law, two substrates.

---

## 8. The cross-substrate parallel (why both arms tell the same story)

Lay the two reports side by side and a single structure appears:

| timescale | biology (this arm) | optimization (companion report) |
|---|---|---|
| fast `z̄` (breeder's eq / gradient step) | exact | exact |
| medium `G` (genetic covariance / preconditioner) | **clean** (closure cos>0.99, Σ_∞∝A⁻¹, Force-2 compression, CMA-ES→A⁻¹) | **clean** (SOAP whitening works; α*(N*) the optimum) |
| slow `M` (mutational architecture / learned source) | **unconfirmed as a process** — too slow to observe; exploration drift cos ≈ −0.3 | **unconfirmed as a self-tuner** — `evolve_M` targets A⁻¹ regardless of N*, the wrong optimizer objective |

So the project's two genuinely hard, genuinely unfinished pieces are *the same piece*: **the slow,
second-order evolution of the mutational architecture / preconditioner source M.** In biology it's
too slow to watch; in optimization it optimizes the wrong (exploration) objective. The fast/medium
correspondence — the part the paper actually rests on — is solid in both.

---

## 9. What this means for the plan, concretely

- **The keystone "M evolves to A⁻¹" is supported at the level of fixed points, regime structure, and
  the two channels — but not as an observed slow dynamical process.** Report it as: the *load
  surface and its fixed points* are validated; the *emergent bet-hedging mechanism* reaches the A⁻¹
  ratio at equilibrium; the *rate* of the second-order flow is below the resolution of the IBM at
  accessible compute. Don't claim a watched M→A⁻¹ trajectory in the exploration regime; V3/V5 don't
  support it.
- **The two strongest, most defensible biological results** are (i) the `(z̄,G)` closure
  `G=Ĝ(M)`, `Σ_∞∝A⁻¹` up to a single ~2.8× Gaussian-closure constant (V2), and (ii) the **two-channel
  landscape law** (anisotropic Ω → track; isotropic → hedge to A⁻¹), IBM-confirmed. These are the
  publishable spine of the arm.
- **V3-exploration and V5-adiabaticity need either much larger scale or a reframing.** The honest
  options: scale the burst sweeps far up (more replicates, more generations, tuned ε) to try to
  resolve the slow exploration drift, OR present the exploration result via the *equilibrium ratio*
  (three_regimes / channels) rather than the *drift-direction cosine*, which is where the signal is.
- **Sim A's "M-evolution too slow to observe" should be stated up front**, not buried — it's the
  honest counterpart to the optimizer arm's "evolve_M targets the wrong objective," and together
  they make the clean claim: *the first two timescales transfer; the third (architecture/source M
  evolution) is the open problem on both sides.*

---

## 10. Status table (validated thresholds → result)

| component | claim | result | status |
|---|---|---|---|
| theory_load_surface | ĜAĜ=M, Σ_∞ Lyapunov, regime fixed points | exact <1e-9 | ✅ |
| evolvable_m | realized M == encoded M | <1e-6 | ✅ |
| V0 drift_estimator | recovery cos, metric necessity, split | >0.999 / >0.99 / >0.99 | ✅ |
| V1 fitness_scan | G diag, U vs −ℒ_eff, closure shape | cos>0.55, ratio>1.2 const | ✅ (scale const) |
| V2 closure | G∝Ĝ, Σ_∞∝A⁻¹, Gaussian closure | cos 0.99 / 0.96; scale ≈2.9; skew<0.4 | ✅ (✚ ~2.8× constant) |
| V3 static | canalization drift cos | 0.98 | ✅ |
| **V3 exploration** | exploration drift cos | **−0.30** | ⚠️ unresolved at scale |
| V4 curl | aligned curl small | <0.35 | ✅ |
| **V5 adiabatic** | cos improves as ε→0 | **degrades 0.98→−0.91** | ⚠️ not confirmed (prod sweep) |
| V6 trajectory | SDE≈IBM (static) | sliced-W 0.011 | ✅ (static) |
| three_regimes | exploration ratio→A⁻¹ (target 4) | 4.97 | ✅ |
| landscape channels | aniso→track Ω, iso→A⁻¹ | 0.31/0.04 vs ≈4 | ✅ (the headline) |
| Sim A | G→A⁻¹ stable | α≈0.84 | ✅ |
| Sim B compression | Force-2 (anisotropy↓ as noise↑) | 1.51→1.06 | ✅ (incoherent route) |
| Sim C | CMA-ES→A⁻¹ stable | cos>0.9 | ✅ |
| **slow M-evolution as a process** | M dynamically evolves to A⁻¹ | **too slow to observe** | ⚠️ conjecture |

(Provenance: every threshold above is taken **directly from the assertions in `tests/test_*.py`**
and every number from the **production CSVs in `runs/`** — these are the ground truth, independent
of a suite re-run. The ⚠️ rows are production-CSV findings the pilot-config unit tests are
deliberately written *not* to hard-assert (their docstrings say so: V3/V5 unit tests check only a
robust/weak monotone at small scale, leaving the clean exploration-drift cosine to the burst sweep —
which is where it doesn't hold). A full CPU re-run of the suite confirmed green:
**60 passed (12m49s)**. Note this is *consistent* with the ⚠️ rows — those unit tests pass because
they assert only weak/robust monotones at pilot scale, so "60 passed" and "the production
exploration-drift cosine is −0.30 / V5 degrades as ε→0" are both true.)

## Artifacts
`sim/{evolvable_m,theory_load_surface,drift_estimator,v1_fitness_scan…v6_trajectory,three_regimes,
sim_a_exponent,sim_b_compression,sim_c_naturalgrad,directional_alignment,exploration_scaling}.py`;
`tests/test_*` (matching); `runs/{v2_closure,v3_drift,v4_curl,v5_adiabatic,v6_trajectory,
three_regimes,sim_a,sim_b,sim_b_compression}/`; notes
`notes_landscape_fluctuation_benefit.md`, `notes_evolution_vs_ml.md`.

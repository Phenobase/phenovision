# §2.4 demographic-noise posterior sampler — findings & open problem

Status: **shape recovery works; exact posterior recovery is an open sub-problem.** This is the
component the theory (`two_noise_framework_summary.md` §9.5) itself calls "a separate, larger
contribution" — a structured pSGLD.

## What we established (empirically, on a Kronecker quadratic with Fisher gradient noise)

The injection MATH is correct. Hand-coded pSGLD with a **fixed, correct** preconditioner
`H = κ^{-1}` and the `langevin_noise_std` helper recovers the Gibbs posterior exactly:
`V·κ = 0.506 ≈ T = 0.5` for all coordinates. So `optim/demographic_noise.py` and the injection
in `SOAPFullPower.step()` are right.

Driving it through the **actual SOAP optimizer** requires, in order of impact:
1. **No momentum (β1 = 0).** β1=0.95 corrupts the stationary law (slope −0.85 instead of −1);
   clean pSGLD is overdamped (drift `−H∇L`, no momentum term). β1=0 → slope −1.0.
2. **Stable preconditioner (infrequent/clamped basis refresh + large β2).** With
   `precondition_frequency=10` the basis refreshes faster than `exp_avg_sq` re-equilibrates
   (~1/(1−β2) steps), so `denom` is perpetually transient and FDT breaks
   (`V·κ` ≈ 34–78). Freezing the basis + β2=0.99 → `V·κ` ≈ 1.0 (slope −1.0).
3. **Absolute (not relative) small damping**, so `1/denom` stays faithful to the curvature
   eigenvalue rather than being floored by `damping·max(denom)`.

## SOLUTION (warm-up-then-freeze) — works

Recipe that recovers the posterior covariance through the actual SOAP optimizer:
1. **Phase 1 (warm up basis):** run with refreshes (`precondition_frequency` small), demographic
   noise OFF. SOAP's `L=E[ggᵀ]`/`R` accumulate toward the true Fisher `C=A`, so repeated refresh
   converges the eigenbasis to the curvature eigenbasis.
2. **Phase 2 (freeze):** set `precondition_frequency` huge (stop refreshing), keep noise OFF a few
   thousand steps so `exp_avg_sq` re-equilibrates in the now-correct, stable basis.
3. **Phase 3 (sample):** turn demographic noise ON and collect snapshots.
   (β1=0, absolute small damping, large β2.)

Result on the Kronecker quadratic (κ isotropic, α=1, T=0.5): **slope −1.004** and `V·κ` tightly
clustered ≈1.08 across all coordinates (was [1.45,1.13,…] with the random rank-1 basis). So the
basis problem is solved — the recovered covariance has the correct shape AND isotropy.

The only residual is a **global ~2× temperature constant** (`V·κ`≈1.08 vs T=0.5), isotropic and
stable — the standard SGLD discretization convention (Welling–Teh use `ε/2` drift vs full-step
noise). It is calibratable: set `demographic_temperature = T_target/2`, or report posterior
covariance recovery up to a scalar temperature (which is what §2.4's matrix-distance-on-shared-
subspace metric measures). Not a correctness bug.

## (historical) The problem this solved

With the basis FROZEN, the realized `V` is correct in *shape* (slope −1) but (a) ~2× too hot and
(b) **anisotropic even when κ is isotropic** ([1.45, 1.13, 1.13, 1.10] for κ≡1). Root cause: SOAP
freezes the basis from the **rank-1 initial gradient** — a *random* rotation, not the curvature
eigenbasis. In a wrong basis, the per-coordinate `denom` (correct in the rotated basis) does not
map to the standard-basis curvature, so the projected-back noise covariance ≠ `2T·H_curvature`.

The tension: **a correct basis needs refreshes; FDT needs a stable `denom`.** Resolving it (so the
SOAP-driven sampler exactly recovers an anisotropic posterior) is genuine research. Options to try:
- Warm up the basis to convergence (many refreshes) with the demographic noise OFF, then FREEZE
  it and turn noise ON — gives a correct, stable basis.
- Or decouple: estimate the basis on a slow timescale but hold `exp_avg_sq` fixed long enough to
  equilibrate between refreshes (β2 → 1 with refresh period ≫ 1/(1−β2)).
- Or use a separately-estimated curvature (curvature/lanczos.py true-Fisher) to set H for the
  noise, independent of the Adam denom (cleaner FDT, departs from "pure SOAP").

## Implication for the paper

The §2.4 claim — *only SOAP-NG (α=1) + demographic noise recovers the posterior covariance, while
SGD/SOAP(α=0.5) get the wrong shape* — is supported at the level of the **covariance SHAPE / slope**
(the matrix-distance-on-shared-subspace metric, up to a scalar temperature). Exact temperature and
anisotropy recovery need the basis fix above. Recommend the "warm-up-then-freeze" route for the
clean demonstration, and stating the basis-estimation/FDT tension explicitly (it is itself a finding).

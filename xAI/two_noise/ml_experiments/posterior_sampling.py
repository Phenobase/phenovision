"""§2.4 — the posterior-sampling demonstration (the FDT payoff).

Claim: SOAP-NG (alpha=1) + demographic noise samples the Bayesian posterior covariance, while
SGD / SOAP (alpha=0.5) recover the wrong shape, because they sample V_stat ∝ C^{-alpha} not the
true C^{-1} posterior scale. The demographic term is the FDT-restoring noise SGD lacks
(framework §9.5).

Toy: multinomial logistic regression with a Gaussian prior. The weight is a 2D matrix W (d x K)
so the optimizer takes its Kronecker (precond_power=alpha) path and the demographic hook applies.
We use the FULL-batch potential U(W) = NLL(W) + ||W||^2/(2*prior_var); all stochasticity is the
demographic injection -> pure preconditioned SGLD. NumPyro NUTS gives the reference posterior.

Sampler recipe (from notes_posterior_sampling.md): beta1=0 (overdamped), warm up the SOAP basis
with refreshes (noise off) so it converges to the curvature eigenbasis, then FREEZE it and turn
the demographic noise on so the per-coordinate denom equilibrates in a correct, stable basis ->
FDT holds. Absolute small damping. The realized temperature carries a ~2x SGLD discretization
constant; we therefore report posterior-covariance recovery as SHAPE (eigenvalue scaling /
matrix cosine), which is what the framework's "matrix distance on the shared subspace" measures.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # NUTS reference on CPU jax (this is the torch env)

import torch  # noqa: E402

import sys  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optim.soap_full_power import SOAPFullPower  # noqa: E402


# --------------------------------------------------------------------------- data + potential
@dataclass
class LogRegData:
    X: torch.Tensor   # (n, d)
    y: torch.Tensor   # (n,) in {0..K-1}
    K: int
    prior_var: float


def make_logreg_data(n=400, d=5, K=3, prior_var=1.0, seed=0) -> LogRegData:
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d, generator=g, dtype=torch.float64)
    W_true = torch.randn(d, K, generator=g, dtype=torch.float64)
    logits = X @ W_true
    probs = torch.softmax(logits, dim=1)
    y = torch.multinomial(probs, 1, generator=g).squeeze(1)
    return LogRegData(X=X, y=y, K=K, prior_var=prior_var)


def neg_log_posterior(W, data: LogRegData):
    """U(W) = sum_i CE_i + ||W||^2 / (2 prior_var)  (unnormalized negative log posterior)."""
    logits = data.X @ W
    ll = torch.nn.functional.cross_entropy(logits, data.y, reduction="sum")
    prior = (W ** 2).sum() / (2.0 * data.prior_var)
    return ll + prior


# --------------------------------------------------------------------------- reference posterior
def numpyro_reference(data: LogRegData, num_warmup=800, num_samples=2000, seed=0):
    """Gold-standard posterior mean + covariance over vec(W) via NUTS."""
    import jax, jax.numpy as jnp
    import numpyro, numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS

    Xj = jnp.asarray(data.X.numpy()); yj = jnp.asarray(data.y.numpy())
    d, K = data.X.shape[1], data.K

    def model(X, y):
        W = numpyro.sample("W", dist.Normal(jnp.zeros((d, K)),
                                            jnp.sqrt(data.prior_var)).to_event(2))
        numpyro.sample("y", dist.Categorical(logits=X @ W), obs=y)

    mcmc = MCMC(NUTS(model), num_warmup=num_warmup, num_samples=num_samples, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), Xj, yj)
    W = np.asarray(mcmc.get_samples()["W"]).reshape(num_samples, -1)  # (S, d*K)
    return W.mean(0), np.cov(W, rowvar=False)


# --------------------------------------------------------------------------- optimizer samplers
def run_sampler(data: LogRegData, alpha, demographic, T=1.0, lr=2e-3, seed=0,
                warmup=4000, freeze_after=True, sample_steps=20000, thin=10):
    """Optimizer-as-sampler. Returns (mean, cov) over vec(W) from post-warmup snapshots.

    alpha: precond_power (0 -> SGD-like isotropic, 0.5 -> whitening, 1.0 -> natural gradient).
    demographic: inject the §9.5 noise (the FDT-restoring term). With demographic=False and
    full-batch gradients the chain just descends to the MAP (no posterior exploration).
    """
    d, K = data.X.shape[1], data.K
    W = torch.nn.Parameter(torch.zeros(d, K, dtype=torch.float64))
    demo_gen = torch.Generator().manual_seed(seed + 7)
    opt = SOAPFullPower([W], lr=lr, betas=(0.0, 0.99), precond_power=alpha,
                        damping=1e-5, relative_damping=False, precondition_frequency=10,
                        weight_decay=0.0, eps=1e-12,
                        demographic_noise=False, demographic_temperature=T,
                        demographic_generator=demo_gen, demographic_warmup=0)
    grp = opt.param_groups[0]

    def closure_step():
        opt.zero_grad()
        U = neg_log_posterior(W, data)
        U.backward()
        opt.step()

    # Phase 1: warm up basis (refreshes on, noise off) -> converge to MAP + curvature eigenbasis.
    for _ in range(warmup):
        closure_step()
    # Phase 2: freeze the basis so denom equilibrates in a stable, correct basis (FDT).
    if freeze_after:
        grp["precondition_frequency"] = 10 ** 9
        for _ in range(2000):
            closure_step()
    # Phase 3: turn on the demographic noise and sample.
    grp["demographic_noise"] = demographic
    samples = []
    for t in range(sample_steps):
        closure_step()
        if t % thin == 0:
            samples.append(W.detach().reshape(-1).clone().numpy())
    S = np.stack(samples)
    return S.mean(0), np.cov(S, rowvar=False)


# --------------------------------------------------------------------------- metrics
def covariance_shape_recovery(cov_sampler, cov_ref):
    """How well the sampler's covariance matches the reference posterior covariance, in SHAPE.

    Returns: matrix cosine <C_s, C_ref>/(||.||||.||) (1 = same shape up to scale); and the
    log-log slope of eig(C_sampler) vs eig(C_ref) paired by descending order (1 = same scaling).
    """
    a = cov_sampler / np.linalg.norm(cov_sampler)
    b = cov_ref / np.linalg.norm(cov_ref)
    cos = float((a * b).sum())
    es = np.sort(np.linalg.eigvalsh(cov_sampler))[::-1]
    er = np.sort(np.linalg.eigvalsh(cov_ref))[::-1]
    keep = (es > 1e-12) & (er > 1e-12)
    slope = float(np.polyfit(np.log(er[keep]), np.log(es[keep]), 1)[0]) if keep.sum() > 1 else np.nan
    return cos, slope


def run_comparison(data: LogRegData, seed=0, **kw):
    """Compare four samplers against the NUTS reference; return a tidy list of dicts."""
    ref_mean, ref_cov = numpyro_reference(data, seed=seed)
    specs = [
        ("SGD (no demo)",        0.0, False),
        ("SOAP a=0.5 + demo",    0.5, True),
        ("SOAP-NG a=1 (no demo)", 1.0, False),
        ("SOAP-NG a=1 + demo",   1.0, True),
    ]
    rows = []
    for name, alpha, demo in specs:
        m, c = run_sampler(data, alpha=alpha, demographic=demo, seed=seed, **kw)
        cos, slope = covariance_shape_recovery(c, ref_cov)
        rows.append({"sampler": name, "alpha": alpha, "demographic": demo,
                     "cov_cosine_to_posterior": round(cos, 4),
                     "cov_eig_slope_vs_posterior": round(slope, 4),
                     "mean_err": round(float(np.linalg.norm(m - ref_mean)), 4),
                     "total_var": round(float(np.trace(c)), 6)})
    return rows, ref_cov


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400); ap.add_argument("--d", type=int, default=5)
    ap.add_argument("--K", type=int, default=3); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample-steps", type=int, default=20000)
    ap.add_argument("--out", default="runs/posterior/comparison.csv")
    a = ap.parse_args()
    data = make_logreg_data(n=a.n, d=a.d, K=a.K, seed=a.seed)
    rows, _ = run_comparison(data, seed=a.seed, sample_steps=a.sample_steps)
    import csv
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"[posterior] wrote {a.out}")
    for r in rows:
        print(f"  {r['sampler']:24s} cov_cosine={r['cov_cosine_to_posterior']:.3f} "
              f"eig_slope={r['cov_eig_slope_vs_posterior']:.3f} mean_err={r['mean_err']:.3f}")

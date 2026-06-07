"""§2.4 posterior-sampling tests — the ROBUST claims only.

Demonstrated robustly: the demographic (FDT-restoring) term is what enables posterior sampling.
Without it the optimizer collapses to the MAP (~zero variance); with it the chain explores and
its covariance is positively aligned with the true posterior. The sharp 'alpha=1 is best' claim
needs true-Fisher preconditioning + temperature calibration (see notes_posterior_sampling.md) and
is intentionally NOT asserted here.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

from ml_experiments.posterior_sampling import (make_logreg_data, run_sampler,
                                               numpyro_reference, covariance_shape_recovery)


def test_demographic_term_enables_sampling():
    """No-demo samplers collapse to the MAP (tiny variance); demo samplers explore."""
    data = make_logreg_data(n=300, d=4, K=3, seed=0)
    _, cov_nodemo = run_sampler(data, alpha=1.0, demographic=False, seed=0,
                                warmup=2000, sample_steps=4000, thin=5)
    _, cov_demo = run_sampler(data, alpha=1.0, demographic=True, seed=0,
                              warmup=2000, sample_steps=4000, thin=5)
    var_nodemo = float(np.trace(cov_nodemo))
    var_demo = float(np.trace(cov_demo))
    assert var_demo > 20 * var_nodemo, f"demo var {var_demo:.4f} not >> no-demo {var_nodemo:.4f}"
    assert var_demo > 1e-2, f"demo sampler barely moved: {var_demo:.4f}"


def test_demo_sampler_covariance_aligns_with_posterior():
    """A demographic sampler's covariance is positively aligned with the NUTS posterior
    (matrix cosine well above chance), unlike SGD-no-demo."""
    data = make_logreg_data(n=300, d=4, K=3, seed=0)
    _, ref_cov = numpyro_reference(data, num_warmup=400, num_samples=1000, seed=0)
    _, cov_demo = run_sampler(data, alpha=0.5, demographic=True, seed=0,
                              warmup=2000, sample_steps=6000, thin=5)
    cos, _ = covariance_shape_recovery(cov_demo, ref_cov)
    _, cov_sgd = run_sampler(data, alpha=0.0, demographic=False, seed=0,
                             warmup=2000, sample_steps=6000, thin=5)
    cos_sgd, _ = covariance_shape_recovery(cov_sgd, ref_cov)
    assert cos > 0.4, f"demo sampler cov cosine to posterior too low: {cos:.3f}"
    assert cos > cos_sgd, f"demo cosine {cos:.3f} not better than SGD-no-demo {cos_sgd:.3f}"

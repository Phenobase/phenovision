"""True (sampled-label) Fisher vs empirical Fisher — the empirical-vs-true-Fisher caveat made
operational (framework §3; Morwani et al. 2024 §2.1.2).

- empirical Fisher: per-sample gradients on the *observed* labels (what SOAP's L=E[ggᵀ] uses).
- true Fisher: gradients on labels *sampled from the model's predictive distribution* — equals the
  Hessian (curvature) at a correctly-specified optimum, the C=A identity the framework relies on.

`sampled_label_gradient` produces a gradient suitable for the optimizer's `_soap_precond_grad`
hook so the preconditioner statistics can be driven by the true Fisher instead of empirical.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def sampled_label_gradient(model, inputs, generator=None, loss_kind="classification"):
    """Gradient of the loss on labels SAMPLED from the model's output distribution (true Fisher).

    classification: sample y ~ Categorical(softmax(logits)); regression: y ~ N(mean, 1).
    Returns a dict {param: grad} (per-parameter) for assignment to p._soap_precond_grad.
    """
    model.zero_grad(set_to_none=True)
    logits = model(inputs)
    if loss_kind == "classification":
        with torch.no_grad():
            probs = F.softmax(logits, dim=-1)
            y = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
        loss = F.cross_entropy(logits, y)
    else:
        with torch.no_grad():
            y = logits.detach() + torch.randn(logits.shape, generator=generator,
                                              device=logits.device, dtype=logits.dtype)
        loss = 0.5 * F.mse_loss(logits, y)
    grads = torch.autograd.grad(loss, [p for p in model.parameters() if p.requires_grad])
    return {p: g.detach() for p, g in zip([p for p in model.parameters() if p.requires_grad], grads)}


def empirical_gradient(model, inputs, targets, loss_fn):
    """Gradient on the OBSERVED labels (empirical Fisher direction)."""
    model.zero_grad(set_to_none=True)
    loss = loss_fn(model(inputs), targets)
    grads = torch.autograd.grad(loss, [p for p in model.parameters() if p.requires_grad])
    return {p: g.detach() for p, g in zip([p for p in model.parameters() if p.requires_grad], grads)}


def assign_precond_grad(model, grad_dict):
    """Attach grads to params as `_soap_precond_grad` for SOAPFullPower to use as preconditioner
    statistics. Pass grad_dict=None to clear."""
    for p in model.parameters():
        if grad_dict is None:
            if hasattr(p, "_soap_precond_grad"):
                delattr(p, "_soap_precond_grad")
        elif p in grad_dict:
            p._soap_precond_grad = grad_dict[p]

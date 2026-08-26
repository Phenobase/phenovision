"""Curvature estimation: top-k Hessian eigenpairs (Lanczos), Hutchinson trace, principal
angles. Used by §2.3 (stationary-variance) and §2.5 (curvature tracking).

Implemented from scratch (Lanczos + Hutchinson via Hessian-vector products) so it has no
external dependency beyond torch; PyHessian is vendored as a cross-check if desired. HVPs use
double-backprop on a supplied loss closure.
"""

from __future__ import annotations

from typing import Callable, List

import torch
from torch import Tensor


def _flatten(vs: List[Tensor]) -> Tensor:
    return torch.cat([v.reshape(-1) for v in vs])


def _hvp(loss_fn: Callable[[], Tensor], params: List[Tensor], vec: Tensor) -> Tensor:
    """Hessian-vector product H @ vec for the scalar loss returned by loss_fn()."""
    loss = loss_fn()
    grads = torch.autograd.grad(loss, params, create_graph=True)
    g = _flatten(grads)
    gv = (g * vec).sum()
    hv = torch.autograd.grad(gv, params, retain_graph=False)
    return _flatten(hv).detach()


def lanczos_tridiag(loss_fn, params, k: int, generator=None):
    """k steps of Lanczos on the Hessian; returns (alphas, betas) of the tridiagonal T."""
    params = [p for p in params if p.requires_grad]
    dim = sum(p.numel() for p in params)
    device = params[0].device
    dtype = params[0].dtype
    q = torch.randn(dim, generator=generator, device=device, dtype=dtype)
    q = q / q.norm()
    alphas, betas, Q = [], [], [q]
    beta = 0.0
    q_prev = torch.zeros_like(q)
    for _ in range(k):
        w = _hvp(loss_fn, params, q)
        alpha = torch.dot(w, q)
        w = w - alpha * q - beta * q_prev
        # full reorthogonalization (k is small)
        for qj in Q:
            w = w - torch.dot(w, qj) * qj
        beta = w.norm()
        alphas.append(alpha)
        if beta < 1e-8 or len(alphas) == k:
            break
        q_prev = q
        q = w / beta
        betas.append(beta)
        Q.append(q)
    return torch.stack(alphas), (torch.stack(betas) if betas else torch.tensor([], device=device)), Q


def top_k_eigenvalues(loss_fn, params, k: int = 20, n_iter: int = None, generator=None):
    """Top-k Hessian eigenvalue estimates via Lanczos (Ritz values)."""
    n_iter = n_iter or max(2 * k, 20)
    alphas, betas, _ = lanczos_tridiag(loss_fn, params, n_iter, generator)
    T = torch.diag(alphas)
    if betas.numel():
        idx = torch.arange(betas.numel())
        T[idx, idx + 1] = betas
        T[idx + 1, idx] = betas
    evals = torch.linalg.eigvalsh(T)
    return torch.sort(evals, descending=True).values[:k]


def top_k_eigenpairs(loss_fn, params, k: int = 10, n_iter: int = None, generator=None):
    """Top-k Hessian (Ritz value, Ritz vector) pairs via Lanczos. Returns (evals (k,),
    evecs (dim, k)) sorted by descending eigenvalue. Ritz vectors = Q @ eig(T)."""
    params = [p for p in params if p.requires_grad]
    n_iter = n_iter or max(3 * k, 30)
    alphas, betas, Q = lanczos_tridiag(loss_fn, params, n_iter, generator)
    m = alphas.numel()
    T = torch.diag(alphas)
    if betas.numel():
        idx = torch.arange(betas.numel())
        T[idx, idx + 1] = betas
        T[idx + 1, idx] = betas
    evals, S = torch.linalg.eigh(T)                 # ascending; S columns = eigvecs in Lanczos basis
    Qm = torch.stack(Q[:m], dim=1)                  # (dim, m)
    ritz = Qm @ S                                   # (dim, m) Ritz vectors
    order = torch.argsort(evals, descending=True)[:k]
    return evals[order], ritz[:, order]


def hutchinson_trace(loss_fn, params, n_probes: int = 100, generator=None) -> float:
    """Stochastic estimate of tr(H) via Rademacher probes."""
    params = [p for p in params if p.requires_grad]
    dim = sum(p.numel() for p in params)
    device = params[0].device
    dtype = params[0].dtype
    total = 0.0
    for _ in range(n_probes):
        v = torch.randint(0, 2, (dim,), generator=generator, device=device, dtype=dtype) * 2 - 1
        hv = _hvp(loss_fn, params, v)
        total += torch.dot(hv, v).item()
    return total / n_probes


def principal_angles(U: Tensor, V: Tensor) -> Tensor:
    """Principal angles (radians, ascending) between the column spaces of U and V.

    U: (d, k1), V: (d, k2) with orthonormal-ish columns. Returns min(k1,k2) angles; the first
    (smallest) angle ~0 means the leading directions align. Used for §2.5 eigenbasis alignment.
    """
    Uq, _ = torch.linalg.qr(U)
    Vq, _ = torch.linalg.qr(V)
    s = torch.linalg.svdvals(Uq.t() @ Vq).clamp(-1.0, 1.0)
    return torch.arccos(s)

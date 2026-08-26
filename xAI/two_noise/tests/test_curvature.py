"""Curvature module tests: Lanczos vs dense eigh, Hutchinson trace, principal angles,
and the true-vs-empirical Fisher distinction."""
import torch
import torch.nn as nn

from curvature.lanczos import top_k_eigenvalues, hutchinson_trace, principal_angles
from curvature.true_fisher import sampled_label_gradient, empirical_gradient


def _quadratic_loss(A, theta):
    return lambda: 0.5 * theta @ A @ theta


def test_lanczos_topk_matches_dense_eigh():
    torch.manual_seed(0)
    d = 30
    M = torch.randn(d, d, dtype=torch.float64)
    A = M @ M.t() + 0.1 * torch.eye(d, dtype=torch.float64)  # PSD known Hessian
    theta = nn.Parameter(torch.randn(d, dtype=torch.float64))
    gen = torch.Generator().manual_seed(1)
    ritz = top_k_eigenvalues(_quadratic_loss(A, theta), [theta], k=5, n_iter=60, generator=gen)
    true_top = torch.linalg.eigvalsh(A).flip(0)[:5]
    rel = (ritz - true_top).abs() / true_top
    assert rel.max() < 1e-3, f"top-5 eigvals rel err {rel.max():.2e}: {ritz} vs {true_top}"


def test_hutchinson_trace():
    torch.manual_seed(0)
    d = 40
    M = torch.randn(d, d, dtype=torch.float64)
    A = M @ M.t() + 0.1 * torch.eye(d, dtype=torch.float64)
    theta = nn.Parameter(torch.zeros(d, dtype=torch.float64))
    gen = torch.Generator().manual_seed(2)
    tr = hutchinson_trace(_quadratic_loss(A, theta), [theta], n_probes=300, generator=gen)
    assert abs(tr - torch.trace(A).item()) / torch.trace(A).item() < 0.1


def test_principal_angles():
    torch.manual_seed(0)
    d = 20
    Q, _ = torch.linalg.qr(torch.randn(d, d, dtype=torch.float64))
    U = Q[:, :3]
    # identical subspace -> angles ~0
    ang_same = principal_angles(U, U.clone())
    assert ang_same.max() < 1e-5
    # orthogonal subspace -> angles ~pi/2
    V = Q[:, 3:6]
    ang_orth = principal_angles(U, V)
    assert (ang_orth - torch.pi / 2).abs().max() < 1e-4


def test_true_vs_empirical_fisher_differ_off_optimum():
    """Away from the optimum the sampled-label (true) Fisher gradient differs from the empirical
    one; averaged over many label samples the true-Fisher gradient mean ~ 0 (labels ~ model),
    while the empirical gradient on a fixed (wrong) label is systematically nonzero."""
    torch.manual_seed(0)
    model = nn.Linear(8, 4)
    x = torch.randn(16, 8)
    y = torch.randint(0, 4, (16,))
    emp = empirical_gradient(model, x, y, nn.CrossEntropyLoss())
    gen = torch.Generator().manual_seed(3)
    # average sampled-label gradient over many draws
    acc = {p: torch.zeros_like(p) for p in model.parameters()}
    N = 200
    for _ in range(N):
        gd = sampled_label_gradient(model, x, generator=gen)
        for p in acc:
            acc[p] += gd[p] / N
    w = model.weight
    emp_norm = emp[w].norm().item()
    true_mean_norm = acc[w].norm().item()
    # empirical (fixed labels) is clearly nonzero; mean sampled-label gradient is much smaller
    assert emp_norm > 1e-3
    assert true_mean_norm < 0.5 * emp_norm, f"true-Fisher mean {true_mean_norm:.3e} not < empirical {emp_norm:.3e}"

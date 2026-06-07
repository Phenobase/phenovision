"""GPU coexistence smoke test: PyTorch and JAX must both see the GPU in one process.

Decides the single-env vs split-env question (CONVENTIONS / plan Step 0). If JAX fails to
initialize the GPU here (e.g. cudnn 9.1 vs 9.23 clash after torch downgraded it), we split
into two_noise_torch + two_noise_jax. Also runs a one-step SOAPFullPower on CUDA to confirm
the optimizer works on GPU.
"""
import sys


def main():
    ok = True

    print("=== PyTorch CUDA ===")
    import torch
    print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name(0))
        x = torch.randn(512, 512, device="cuda")
        y = (x @ x.t()).sum().item()
        print("torch matmul on cuda ok, sum=", y)
    else:
        ok = False
        print("FAIL: torch.cuda not available")

    print("\n=== JAX GPU ===")
    try:
        import jax
        import jax.numpy as jnp
        devs = jax.devices()
        print("jax", jax.__version__, "devices:", devs)
        gpu_devs = [d for d in devs if d.platform == "gpu"]
        if gpu_devs:
            a = jnp.ones((512, 512))
            z = float((a @ a.T).sum())
            print("jax matmul on gpu ok, sum=", z)
        else:
            ok = False
            print("FAIL: jax sees no GPU device")
    except Exception as e:
        ok = False
        print(f"FAIL: jax GPU init raised {type(e).__name__}: {e}")

    print("\n=== SOAPFullPower one step on CUDA ===")
    try:
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from optim.soap_full_power import SOAPFullPower
        p = torch.nn.Parameter(torch.randn(64, 32, device="cuda"))
        opt = SOAPFullPower([p], lr=1e-3, precond_power=1.0, damping=1e-2)
        for _ in range(3):
            p.grad = torch.randn_like(p)
            opt.step()
        print("SOAPFullPower cuda step ok, param finite:", bool(torch.isfinite(p).all()))
    except Exception as e:
        ok = False
        print(f"FAIL: SOAPFullPower cuda raised {type(e).__name__}: {e}")

    print("\nRESULT:", "COEXISTENCE OK (single env works)" if ok else "COEXISTENCE FAILED (split envs)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

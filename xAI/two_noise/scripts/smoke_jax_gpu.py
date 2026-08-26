"""JAX-only GPU smoke test for the two_noise_jax env (Strand 2)."""
import sys
import jax, jax.numpy as jnp
print("jax", jax.__version__, "devices:", jax.devices())
gpu = [d for d in jax.devices() if d.platform == "gpu"]
if not gpu:
    print("FAIL: no GPU"); sys.exit(1)
# a real jit+vmap workload (mirrors the IBM sim pattern: vmap over replicates)
@jax.jit
def step(key, x):
    k1, k2 = jax.random.split(key)
    return x * 0.99 + 0.01 * jax.random.normal(k2, x.shape), k1
keys = jax.random.split(jax.random.PRNGKey(0), 64)
xs = jnp.ones((64, 128))
vstep = jax.vmap(step)
for _ in range(100):
    xs, keys = vstep(keys, xs)
print("vmap+jit on gpu ok, mean:", float(xs.mean()))
print("RESULT: JAX GPU OK")

#!/bin/bash
# Strand 2 env: JAX (GPU) only, no PyTorch, so cudnn stays at jax's required 9.23.
# The two strands meet only at CSV/figure time, so a separate env is painless.
set -eo pipefail
ENV=two_noise_jax
RUN="mamba run -n $ENV"

echo "=== create base ==="
mamba create -y -n "$ENV" -c conda-forge \
    python=3.11 numpy scipy pandas matplotlib pytest pyyaml tqdm

echo "=== jax[cuda12] + pure-python deps ==="
$RUN pip install --no-input "jax[cuda12]" cma

echo "=== import check ==="
$RUN python - <<'PY'
import importlib
for m in ["numpy","scipy","pandas","jax","cma"]:
    mod = importlib.import_module(m)
    print(f"OK {m:8s} {getattr(mod,'__version__','?')}")
PY

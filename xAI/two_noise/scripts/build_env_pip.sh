#!/bin/bash
# Resume the two_noise env build: pip-install the GPU stack into the already-created
# conda env. Uses `mamba run` (no bashrc activation) to avoid the BASHRCSOURCED pitfall.
# NOTE: deliberately NOT using `set -u` (it trips /etc/bashrc); use -eo pipefail only.
set -eo pipefail
ENV=two_noise
RUN="mamba run -n $ENV"

echo "=== PyTorch (cu124) ==="
$RUN pip install --no-input torch torchvision --index-url https://download.pytorch.org/whl/cu124

echo "=== JAX (cuda12) ==="
$RUN pip install --no-input "jax[cuda12]"

echo "=== remaining deps ==="
$RUN pip install --no-input numpyro laplace-torch pyhessian cma timm transformers safetensors

echo "=== DONE: import check ==="
$RUN python - <<'PY'
import importlib
for m in ["numpy","scipy","pandas","matplotlib","torch","jax","numpyro","laplace","pyhessian","cma","timm","transformers","einops","hydra"]:
    try:
        mod = importlib.import_module(m)
        print(f"OK   {m:14s} {getattr(mod,'__version__','?')}")
    except Exception as e:
        print(f"FAIL {m:14s} {type(e).__name__}: {e}")
PY
